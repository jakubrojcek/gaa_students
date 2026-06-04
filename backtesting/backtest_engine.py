"""
backtest_engine.py
------------------
Core backtest loop for asset allocation strategies.

The ``Backtester`` receives a price DataFrame and a ``Strategy``, simulates
periodic rebalancing with weight drift and optional transaction costs, and
returns a ``BacktestResult`` whose NAV series plugs directly into
``TimeseriesAnalyzer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.backtest_result import BacktestResult
from backtesting.rebalance_schedule import RebalanceFreq, make_rebalance_dates
from backtesting.strategy_protocols import Strategy, Weights
from backtesting.utils import apply_transaction_cost, calc_turnover, drift_weights


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    Attributes
    ----------
    initial_capital : float
        Starting NAV level (default 100.0).
    transaction_cost_bps : float
        One-way transaction cost in basis points applied at each rebalance.
    rebalance_freq : str
        Pandas offset alias controlling how often the portfolio rebalances.
    long_short : bool
        If ``True``, drift long and short legs independently between
        rebalances so each leg's gross exposure is preserved.
    initial_weights_from_strategy : bool
        If ``True``, call ``strategy.compute_weights()`` on the first
        returns date to set the starting weights.  Otherwise start with
        equal weights across all assets.
    start_date : pd.Timestamp | None
        First date to include (inclusive).  ``None`` uses the earliest
        available date in the price data.
    end_date : pd.Timestamp | None
        Last date to include (inclusive).  ``None`` uses the latest
        available date.
    """

    initial_capital: float = 100.0
    transaction_cost_bps: float = 0.0
    rebalance_freq: RebalanceFreq | str = "ME"
    long_short: bool = False
    initial_weights_from_strategy: bool = False
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None


class Backtester:
    """General-purpose backtester for asset allocation strategies.

    Parameters
    ----------
    config : BacktestConfig | None
        Backtest settings.  Uses defaults when ``None``.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
    ) -> BacktestResult:
        """Execute a backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Price-level DataFrame with a ``DatetimeIndex``.  Each column
            represents an asset in the investment universe (e.g.
            ``"EQ_WL-i"``, ``"BD_US-i"``).  The backtester converts
            prices to returns internally via ``pct_change()``.
        strategy : Strategy
            Object implementing ``compute_weights(data, as_of,
            current_weights) -> Weights``.

        Returns
        -------
        BacktestResult
        """
        self._validate_inputs(prices, strategy)

        # --- Convert prices to returns -----------------------------------
        returns = prices.pct_change(fill_method=None)

        # --- Slice to requested window -----------------------------------
        start = self.config.start_date or returns.index[0]
        end = self.config.end_date or returns.index[-1]
        returns = returns.loc[start:end]

        # Drop leading row of NaNs from pct_change and any all-NaN rows
        returns = returns.dropna(how="all")

        # --- Determine rebalance schedule --------------------------------
        rebalance_dates = make_rebalance_dates(
            returns.index,
            self.config.rebalance_freq,
        )
        rebalance_set: set[pd.Timestamp] = set(rebalance_dates)

        # --- Initialise state --------------------------------------------
        n_assets = len(returns.columns)
        first_date = returns.index[0]

        if self.config.initial_weights_from_strategy:
            current_weights: Weights = strategy.compute_weights(
                prices.loc[:first_date],
                first_date,
                None,
            )
        else:
            current_weights: Weights = pd.Series(
                1.0 / n_assets,
                index=returns.columns,
            )

        nav_values: list[float] = []
        return_values: list[float] = []
        weight_records: list[pd.Series] = []
        turnover_values: list[float] = []
        nav = self.config.initial_capital

        # --- Main loop ---------------------------------------------------
        for date, row_returns in returns.iterrows():
            # Fill NaN returns with 0 for assets without data yet
            row_returns = row_returns.fillna(0.0)

            # Portfolio return for this period
            port_ret = float(current_weights @ row_returns)

            # Update NAV
            nav = nav * (1.0 + port_ret)

            # Drift weights
            current_weights = drift_weights(
                current_weights,
                row_returns,
                port_ret,
                long_short=self.config.long_short,
            )

            # Rebalance if scheduled
            period_turnover = 0.0
            if date in rebalance_set:
                target_weights = strategy.compute_weights(
                    prices.loc[:date],
                    date,
                    current_weights,
                )

                period_turnover = calc_turnover(current_weights, target_weights)

                # Apply transaction costs
                nav = apply_transaction_cost(
                    nav,
                    period_turnover,
                    self.config.transaction_cost_bps,
                )

                current_weights = target_weights

            # Record
            nav_values.append(nav)
            return_values.append(port_ret)
            weight_records.append(current_weights.copy())
            turnover_values.append(period_turnover)

        # --- Build result ------------------------------------------------
        nav_series = pd.Series(nav_values, index=returns.index, name="NAV")
        returns_series = pd.Series(
            return_values,
            index=returns.index,
            name="returns",
        )
        weights_df = pd.DataFrame(weight_records, index=returns.index)
        turnover_series = pd.Series(
            turnover_values,
            index=returns.index,
            name="turnover",
        )

        return BacktestResult(
            nav=nav_series,
            returns=returns_series,
            weights_history=weights_df,
            turnover=turnover_series,
            rebalance_dates=rebalance_dates,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        prices: pd.DataFrame,
        strategy: Strategy,
    ) -> None:
        """Guard-clause validation of ``run()`` arguments."""
        if not isinstance(prices, pd.DataFrame):
            raise TypeError(
                f"prices must be a pd.DataFrame, got {type(prices).__name__}."
            )
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError(
                f"prices must have a DatetimeIndex, got {type(prices.index).__name__}."
            )
        if prices.empty:
            raise ValueError("prices DataFrame is empty.")
        if not isinstance(strategy, Strategy):
            raise TypeError(
                "strategy must implement the Strategy protocol "
                "(must have a compute_weights method)."
            )
