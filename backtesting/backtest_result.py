"""
backtest_result.py
------------------
Immutable container for backtest outputs.

The ``BacktestResult`` bridges the backtester and the analytics layer:
calling ``to_prices_df()`` produces a DataFrame that can be passed
directly to ``TimeseriesAnalyzer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Immutable output of a backtest run.

    Attributes
    ----------
    nav : pd.Series
        Indexed performance (NAV) series starting at ``initial_capital``.
    returns : pd.Series
        Period portfolio returns.
    weights_history : pd.DataFrame
        Rows = dates, columns = assets, values = portfolio weights.
    turnover : pd.Series
        Per-period turnover (sum of absolute weight changes at rebalances,
        zero on non-rebalance dates).
    rebalance_dates : list[pd.Timestamp]
        Dates on which the portfolio was rebalanced.
    """

    nav: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    rebalance_dates: list[pd.Timestamp]

    def to_prices_df(self, name: str = "Strategy") -> pd.DataFrame:
        """Convert the NAV series to a single-column DataFrame.

        The output is directly compatible with
        ``analytics.timeseries_analyzer.TimeseriesAnalyzer(prices)``.

        Parameters
        ----------
        name : str
            Column name for the NAV series.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame with a DatetimeIndex.
        """
        return self.nav.to_frame(name=name)
