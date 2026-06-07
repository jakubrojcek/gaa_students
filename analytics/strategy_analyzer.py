"""
strategy_analyzer.py
--------------------
Strategy-specific analytics that go beyond the pure performance metrics
in :mod:`analytics.timeseries_analyzer`.

These functions and the :class:`StrategyAnalyzer` class compute:
    - Active-return metrics vs a benchmark: tracking error, information ratio,
      hit rate, up/down capture ratios.
    - Running drawdown time series (for plotting).
    - Information coefficient between a signal and forward returns.
    - Accumulated transaction-cost dollar series.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from analytics.timeseries_analyzer import (
    _validate_datetime_index,
    infer_periods_per_year,
)


# ---------------------------------------------------------------------------
# Active-return metrics
# ---------------------------------------------------------------------------


def calc_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
) -> float:
    """Standard deviation of active (strategy − benchmark) returns.

    Parameters
    ----------
    returns : pd.Series
        Strategy period returns.
    benchmark_returns : pd.Series
        Benchmark period returns aligned on the same index.
    annualize : bool
        If True, scale by ``sqrt(periods_per_year)``.
    """
    _validate_datetime_index(returns)
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te: float = float(active.std(ddof=1))

    if annualize:
        te *= float(np.sqrt(infer_periods_per_year(aligned.index)))

    return te


def calc_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
) -> float:
    """Information ratio: mean active return / std active return.

    Parameters
    ----------
    returns : pd.Series
        Strategy period returns.
    benchmark_returns : pd.Series
        Benchmark period returns aligned on the same index.
    annualize : bool
        If True, scale by ``sqrt(periods_per_year)``.
    """
    _validate_datetime_index(returns)
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    std = float(active.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return float("nan")

    ir: float = float(active.mean()) / std
    if annualize:
        ir *= float(np.sqrt(infer_periods_per_year(aligned.index)))

    return ir


def calc_hit_rate(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Fraction of periods where the strategy return exceeds the benchmark."""
    _validate_datetime_index(returns)
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return float("nan")
    wins = (aligned.iloc[:, 0] > aligned.iloc[:, 1]).sum()
    return float(wins) / float(len(aligned))


def calc_capture_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    side: Literal["up", "down"] = "up",
) -> float:
    """Up- or down-capture ratio vs the benchmark.

    Up-capture: mean strategy return on benchmark-up periods / mean benchmark
    return on those same periods.  Down-capture is the symmetric measure
    over benchmark-down periods.
    """
    _validate_datetime_index(returns)
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["strategy", "benchmark"]

    if side == "up":
        mask = aligned["benchmark"] > 0
    elif side == "down":
        mask = aligned["benchmark"] < 0
    else:
        raise ValueError(f"side must be 'up' or 'down', got {side!r}.")

    subset = aligned.loc[mask]
    if subset.empty:
        return float("nan")

    bench_mean = float(subset["benchmark"].mean())
    if bench_mean == 0.0:
        return float("nan")

    return float(subset["strategy"].mean()) / bench_mean


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------


def calc_information_coefficient(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: Literal["pearson", "spearman"] = "spearman",
    rolling_window: int | None = None,
) -> float | pd.Series:
    """Correlation between a signal at t and the realised forward return.

    Parameters
    ----------
    signal : pd.Series
        Signal values indexed by date.
    forward_returns : pd.Series
        Realised returns over the holding period starting at each date.
        The caller is responsible for aligning — e.g. passing
        ``returns.shift(-1)`` if the signal at t is applied to ``r_{t+1}``.
    method : {"pearson", "spearman"}
        Correlation method.
    rolling_window : int | None
        When provided, returns a rolling correlation series of that window
        length; otherwise a single scalar correlation over the full sample.
    """
    _validate_datetime_index(signal)
    aligned = pd.concat(
        [signal.rename("signal"), forward_returns.rename("fwd")],
        axis=1,
        join="inner",
    ).dropna()

    if rolling_window is None:
        if len(aligned) < 2:
            return float("nan")
        return float(aligned["signal"].corr(aligned["fwd"], method=method))

    if rolling_window < 2:
        raise ValueError(f"rolling_window must be >= 2, got {rolling_window}.")
    return aligned["signal"].rolling(rolling_window).corr(aligned["fwd"])


# ---------------------------------------------------------------------------
# Time-series helpers for plotting
# ---------------------------------------------------------------------------


def calc_running_drawdown(returns: pd.Series) -> pd.Series:
    """Peak-to-current drawdown at every date (zero when at a new high)."""
    _validate_datetime_index(returns)
    cum = (1.0 + returns.fillna(0.0)).cumprod()
    peak = cum.cummax()
    return cum / peak - 1.0


def calc_accumulated_transaction_costs(
    turnover: pd.Series,
    cost_bps: float,
    nav: pd.Series,
    annualize: bool = False,
) -> pd.Series | float:
    """Running sum of dollar transaction costs through time.

    At each date the incremental cost is ``nav[t] * turnover[t] * cost_bps / 10_000``;
    by default this function returns the cumulative-sum series.  When
    *annualize* is True it instead returns a single scalar equal to the
    total accumulated cost divided by the number of years spanned by the
    aligned series (a per-year cost rate).

    Parameters
    ----------
    turnover : pd.Series
        Per-period turnover (sum of absolute weight changes) aligned on
        the same index as ``nav``.  Zero on non-rebalance dates.
    cost_bps : float
        One-way transaction cost in basis points.
    nav : pd.Series
        Portfolio NAV series.
    annualize : bool
        If True, return a scalar annualized cost (total cost / years).
        Otherwise return the cumulative cost time series.
    """
    _validate_datetime_index(nav)
    aligned = pd.concat([turnover.rename("to"), nav.rename("nav")], axis=1).dropna()
    period_cost = aligned["nav"] * aligned["to"] * (cost_bps / 10_000.0)
    cumulative = period_cost.cumsum().rename("accumulated_transaction_cost")

    if not annualize:
        return cumulative

    if cumulative.empty:
        return float("nan")
    total = float(cumulative.iloc[-1])
    n_years = len(aligned) / infer_periods_per_year(aligned.index)
    return total / n_years if n_years > 0 else float("nan")


# ---------------------------------------------------------------------------
# StrategyAnalyzer
# ---------------------------------------------------------------------------


class StrategyAnalyzer:
    """Bundle strategy-vs-benchmark analytics for a single backtest result.

    Parameters
    ----------
    nav : pd.Series
        Strategy NAV series (from ``BacktestResult.nav``).
    benchmark_nav : pd.Series
        Benchmark NAV aligned on the same index.
    weights_history : pd.DataFrame
        Per-date target weights (from ``BacktestResult.weights_history``).
    turnover : pd.Series
        Per-date turnover (from ``BacktestResult.turnover``).
    signal_history : pd.Series | None
        Optional per-date signal values.  When supplied, the information
        coefficient and signal-vs-weight overlays become available.
    cost_bps : float
        Transaction-cost assumption used by the backtest, for the
        accumulated-cost calculation.
    """

    def __init__(
        self,
        nav: pd.Series,
        benchmark_nav: pd.Series,
        weights_history: pd.DataFrame,
        turnover: pd.Series,
        signal_history: pd.Series | None = None,
        cost_bps: float = 0.0,
    ) -> None:
        _validate_datetime_index(nav)
        _validate_datetime_index(benchmark_nav)

        self.nav: pd.Series = nav
        self.benchmark_nav: pd.Series = benchmark_nav
        self.weights_history: pd.DataFrame = weights_history
        self.turnover: pd.Series = turnover
        self.signal_history: pd.Series | None = signal_history
        self.cost_bps: float = cost_bps

        self._returns: pd.Series = nav.pct_change()
        self._benchmark_returns: pd.Series = benchmark_nav.pct_change()

    # ------------------------------------------------------------------
    # Active-return summary
    # ------------------------------------------------------------------

    def apply_standard_functions(self) -> pd.DataFrame:
        """Return a one-column DataFrame of strategy-vs-benchmark metrics."""
        r = self._returns
        b = self._benchmark_returns
        metrics: dict[str, Any] = {
            "tracking_error": calc_tracking_error(r, b, annualize=True),
            "information_ratio": calc_information_ratio(r, b, annualize=True),
            "hit_rate": calc_hit_rate(r, b),
            "up_capture": calc_capture_ratio(r, b, side="up"),
            "down_capture": calc_capture_ratio(r, b, side="down"),
        }
        if self.signal_history is not None:
            metrics["information_coefficient"] = self.information_coefficient()

        return pd.DataFrame({"Strategy vs Benchmark": metrics})

    # ------------------------------------------------------------------
    # Series helpers (used by the PDF report)
    # ------------------------------------------------------------------

    def running_drawdown(self) -> pd.DataFrame:
        """Running drawdown for strategy and benchmark as a two-column frame."""
        return pd.concat(
            {
                "strategy": calc_running_drawdown(self._returns),
                "benchmark": calc_running_drawdown(self._benchmark_returns),
            },
            axis=1,
        )

    def accumulated_transaction_costs(
        self, annualize: bool = False
    ) -> pd.Series | float:
        """Cumulative (default) or annualized dollar transaction costs."""
        return calc_accumulated_transaction_costs(
            self.turnover,
            self.cost_bps,
            self.nav,
            annualize=annualize,
        )

    def information_coefficient(
        self,
        method: Literal["pearson", "spearman"] = "spearman",
    ) -> float:
        """Single-number IC of signal vs the next-period return."""
        if self.signal_history is None:
            raise ValueError("signal_history is required to compute IC.")
        fwd = self._returns.shift(-1)
        return float(
            calc_information_coefficient(
                self.signal_history,
                fwd,
                method=method,
                rolling_window=None,
            )
        )

    def rolling_information_coefficient(
        self,
        window: int = 252,
        method: Literal["pearson", "spearman"] = "spearman",
    ) -> pd.Series:
        """Rolling IC series of signal vs the next-period return."""
        if self.signal_history is None:
            raise ValueError("signal_history is required to compute IC.")
        fwd = self._returns.shift(-1)
        result = calc_information_coefficient(
            self.signal_history,
            fwd,
            method=method,
            rolling_window=window,
        )
        assert isinstance(result, pd.Series)
        return result
