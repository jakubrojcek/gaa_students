"""
momentum.py
-----------
Trend (time-series) and cross-sectional momentum strategies with a
research-grade ``fit`` / ``predict`` interface for backtest-validation
studies (walk-forward, CPCV, bootstrap reality check).

Two strategies are provided:

* :class:`TimeSeriesMomentum` — single asset.  The position is the sign of
  the trailing ``lookback``-period return, in ``{-1, 0, +1}``; the strategy
  return is ``position · forward_return``.  The searched parameter is
  ``lookback``.
* :class:`CrossSectionalMomentum` — multi asset.  Assets are ranked by their
  trailing ``lookback``-period return; the top ``quantile`` fraction is held
  long and the bottom ``quantile`` short, equal-weighted within each leg and
  **dollar-neutral** (long leg sums to ``+1``, short leg to ``-1``).  The
  searched parameters are ``lookback`` and ``quantile``.

No look-ahead by construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The position formed at date ``t`` uses only the trailing window ending at
``t`` (``rolling`` is backward-looking); it is then ``shift``-ed by one and
multiplied by the return realised at ``t + 1``.  Strategy returns are
therefore indexed by their **realisation** date and are causal even when a
window is computed over the full sample and sliced afterwards — the property
that lets purged cross-validation slice the series by arbitrary
(non-contiguous) test groups.

Research interface
~~~~~~~~~~~~~~~~~~
Every strategy exposes:

* ``param_grid()`` — the list of candidate parameter dicts.
* ``strategy_returns(returns, **params)`` — the full-sample per-period
  strategy returns for one fixed parameter set.
* ``fit(returns, train_index)`` — select the parameter set maximising the
  in-sample (train-window) Sharpe; stored on the instance.
* ``predict(returns, test_index)`` — the selected-parameter strategy returns
  restricted to the test window (out-of-sample once parameters were chosen on
  a disjoint train window).

Both classes also implement the backtester :class:`Strategy` protocol
(``compute_weights``) so the *same* logic can drive a headline full-sample
backtest through :class:`backtesting.Backtester`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from backtesting.strategy_protocols import Weights

# ---------------------------------------------------------------------------
# Pure functions — the signal-to-returns maps
# ---------------------------------------------------------------------------


def time_series_momentum_returns(returns: pd.Series, lookback: int) -> pd.Series:
    """Per-period returns of a single-asset trend-following rule.

    ``position_t = sign(Σ r over the trailing ``lookback`` periods ending t)``
    ∈ ``{-1, 0, +1}``; the strategy return realised at ``t`` is
    ``position_{t-1} · r_t``.  The first ``lookback`` observations (and the
    first realised period after them) are ``NaN`` — the warm-up has no
    trailing window.

    Parameters
    ----------
    returns : pd.Series
        Single-asset period returns with a ``DatetimeIndex``.
    lookback : int
        Trailing window length in observations (``>= 1``).
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}.")
    trailing = returns.rolling(lookback, min_periods=lookback).sum()
    position = np.sign(trailing)
    return (position.shift(1) * returns).rename("ts_momentum")


def _cross_sectional_weights(
    trailing_row: pd.Series,
    columns: pd.Index,
    quantile: float,
    long_only: bool = False,
) -> pd.Series:
    """Cross-sectional momentum weights for one period.

    With ``long_only=False`` (default) the portfolio is **dollar-neutral**:
    long the top ``quantile`` (sum ``+1``), short the bottom ``quantile`` (sum
    ``-1``), equal-weighted within each leg.  With ``long_only=True`` only the
    top ``quantile`` is held, **fully invested** (sum ``+1``) and **weighted
    linearly by rank** — the highest-momentum name gets weight ``k`` units, the
    ``k``-th gets ``1`` unit (normalised by ``k(k+1)/2``), so allocation tapers
    smoothly across the selection rather than concentrating in a single name.

    Returns an all-``NaN`` row when a portfolio cannot be formed (fewer than
    two assets with a trailing return, or a quantile too small to select even
    one name per leg) so that the period is excluded rather than counted as a
    flat (zero) return.
    """
    valid = trailing_row.dropna()
    n = len(valid)
    if n < 2:
        return pd.Series(np.nan, index=columns)

    k = max(1, int(np.floor(quantile * n)))
    k = min(k, n // 2)  # keep the long and short legs disjoint
    if k < 1:
        return pd.Series(np.nan, index=columns)

    weights = pd.Series(0.0, index=columns)
    if long_only:
        # Rank-linear long-only: nlargest is descending, so [k, k-1, ..., 1]
        # gives the top name the largest share, tapering to the k-th.
        longs = valid.nlargest(k)
        rank_units = np.arange(k, 0, -1, dtype=float)
        weights.loc[longs.index] = rank_units / rank_units.sum()
    else:
        weights.loc[valid.nlargest(k).index] = 1.0 / k
        weights.loc[valid.nsmallest(k).index] = -1.0 / k
    return weights


def cross_sectional_momentum_weights(
    returns: pd.DataFrame,
    lookback: int,
    quantile: float,
    long_only: bool = False,
) -> pd.DataFrame:
    """History of cross-sectional momentum weights.

    Row ``t`` holds the weights *formed* at ``t`` from the trailing
    ``lookback``-period return; warm-up rows are ``NaN``.  See
    :func:`_cross_sectional_weights` for the ``long_only`` convention.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}.")
    if not 0.0 < quantile <= 0.5:
        raise ValueError(f"quantile must be in (0, 0.5], got {quantile}.")
    trailing = returns.rolling(lookback, min_periods=lookback).sum()
    return trailing.apply(
        lambda row: _cross_sectional_weights(
            row, returns.columns, quantile, long_only
        ),
        axis=1,
    )


def cross_sectional_momentum_returns(
    returns: pd.DataFrame,
    lookback: int,
    quantile: float,
    long_only: bool = False,
) -> pd.Series:
    """Per-period returns of a cross-sectional momentum rule.

    The weights formed at ``t`` are applied to the returns realised at
    ``t + 1`` (``weights.shift(1)``).  Periods without a formable portfolio
    are ``NaN``.  ``long_only`` toggles between the dollar-neutral and the
    long-only top-quantile portfolio.
    """
    weights = cross_sectional_momentum_weights(returns, lookback, quantile, long_only)
    return (weights.shift(1) * returns).sum(axis=1, min_count=1).rename("xs_momentum")


def _in_sample_sharpe(returns: pd.Series) -> float:
    """Per-period Sharpe of an in-sample return slice (``-inf`` if degenerate)."""
    series = returns.dropna()
    if len(series) < 2:
        return float("-inf")
    std = float(series.std(ddof=1))
    if std == 0.0:
        return float("-inf")
    return float(series.mean()) / std


# ---------------------------------------------------------------------------
# Shared fit / predict machinery
# ---------------------------------------------------------------------------


class _MomentumStrategy:
    """Parameter-search, fit/predict and caching shared by both strategies.

    Subclasses implement :meth:`param_grid` and :meth:`strategy_returns`.
    The per-parameter full-sample return series is memoised per ``returns``
    object so that a parameter sweep (``fit``) and the subsequent ``predict``
    — and every CPCV combination sharing the same ``returns`` — reuse one
    computation.  The cache assumes the ``returns`` object is stable for the
    duration of a study; call :meth:`clear_cache` otherwise.
    """

    def __init__(self) -> None:
        self.selected_params_: dict[str, float | int] | None = None
        self._cache: dict[tuple, pd.Series] = {}

    # -- to be implemented by subclasses --------------------------------
    def param_grid(self) -> list[dict[str, float | int]]:
        raise NotImplementedError

    def strategy_returns(
        self,
        returns: pd.Series | pd.DataFrame,
        **params: float | int,
    ) -> pd.Series:
        raise NotImplementedError

    # -- caching --------------------------------------------------------
    def _cached_returns(
        self,
        returns: pd.Series | pd.DataFrame,
        params: dict[str, float | int],
    ) -> pd.Series:
        key = (id(returns), tuple(sorted(params.items())))
        if key not in self._cache:
            self._cache[key] = self.strategy_returns(returns, **params)
        return self._cache[key]

    def clear_cache(self) -> None:
        """Drop memoised per-parameter return series."""
        self._cache.clear()

    # -- research interface ---------------------------------------------
    def fit(
        self,
        returns: pd.Series | pd.DataFrame,
        train_index: pd.DatetimeIndex,
    ) -> _MomentumStrategy:
        """Select the parameter set maximising the train-window Sharpe."""
        grid = self.param_grid()
        best_params = grid[0]
        best_sharpe = float("-inf")
        for params in grid:
            in_sample = self._cached_returns(returns, params).reindex(train_index)
            sharpe = _in_sample_sharpe(in_sample)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        self.selected_params_ = best_params
        return self

    def predict(
        self,
        returns: pd.Series | pd.DataFrame,
        test_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Selected-parameter strategy returns restricted to ``test_index``."""
        if self.selected_params_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._cached_returns(returns, self.selected_params_).reindex(test_index)


# ---------------------------------------------------------------------------
# Single-asset time-series momentum
# ---------------------------------------------------------------------------


class TimeSeriesMomentum(_MomentumStrategy):
    """Single-asset trend follower searched over ``lookback``.

    Parameters
    ----------
    lookbacks : Sequence[int]
        Candidate trailing-window lengths (observations) to search over.
        A single-element sequence pins the strategy to one lookback (handy
        for a headline backtest through :class:`backtesting.Backtester`).
    """

    def __init__(self, lookbacks: Sequence[int]) -> None:
        super().__init__()
        lookbacks = [int(lb) for lb in lookbacks]
        if not lookbacks:
            raise ValueError("lookbacks must be non-empty.")
        if any(lb < 1 for lb in lookbacks):
            raise ValueError("all lookbacks must be >= 1.")
        self.lookbacks = lookbacks

    def param_grid(self) -> list[dict[str, float | int]]:
        return [{"lookback": lb} for lb in self.lookbacks]

    def strategy_returns(
        self,
        returns: pd.Series | pd.DataFrame,
        **params: float | int,
    ) -> pd.Series:
        series = _as_single_series(returns)
        return time_series_momentum_returns(series, int(params["lookback"]))

    # -- backtester Strategy protocol -----------------------------------
    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Sign-of-trailing-return position for the single asset in ``data``."""
        lookback = int(
            (self.selected_params_ or self.param_grid()[0])["lookback"]
        )
        returns = data.loc[:as_of].pct_change(fill_method=None)
        recent = returns.iloc[-lookback:]
        if len(recent) < lookback:
            return pd.Series(0.0, index=data.columns)
        position = np.sign(recent.sum())
        return position.reindex(data.columns).fillna(0.0)


# ---------------------------------------------------------------------------
# Multi-asset cross-sectional momentum
# ---------------------------------------------------------------------------


class CrossSectionalMomentum(_MomentumStrategy):
    """Cross-sectional momentum searched over ``lookback`` × ``quantile``.

    Parameters
    ----------
    lookbacks : Sequence[int]
        Candidate trailing-window lengths (observations).
    quantiles : Sequence[float]
        Candidate top/bottom fractions in ``(0, 0.5]`` (e.g. ``0.2`` →
        long the top quintile, short the bottom quintile).
    long_only : bool
        When ``False`` (default) the portfolio is dollar-neutral (long top,
        short bottom, equal-weighted legs).  When ``True`` only the top
        ``quantile`` is held, fully invested and weighted **linearly by rank**
        (best momentum gets the largest share) — comparable to a long-only
        benchmark.
    """

    def __init__(
        self,
        lookbacks: Sequence[int],
        quantiles: Sequence[float],
        long_only: bool = False,
    ) -> None:
        super().__init__()
        lookbacks = [int(lb) for lb in lookbacks]
        quantiles = [float(q) for q in quantiles]
        if not lookbacks or not quantiles:
            raise ValueError("lookbacks and quantiles must be non-empty.")
        if any(lb < 1 for lb in lookbacks):
            raise ValueError("all lookbacks must be >= 1.")
        if any(not 0.0 < q <= 0.5 for q in quantiles):
            raise ValueError("all quantiles must be in (0, 0.5].")
        self.lookbacks = lookbacks
        self.quantiles = quantiles
        self.long_only = long_only

    def param_grid(self) -> list[dict[str, float | int]]:
        return [
            {"lookback": lb, "quantile": q}
            for lb in self.lookbacks
            for q in self.quantiles
        ]

    def strategy_returns(
        self,
        returns: pd.Series | pd.DataFrame,
        **params: float | int,
    ) -> pd.Series:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("CrossSectionalMomentum requires a DataFrame of returns.")
        return cross_sectional_momentum_returns(
            returns,
            int(params["lookback"]),
            float(params["quantile"]),
            long_only=self.long_only,
        )

    # -- backtester Strategy protocol -----------------------------------
    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Rank by trailing return at ``as_of`` and form the momentum portfolio."""
        params = self.selected_params_ or self.param_grid()[0]
        lookback = int(params["lookback"])
        quantile = float(params["quantile"])
        returns = data.loc[:as_of].pct_change(fill_method=None)
        recent = returns.iloc[-lookback:]
        if len(recent) < lookback:
            return pd.Series(0.0, index=data.columns)
        trailing = recent.sum()
        return _cross_sectional_weights(
            trailing, data.columns, quantile, self.long_only
        ).fillna(0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_single_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    """Coerce a one-column frame to a Series; reject genuinely multi-asset input."""
    if isinstance(returns, pd.Series):
        return returns
    if returns.shape[1] != 1:
        raise ValueError(
            "TimeSeriesMomentum is single-asset; got a DataFrame with "
            f"{returns.shape[1]} columns."
        )
    return returns.iloc[:, 0]
