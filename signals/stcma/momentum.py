"""
momentum.py
-----------
Trend (time-series) and cross-sectional momentum strategies that implement the
backtester :class:`~backtesting.strategy_protocols.Strategy` protocol, plus
:func:`time_series_momentum_returns` — a pure single-asset signal-to-returns map
kept for diagnostic benchmarks.

Two strategies are provided, each pinned to a **single** parameter set so it can
drive the engine directly:

* :class:`TimeSeriesMomentum` — single asset.  The position is the sign of the
  trailing ``lookback``-period return, in ``{-1, 0, +1}``.
* :class:`CrossSectionalMomentum` — multi asset.  Assets are ranked by their
  trailing ``lookback``-period return; the top/bottom ``quantile`` fractions form
  the long/short legs (see
  :func:`signals.stcma.signal_strategy.cross_sectional_weights` for the
  ``long_only`` and ``leg_weighting`` conventions).

Parameter search and walk-forward / CPCV validation are **not** done here: a
:class:`~signals.ml_pipeline.signal_research_pipeline.ParameterGrid` and one of
the ``*_factory`` functions below are handed to the backtester-driven research
helpers in :mod:`signals.ml_pipeline` (``select_best_param_cell``, ``run_cpcv``,
``WalkForwardStrategy``), so every reported number flows through
:class:`backtesting.Backtester`.

No look-ahead by construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The position formed at date ``t`` uses only the trailing window ending at ``t``
(``rolling`` / ``iloc[-lookback:]`` are backward-looking).  Backtested per-period
returns are therefore causal — the return at ``t`` depends only on prices up to
``t`` — which lets purged cross-validation reindex a full-sample backtest by
arbitrary (non-contiguous) test groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtesting.strategy_protocols import Weights
from signals.stcma.signal_strategy import LegWeighting, cross_sectional_weights

if TYPE_CHECKING:
    from signals.ml_pipeline.signal_research_pipeline import CellParams

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


# ---------------------------------------------------------------------------
# Single-asset time-series momentum
# ---------------------------------------------------------------------------


class TimeSeriesMomentum:
    """Single-asset trend follower (engine ``Strategy``) at one ``lookback``.

    Parameters
    ----------
    lookback : int
        Trailing-window length in observations (``>= 1``).  The position is the
        sign of the trailing-``lookback`` return; with
        ``BacktestConfig(long_short=True)`` that maps to a fully long, flat, or
        fully short single-asset book.
    """

    def __init__(self, lookback: int) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}.")
        self.lookback = int(lookback)

    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Sign-of-trailing-return position for the single asset in ``data``."""
        # Only the last ``lookback + 1`` prices feed the trailing return, so
        # slice before ``pct_change`` — O(lookback) instead of O(history).
        window = data.loc[:as_of].iloc[-(self.lookback + 1) :]
        recent = window.pct_change(fill_method=None).iloc[-self.lookback :]
        if len(recent) < self.lookback:
            return pd.Series(0.0, index=data.columns)
        position = np.sign(recent.sum())
        return position.reindex(data.columns).fillna(0.0)


# ---------------------------------------------------------------------------
# Multi-asset cross-sectional momentum
# ---------------------------------------------------------------------------


class CrossSectionalMomentum:
    """Cross-sectional momentum (engine ``Strategy``) at one ``lookback`` × ``quantile``.

    Parameters
    ----------
    lookback : int
        Trailing-window length in observations (``>= 1``).
    quantile : float
        Per-leg top/bottom fraction in ``(0, 1.0]`` (e.g. ``0.2`` → long the top
        quintile, short the bottom quintile).  Each leg is capped at half the
        cross-section internally, so ``1.0`` means "top half vs bottom half".
    long_only : bool
        When ``False`` (default) the portfolio is the dollar-neutral long/short
        overlay (long top, short bottom).  When ``True`` that overlay is added
        to a fully-invested 1/n book and scaled to stay non-negative, giving a
        long-only portfolio comparable to an equal-weight benchmark.
    leg_weighting : LegWeighting
        How each leg is weighted across its ranked names —
        ``"equally_weighted"`` (default), ``"linear"`` or ``"square_root"``
        (best-ranked name heaviest for the latter two).
    """

    def __init__(
        self,
        lookback: int,
        quantile: float,
        long_only: bool = False,
        leg_weighting: LegWeighting = "equally_weighted",
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}.")
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1.0], got {quantile}.")
        self.lookback = int(lookback)
        self.quantile = float(quantile)
        self.long_only = long_only
        self.leg_weighting = leg_weighting

    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Rank by trailing return at ``as_of`` and form the momentum portfolio."""
        # Only the last ``lookback + 1`` prices feed the trailing return, so
        # slice before ``pct_change`` — O(lookback) instead of O(history).
        window = data.loc[:as_of].iloc[-(self.lookback + 1) :]
        recent = window.pct_change(fill_method=None).iloc[-self.lookback :]
        if len(recent) < self.lookback:
            return pd.Series(0.0, index=data.columns)
        trailing = recent.sum()
        return cross_sectional_weights(
            trailing, data.columns, self.quantile, self.long_only, self.leg_weighting
        ).fillna(0.0)


# ---------------------------------------------------------------------------
# Cell factories — map a research grid cell onto a concrete strategy
# ---------------------------------------------------------------------------


def time_series_momentum_factory(cell: CellParams) -> TimeSeriesMomentum:
    """Build a :class:`TimeSeriesMomentum` from a grid cell's ``strategy`` params."""
    return TimeSeriesMomentum(lookback=int(cell.strategy["lookback"]))


def cross_sectional_momentum_factory(cell: CellParams) -> CrossSectionalMomentum:
    """Build a :class:`CrossSectionalMomentum` from a grid cell's ``strategy`` params.

    Reads ``lookback`` and ``quantile`` (required) plus optional ``long_only`` and
    ``leg_weighting`` overrides.
    """
    params = cell.strategy
    return CrossSectionalMomentum(
        lookback=int(params["lookback"]),
        quantile=float(params["quantile"]),
        long_only=bool(params.get("long_only", False)),
        leg_weighting=params.get("leg_weighting", "equally_weighted"),
    )
