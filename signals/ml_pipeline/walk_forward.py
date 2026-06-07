"""
walk_forward.py
---------------
Walk-forward analysis driven *by the backtest engine itself*.

The :class:`backtesting.Backtester` already is the walk-forward loop: at every
rebalance it calls ``strategy.compute_weights(data.loc[:as_of], as_of, ...)``, so
anything a strategy decides from ``data`` is physically causal.  :class:`WalkForwardStrategy`
exploits that — it wraps a ``strategy_factory`` + parameter grid and, on a refit
schedule, calls :meth:`WalkForwardStrategy.fit` to re-select the best parameter
cell on the train window ending at ``as_of`` (rolling or expanding), then
delegates weight computation to that concrete strategy.  Running a single
backtest with this strategy therefore *is* the out-of-sample walk-forward path.

:func:`make_train_test_windows` remains as a standalone helper for ad-hoc
train/predict window enumeration (used by ML diagnostics).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from backtesting.backtest_engine import BacktestConfig
from backtesting.strategy_protocols import Strategy, Weights
from signals.ml_pipeline.signal_research_pipeline import (
    CellParams,
    ParameterGrid,
    cell_param_values,
    max_grid_lookback,
    select_best_param_cell,
)


@dataclass(frozen=True)
class WalkForwardWindow:
    """One train + predict block drawn from a time index."""

    train: pd.DatetimeIndex
    predict: pd.DatetimeIndex


def make_train_test_windows(
    index: pd.DatetimeIndex,
    mode: Literal["expanding", "rolling"] = "expanding",
    min_train_size: int = 252,
    retrain_every: int = 21,
    test_horizon: int = 21,
    train_window: int | None = None,
) -> list[WalkForwardWindow]:
    """Split ``index`` into a list of walk-forward train/predict windows.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Full sample index (e.g. a price-frame index).
    mode : {"expanding", "rolling"}
        ``"expanding"`` keeps all past observations in the train set;
        ``"rolling"`` uses only the most recent ``train_window`` rows.
    min_train_size : int
        Minimum number of observations before the first fit is attempted.
    retrain_every : int
        Gap (in observations) between consecutive refits.
    test_horizon : int
        Length of each predict block.
    train_window : int | None
        Required when ``mode == "rolling"``; ignored for ``"expanding"``.

    Returns
    -------
    list[WalkForwardWindow]
        Successive windows with disjoint predict blocks.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}.")
    if min_train_size < 2:
        raise ValueError(f"min_train_size must be >= 2, got {min_train_size}.")
    if retrain_every < 1:
        raise ValueError(f"retrain_every must be >= 1, got {retrain_every}.")
    if test_horizon < 1:
        raise ValueError(f"test_horizon must be >= 1, got {test_horizon}.")
    if mode == "rolling":
        if train_window is None:
            raise ValueError("train_window is required when mode='rolling'.")
        if train_window < 2:
            raise ValueError(f"train_window must be >= 2, got {train_window}.")

    windows: list[WalkForwardWindow] = []
    n = len(index)
    start = min_train_size
    while start + test_horizon <= n:
        if mode == "expanding":
            train_idx = index[:start]
        else:
            assert train_window is not None
            lo = max(0, start - train_window)
            train_idx = index[lo:start]
        predict_idx = index[start : start + test_horizon]
        windows.append(WalkForwardWindow(train=train_idx, predict=predict_idx))
        start += retrain_every

    return windows


# ---------------------------------------------------------------------------
# Walk-forward strategy (self-refitting ``Strategy``)
# ---------------------------------------------------------------------------


@dataclass
class _WalkForwardState:
    """Mutable per-run state kept on the strategy instance."""

    rebalance_count: int = 0
    rebalances_since_fit: int | None = None  # None → never fit yet
    selected: Strategy | None = None
    selected_params: dict[pd.Timestamp, dict[str, object]] = field(default_factory=dict)
    first_fit_date: pd.Timestamp | None = None


class WalkForwardStrategy:
    """``Strategy`` wrapper that re-selects its parameters walk-forward in-backtest.

    The Backtester supplies the walk-forward loop: ``compute_weights`` receives
    ``data.loc[:as_of]`` at each rebalance, so any selection made here is causal.
    On a refit (the first eligible rebalance after a ``min_train_size`` warm-up,
    then every ``retrain_every_rebalances`` rebalances), :meth:`fit` chooses the
    grid cell with the highest annualised train Sharpe on the train window
    (rolling or expanding, ending at ``as_of``) and delegates weight computation
    to that concrete strategy until the next refit.  During warm-up the strategy
    holds cash (zero weights).

    Parameters
    ----------
    strategy_factory : Callable[[CellParams], Strategy]
        Builds a concrete engine ``Strategy`` from one grid cell (e.g.
        :func:`signals.stcma.momentum.time_series_momentum_factory`).
    param_grid : ParameterGrid
        Parameter grid searched on every refit.  A swept ``lookback`` raises the
        effective warm-up so the train window can always form a signal.
    base_config : BacktestConfig
        Backtest settings used for the *selection* backtests on each train
        window.
    train_mode : {"expanding", "rolling"}
        ``"expanding"`` selects on all history up to ``as_of``; ``"rolling"``
        on the last ``train_window`` observations.
    min_train_size : int
        Observations required before the first refit (raised to the grid's max
        lookback so selection always has at least one signal period).
    retrain_every_rebalances : int
        Refit cadence in rebalance counts (``1`` → refit every rebalance).
    train_window : int | None
        Required when ``train_mode == "rolling"``; ignored otherwise.
    periods_per_year : float | None
        Annualisation factor for the selection Sharpe; inferred from the index
        when ``None``.
    cell_returns : dict[str, pd.Series] | None
        Optional precomputed full-sample per-cell backtest returns (see
        :func:`signals.ml_pipeline.backtest_param_cells`).  When supplied, each
        refit *slices* this cache to the train window instead of re-backtesting
        every cell from scratch — O(N) rather than O(N²) over a run.  Because
        cell returns are causal (the return at ``t`` depends only on prices up
        to ``t``) and only the train-window slice is read at each refit, the
        cache changes nothing about the selection decisions; it is purely a
        speed-up.  When ``None`` the train window is re-backtested each refit.
    """

    def __init__(
        self,
        strategy_factory: Callable[[CellParams], Strategy],
        param_grid: ParameterGrid,
        base_config: BacktestConfig,
        train_mode: Literal["expanding", "rolling"] = "expanding",
        min_train_size: int = 60,
        retrain_every_rebalances: int = 1,
        train_window: int | None = None,
        periods_per_year: float | None = None,
        cell_returns: dict[str, pd.Series] | None = None,
    ) -> None:
        if train_mode not in ("expanding", "rolling"):
            raise ValueError(
                f"train_mode must be 'expanding' or 'rolling', got {train_mode!r}."
            )
        if min_train_size < 2:
            raise ValueError(f"min_train_size must be >= 2, got {min_train_size}.")
        if retrain_every_rebalances < 1:
            raise ValueError(
                f"retrain_every_rebalances must be >= 1, got {retrain_every_rebalances}."
            )
        if train_mode == "rolling" and (train_window is None or train_window < 2):
            raise ValueError("train_window must be >= 2 when train_mode='rolling'.")

        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.base_config = base_config
        self.train_mode = train_mode
        self.retrain_every_rebalances = retrain_every_rebalances
        self.train_window = train_window
        self.periods_per_year = periods_per_year
        self._cell_returns = cell_returns

        self._max_lookback = max_grid_lookback(param_grid)
        # Selection needs at least one signal period, so warm up past the lookback.
        self.min_train_size = max(int(min_train_size), self._max_lookback + 1)
        self._state = _WalkForwardState()

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Refit on schedule, then delegate to the selected concrete strategy."""
        self._state.rebalance_count += 1
        self.fit(data, as_of)
        if self._state.selected is None:
            return _zero_weights(data.columns)  # warm-up: hold cash
        return self._state.selected.compute_weights(data, as_of, current_weights)

    # ------------------------------------------------------------------
    # Refit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """Re-select parameters on data up to ``as_of`` if warm-up + cadence allow.

        ``data`` is the price history through ``as_of``.  No-op during warm-up or
        between scheduled refits (the previous selection is reused).
        """
        if len(data) < self.min_train_size:
            return  # warm-up: not enough history to select yet
        if not self._schedule_says_refit():
            return  # reuse the current selection until the next scheduled refit

        train = self._train_window(data)
        best_cell, _ = select_best_param_cell(
            self.param_grid,
            self.strategy_factory,
            train,
            self.base_config,
            metric_index=train.index,
            periods_per_year=self.periods_per_year,
            cell_returns=self._cell_returns,
        )
        self._state.selected = self.strategy_factory(best_cell)
        self._state.rebalances_since_fit = 0
        as_of = pd.Timestamp(as_of)
        self._state.selected_params[as_of] = cell_param_values(best_cell)
        if self._state.first_fit_date is None:
            self._state.first_fit_date = as_of

    def _schedule_says_refit(self) -> bool:
        """Return ``True`` iff the cadence requires a refit this call."""
        if self._state.rebalances_since_fit is None:
            return True  # first eligible rebalance
        self._state.rebalances_since_fit += 1
        return self._state.rebalances_since_fit >= self.retrain_every_rebalances

    def _train_window(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.train_mode == "rolling":
            assert self.train_window is not None
            return data.iloc[-self.train_window :]
        return data

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def selected_params_(self) -> pd.DataFrame:
        """One row per refit (indexed by refit date) of the chosen parameters."""
        if not self._state.selected_params:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(
            self._state.selected_params, orient="index"
        ).sort_index()

    @property
    def first_fit_date_(self) -> pd.Timestamp | None:
        """Date of the first refit (start of the out-of-sample path)."""
        return self._state.first_fit_date

    def reset(self) -> None:
        """Clear the in-memory selection state (for a fresh backtest run)."""
        self._state = _WalkForwardState()


def _zero_weights(columns: pd.Index) -> Weights:
    return pd.Series(0.0, index=columns, dtype=float)
