"""
walk_forward.py
---------------
Utilities for building expanding / rolling walk-forward train-test windows.

The output is a list of ``(train_slice, predict_slice)`` index pairs that
a driver can consume to run one fit + one prediction per window.  The
:class:`MLStrategy` class does *not* use these slices directly — it
decides per-rebalance whether to refit — but the helpers are kept here
for ad-hoc walk-forward evaluations and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FitPredictStrategy(Protocol):
    """Research strategy consumed by the validation schemes.

    A strategy selects parameters on a *train* index and returns per-period
    strategy returns on a *test* index.  The momentum strategies in
    :mod:`signals.stcma.momentum` implement this interface, as does any
    object exposing the two methods below.
    """

    def fit(
        self,
        returns: pd.Series | pd.DataFrame,
        train_index: pd.DatetimeIndex,
    ) -> FitPredictStrategy: ...

    def predict(
        self,
        returns: pd.Series | pd.DataFrame,
        test_index: pd.DatetimeIndex,
    ) -> pd.Series: ...


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
        Full sample index (e.g. a returns-frame index).
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


@dataclass(frozen=True)
class WalkForwardResult:
    """Output of :func:`walk_forward_strategy_returns`.

    Attributes
    ----------
    oos_returns : pd.Series
        Concatenated out-of-sample per-period strategy returns across all
        predict blocks (one continuous path when ``retrain_every ==
        test_horizon``; later blocks win on any overlap).
    selected_params : pd.DataFrame
        One row per window, indexed by the predict block's first date, with
        the parameters chosen on that window's train set.
    windows : list[WalkForwardWindow]
        The train/predict windows that were evaluated.
    """

    oos_returns: pd.Series
    selected_params: pd.DataFrame
    windows: list[WalkForwardWindow]


def walk_forward_strategy_returns(
    strategy: FitPredictStrategy,
    returns: pd.Series | pd.DataFrame,
    mode: Literal["expanding", "rolling"] = "expanding",
    min_train_size: int = 60,
    retrain_every: int = 12,
    test_horizon: int = 12,
    train_window: int | None = None,
) -> WalkForwardResult:
    """Run anchored / rolling walk-forward analysis end to end.

    For each window the strategy's parameters are chosen on the train block
    and the resulting strategy returns are collected on the disjoint predict
    block — the textbook single OOS path of walk-forward analysis.  Set
    ``retrain_every == test_horizon`` (the default) so the predict blocks
    tile the post-warm-up sample exactly once.

    Parameters
    ----------
    strategy : FitPredictStrategy
        Object with ``fit(returns, train_index)`` and
        ``predict(returns, test_index)``.  It is refit on every window.
    returns : pd.Series | pd.DataFrame
        Full-sample returns (single asset → Series, cross-section → frame).
    mode, min_train_size, retrain_every, test_horizon, train_window
        Forwarded to :func:`make_train_test_windows`.

    Returns
    -------
    WalkForwardResult
    """
    windows = make_train_test_windows(
        returns.index,
        mode=mode,
        min_train_size=min_train_size,
        retrain_every=retrain_every,
        test_horizon=test_horizon,
        train_window=train_window,
    )
    if not windows:
        raise ValueError(
            "No walk-forward windows produced; the sample is too short for the "
            "requested min_train_size / test_horizon."
        )

    oos_segments: list[pd.Series] = []
    param_rows: dict[pd.Timestamp, dict[str, float | int]] = {}
    for window in windows:
        strategy.fit(returns, window.train)
        oos_segments.append(strategy.predict(returns, window.predict))
        param_rows[window.predict[0]] = dict(
            getattr(strategy, "selected_params_", None) or {}
        )

    # Concatenate; on any overlap keep the most recent window's prediction.
    oos = pd.concat(oos_segments)
    oos = oos[~oos.index.duplicated(keep="last")].sort_index()
    selected = pd.DataFrame.from_dict(param_rows, orient="index").sort_index()
    return WalkForwardResult(oos_returns=oos, selected_params=selected, windows=windows)
