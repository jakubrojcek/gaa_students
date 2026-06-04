"""
combinatorial_purged_cv.py
--------------------------
Combinatorial Purged Cross-Validation (CPCV) — López de Prado, *Advances in
Financial Machine Learning* (2018), ch. 7 (purging/embargo) and ch. 12 (CPCV).

Motivation
~~~~~~~~~~
Walk-forward analysis yields a **single** out-of-sample path, so its Sharpe is
one draw from a sampling distribution — easy to over-read.  CPCV instead
splits the timeline into ``N`` contiguous groups, tests every combination of
``k`` groups (training on the remaining ``N − k``), and recombines the
out-of-sample predictions into

.. math:: \\varphi = \\binom{N-1}{k-1}

distinct full-length OOS paths.  Each group is tested in exactly ``φ``
combinations, so each contributes one segment to each of the ``φ`` paths.  The
Sharpe of each path is one observation, turning a point estimate into a
**distribution** whose dispersion exposes how lucky (or fragile) a backtest is.

Leakage control
~~~~~~~~~~~~~~~
Because test groups can be interior, the train set straddles them in time and
would otherwise leak through serial correlation and overlapping signal
windows.  Two guards are applied to every split:

* **Purge** — drop train observations within ``label_horizon`` observations of
  either edge of a contiguous test block.  For a momentum rule the information
  span of one observation is its trailing signal window, so a sensible value
  is the strategy's lookback (the example passes it explicitly).
* **Embargo** — additionally drop ``embargo_pct · T`` observations immediately
  *after* each test block, absorbing residual serial correlation that purging
  on the deterministic window alone does not cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, sqrt

import numpy as np
import pandas as pd

from analytics.timeseries_analyzer import infer_periods_per_year
from signals.ml_pipeline.walk_forward import FitPredictStrategy


@dataclass(frozen=True)
class CPCVSplit:
    """One CPCV combination: which groups are tested, with purged train rows."""

    test_group_ids: tuple[int, ...]
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    # Per-group test slices (group id → its dates), needed for path assembly.
    test_groups: dict[int, pd.DatetimeIndex]


class CombinatorialPurgedCV:
    """Generate purged/embargoed CPCV splits and count the recombined paths.

    Parameters
    ----------
    n_groups : int
        Number ``N`` of contiguous groups the timeline is split into.
    n_test_groups : int
        Number ``k`` of groups held out for testing in each combination
        (``1 <= k < N``).
    embargo_pct : float
        Fraction of the total sample length embargoed after each test block.
    label_horizon : int
        Observations of overlap purged on each edge of a test block; for a
        momentum strategy set this to the (maximum) signal lookback.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        label_horizon: int = 1,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}.")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError(
                f"n_test_groups must be in [1, n_groups), got {n_test_groups}."
            )
        if embargo_pct < 0.0:
            raise ValueError(f"embargo_pct must be >= 0, got {embargo_pct}.")
        if label_horizon < 0:
            raise ValueError(f"label_horizon must be >= 0, got {label_horizon}.")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.label_horizon = label_horizon

    @property
    def n_splits(self) -> int:
        """Number of train/test combinations, ``C(N, k)``."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of recombined OOS paths, ``φ = C(N − 1, k − 1)``."""
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def split(self, index: pd.DatetimeIndex) -> list[CPCVSplit]:
        """Build every purged/embargoed CPCV split for ``index``."""
        n = len(index)
        if n < self.n_groups:
            raise ValueError(
                f"Need at least n_groups={self.n_groups} observations, got {n}."
            )
        # Contiguous, near-equal groups of positional indices.
        group_positions = [np.asarray(g) for g in np.array_split(np.arange(n), self.n_groups)]
        embargo_len = int(np.ceil(self.embargo_pct * n))

        splits: list[CPCVSplit] = []
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_pos = np.concatenate([group_positions[g] for g in combo])
            test_pos.sort()
            blocked = self._purge_and_embargo(test_pos, n, embargo_len)
            train_pos = np.setdiff1d(np.arange(n), blocked, assume_unique=False)

            test_groups = {
                g: index[group_positions[g]] for g in combo
            }
            splits.append(
                CPCVSplit(
                    test_group_ids=combo,
                    train_index=index[train_pos],
                    test_index=index[np.sort(test_pos)],
                    test_groups=test_groups,
                )
            )
        return splits

    def _purge_and_embargo(
        self,
        test_pos: np.ndarray,
        n: int,
        embargo_len: int,
    ) -> np.ndarray:
        """Positions removed from train: the test rows plus purge + embargo bands."""
        blocked: set[int] = set(int(p) for p in test_pos)
        for start, end in _contiguous_segments(test_pos):
            lo = max(0, start - self.label_horizon)
            hi = min(n - 1, end + self.label_horizon)
            blocked.update(range(lo, hi + 1))  # purge band on both edges
            embargo_hi = min(n - 1, end + embargo_len)
            blocked.update(range(end + 1, embargo_hi + 1))  # embargo after block
        return np.array(sorted(blocked), dtype=int)


def _contiguous_segments(positions: np.ndarray) -> list[tuple[int, int]]:
    """Collapse a sorted position array into ``(start, end)`` inclusive runs."""
    if positions.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = int(positions[0])
    for p in positions[1:]:
        p = int(p)
        if p == prev + 1:
            prev = p
            continue
        segments.append((start, prev))
        start = prev = p
    segments.append((start, prev))
    return segments


# ---------------------------------------------------------------------------
# Path reconstruction + driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCVResult:
    """Aggregated CPCV output.

    Attributes
    ----------
    paths : pd.DataFrame
        ``φ`` columns (``path_0 ...``), each a full-length OOS per-period
        return series recombined from the test-group predictions.
    path_sharpes : pd.Series
        Annualised Sharpe of each path — the distribution of headline
        performance the backtest could have produced.
    selected_params : pd.DataFrame
        Parameters chosen on the train set of each combination (one row per
        split, indexed by the tested group ids).
    n_groups, n_test_groups, n_splits, n_paths : int
        The CPCV geometry that produced the result.
    """

    paths: pd.DataFrame
    path_sharpes: pd.Series
    selected_params: pd.DataFrame
    n_groups: int
    n_test_groups: int
    n_splits: int
    n_paths: int


def reconstruct_paths(
    group_segments: dict[int, list[pd.Series]],
    n_paths: int,
) -> pd.DataFrame:
    """Recombine per-group OOS segments into ``n_paths`` full-length paths.

    ``group_segments[g]`` must hold exactly ``n_paths`` segments for every
    group ``g`` (one per combination in which ``g`` was a test group, in
    enumeration order).  Path ``j`` concatenates the ``j``-th segment of every
    group, then sorts by date.

    Parameters
    ----------
    group_segments : dict[int, list[pd.Series]]
        Map of group id → list of OOS return segments.
    n_paths : int
        Expected number of segments per group, ``φ = C(N − 1, k − 1)``.
    """
    for g, segs in group_segments.items():
        if len(segs) != n_paths:
            raise ValueError(
                f"Group {g} has {len(segs)} segments, expected n_paths={n_paths}."
            )
    paths: dict[str, pd.Series] = {}
    for j in range(n_paths):
        path = pd.concat([group_segments[g][j] for g in sorted(group_segments)])
        paths[f"path_{j}"] = path.sort_index()
    return pd.DataFrame(paths)


def run_cpcv(
    strategy: FitPredictStrategy,
    returns: pd.Series | pd.DataFrame,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    label_horizon: int = 1,
    periods_per_year: float | None = None,
) -> CPCVResult:
    """Run CPCV end to end and return the distribution of path Sharpes.

    For every combination the strategy is refit on the purged train index and
    used to predict each held-out group; the predictions are recombined into
    ``φ`` OOS paths whose annualised Sharpes form the reported distribution.

    Parameters
    ----------
    strategy : FitPredictStrategy
        Object with ``fit`` / ``predict`` (see
        :class:`signals.stcma.momentum.TimeSeriesMomentum`).
    returns : pd.Series | pd.DataFrame
        Full-sample returns driving the strategy.
    n_groups, n_test_groups, embargo_pct, label_horizon
        Forwarded to :class:`CombinatorialPurgedCV`.
    periods_per_year : float | None
        Annualisation factor for the path Sharpes; inferred from the index
        when ``None``.

    Returns
    -------
    CPCVResult
    """
    cv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        label_horizon=label_horizon,
    )
    splits = cv.split(returns.index)
    ppy = periods_per_year or infer_periods_per_year(returns.index)

    group_segments: dict[int, list[pd.Series]] = {g: [] for g in range(n_groups)}
    param_rows: dict[tuple[int, ...], dict[str, float | int]] = {}
    for split in splits:
        strategy.fit(returns, split.train_index)
        param_rows[split.test_group_ids] = dict(
            getattr(strategy, "selected_params_", None) or {}
        )
        for g in split.test_group_ids:
            group_segments[g].append(strategy.predict(returns, split.test_groups[g]))

    paths = reconstruct_paths(group_segments, cv.n_paths)
    path_sharpes = paths.apply(lambda col: _annualised_sharpe(col, ppy))

    selected_params = pd.DataFrame.from_dict(param_rows, orient="index")
    selected_params.index = pd.MultiIndex.from_tuples(
        selected_params.index, names=[f"test_g{i}" for i in range(n_test_groups)]
    )

    return CPCVResult(
        paths=paths,
        path_sharpes=path_sharpes,
        selected_params=selected_params,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        n_splits=cv.n_splits,
        n_paths=cv.n_paths,
    )


def _annualised_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    """Annualised Sharpe of a return path (``NaN`` if degenerate)."""
    series = returns.dropna()
    if len(series) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(series.mean()) / std * sqrt(periods_per_year)
