"""
reality_check.py
----------------
Data-snooping tests for a *set* of trading rules, built on the stationary
bootstrap.

When many strategies (here: a parameter grid) are searched and the best is
reported, its in-sample performance is biased upward — the maximum of many
noisy estimates.  These tests ask whether the **best** rule's edge survives
that selection:

* :func:`whites_reality_check` — White's Reality Check (2000).  H0: the best
  rule does not outperform the benchmark.  Statistic
  ``V = max_m √T · d̄_m`` with the null distribution from the stationary
  bootstrap of the relative-performance series ``d``.
* :func:`hansens_spa` — Hansen's Superior Predictive Ability test (2005).
  Studentises each rule and re-centres only competitive rules, which removes
  the sensitivity of White's RC to poor and irrelevant alternatives.  Returns
  the *consistent* p-value plus the *lower* / *upper* bounds.

The stationary bootstrap (Politis & Romano, 1994) resamples blocks of random
(geometric) length so that serial dependence in the return series is preserved
under H0 — essential for honest p-values on autocorrelated strategy returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int,
    block_mean_length: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resampled row indices for the stationary bootstrap.

    Each resample is built from blocks of i.i.d. geometric length with mean
    ``block_mean_length`` (restart probability ``p = 1 / block_mean_length``),
    wrapping circularly so every block has the same length distribution.

    Parameters
    ----------
    n : int
        Length of the original series.
    block_mean_length : float
        Mean block length; larger values preserve more serial dependence.
        ``1.0`` reduces to the i.i.d. bootstrap.
    n_resamples : int
        Number of bootstrap resamples ``B``.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(B, n)`` of positions into ``[0, n)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if block_mean_length < 1.0:
        raise ValueError(f"block_mean_length must be >= 1, got {block_mean_length}.")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}.")

    p = 1.0 / block_mean_length
    # A new block starts at t=0 and wherever a Bernoulli(p) draw fires.
    restart = rng.random((n_resamples, n)) < p
    restart[:, 0] = True

    positions = np.broadcast_to(np.arange(n), (n_resamples, n))
    # Position of the most recent restart at or before each t.
    last_restart = np.maximum.accumulate(np.where(restart, positions, 0), axis=1)
    offset = positions - last_restart  # 0 at a restart, +1 each step within a block

    block_id = np.cumsum(restart, axis=1) - 1  # 0-based block index per (b, t)
    block_starts = rng.integers(0, n, size=(n_resamples, n))  # random start per block
    start_for_t = np.take_along_axis(block_starts, block_id, axis=1)
    return (start_for_t + offset) % n


# ---------------------------------------------------------------------------
# Relative performance + bootstrap means
# ---------------------------------------------------------------------------


def relative_performance(
    trial_returns: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-period performance of each trial relative to a benchmark.

    ``d_{t,m} = r_{m,t} − benchmark_t`` (or just ``r_{m,t}`` when ``benchmark``
    is ``None``, i.e. testing skill against a zero return).  Rows with any
    missing value are dropped so all trials share one complete sample — a
    requirement for a fair maximum across trials.
    """
    if benchmark is None:
        performance = trial_returns.copy()
    else:
        performance = trial_returns.sub(benchmark, axis=0)
    return performance.dropna(how="any")


def _bootstrap_means(values: np.ndarray, boot_indices: np.ndarray) -> np.ndarray:
    """Mean of each column over each bootstrap resample (memory-frugal).

    ``values`` is ``(T, M)``; ``boot_indices`` is ``(B, T)``; returns ``(B, M)``
    using per-resample row counts to avoid materialising the ``(B, T, M)``
    gather.
    """
    n_rows = values.shape[0]
    means = np.empty((boot_indices.shape[0], values.shape[1]), dtype=float)
    for b, rows in enumerate(boot_indices):
        counts = np.bincount(rows, minlength=n_rows).astype(float)
        means[b] = counts @ values / n_rows
    return means


# ---------------------------------------------------------------------------
# White's Reality Check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealityCheckResult:
    """Output of :func:`whites_reality_check`."""

    p_value: float
    statistic: float
    best_trial: str
    trial_means: pd.Series
    bootstrap_statistics: np.ndarray


def whites_reality_check(
    performance: pd.DataFrame,
    benchmark: pd.Series | None = None,
    block_mean_length: float = 10.0,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> RealityCheckResult:
    """White's Reality Check p-value for the best of many trials.

    Parameters
    ----------
    performance : pd.DataFrame
        Columns = trials, rows = periods (e.g. each parameterisation's
        per-period strategy returns).
    benchmark : pd.Series | None
        Subtracted from every trial before testing; ``None`` tests against a
        zero return (pure skill).
    block_mean_length : float
        Mean block length for the stationary bootstrap.
    n_resamples : int
        Number of bootstrap resamples.
    seed : int | None
        Seed for reproducibility.

    Returns
    -------
    RealityCheckResult
    """
    perf = relative_performance(performance, benchmark)
    values = perf.to_numpy(dtype=float)
    n_obs = values.shape[0]
    if n_obs < 2:
        raise ValueError("Need at least 2 complete observations for the bootstrap.")

    mean = values.mean(axis=0)
    statistic = float(np.sqrt(n_obs) * mean.max())

    rng = np.random.default_rng(seed)
    boot_indices = stationary_bootstrap_indices(n_obs, block_mean_length, n_resamples, rng)
    boot_means = _bootstrap_means(values, boot_indices)
    boot_stats = np.sqrt(n_obs) * (boot_means - mean).max(axis=1)

    p_value = float((1 + np.sum(boot_stats >= statistic)) / (n_resamples + 1))
    return RealityCheckResult(
        p_value=p_value,
        statistic=statistic,
        best_trial=str(perf.columns[mean.argmax()]),
        trial_means=pd.Series(mean, index=perf.columns),
        bootstrap_statistics=boot_stats,
    )


# ---------------------------------------------------------------------------
# Hansen's SPA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SPAResult:
    """Output of :func:`hansens_spa` (consistent p-value plus bounds)."""

    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    statistic: float
    best_trial: str
    trial_means: pd.Series
    trial_std: pd.Series


def hansens_spa(
    performance: pd.DataFrame,
    benchmark: pd.Series | None = None,
    block_mean_length: float = 10.0,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> SPAResult:
    """Hansen's Superior Predictive Ability test.

    Studentises each trial by its bootstrap standard deviation and re-centres
    the bootstrap distribution by a rule-specific mean ``g_m``.  Three
    re-centrings give the *lower* (liberal), *consistent* (recommended) and
    *upper* (conservative) p-values; the consistent one excludes only rules
    that are too far below the benchmark to matter, using the threshold
    ``A_T = √(2·ln ln T)``.

    Parameters
    ----------
    performance, benchmark, block_mean_length, n_resamples, seed
        As in :func:`whites_reality_check`.

    Returns
    -------
    SPAResult
    """
    perf = relative_performance(performance, benchmark)
    values = perf.to_numpy(dtype=float)
    n_obs = values.shape[0]
    if n_obs < 16:
        raise ValueError(
            "Hansen's SPA needs a reasonable sample (>= 16 observations); "
            f"got {n_obs}."
        )

    mean = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot_indices = stationary_bootstrap_indices(n_obs, block_mean_length, n_resamples, rng)
    boot_means = _bootstrap_means(values, boot_indices)

    # ω_m: bootstrap standard deviation of √T·d̄_m.
    omega = np.sqrt(n_obs) * (boot_means - mean).std(axis=0, ddof=1)
    omega = np.where(omega > 0.0, omega, np.nan)

    studentised = np.sqrt(n_obs) * mean / omega
    statistic = float(max(np.nanmax(studentised), 0.0))

    threshold = sqrt(2.0 * log(log(n_obs)))
    recentre = {
        "consistent": np.where(studentised >= -threshold, mean, 0.0),
        "lower": np.where(mean >= 0.0, mean, 0.0),
        "upper": mean,
    }

    sqrt_t = np.sqrt(n_obs)
    p_values: dict[str, float] = {}
    for name, g in recentre.items():
        boot_studentised = sqrt_t * (boot_means - g) / omega
        boot_stat = np.maximum(np.nanmax(boot_studentised, axis=1), 0.0)
        p_values[name] = float((1 + np.sum(boot_stat >= statistic)) / (n_resamples + 1))

    return SPAResult(
        p_value_consistent=p_values["consistent"],
        p_value_lower=p_values["lower"],
        p_value_upper=p_values["upper"],
        statistic=statistic,
        best_trial=str(perf.columns[np.nanargmax(studentised)]),
        trial_means=pd.Series(mean, index=perf.columns),
        trial_std=pd.Series(omega, index=perf.columns),
    )
