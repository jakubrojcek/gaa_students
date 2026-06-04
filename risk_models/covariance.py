"""
covariance.py
-------------
Covariance matrix estimators for portfolio risk models.

Provides stateless estimator functions and a rolling precomputation
helper that efficiently produces covariance snapshots at specified dates.

Estimators (``method`` values for :func:`compute_covariance`)
-------------------------------------------------------------
- ``"sample"``         — unbiased sample covariance (``ddof=1``)
- ``"ledoit_wolf_cc"`` — Ledoit–Wolf shrinkage, constant-correlation target
- ``"ledoit_wolf_sf"`` — Ledoit–Wolf shrinkage, single-factor target
- ``"ledoit_wolf_id"`` — Ledoit–Wolf shrinkage, scaled-identity target
- ``"ewma"``           — exponentially weighted moving average
- ``"semicovariance"`` — downside-only covariance (below threshold)

All estimators return an N×N ``pd.DataFrame`` indexed and columned
by asset name.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from analytics.timeseries_analyzer import infer_periods_per_year
from risk_models.utils import map_dates_to_index

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CovMethod = Literal[
    "sample",
    "ledoit_wolf_cc",
    "ledoit_wolf_sf",
    "ledoit_wolf_id",
    "ewma",
    "semicovariance",
]
"""Supported covariance estimation methods."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_covariance(
    returns: pd.DataFrame,
    method: CovMethod = "sample",
    **kwargs,
) -> pd.DataFrame:
    """Compute a covariance matrix from a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        T×N DataFrame of asset returns (rows = dates, columns = assets).
        NaN rows are dropped before estimation.
    method : CovMethod
        Estimation method.
    **kwargs
        Forwarded to the underlying estimator:

        - ``"ewma"``: ``halflife`` (*int*, default ``60``)
        - ``"semicovariance"``: ``threshold`` (*float*, default ``0.0``)
        - ``"ledoit_wolf_sf"``: ``market_ticker`` (*str | None*, default
          ``"EQ_WL"``).  Ticker of the market factor column in *returns*.
          Falls back to the equal-weighted (1/N) portfolio when ``None``
          or when the ticker is absent from *returns*.

    Returns
    -------
    pd.DataFrame
        N×N covariance matrix indexed and columned by asset name.

    Raises
    ------
    ValueError
        If fewer than 2 non-NaN observations remain or *method* is
        unknown.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(clean)}")

    if method == "sample":
        return _sample_cov(clean)
    if method == "ledoit_wolf_cc":
        return _ledoit_wolf(clean, target="constant_correlation")
    if method == "ledoit_wolf_sf":
        return _ledoit_wolf(clean, target="single_factor", **kwargs)
    if method == "ledoit_wolf_id":
        return _ledoit_wolf(clean, target="identity")
    if method == "ewma":
        return _ewma_cov(clean, **kwargs)
    if method == "semicovariance":
        return _semicovariance(clean, **kwargs)

    raise ValueError(
        f"Unknown method {method!r}. Choose from {list(CovMethod.__args__)}"
    )  # type: ignore[attr-defined]


def compute_rolling_covariance(
    returns: pd.DataFrame,
    dates: list[pd.Timestamp] | None = None,
    freq: str | None = None,
    method: CovMethod = "sample",
    lookback: int | None = None,
    annualize: bool = True,
    start_date: pd.Timestamp | str | None = None,
    min_obs: int = 2,
    **kwargs,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Precompute covariance matrices at specific dates.

    For non-EWMA methods each snapshot uses the most recent *lookback*
    observations (or all available history when *lookback* is ``None``).
    For ``"ewma"`` a single forward pass through the data produces all
    snapshots in O(T·N²) — *lookback* is ignored (the effective memory
    is controlled by *halflife*).

    .. note:: **Minimum-observation gating**

       The ``min_obs`` parameter (default 2) controls the expanding-universe
       path only.  The windowed path (``_rolling_windowed``) hard-codes a
       minimum of 2 non-NaN rows because a covariance matrix is undefined
       with fewer.  ``compute_rolling_asset_factor_cov`` uses a higher
       default (24) because factor regression requires ``K + 2`` rows to
       be well-determined and factor models are unreliable with very few
       observations.

    Parameters
    ----------
    returns : pd.DataFrame
        Date-indexed T×N DataFrame of asset returns.
    dates : list[pd.Timestamp] | None
        Explicit dates at which to snapshot the covariance matrix.
        Dates that fall between observations are mapped to the nearest
        prior date in the returns index.
    freq : str | None
        Pandas frequency string (e.g. ``"ME"``, ``"QE"``).  Used to
        generate snapshot dates from the returns index when *dates* is
        not provided.  Ignored when *dates* is given.
    method : CovMethod
        Estimation method.
    lookback : int | None
        Window size (number of observations).  ``None`` uses all
        available history up to each date.  Ignored for ``"ewma"``.
    annualize : bool
        When ``True`` (default) multiply each covariance matrix by the
        inferred number of periods per year so that the result is in
        annualised units.
    start_date : pd.Timestamp | str | None
        When provided, activates *expanding-universe* mode.  Snapshots
        before this date are skipped.  At each snapshot date ``d``, the
        asset universe is the set of columns that have at least
        *min_obs* non-NaN observations in ``[start_date, d]``.  As new
        assets accumulate enough history, they are added to the
        covariance matrix — so the matrix grows in size over time.
        The estimation window per snapshot still respects *lookback*
        (capped to begin no earlier than *start_date*).
    min_obs : int
        Minimum number of non-NaN observations an asset needs (within
        ``[start_date, d]``) to be included in the universe at date
        ``d``.  Only used when *start_date* is provided.  Default 2.
    **kwargs
        Forwarded to :func:`compute_covariance`.

    Returns
    -------
    dict[pd.Timestamp, pd.DataFrame]
        Mapping from each requested date to its N×N covariance matrix.

    Raises
    ------
    ValueError
        If neither *dates* nor *freq* is provided, or if a requested
        date precedes the first observation.
    """
    if dates is None and freq is None:
        raise ValueError("Provide either 'dates' or 'freq'.")

    if dates is None:
        dates = pd.date_range(
            returns.index.min(),
            returns.index.max(),
            freq=freq,
        ).tolist()

    if not dates:
        return {}

    if start_date is not None:
        result = _rolling_expanding_universe(
            returns,
            dates,
            pd.Timestamp(start_date),
            method,
            lookback,
            min_obs,
            **kwargs,
        )
    elif method == "ewma":
        result = _rolling_ewma(returns, dates, **kwargs)
    else:
        result = _rolling_windowed(returns, dates, method, lookback, **kwargs)

    if annualize:
        periods_per_year = infer_periods_per_year(returns.index)
        result = {d: cov * periods_per_year for d, cov in result.items()}

    return result


def nearest_psd(matrix: pd.DataFrame) -> pd.DataFrame:
    """Project a symmetric matrix to the nearest positive semi-definite matrix.

    Clips negative eigenvalues to zero — the exact Frobenius-norm
    projection onto the PSD cone.

    Parameters
    ----------
    matrix : pd.DataFrame
        Symmetric N×N matrix that may have small negative eigenvalues.

    Returns
    -------
    pd.DataFrame
        Nearest PSD matrix (same index/columns as input).
    """
    A = matrix.values.astype(float)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    X = (eigvecs * eigvals) @ eigvecs.T
    X = (X + X.T) / 2.0
    return pd.DataFrame(X, index=matrix.index, columns=matrix.columns)


# ---------------------------------------------------------------------------
# Estimators (private)
# ---------------------------------------------------------------------------


def _sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Unbiased sample covariance (ddof=1)."""
    return returns.cov()


def _ewma_cov(returns: pd.DataFrame, halflife: int = 60) -> pd.DataFrame:
    """Exponentially weighted covariance matrix.

    Recursion: ``S_t = (1 - α)·S_{t-1} + α·r_t·r_tᵀ`` with
    ``α = 1 - exp(-ln2 / halflife)``.  Mean is **not** subtracted
    (standard in finance where E[r] ≈ 0 at high frequency).

    The first observation seeds the recursion; allow at least one
    *halflife* of warm-up data for a stable estimate.
    """
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    arr = returns.values  # T×N

    S = np.outer(arr[0], arr[0])
    for t in range(1, len(arr)):
        r = arr[t]
        S = (1 - alpha) * S + alpha * np.outer(r, r)

    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def _ledoit_wolf(
    returns: pd.DataFrame,
    target: str,
    market_ticker: str | None = "EQ_WL",
) -> pd.DataFrame:
    """Ledoit–Wolf shrinkage: ``Σ̂ = (1 - δ)·S + δ·F``."""
    X = returns.values  # T×N
    Xc = X - X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)  # N×N

    if target == "single_factor":
        # Use the specified ticker as the market factor; fall back to 1/N.
        if market_ticker is not None and market_ticker in returns.columns:
            col = returns[market_ticker].values
            market = col - col.mean()
        else:
            market = Xc.mean(axis=1)  # equal-weighted 1/N portfolio
        F = _target_single_factor(Xc, S, market)
    elif target == "constant_correlation":
        F = _target_constant_correlation(Xc, S)
    elif target == "identity":
        F = _target_identity(Xc, S)
    else:
        raise ValueError(f"Unknown shrinkage target {target!r}")

    delta = _shrinkage_intensity(Xc, S, F)
    shrunk = (1 - delta) * S + delta * F

    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def _semicovariance(
    returns: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Semicovariance (downside only).

    Returns above *threshold* are zeroed out before computing the
    covariance, capturing only joint downside risk.
    """
    below = returns.where(returns <= threshold, 0.0)
    return below.cov()


# ---------------------------------------------------------------------------
# Shrinkage targets (private)
# ---------------------------------------------------------------------------


def _target_constant_correlation(
    Xc: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """``F_ij = r̄·√(s_ii·s_jj)`` for i ≠ j; ``F_ii = s_ii``."""
    N = S.shape[0]
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    r_bar = (corr.sum() - N) / (N * (N - 1))
    F = r_bar * np.outer(std, std)
    np.fill_diagonal(F, np.diag(S))
    return F


def _target_single_factor(
    Xc: np.ndarray,
    S: np.ndarray,
    market: np.ndarray,
) -> np.ndarray:
    """Single-factor target: ``F = β·βᵀ·σ²_m + diag(residual)``.

    Parameters
    ----------
    market : np.ndarray
        Length-T centred market return series (either a specific ticker
        or the 1/N equal-weighted portfolio).
    """
    T = Xc.shape[0]
    var_m = np.var(market, ddof=1)

    if var_m < 1e-14:
        return np.diag(np.diag(S))

    betas = (Xc.T @ market) / ((T - 1) * var_m)
    residuals = Xc - np.outer(market, betas)
    resid_var = np.var(residuals, axis=0, ddof=1)

    return var_m * np.outer(betas, betas) + np.diag(resid_var)


def _target_identity(
    Xc: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """``F = (tr S / N)·I``."""
    N = S.shape[0]
    return (np.trace(S) / N) * np.eye(N)


# ---------------------------------------------------------------------------
# Shrinkage intensity (Ledoit & Wolf 2004)
# ---------------------------------------------------------------------------


def _shrinkage_intensity(
    Xc: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
) -> float:
    """Optimal δ* ∈ [0, 1] minimising ``E[‖Σ̂ − Σ‖²_F]``.

    Vectorised derivation (ρ ≈ 0 approximation):

    ``π̂_ij = [(Xc⊙Xc)ᵀ(Xc⊙Xc) / T]_ij − ((T−2)/T)·s²_ij``

    ``δ* = clip(Σ π̂_ij / (T · ‖F − S‖²_F), 0, 1)``
    """
    T = Xc.shape[0]

    Xc2 = Xc * Xc
    sum_xx2 = float(np.einsum("ti,tj->", Xc2, Xc2))  # Σ_ij Σ_t xc²_ti·xc²_tj
    sum_S2 = float(np.sum(S * S))
    pi_hat = sum_xx2 / T - (T - 2) / T * sum_S2

    gamma = float(np.sum((F - S) ** 2))
    if gamma < 1e-14:
        return 1.0

    return float(np.clip(pi_hat / (T * gamma), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rolling helpers (private)
# ---------------------------------------------------------------------------


def _rolling_ewma(
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    halflife: int = 60,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Single-pass EWMA producing snapshots only at *dates*.

    Time complexity: O(T·N²) regardless of how many dates are requested.
    """
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    assets = returns.columns

    # Drop rows with any NaN so that the EWMA recursion is never seeded
    # or updated with NaN values (a single NaN propagates to all future S).
    clean = returns.dropna()
    if clean.empty:
        return {}
    arr = clean.values
    idx = clean.index

    mapping = map_dates_to_index(dates, idx)
    reverse: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for req, actual in mapping.items():
        reverse.setdefault(actual, []).append(req)
    snapshot_set = set(reverse)

    result: dict[pd.Timestamp, pd.DataFrame] = {}

    S = np.outer(arr[0], arr[0])
    if idx[0] in snapshot_set:
        cov = pd.DataFrame(S.copy(), index=assets, columns=assets)
        for req in reverse[idx[0]]:
            result[req] = cov

    for t in range(1, len(arr)):
        r = arr[t]
        S = (1 - alpha) * S + alpha * np.outer(r, r)
        if idx[t] in snapshot_set:
            cov = pd.DataFrame(S.copy(), index=assets, columns=assets)
            for req in reverse[idx[t]]:
                result[req] = cov

    return result


def _rolling_expanding_universe(
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    start_date: pd.Timestamp,
    method: CovMethod,
    lookback: int | None,
    min_obs: int,
    **kwargs,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Expanding-universe rolling estimation.

    At each snapshot date the asset universe is restricted to columns
    with at least *min_obs* non-NaN observations within
    ``[start_date, snapshot_date]``.  The covariance matrix therefore
    grows over time as new assets accumulate enough history.

    Snapshot dates earlier than *start_date* are skipped.
    """
    base = returns.loc[start_date:]
    if base.empty:
        return {}

    snapshot_dates = [d for d in dates if d >= start_date]
    if not snapshot_dates:
        return {}

    mapping = map_dates_to_index(snapshot_dates, base.index)
    result: dict[pd.Timestamp, pd.DataFrame] = {}

    for req, actual in mapping.items():
        loc = base.index.get_loc(actual)
        if lookback is None:
            window = base.iloc[: loc + 1]
        else:
            start = max(0, loc + 1 - lookback)
            window = base.iloc[start : loc + 1]

        counts = window.count()
        universe = counts[counts >= min_obs].index.tolist()
        if len(universe) < 2:
            continue

        sub = window[universe].dropna()
        if len(sub) < min_obs:
            continue

        result[req] = compute_covariance(sub, method=method, **kwargs)

    return result


def _rolling_windowed(
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    method: CovMethod,
    lookback: int | None,
    **kwargs,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Windowed rolling estimation for non-EWMA methods."""
    mapping = map_dates_to_index(dates, returns.index)
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    cache: dict[pd.Timestamp, pd.DataFrame] = {}

    for req, actual in mapping.items():
        if actual in cache:
            result[req] = cache[actual]
            continue

        loc = returns.index.get_loc(actual)
        if lookback is None:
            window = returns.iloc[: loc + 1]
        else:
            start = max(0, loc + 1 - lookback)
            window = returns.iloc[start : loc + 1]

        if len(window.dropna()) < 2:
            continue  # not enough history yet — skip this date

        cov = compute_covariance(window, method=method, **kwargs)
        cache[actual] = cov
        result[req] = cov

    return result
