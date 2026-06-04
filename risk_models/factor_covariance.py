"""
factor_covariance.py
--------------------
Factor-based covariance estimation for asset allocation.

Builds on :func:`~risk_models.covariance.compute_rolling_covariance` to
decompose asset covariance into systematic (factor) and idiosyncratic
components:

    Σ_asset = B @ Σ_factor @ B.T + diag(σ²_ε)

Beta estimation methods (``beta_method`` values)
-------------------------------------------------
- ``"static"``      — OLS on the full lookback window
- ``"ewma"``        — exponentially weighted betas (single-pass)
- ``"group_lasso"`` — hierarchical group lasso for sparse factor selection
- ``"lasso"``       — standard LASSO (L1 penalty) via coordinate descent
- ``"elastic_net"`` — elastic net (L1 + L2 penalty) via coordinate descent

For all methods, the K×K factor covariance is estimated separately using
``cov_method`` (e.g. ``"ledoit_wolf_cc"``, ``"ewma"``), not derived from
the beta estimator's internal matrices.

All methods produce per-date snapshots of the N×K loading matrix,
K×K factor covariance, and N-vector of idiosyncratic variances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from analytics.timeseries_analyzer import infer_periods_per_year
from risk_models.covariance import (
    CovMethod,
    compute_covariance,
)
from risk_models.utils import map_dates_to_index

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BetaMethod = Literal["static", "ewma", "group_lasso", "lasso", "elastic_net"]
"""Supported beta estimation methods."""

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorCovResult:
    """Snapshot of a factor-based covariance decomposition at one date.

    Attributes
    ----------
    loadings : pd.DataFrame
        N×K factor loading (beta) matrix.  Index = assets, columns = factors.
    factor_cov : pd.DataFrame
        K×K factor covariance matrix.
    idiosyncratic_var : pd.Series
        N-vector of residual (idiosyncratic) variances per asset.
    asset_cov : pd.DataFrame
        N×N reconstructed asset covariance: ``B @ Σ_f @ B.T + diag(σ²_ε)``.
    """

    loadings: pd.DataFrame
    factor_cov: pd.DataFrame
    idiosyncratic_var: pd.Series
    asset_cov: pd.DataFrame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_rolling_asset_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp] | None = None,
    freq: str | None = None,
    beta_method: BetaMethod = "static",
    lookback: int | None = None,
    cov_method: CovMethod = "sample",
    annualize: bool = True,
    start_date: pd.Timestamp | str | None = None,
    min_obs: int = 24,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Compute rolling factor-based asset covariance decompositions.

    At each requested date the function:

    1. Estimates N×K factor loadings (betas) using ``beta_method``.
    2. Estimates the K×K factor covariance using ``cov_method``.
    3. Computes idiosyncratic variances from regression residuals.
    4. Reconstructs the N×N asset covariance as
       ``B @ Σ_f @ B.T + diag(σ²_ε)``.

    The factor covariance is always estimated via ``cov_method`` on the
    lookback window of factor returns — independent of the beta estimator's
    internal matrices.

    .. note:: **Minimum-observation gating**

       ``min_obs`` defaults to 24 (vs. 2 for ``compute_rolling_covariance``)
       because factor regression requires at least ``K + 2`` rows and is
       unreliable with very few observations.  Per-snapshot, windows with
       fewer than ``K + 2`` clean factor rows are silently skipped.

    Parameters
    ----------
    returns : pd.DataFrame
        T×N asset returns (rows = dates, columns = assets).
    factor_returns : pd.DataFrame
        T×K factor returns, date-aligned with *returns*.
        Columns that also appear in *returns* are valid (e.g. using
        some assets as factors).
    dates : list[pd.Timestamp] | None
        Explicit snapshot dates.
    freq : str | None
        Pandas frequency string to auto-generate dates (e.g. ``"ME"``).
    beta_method : BetaMethod
        How to estimate factor loadings:

        - ``"static"``      — OLS on the lookback window.
        - ``"ewma"``        — exponentially weighted betas.
        - ``"group_lasso"`` — hierarchical group lasso (sparse).
        - ``"lasso"``       — standard LASSO via coordinate descent.
        - ``"elastic_net"`` — elastic net via coordinate descent.
    lookback : int | None
        Observation window for betas and factor covariance.  ``None``
        uses all available history.  For ``"ewma"`` this controls the
        lookback used for factor covariance computation; beta memory
        is controlled by ``halflife``.
    cov_method : CovMethod
        Method for estimating the K×K factor covariance matrix.
    annualize : bool
        Annualise factor covariance and idiosyncratic variance.
    start_date : pd.Timestamp | str | None
        When provided, activates *expanding-universe* mode.  Snapshots
        before this date are skipped.  At each snapshot date ``d``, the
        asset universe is the set of columns that have at least
        *min_obs* non-NaN observations in ``[start_date, d]``.  As new
        assets accumulate enough history, the factor model grows to
        cover them.  Factors themselves are **not** filtered — they are
        expected to be available throughout the requested window.
    min_obs : int
        Minimum number of non-NaN observations an asset needs (within
        ``[start_date, d]``) to be included in the universe at date
        ``d``.  Only used when *start_date* is provided.  Default 24.
    **kwargs
        Extra parameters forwarded to beta estimators and covariance:

        - ``halflife`` (*int*, default ``60``): for ``"ewma"`` betas
          and ``cov_method="ewma"``.
        - ``alpha`` (*float*, default ``0.01``): regularisation strength
          for ``"group_lasso"``, ``"lasso"``, and ``"elastic_net"``.
        - ``l1_ratio`` (*float*, default ``0.5``): fraction of L1 in the
          elastic net penalty (``1.0`` = pure LASSO, ``0.0`` = ridge).
        - ``n_clusters`` (*int* | None): number of factor groups for HCGL.
          Defaults to ``max(1, floor(sqrt(K)))``.
        - ``linkage_method`` (*str*, default ``"ward"``): linkage criterion
          for hierarchical clustering (``"ward"``, ``"complete"``, etc.).
        - ``max_iter`` (*int*, default ``1000``): max iterations for
          regularised estimators.

    Returns
    -------
    dict[pd.Timestamp, FactorCovResult]
        Mapping from each date to its factor covariance decomposition.

    Raises
    ------
    ValueError
        If neither *dates* nor *freq* is given, or columns are missing.
    """
    if dates is None and freq is None:
        raise ValueError("Provide either 'dates' or 'freq'.")

    # Expanding-universe mode ------------------------------------------------
    if start_date is not None:
        return _rolling_expanding_universe_factor_cov(
            returns=returns,
            factor_returns=factor_returns,
            dates=dates,
            freq=freq,
            beta_method=beta_method,
            lookback=lookback,
            cov_method=cov_method,
            annualize=annualize,
            start_date=pd.Timestamp(start_date),
            min_obs=min_obs,
            **kwargs,
        )

    # Align indices ----------------------------------------------------------
    common_idx = returns.index.intersection(factor_returns.index)
    if common_idx.empty:
        raise ValueError("No overlapping dates between returns and factor_returns.")
    returns = returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]

    # Generate dates ---------------------------------------------------------
    if dates is None:
        dates = pd.date_range(
            returns.index.min(),
            returns.index.max(),
            freq=freq,
        ).tolist()
    if not dates:
        return {}

    # Dispatch beta method ---------------------------------------------------
    if beta_method == "ewma":
        return _rolling_ewma_beta_factor_cov(
            returns,
            factor_returns,
            dates,
            lookback,
            cov_method,
            annualize,
            **kwargs,
        )
    if beta_method == "group_lasso":
        return _rolling_group_lasso_beta_factor_cov(
            returns,
            factor_returns,
            dates,
            lookback,
            cov_method,
            annualize,
            **kwargs,
        )
    if beta_method == "static":
        return _rolling_static_beta_factor_cov(
            returns,
            factor_returns,
            dates,
            lookback,
            cov_method,
            annualize,
            **kwargs,
        )
    if beta_method == "lasso":
        return _rolling_lasso_beta_factor_cov(
            returns,
            factor_returns,
            dates,
            lookback,
            cov_method,
            annualize,
            **kwargs,
        )
    if beta_method == "elastic_net":
        return _rolling_elastic_net_beta_factor_cov(
            returns,
            factor_returns,
            dates,
            lookback,
            cov_method,
            annualize,
            **kwargs,
        )

    raise ValueError(
        f"Unknown beta_method {beta_method!r}. Choose from {list(BetaMethod.__args__)}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Beta estimators — static (OLS, optionally windowed)
# ---------------------------------------------------------------------------


def _estimate_static_betas(
    asset_ret: pd.DataFrame,
    factor_ret: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-asset OLS betas and idiosyncratic variances.

    Regression is fit **without an intercept**, assuming mean-zero returns
    (standard at high frequency).

    Each asset is regressed independently against all factors using only
    the rows where both that asset **and** all factors are non-null.
    This allows assets with different start dates to coexist in the same
    snapshot without one short-history asset zeroing out all other assets.

    Assets with fewer observations than ``K + 2`` (under-determined) receive
    NaN loadings and are excluded from ``asset_cov`` reconstruction.

    Returns
    -------
    loadings : pd.DataFrame
        N×K loading matrix (NaN rows for under-determined assets).
    idio_var : pd.Series
        N-vector of residual variances (NaN for under-determined assets).
    """
    K = factor_ret.shape[1]
    # Factor-side clean mask: rows where all factors are non-null
    factor_ok = factor_ret.notna().all(axis=1)
    X_full = factor_ret.loc[factor_ok].values  # T_f×K

    B_rows: list[np.ndarray] = []
    idio_vals: list[float] = []

    for col in asset_ret.columns:
        y_series = asset_ret.loc[factor_ok, col]
        asset_ok = y_series.notna()
        y = y_series.loc[asset_ok].values  # T_i
        X = X_full[asset_ok.values]  # T_i×K

        if len(y) < K + 2:
            B_rows.append(np.full(K, np.nan))
            idio_vals.append(np.nan)
            continue

        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]  # K
        except np.linalg.LinAlgError:
            B_rows.append(np.full(K, np.nan))
            idio_vals.append(np.nan)
            continue

        resid = y - X @ b
        idio_vals.append(float(np.var(resid, ddof=1)))
        B_rows.append(b)

    loadings = pd.DataFrame(
        np.array(B_rows),
        index=asset_ret.columns,
        columns=factor_ret.columns,
    )
    idio_series = pd.Series(idio_vals, index=asset_ret.columns)
    return loadings, idio_series


def _rolling_static_beta_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lookback: int | None,
    cov_method: CovMethod,
    annualize: bool,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Rolling factor model with static (per-asset windowed OLS) betas.

    Each asset is regressed on the factor window independently so that
    short-history assets do not suppress longer-history ones.
    Assets with insufficient observations in the window are assigned NaN
    loadings and excluded from the reconstructed covariance.
    Factor covariance is estimated via ``cov_method`` on the same window.
    """
    periods_per_year = infer_periods_per_year(returns.index) if annualize else 1.0
    mapping = map_dates_to_index(dates, returns.index)
    result: dict[pd.Timestamp, FactorCovResult] = {}
    cache: dict[pd.Timestamp, FactorCovResult] = {}
    K = factor_returns.shape[1]

    for req, actual in mapping.items():
        if actual in cache:
            result[req] = cache[actual]
            continue

        loc = returns.index.get_loc(actual)
        start = 0 if lookback is None else max(0, loc + 1 - lookback)
        end = loc + 1

        asset_window = returns.iloc[start:end]
        factor_window = factor_returns.iloc[start:end]

        factor_clean = factor_window.dropna(how="any")
        if len(factor_clean) < K + 2:
            continue

        factor_cov = compute_covariance(factor_clean, method=cov_method, **kwargs)
        if annualize:
            factor_cov = factor_cov * periods_per_year

        loadings, idio_var = _fit_per_block(
            asset_window=asset_window,
            factor_window=factor_window,
            K=K,
            beta_fn=_estimate_static_betas,
            annualize=annualize,
            base_ppy=periods_per_year,
        )
        if loadings is None:
            continue
        asset_cov = _reconstruct_asset_cov(loadings, factor_cov, idio_var)

        fcr = FactorCovResult(
            loadings=loadings,
            factor_cov=factor_cov,
            idiosyncratic_var=idio_var,
            asset_cov=asset_cov,
        )
        cache[actual] = fcr
        result[req] = fcr

    return result


# ---------------------------------------------------------------------------
# Beta estimators — EWMA
# ---------------------------------------------------------------------------


def _estimate_ewma_betas(
    asset_ret: pd.DataFrame,
    factor_ret: pd.DataFrame,
    halflife: float = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-asset exponentially weighted OLS betas.

    Regression is fit **without an intercept**, assuming mean-zero returns
    (standard at high frequency).

    Each asset is regressed independently against all factors using EWMA
    weights ``w_t = (1−α)^(T−t)`` with ``α = 1 − exp(−ln 2 / halflife)``.
    Closed-form weighted OLS is applied per asset over the rows where both
    that asset and all factors are non-null, so assets with different start
    dates / sampling frequencies coexist cleanly within the same window.

    Returns
    -------
    loadings : pd.DataFrame
        N×K loading matrix.
    idio_var : pd.Series
        N-vector of (weighted) residual variances.
    """
    alpha = 1.0 - np.exp(-np.log(2) / float(halflife))
    K = factor_ret.shape[1]

    factor_ok = factor_ret.notna().all(axis=1)
    X_full = factor_ret.loc[factor_ok].values
    T_f = X_full.shape[0]
    # EWMA weights newest=largest
    w_full = (1 - alpha) ** np.arange(T_f - 1, -1, -1)

    B_rows: list[np.ndarray] = []
    idio_vals: list[float] = []

    for col in asset_ret.columns:
        y_series = asset_ret.loc[factor_ok, col]
        asset_ok = y_series.notna().values
        y = y_series.values[asset_ok]
        X = X_full[asset_ok]
        w = w_full[asset_ok]

        if len(y) < K + 2:
            B_rows.append(np.full(K, np.nan))
            idio_vals.append(np.nan)
            continue

        Wsqrt = np.sqrt(w)
        try:
            b = np.linalg.lstsq(X * Wsqrt[:, None], y * Wsqrt, rcond=None)[0]
        except np.linalg.LinAlgError:
            B_rows.append(np.full(K, np.nan))
            idio_vals.append(np.nan)
            continue

        resid = y - X @ b
        # Weighted residual variance
        wsum = w.sum()
        idio_vals.append(float(np.sum(w * resid * resid) / max(wsum - 1.0, 1e-12)))
        B_rows.append(b)

    loadings = pd.DataFrame(
        np.array(B_rows), index=asset_ret.columns, columns=factor_ret.columns
    )
    idio_series = pd.Series(idio_vals, index=asset_ret.columns)
    return loadings, idio_series


def _rolling_ewma_beta_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lookback: int | None,
    cov_method: CovMethod,
    annualize: bool,
    halflife: int = 60,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Rolling factor model with exponentially weighted (per-asset) betas.

    Each snapshot performs a weighted OLS per asset over its non-NaN rows
    in the lookback window, with weights decaying exponentially toward the
    past.  This per-asset formulation lets the same routine handle assets
    sampled at different native frequencies (the dispatcher splits assets
    by observation pattern via ``_fit_per_block`` and the halflife is
    rescaled per block to keep the effective decay horizon constant).

    Factor covariance is estimated via ``cov_method`` on the lookback
    window of factor returns at each snapshot date.
    """
    ppy = infer_periods_per_year(returns.index) if annualize else 1.0
    mapping = map_dates_to_index(dates, returns.index)
    result: dict[pd.Timestamp, FactorCovResult] = {}
    cache: dict[pd.Timestamp, FactorCovResult] = {}
    K = factor_returns.shape[1]

    for req, actual in mapping.items():
        if actual in cache:
            result[req] = cache[actual]
            continue

        loc = returns.index.get_loc(actual)
        start = 0 if lookback is None else max(0, loc + 1 - lookback)
        end = loc + 1

        asset_window = returns.iloc[start:end]
        factor_window = factor_returns.iloc[start:end]

        factor_clean = factor_window.dropna(how="any")
        if len(factor_clean) < K + 2:
            continue

        cov_kwargs = (
            {"halflife": halflife, **kwargs} if cov_method == "ewma" else kwargs
        )
        try:
            factor_cov = compute_covariance(
                factor_clean, method=cov_method, **cov_kwargs
            )
        except ValueError:
            continue
        if annualize:
            factor_cov = factor_cov * ppy

        loadings, idio_var = _fit_per_block(
            asset_window=asset_window,
            factor_window=factor_window,
            K=K,
            beta_fn=_estimate_ewma_betas,
            annualize=annualize,
            base_ppy=ppy,
            fit_kwargs={"halflife": halflife},
            rescale_halflife=True,
        )
        if loadings is None:
            continue
        asset_cov = _reconstruct_asset_cov(loadings, factor_cov, idio_var)

        fcr = FactorCovResult(
            loadings=loadings,
            factor_cov=factor_cov,
            idiosyncratic_var=idio_var,
            asset_cov=asset_cov,
        )
        cache[actual] = fcr
        result[req] = fcr

    return result


# ---------------------------------------------------------------------------
# Beta estimators — Hierarchical Group Lasso
# ---------------------------------------------------------------------------


def _group_lasso_betas(
    asset_ret: pd.DataFrame,
    factor_ret: pd.DataFrame,
    alpha: float = 0.01,
    groups: list[list[str]] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate sparse factor loadings via group lasso.

    Regression is fit **without an intercept**, assuming mean-zero returns
    (standard at high frequency).

    The group lasso penalises entire groups of factor loadings to zero,
    encouraging parsimonious factor selection per asset.  Groups default
    to one factor per group (equivalent to multi-task lasso).

    Minimises for each asset i:

        (1/2T) ||y_i - X @ b_i||² + α Σ_g √|g| ||b_i[g]||₂

    Solved via block coordinate descent.

    Parameters
    ----------
    asset_ret : pd.DataFrame
        T×N asset returns.
    factor_ret : pd.DataFrame
        T×K factor returns.
    alpha : float
        Regularisation strength.
    groups : list[list[str]] | None
        Factor groupings.  Each inner list contains factor column names
        that form one group.  Defaults to one group per factor.
    max_iter : int
        Maximum coordinate descent iterations.
    tol : float
        Convergence tolerance on coefficient change.

    Returns
    -------
    loadings : pd.DataFrame
        N×K sparse loading matrix.
    idio_var : pd.Series
        N-vector of residual variances.

    .. note::

       Assumes the caller has already aligned and cleaned inputs (see
       ``_estimate_lasso_betas`` note on NaN handling).
    """
    X = factor_ret.values.copy()  # T×K
    Y = asset_ret.values.copy()  # T×N
    T, K = X.shape
    N = Y.shape[1]

    factors = list(factor_ret.columns)

    # Build group index lists
    if groups is None:
        group_idxs = [[j] for j in range(K)]
    else:
        group_idxs = []
        for g in groups:
            idxs = [factors.index(f) for f in g if f in factors]
            if idxs:
                group_idxs.append(idxs)

    # Standardise factors for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)
    X_std[X_std < 1e-12] = 1.0
    X_scaled = (X - X_mean) / X_std

    B = np.zeros((K, N))  # coefficients

    for n_iter in range(max_iter):
        B_old = B.copy()

        for g_idx in group_idxs:
            g_size = len(g_idx)
            sqrt_g = np.sqrt(g_size)

            # Partial residual
            mask = np.ones(K, dtype=bool)
            mask[g_idx] = False
            R = Y - X_scaled[:, mask] @ B[mask, :]  # T×N

            # OLS solution for this group
            Xg = X_scaled[:, g_idx]  # T×|g|
            XgTR = Xg.T @ R / T  # |g|×N

            # Group-wise soft thresholding
            for i in range(N):
                z = XgTR[:, i]
                norm_z = np.linalg.norm(z)
                threshold = alpha * sqrt_g
                if norm_z <= threshold:
                    B[g_idx, i] = 0.0
                else:
                    # Shrinkage factor
                    XgTXg = Xg.T @ Xg / T  # |g|×|g|
                    shrink = 1.0 - threshold / norm_z
                    try:
                        B[g_idx, i] = shrink * np.linalg.solve(XgTXg, z)
                    except np.linalg.LinAlgError:
                        B[g_idx, i] = shrink * np.linalg.lstsq(XgTXg, z, rcond=None)[0]

        # Check convergence
        if np.max(np.abs(B - B_old)) < tol:
            break

    # Un-scale coefficients: B was fit on X_scaled = (X - X_mean) / X_std
    # So y ≈ X_scaled @ B  →  y ≈ X_orig @ (B / X_std)  (ignoring intercept)
    B = B / X_std[:, np.newaxis]

    # Residuals using original (un-centred) factor returns
    residuals = asset_ret.values - factor_ret.values @ B

    idio_var = np.var(residuals, axis=0, ddof=1)

    loadings = pd.DataFrame(B.T, index=asset_ret.columns, columns=factor_ret.columns)
    idio_series = pd.Series(idio_var, index=asset_ret.columns)
    return loadings, idio_series


def _compute_hc_groups(
    factor_ret: pd.DataFrame,
    n_clusters: int | None = None,
    linkage_method: str = "ward",
) -> list[list[str]]:
    """Derive factor groups from hierarchical clustering of factor correlations.

    Factors with high absolute correlation are placed in the same group so
    that the group lasso can select or zero them together.

    Parameters
    ----------
    factor_ret : pd.DataFrame
        T×K factor returns for the current lookback window.
    n_clusters : int | None
        Number of groups to form.  Defaults to ``max(1, floor(sqrt(K)))``.
    linkage_method : str
        Linkage criterion passed to :func:`scipy.cluster.hierarchy.linkage`
        (e.g. ``"ward"``, ``"complete"``, ``"average"``).

    Returns
    -------
    list[list[str]]
        Each inner list contains the factor column names belonging to one group.
    """
    factors = list(factor_ret.columns)
    K = len(factors)
    if K == 1:
        return [factors]

    corr = factor_ret.corr().fillna(0.0).values
    # Distance: highly correlated factors (in either direction) should cluster
    dist = np.clip(1.0 - np.abs(corr), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)

    k = min(n_clusters if n_clusters is not None else max(1, int(np.sqrt(K))), K)
    labels = fcluster(Z, k, criterion="maxclust")

    groups: list[list[str]] = [[] for _ in range(k)]
    for j, lbl in enumerate(labels):
        groups[lbl - 1].append(factors[j])

    return [g for g in groups if g]


def _rolling_group_lasso_beta_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lookback: int | None,
    cov_method: CovMethod,
    annualize: bool,
    alpha: float = 0.01,
    n_clusters: int | None = None,
    linkage_method: str = "ward",
    max_iter: int = 1000,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Rolling factor model with hierarchical clustering group lasso (HCGL) betas.

    For each date window, factor groups are derived from hierarchical clustering
    of the factor correlation matrix, then passed to the group lasso estimator.
    This implements the HCGL model: groups are data-driven and updated at every
    rebalance date rather than fixed in advance.

    Factor covariance is estimated via ``cov_method`` on the same lookback
    window of factor returns.
    """
    ppy = infer_periods_per_year(returns.index) if annualize else 1.0
    mapping = map_dates_to_index(dates, returns.index)
    result: dict[pd.Timestamp, FactorCovResult] = {}
    cache: dict[pd.Timestamp, FactorCovResult] = {}
    K = factor_returns.shape[1]

    def _glasso_fit(a, f, **kw):
        groups = _compute_hc_groups(
            f, n_clusters=n_clusters, linkage_method=linkage_method
        )
        return _group_lasso_betas(a, f, groups=groups, **kw)

    for req, actual in mapping.items():
        if actual in cache:
            result[req] = cache[actual]
            continue

        loc = returns.index.get_loc(actual)
        start = 0 if lookback is None else max(0, loc + 1 - lookback)
        end = loc + 1

        asset_window = returns.iloc[start:end]
        factor_window = factor_returns.iloc[start:end]

        factor_clean = factor_window.dropna(how="any")
        if len(factor_clean) < K + 2:
            continue

        factor_cov = compute_covariance(factor_clean, method=cov_method, **kwargs)
        if annualize:
            factor_cov = factor_cov * ppy

        loadings, idio_var = _fit_per_block(
            asset_window=asset_window,
            factor_window=factor_window,
            K=K,
            beta_fn=_glasso_fit,
            annualize=annualize,
            base_ppy=ppy,
            fit_kwargs={"alpha": alpha, "max_iter": max_iter},
        )
        if loadings is None:
            continue
        asset_cov = _reconstruct_asset_cov(loadings, factor_cov, idio_var)

        fcr = FactorCovResult(
            loadings=loadings,
            factor_cov=factor_cov,
            idiosyncratic_var=idio_var,
            asset_cov=asset_cov,
        )
        cache[actual] = fcr
        result[req] = fcr

    return result


# ---------------------------------------------------------------------------
# Beta estimators — LASSO (standard, per-asset coordinate descent)
# ---------------------------------------------------------------------------


def _estimate_lasso_betas(
    asset_ret: pd.DataFrame,
    factor_ret: pd.DataFrame,
    alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate factor loadings via standard LASSO coordinate descent.

    Regression is fit **without an intercept**, assuming mean-zero returns
    (standard at high frequency).

    Minimises for each asset i:

        (1/2T) ||y_i - X @ b_i||² + α ||b_i||₁

    Solved per-column via cyclic coordinate descent with soft thresholding.
    Factors are standardised before fitting and coefficients are un-scaled
    afterwards so that loadings are in the original return scale.

    Parameters
    ----------
    asset_ret : pd.DataFrame
        T×N asset returns.
    factor_ret : pd.DataFrame
        T×K factor returns.
    alpha : float
        L1 regularisation strength, interpreted in the **original** (un-scaled)
        coefficient space.  A loading whose absolute value would be below
        ``alpha`` is shrunk to zero.
    max_iter : int
        Maximum coordinate descent iterations.
    tol : float
        Convergence tolerance on max absolute coefficient change.

    Returns
    -------
    loadings : pd.DataFrame
        N×K sparse loading matrix.
    idio_var : pd.Series
        N-vector of residual variances.

    .. note::

       Unlike ``_estimate_static_betas``, this estimator does **not**
       handle per-asset NaN patterns internally.  It assumes the caller
       (``_fit_per_block`` → ``_split_assets_by_obs_pattern``) has already
       grouped assets by observation pattern and aligned factor returns,
       so that ``asset_ret`` and ``factor_ret`` are fully non-NaN.
    """
    X = factor_ret.values.copy()  # T×K
    Y = asset_ret.values.copy()  # T×N
    T, K = X.shape

    # Standardise factors for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)
    X_std[X_std < 1e-12] = 1.0
    X_scaled = (X - X_mean) / X_std

    # Pre-compute per-column norms squared (denominator in coordinate updates)
    col_norms_sq = (X_scaled**2).sum(axis=0) / T  # K

    # Scale alpha per factor so that the threshold applies to original-space
    # betas, not scaled ones.  Since b_orig = b_scaled / X_std, applying
    # threshold α*X_std[j] in scaled space zeros out any b_orig whose
    # magnitude would be below α.
    alpha_scaled = alpha * X_std  # K — per-factor threshold in scaled space

    B = np.zeros((K, Y.shape[1]))

    for _ in range(max_iter):
        B_old = B.copy()

        for j in range(K):
            if col_norms_sq[j] < 1e-12:
                B[j, :] = 0.0
                continue

            # Partial residual: Y minus contributions of all factors except j
            R = Y - X_scaled @ B + np.outer(X_scaled[:, j], B[j, :])  # T×N
            z = X_scaled[:, j] @ R / T  # N — unconstrained update

            # Soft threshold in scaled space with per-factor threshold α·σ_j
            abs_z = np.abs(z)
            B[j, :] = (
                np.sign(z) * np.maximum(abs_z - alpha_scaled[j], 0.0) / col_norms_sq[j]
            )

        if np.max(np.abs(B - B_old)) < tol:
            break

    # Un-scale: coefficients fit on X_scaled; convert back to original scale
    B = B / X_std[:, np.newaxis]

    residuals = asset_ret.values - factor_ret.values @ B
    idio_var = np.var(residuals, axis=0, ddof=1)

    loadings = pd.DataFrame(B.T, index=asset_ret.columns, columns=factor_ret.columns)
    idio_series = pd.Series(idio_var, index=asset_ret.columns)
    return loadings, idio_series


def _rolling_lasso_beta_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lookback: int | None,
    cov_method: CovMethod,
    annualize: bool,
    alpha: float = 0.01,
    max_iter: int = 1000,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Rolling factor model with LASSO betas.

    Factor covariance is estimated via ``cov_method`` on the same lookback
    window of factor returns.
    """
    ppy = infer_periods_per_year(returns.index) if annualize else 1.0
    mapping = map_dates_to_index(dates, returns.index)
    result: dict[pd.Timestamp, FactorCovResult] = {}
    cache: dict[pd.Timestamp, FactorCovResult] = {}
    K = factor_returns.shape[1]

    for req, actual in mapping.items():
        if actual in cache:
            result[req] = cache[actual]
            continue

        loc = returns.index.get_loc(actual)
        start = 0 if lookback is None else max(0, loc + 1 - lookback)
        end = loc + 1

        asset_window = returns.iloc[start:end]
        factor_window = factor_returns.iloc[start:end]

        factor_clean = factor_window.dropna(how="any")
        if len(factor_clean) < K + 2:
            continue

        factor_cov = compute_covariance(factor_clean, method=cov_method, **kwargs)
        if annualize:
            factor_cov = factor_cov * ppy

        loadings, idio_var = _fit_per_block(
            asset_window=asset_window,
            factor_window=factor_window,
            K=K,
            beta_fn=_estimate_lasso_betas,
            annualize=annualize,
            base_ppy=ppy,
            fit_kwargs={"alpha": alpha, "max_iter": max_iter},
        )
        if loadings is None:
            continue
        asset_cov = _reconstruct_asset_cov(loadings, factor_cov, idio_var)

        fcr = FactorCovResult(
            loadings=loadings,
            factor_cov=factor_cov,
            idiosyncratic_var=idio_var,
            asset_cov=asset_cov,
        )
        cache[actual] = fcr
        result[req] = fcr

    return result


# ---------------------------------------------------------------------------
# Beta estimators — Elastic Net (L1 + L2 coordinate descent)
# ---------------------------------------------------------------------------


def _estimate_elastic_net_betas(
    asset_ret: pd.DataFrame,
    factor_ret: pd.DataFrame,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate factor loadings via elastic net coordinate descent.

    Regression is fit **without an intercept**, assuming mean-zero returns
    (standard at high frequency).

    Minimises for each asset i:

        (1/2T) ||y_i - X @ b_i||² + α·l1_ratio·||b_i||₁
                                   + (α·(1−l1_ratio)/2)·||b_i||²

    The combined L1 + L2 penalty encourages sparsity (via L1) while
    stabilising correlated factors (via L2).

    Parameters
    ----------
    asset_ret : pd.DataFrame
        T×N asset returns.
    factor_ret : pd.DataFrame
        T×K factor returns.
    alpha : float
        Total regularisation strength.
    l1_ratio : float
        Mix parameter: ``1.0`` = pure LASSO, ``0.0`` = pure ridge.
    max_iter : int
        Maximum coordinate descent iterations.
    tol : float
        Convergence tolerance on max absolute coefficient change.

    Returns
    -------
    loadings : pd.DataFrame
        N×K loading matrix (partially sparse).
    idio_var : pd.Series
        N-vector of residual variances.

    .. note::

       Assumes the caller has already aligned and cleaned inputs (see
       ``_estimate_lasso_betas`` note on NaN handling).
    """
    X = factor_ret.values.copy()  # T×K
    Y = asset_ret.values.copy()  # T×N
    T, K = X.shape

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)
    X_std[X_std < 1e-12] = 1.0
    X_scaled = (X - X_mean) / X_std

    col_norms_sq = (X_scaled**2).sum(axis=0) / T  # K

    # Scale L1 threshold per factor so that it applies to original-space betas.
    lambda1 = alpha * l1_ratio * X_std  # K — per-factor L1 threshold
    lambda2 = alpha * (1.0 - l1_ratio)  # L2 penalty coefficient (uniform)

    B = np.zeros((K, Y.shape[1]))

    for _ in range(max_iter):
        B_old = B.copy()

        for j in range(K):
            # Elastic net denominator includes L2 term
            denom = col_norms_sq[j] + lambda2
            if denom < 1e-12:
                B[j, :] = 0.0
                continue

            R = Y - X_scaled @ B + np.outer(X_scaled[:, j], B[j, :])
            z = X_scaled[:, j] @ R / T

            # Soft threshold for L1 (per-factor) then scale by (norm² + λ₂)
            abs_z = np.abs(z)
            B[j, :] = np.sign(z) * np.maximum(abs_z - lambda1[j], 0.0) / denom

        if np.max(np.abs(B - B_old)) < tol:
            break

    B = B / X_std[:, np.newaxis]

    residuals = asset_ret.values - factor_ret.values @ B
    idio_var = np.var(residuals, axis=0, ddof=1)

    loadings = pd.DataFrame(B.T, index=asset_ret.columns, columns=factor_ret.columns)
    idio_series = pd.Series(idio_var, index=asset_ret.columns)
    return loadings, idio_series


def _rolling_elastic_net_beta_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp],
    lookback: int | None,
    cov_method: CovMethod,
    annualize: bool,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Rolling factor model with elastic net betas.

    Factor covariance is estimated via ``cov_method`` on the same lookback
    window of factor returns.
    """
    ppy = infer_periods_per_year(returns.index) if annualize else 1.0
    mapping = map_dates_to_index(dates, returns.index)
    result: dict[pd.Timestamp, FactorCovResult] = {}
    cache: dict[pd.Timestamp, FactorCovResult] = {}
    K = factor_returns.shape[1]

    for req, actual in mapping.items():
        if actual in cache:
            result[req] = cache[actual]
            continue

        loc = returns.index.get_loc(actual)
        start = 0 if lookback is None else max(0, loc + 1 - lookback)
        end = loc + 1

        asset_window = returns.iloc[start:end]
        factor_window = factor_returns.iloc[start:end]

        factor_clean = factor_window.dropna(how="any")
        if len(factor_clean) < K + 2:
            continue

        factor_cov = compute_covariance(factor_clean, method=cov_method, **kwargs)
        if annualize:
            factor_cov = factor_cov * ppy

        loadings, idio_var = _fit_per_block(
            asset_window=asset_window,
            factor_window=factor_window,
            K=K,
            beta_fn=_estimate_elastic_net_betas,
            annualize=annualize,
            base_ppy=ppy,
            fit_kwargs={"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": max_iter},
        )
        if loadings is None:
            continue
        asset_cov = _reconstruct_asset_cov(loadings, factor_cov, idio_var)

        fcr = FactorCovResult(
            loadings=loadings,
            factor_cov=factor_cov,
            idiosyncratic_var=idio_var,
            asset_cov=asset_cov,
        )
        cache[actual] = fcr
        result[req] = fcr

    return result


# ---------------------------------------------------------------------------
# Mixed-frequency support: per-asset observation grouping + factor aggregation
# ---------------------------------------------------------------------------


def _aggregate_factor_to_index(
    factor_window: pd.DataFrame,
    target_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compound factor returns into bins ending at each *target_idx* date.

    Bin ``k`` spans ``(target_idx[k-1], target_idx[k]]`` — i.e. the factor
    returns realised between consecutive observation dates of an asset
    sampled at lower frequency.  The first bin has no preceding date and
    is dropped.

    .. important::

       This function assumes **simple (arithmetic) returns**: it compounds
       via ``prod(1 + r) - 1``.  Passing continuously-compounded (log)
       returns will produce incorrect bin aggregates.  Callers must ensure
       that factor returns use the ``r`` or ``xr`` transform, **not**
       ``cr``.

    Parameters
    ----------
    factor_window : pd.DataFrame
        T×K **simple** factor returns at the *base* (highest) frequency.
    target_idx : pd.DatetimeIndex
        Observation dates of the lower-frequency asset block.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by a subset of *target_idx* containing the
        compounded factor returns for each bin.  Rows where the factor
        return cannot be computed (target before factor history starts) are
        dropped.
    """
    fw = factor_window.dropna(how="any")
    if fw.empty or len(target_idx) == 0:
        return fw.iloc[0:0]

    # Compound simple returns within each bin: bin_return = prod(1+r) - 1.
    cum = (1.0 + fw).cumprod()
    cum_at_target = (
        cum.reindex(cum.index.union(target_idx)).ffill().reindex(target_idx)
    )
    bins = cum_at_target / cum_at_target.shift(1) - 1.0
    return bins.dropna(how="any")


def _split_assets_by_obs_pattern(
    asset_window: pd.DataFrame,
    factor_window: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Group asset columns by their non-NaN observation date set.

    Each group is a maximal set of assets sharing the same observation
    dates within the window (typical case: one group per native sampling
    frequency, e.g. monthly assets and quarterly assets).  For every group
    the factor returns are aggregated to the group's observation dates via
    :func:`_aggregate_factor_to_index`.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of ``(asset_block, factor_block_aggregated)`` pairs with
        rows already aligned (same DatetimeIndex).  Empty if no group has
        data.
    """
    factor_clean = factor_window.dropna(how="any")
    if factor_clean.empty:
        return []
    factor_lo, factor_hi = factor_clean.index[0], factor_clean.index[-1]

    groups: dict[tuple, list[str]] = {}
    for col in asset_window.columns:
        valid = asset_window.index[asset_window[col].notna()]
        valid = valid[(valid >= factor_lo) & (valid <= factor_hi)]
        if len(valid) == 0:
            continue
        groups.setdefault(tuple(valid), []).append(col)

    out: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for key, cols in groups.items():
        target_idx = pd.DatetimeIndex(key)
        factor_block = _aggregate_factor_to_index(factor_window, target_idx)
        if factor_block.empty:
            continue
        asset_block = asset_window.loc[factor_block.index, cols]
        out.append((asset_block, factor_block))
    return out


def _fit_per_block(
    asset_window: pd.DataFrame,
    factor_window: pd.DataFrame,
    K: int,
    beta_fn,
    annualize: bool,
    base_ppy: float,
    fit_kwargs: dict | None = None,
    rescale_halflife: bool = False,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Run *beta_fn* on each observation-pattern block; concat the results.

    Each block is regressed at its own native frequency: factor returns
    have already been summed into bins matching the block's observation
    dates by :func:`_split_assets_by_obs_pattern`.  Idiosyncratic variances
    are estimated at the block's native frequency and annualised using
    that block's own ``periods_per_year`` so that the final variances live
    on a common (annual) scale.

    .. important:: **Annualisation contract**

       Beta estimators (``_estimate_static_betas``, ``_estimate_ewma_betas``,
       etc.) return idiosyncratic variances in **native-period** units.
       Annualisation is applied here, not inside the estimators.  Do not
       annualise inside a beta estimator — that would double-count.

    Parameters
    ----------
    rescale_halflife : bool
        If True, the ``halflife`` entry in *fit_kwargs* is rescaled to the
        block's native frequency: ``halflife_block = halflife * block_ppy
        / base_ppy``.  Used by the EWMA estimator so that, e.g., a halflife
        of 60 monthly periods becomes 20 for a quarterly block.
    """
    fit_kwargs = dict(fit_kwargs or {})
    blocks = _split_assets_by_obs_pattern(asset_window, factor_window)

    loadings_list: list[pd.DataFrame] = []
    idio_list: list[pd.Series] = []

    for asset_block, factor_block in blocks:
        if len(asset_block) < K + 2:
            continue

        block_kwargs = dict(fit_kwargs)
        block_ppy = (
            infer_periods_per_year(asset_block.index) if annualize else base_ppy
        )
        if rescale_halflife and "halflife" in block_kwargs:
            block_kwargs["halflife"] = max(
                1.0, float(block_kwargs["halflife"]) * (block_ppy / base_ppy)
            )

        loadings, idio_var = beta_fn(asset_block, factor_block, **block_kwargs)

        valid = loadings.index[loadings.notna().all(axis=1)]
        if valid.empty:
            continue
        loadings = loadings.loc[valid]
        idio_var = idio_var.loc[valid]

        if annualize:
            idio_var = idio_var * block_ppy

        loadings_list.append(loadings)
        idio_list.append(idio_var)

    if not loadings_list:
        return None, None

    return pd.concat(loadings_list, axis=0), pd.concat(idio_list, axis=0)


# ---------------------------------------------------------------------------
# Expanding-universe wrapper (private)
# ---------------------------------------------------------------------------


def _rolling_expanding_universe_factor_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    dates: list[pd.Timestamp] | None,
    freq: str | None,
    beta_method: BetaMethod,
    lookback: int | None,
    cov_method: CovMethod,
    annualize: bool,
    start_date: pd.Timestamp,
    min_obs: int,
    **kwargs,
) -> dict[pd.Timestamp, FactorCovResult]:
    """Expanding-universe factor covariance estimation.

    At each snapshot date ``d ≥ start_date`` the asset universe is the
    set of columns with ≥ *min_obs* non-NaN observations in
    ``[start_date, d]``.  The factor model is re-fit on that restricted
    universe via a single-date recursive call into the main dispatcher.

    Less efficient than the single-pass EWMA path (each snapshot is
    fitted independently) but handles all beta methods uniformly and
    keeps per-method logic untouched.
    """
    common_idx = returns.index.intersection(factor_returns.index)
    if common_idx.empty:
        raise ValueError("No overlapping dates between returns and factor_returns.")
    returns = returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]

    if dates is None:
        dates = pd.date_range(
            returns.index.min(),
            returns.index.max(),
            freq=freq,
        ).tolist()
    if not dates:
        return {}

    snapshot_dates = [d for d in dates if d >= start_date]
    if not snapshot_dates:
        return {}

    base_returns = returns.loc[start_date:]
    base_factors = factor_returns.loc[start_date:]
    if base_returns.empty:
        return {}

    result: dict[pd.Timestamp, FactorCovResult] = {}

    for d in snapshot_dates:
        window_rows = base_returns.index[base_returns.index <= d]
        if window_rows.empty:
            continue
        window_returns = base_returns.loc[window_rows]
        window_factors = base_factors.loc[window_rows]

        counts = window_returns.count()
        universe = counts[counts >= min_obs].index.tolist()
        if len(universe) < 1:
            continue

        single = compute_rolling_asset_factor_cov(
            window_returns[universe],
            window_factors,
            dates=[window_rows[-1]],
            beta_method=beta_method,
            lookback=lookback,
            cov_method=cov_method,
            annualize=annualize,
            **kwargs,
        )
        if not single:
            continue
        # The dispatcher may map the requested date to a nearest prior
        # observation; take whatever FactorCovResult it returned.
        result[d] = next(iter(single.values()))

    return result


# ---------------------------------------------------------------------------
# Shared helpers (private)
# ---------------------------------------------------------------------------


def _reconstruct_asset_cov(
    loadings: pd.DataFrame,
    factor_cov: pd.DataFrame,
    idio_var: pd.Series,
) -> pd.DataFrame:
    """Reconstruct N×N asset covariance: B @ Σ_f @ B.T + diag(σ²_ε)."""
    B = loadings.values  # N×K
    Sf = factor_cov.values  # K×K
    systematic = B @ Sf @ B.T  # N×N
    total = systematic + np.diag(idio_var.values)
    return pd.DataFrame(total, index=loadings.index, columns=loadings.index)


