"""
risk_analytics.py
-----------------
Analytics and visualisation for factor-based covariance models.

Provides:

1. **Out-of-sample covariance testing** — compare predicted vs realised
   covariance at multiple horizons using loss functions (Frobenius,
   log-likelihood, portfolio variance).

2. **Risk decomposition** — break portfolio risk into systematic vs
   idiosyncratic, per-factor, and per-asset contributions.

3. **Risk contribution matrix** — (N×K) matrix of each asset's
   contribution to each factor's risk in a portfolio.

4. **Visualisation** — matplotlib figure builders for covariance heatmaps,
   rolling betas, OOS comparison, risk decomposition, loading snapshots.
   All figure functions accept optional ``method_labels`` / ``colors`` /
   ``linestyles`` dicts so that callers supply experiment-specific styling
   without it leaking into this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from risk_models.factor_covariance import FactorCovResult
from risk_models.utils import cov_to_corr

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Weights = pd.Series | np.ndarray

# ---------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskDecomposition:
    """Full risk decomposition of a portfolio under a factor model.

    Attributes
    ----------
    total_var : float
        Total portfolio variance (annualised if inputs are).
    systematic_var : float
        Variance from factor exposures: ``w' B Σ_f B' w``.
    idiosyncratic_var : float
        Variance from residuals: ``w' diag(σ²_ε) w``.
    systematic_pct : float
        Fraction of total variance explained by factors.
    factor_contributions : pd.Series
        Per-factor contribution to systematic variance.  Sums to
        ``systematic_var``.
    asset_contributions : pd.Series
        Per-asset marginal contribution to total risk (MCTR × weight).
        Sums to ``total_var``.
    """

    total_var: float
    systematic_var: float
    idiosyncratic_var: float
    systematic_pct: float
    factor_contributions: pd.Series
    asset_contributions: pd.Series


def decompose_risk(
    weights: Weights,
    factor_result: FactorCovResult,
) -> RiskDecomposition:
    """Decompose portfolio risk into systematic and idiosyncratic components.

    Parameters
    ----------
    weights : pd.Series | np.ndarray
        N-vector of portfolio weights.  If a Series, its index must
        align with ``factor_result.loadings.index``.
    factor_result : FactorCovResult
        Factor model snapshot from
        :func:`~risk_models.factor_covariance.compute_rolling_asset_factor_cov`.

    Returns
    -------
    RiskDecomposition
        Complete risk breakdown.
    """
    B = factor_result.loadings.values  # N×K
    Sf = factor_result.factor_cov.values  # K×K
    eps_var = factor_result.idiosyncratic_var.values  # N

    if isinstance(weights, pd.Series):
        w = weights.reindex(factor_result.loadings.index).values
    else:
        w = np.asarray(weights, dtype=float)

    # Systematic variance: w' B Σ_f B' w
    f_exposure = B.T @ w  # K — portfolio factor exposures
    systematic_var = float(f_exposure @ Sf @ f_exposure)

    # Idiosyncratic variance: w' diag(σ²_ε) w = Σ w²_i σ²_ε_i
    idio_var = float((w**2) @ eps_var)

    total_var = systematic_var + idio_var
    systematic_pct = systematic_var / total_var if total_var > 0 else 0.0

    # Per-factor contributions: decompose f' Σ_f f
    # Factor j contributes: f_j × (Σ_f @ f)_j
    marginal_factor = Sf @ f_exposure  # K
    factor_contribs = f_exposure * marginal_factor
    factor_contributions = pd.Series(
        factor_contribs,
        index=factor_result.factor_cov.columns,
    )

    # Per-asset marginal contribution to total risk (Euler decomposition)
    # MCTR_i = (Σ_asset @ w)_i / σ_p,  contribution_i = w_i × MCTR_i × σ_p
    # Equivalently: contribution_i = w_i × (Σ_asset @ w)_i
    Sigma = factor_result.asset_cov.values  # N×N
    asset_contribs = w * (Sigma @ w)
    asset_contributions = pd.Series(
        asset_contribs,
        index=factor_result.loadings.index,
    )

    return RiskDecomposition(
        total_var=total_var,
        systematic_var=systematic_var,
        idiosyncratic_var=idio_var,
        systematic_pct=systematic_pct,
        factor_contributions=factor_contributions,
        asset_contributions=asset_contributions,
    )


# ---------------------------------------------------------------------------
# Risk contribution matrix
# ---------------------------------------------------------------------------


def risk_contribution_matrix(
    weights: np.ndarray,
    loadings: np.ndarray,
    factor_cov: np.ndarray,
) -> np.ndarray:
    """Create an (N×K) matrix of per-asset, per-factor risk contributions.

    Element ``(i, j)`` is the contribution of asset *i* to factor *j*
    risk in the portfolio.  Row sums give each asset's total systematic
    risk contribution; column sums give each factor's total risk
    contribution.

    Parameters
    ----------
    weights : np.ndarray
        N-vector of portfolio weights.
    loadings : np.ndarray
        N×K factor loading matrix.
    factor_cov : np.ndarray
        K×K factor covariance matrix.

    Returns
    -------
    np.ndarray
        N×K risk contribution matrix.
    """
    f_exposures = loadings.T @ weights  # K
    sys_vol = np.sqrt(f_exposures @ factor_cov @ f_exposures)

    if sys_vol < 1e-14:
        return np.zeros_like(loadings)

    marginal_factor_risk = factor_cov @ f_exposures  # K
    rc_matrix = (weights[:, np.newaxis] * loadings * marginal_factor_risk) / sys_vol

    return rc_matrix


def risk_contribution_matrix_df(
    weights: pd.Series,
    factor_result: FactorCovResult,
) -> pd.DataFrame:
    """DataFrame wrapper for :func:`risk_contribution_matrix`.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (index = asset names).
    factor_result : FactorCovResult
        Factor model snapshot.

    Returns
    -------
    pd.DataFrame
        N×K DataFrame with asset index and factor columns.
    """
    w = weights.reindex(factor_result.loadings.index).values
    rc = risk_contribution_matrix(
        w,
        factor_result.loadings.values,
        factor_result.factor_cov.values,
    )
    return pd.DataFrame(
        rc,
        index=factor_result.loadings.index,
        columns=factor_result.factor_cov.columns,
    )


# ---------------------------------------------------------------------------
# Covariance-based risk contributions (Euler decomposition, no factor model)
# ---------------------------------------------------------------------------


def marginal_risk_contributions(
    weights: pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """Marginal risk contribution per asset under a plain covariance matrix.

    ``MRC_i = (Σw)_i / √(wᵀΣw)`` — the sensitivity of portfolio
    volatility to a marginal increase in asset *i*'s weight.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (index aligned to *cov* columns).
    cov : pd.DataFrame
        N×N covariance matrix.

    Returns
    -------
    pd.Series
        Marginal risk contribution per asset.  ``NaN`` if the portfolio
        has (near-)zero variance.
    """
    w = weights.reindex(cov.columns).values
    sigma_w = cov.values @ w
    vol = float(np.sqrt(max(w @ sigma_w, 0.0)))
    if vol < 1e-16:
        return pd.Series(
            np.nan, index=cov.columns, name="marginal_risk_contribution"
        )
    return pd.Series(
        sigma_w / vol, index=cov.columns, name="marginal_risk_contribution"
    )


def risk_contributions(
    weights: pd.Series,
    cov: pd.DataFrame,
    relative: bool = False,
) -> pd.Series:
    """Risk contribution per asset under a plain covariance matrix.

    Uses the Euler decomposition ``σ_p = Σ_i w_i · MRC_i``:

    - Absolute (default): ``CTR_i = w_i · (Σw)_i / √(wᵀΣw)`` — sums to
      the portfolio volatility ``σ_p``.
    - Relative (``relative=True``): ``CTR_i / σ_p`` — sums to 1.

    This is the covariance-only analogue of
    :meth:`optimization.risk_budget.RiskBudgetOptimizer.risk_contributions`,
    exposed here so callers can compute risk contributions without
    importing the optimization package.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (index aligned to *cov* columns).
    cov : pd.DataFrame
        N×N covariance matrix.
    relative : bool
        Return fractional contributions (sum to 1) instead of absolute
        volatility contributions (sum to ``σ_p``).

    Returns
    -------
    pd.Series
        Risk contribution per asset.
    """
    w = weights.reindex(cov.columns).values
    sigma_w = cov.values @ w
    var = float(w @ sigma_w)
    if var < 1e-16:
        return pd.Series(np.nan, index=cov.columns, name="risk_contribution")
    vol = np.sqrt(var)
    ctr = w * sigma_w / vol
    if relative:
        ctr = ctr / vol
    return pd.Series(ctr, index=cov.columns, name="risk_contribution")


# ---------------------------------------------------------------------------
# Out-of-sample covariance evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OOSResult:
    """Out-of-sample covariance evaluation result for one date.

    Attributes
    ----------
    date : pd.Timestamp
        Evaluation date (start of the OOS window).
    horizon : int
        Number of forward observations used for realised covariance.
    frobenius_loss : float
        Frobenius norm of (predicted − realised).
    log_likelihood_loss : float
        Gaussian negative log-likelihood:
        ``0.5 × (log|Σ| + tr(Σ⁻¹ S) − log|S| − N)``.
    min_var_predicted : float
        Minimum-variance portfolio volatility under predicted Σ.
    min_var_realised : float
        Actual volatility of the min-var portfolio in the OOS window.
    """

    date: pd.Timestamp
    horizon: int
    frobenius_loss: float
    log_likelihood_loss: float
    min_var_predicted: float
    min_var_realised: float


def evaluate_oos_covariance(
    returns: pd.DataFrame,
    rolling_results: dict[pd.Timestamp, FactorCovResult],
    horizon: int | None = None,
    annualize: bool = True,
) -> list[OOSResult]:
    """Evaluate factor covariance predictions out of sample.

    For each date in *rolling_results*, the predicted asset covariance
    is compared against the realised covariance computed from the next
    *horizon* observations.

    Parameters
    ----------
    returns : pd.DataFrame
        Full T×N asset returns (must extend beyond the last prediction
        date by at least *horizon* observations).
    rolling_results : dict[pd.Timestamp, FactorCovResult]
        Output of :func:`~risk_models.factor_covariance.compute_rolling_asset_factor_cov`.
    horizon : int | None
        Number of forward observations for realised covariance.
        Defaults to ``infer_periods_per_year(returns.index)``
        (≈ 1 year of data).
    annualize : bool
        If ``True``, annualise the realised covariance to match the
        predicted (which is typically annualised).

    Returns
    -------
    list[OOSResult]
        One result per date with sufficient forward data.
    """
    from analytics.timeseries_analyzer import infer_periods_per_year

    if horizon is None:
        horizon = int(infer_periods_per_year(returns.index))

    periods_per_year = infer_periods_per_year(returns.index) if annualize else 1.0

    sorted_dates = sorted(rolling_results.keys())
    results: list[OOSResult] = []

    for date in sorted_dates:
        fcr = rolling_results[date]
        # Find forward window
        future_mask = returns.index > date
        future = returns.loc[future_mask]

        if len(future) < horizon:
            continue

        oos_window = future.iloc[:horizon]
        oos_clean = oos_window.dropna()
        if len(oos_clean) < 2:
            continue

        # Intersect predicted and realised asset universes to avoid NaN
        # poisoning when expanding-universe mode adds assets not yet in
        # the OOS window (or vice versa).
        common_assets = fcr.asset_cov.columns.intersection(oos_clean.columns)
        if len(common_assets) < 2:
            continue

        Sigma_pred = fcr.asset_cov.loc[common_assets, common_assets].values
        S_realised = (
            oos_clean[common_assets].cov().values * periods_per_year
        )

        N = len(common_assets)

        # Frobenius loss
        frob = float(np.linalg.norm(Sigma_pred - S_realised, "fro"))

        # Gaussian log-likelihood loss
        try:
            sign_p, logdet_p = np.linalg.slogdet(Sigma_pred)
            sign_r, logdet_r = np.linalg.slogdet(S_realised)
            if sign_p > 0 and sign_r > 0:
                Sigma_inv = np.linalg.inv(Sigma_pred)
                ll_loss = 0.5 * (
                    logdet_p + np.trace(Sigma_inv @ S_realised) - logdet_r - N
                )
            else:
                ll_loss = np.nan
        except np.linalg.LinAlgError:
            ll_loss = np.nan

        # Minimum-variance portfolio comparison
        try:
            Sigma_inv = np.linalg.inv(Sigma_pred)
            ones = np.ones(N)
            w_mv = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

            vol_pred = float(np.sqrt(max(0.0, w_mv @ Sigma_pred @ w_mv)))
            vol_real = float(np.sqrt(max(0.0, w_mv @ S_realised @ w_mv)))
        except np.linalg.LinAlgError:
            vol_pred = np.nan
            vol_real = np.nan

        results.append(
            OOSResult(
                date=date,
                horizon=horizon,
                frobenius_loss=frob,
                log_likelihood_loss=ll_loss,
                min_var_predicted=vol_pred,
                min_var_realised=vol_real,
            )
        )

    return results


def oos_results_to_df(results: list[OOSResult]) -> pd.DataFrame:
    """Convert a list of OOS results to a summary DataFrame.

    Parameters
    ----------
    results : list[OOSResult]
        Output of :func:`evaluate_oos_covariance`.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with one row per evaluation.
    """
    if not results:
        return pd.DataFrame()

    records = []
    for r in results:
        records.append(
            {
                "date": r.date,
                "horizon": r.horizon,
                "frobenius_loss": r.frobenius_loss,
                "log_likelihood_loss": r.log_likelihood_loss,
                "min_var_predicted_vol": r.min_var_predicted,
                "min_var_realised_vol": r.min_var_realised,
                "min_var_vol_ratio": (
                    r.min_var_realised / r.min_var_predicted
                    if r.min_var_predicted > 0
                    else np.nan
                ),
            }
        )

    df = pd.DataFrame(records).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Visualisation helpers (private)
# ---------------------------------------------------------------------------

_DEFAULT_LINESTYLES: list = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
"""Default linestyle cycle for multi-method plots (up to 5 methods)."""


def _style_ax(ax: plt.Axes, ylabel: str = "", title: str = "") -> None:
    """Apply minimal spine/grid styling to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)


def _label(method_labels: dict[str, str] | None, method: str) -> str:
    """Return the display label for *method*, falling back to the key itself."""
    return (method_labels or {}).get(method, method)


def _color(colors: dict[str, str] | None, method: str, idx: int) -> str:
    """Return a colour for *method*, falling back to matplotlib's ``C{idx}``."""
    return (colors or {}).get(method, f"C{idx % 10}")


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def fig_covariance_heatmaps(
    results: dict[str, dict[pd.Timestamp, object]],
    latest: pd.Timestamp,
    cov_method_labels: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """2×N grid: top row = asset correlations, bottom row = factor correlations.

    Asset correlation titles show the beta method (from *method_labels*).
    Factor correlation titles show the covariance estimator (from
    *cov_method_labels*).

    Parameters
    ----------
    results : dict[str, dict[pd.Timestamp, FactorCovResult]]
        Mapping ``{method_key: rolling_results}`` for each method to display.
    latest : pd.Timestamp
        Snapshot date to plot (must be present in every result dict).
    cov_method_labels : dict[str, str] | None
        Human-readable covariance method name per method key.
        Falls back to the method key when absent.
    method_labels : dict[str, str] | None
        Human-readable beta method name per method key.
        Falls back to the method key when absent.
    """
    methods = list(results.keys())
    n = len(methods)
    _cov_labels = cov_method_labels or {}

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(
        f"Correlation matrices at {latest.strftime('%b %Y')}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for col, method in enumerate(methods):
        fcr = results[method][latest]
        for row, (mat, base_subtitle) in enumerate(
            [
                (fcr.asset_cov, "Asset correlations"),
                (fcr.factor_cov, "Factor correlations"),
            ]
        ):
            ax = axes[row, col]
            corr = cov_to_corr(mat)
            labels = list(corr.columns)

            im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)

            if row == 0:
                title = f"{_label(method_labels, method)}\n{base_subtitle}"
            else:
                cov_label = _cov_labels.get(method, method)
                title = f"Cov: {cov_label}\n{base_subtitle}"

            ax.set_title(title, fontweight="bold", fontsize=9)

            if row == 1:
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        v = corr.values[i, j]
                        ax.text(
                            j,
                            i,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black" if abs(v) < 0.7 else "white",
                        )

            fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    fig.tight_layout()
    return fig


def fig_rolling_correlation(
    all_results: dict[str, dict[pd.Timestamp, object]],
    corr_pairs: list[tuple[str, str]],
    burn_date: pd.Timestamp,
    method_labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    linestyles: list | None = None,
) -> plt.Figure:
    """Rolling pairwise asset correlations for all methods, post-burn.

    Parameters
    ----------
    all_results : dict[str, dict[pd.Timestamp, FactorCovResult]]
        Mapping ``{method_key: rolling_results}`` for each method.
    corr_pairs : list[tuple[str, str]]
        Asset pairs ``(a, b)`` to plot.  Both assets must appear in
        ``asset_cov`` for at least one snapshot.
    burn_date : pd.Timestamp
        Earliest snapshot date to include.
    method_labels, colors, linestyles
        Optional display overrides.  Fall back to method key / ``C{i}`` /
        ``_DEFAULT_LINESTYLES`` when ``None``.
    """
    _ls = linestyles or _DEFAULT_LINESTYLES

    def _extract(res: dict, a: str, b: str) -> pd.Series:
        recs = {}
        for date, fcr in res.items():
            if date < burn_date:
                continue
            cov = fcr.asset_cov
            if a in cov.index and b in cov.index:
                sa = np.sqrt(max(cov.loc[a, a], 1e-14))
                sb = np.sqrt(max(cov.loc[b, b], 1e-14))
                recs[date] = cov.loc[a, b] / (sa * sb)
        return pd.Series(recs).sort_index()

    n_pairs = len(corr_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(13, 4 * n_pairs), squeeze=False)
    fig.suptitle(
        f"Rolling pairwise correlations (quarterly, from {burn_date.year})",
        fontsize=13,
        fontweight="bold",
    )

    for row, (a, b) in enumerate(corr_pairs):
        ax = axes[row, 0]
        for i, (method, res) in enumerate(all_results.items()):
            s = _extract(res, a, b)
            if s.empty:
                continue
            ax.plot(
                s.index,
                s.values,
                label=_label(method_labels, method),
                color=_color(colors, method, i),
                linewidth=1.5,
                linestyle=_ls[i % len(_ls)],
            )

        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_ylim(-1.1, 1.1)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        _style_ax(ax, ylabel="Correlation", title=f"Correlation: {a} vs {b}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def fig_rolling_betas(
    all_results: dict[str, dict[pd.Timestamp, object]],
    assets: list[str],
    factors: list[str],
    burn_date: pd.Timestamp,
    method_labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    linestyles: list | None = None,
) -> plt.Figure:
    """Rolling factor betas for selected assets vs factors, all methods overlaid.

    Parameters
    ----------
    all_results : dict[str, dict[pd.Timestamp, FactorCovResult]]
        Mapping ``{method_key: rolling_results}`` for each method.
    assets : list[str]
        Asset names to chart (filtered to those present in the data).
    factors : list[str]
        Factor names to chart (filtered to those present in the data).
    burn_date : pd.Timestamp
        Earliest snapshot date to include.
    method_labels, colors, linestyles
        Optional display overrides.
    """
    _ls = linestyles or _DEFAULT_LINESTYLES
    first_res = next(iter(all_results.values()))
    latest = max(first_res.keys())
    avail_assets = [a for a in assets if a in first_res[latest].loadings.index]
    avail_factors = [f for f in factors if f in first_res[latest].loadings.columns]
    if not avail_assets:
        avail_assets = list(first_res[latest].loadings.index[:3])
    if not avail_factors:
        avail_factors = list(first_res[latest].loadings.columns[:3])

    n_a, n_f = len(avail_assets), len(avail_factors)
    fig, axes = plt.subplots(n_f, n_a, figsize=(4 * n_a, 3 * n_f), sharex=True)
    if n_f == 1:
        axes = axes[np.newaxis, :]
    if n_a == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Rolling factor betas by method (from {burn_date.year})",
        fontsize=13,
        fontweight="bold",
    )

    for col, asset in enumerate(avail_assets):
        for row, factor in enumerate(avail_factors):
            ax = axes[row, col]
            for i, (method, res) in enumerate(all_results.items()):
                betas = {
                    d: fcr.loadings.loc[asset, factor]
                    for d, fcr in res.items()
                    if d >= burn_date
                    and asset in fcr.loadings.index
                    and factor in fcr.loadings.columns
                }
                if not betas:
                    continue
                s = pd.Series(betas).sort_index()
                ax.plot(
                    s.index,
                    s.values,
                    label=_label(method_labels, method),
                    color=_color(colors, method, i),
                    linewidth=1.2,
                    linestyle=_ls[i % len(_ls)],
                )

            ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
            if row == 0:
                ax.set_title(asset, fontweight="bold", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"β {factor}", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            if row == 0 and col == n_a - 1:
                ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    return fig


def fig_oos_comparison(
    oos_data: dict[str, pd.DataFrame],
    burn_date: pd.Timestamp,
    method_labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """Frobenius loss and MinVar vol ratio over time for all methods, post-burn.

    Parameters
    ----------
    oos_data : dict[str, pd.DataFrame]
        Mapping ``{method_key: oos_df}`` where each DataFrame is the output
        of :func:`oos_results_to_df`.
    burn_date : pd.Timestamp
        Earliest date to include in the plots.
    method_labels, colors
        Optional display overrides.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Out-of-sample covariance evaluation — 12-month horizon (from {burn_date.year})",
        fontsize=13,
        fontweight="bold",
    )

    for i, (method, df) in enumerate(oos_data.items()):
        if df.empty:
            continue
        df_plot = df.loc[df.index >= burn_date]
        if df_plot.empty:
            continue
        kw = dict(
            label=_label(method_labels, method),
            color=_color(colors, method, i),
            linewidth=1.4,
        )
        ax1.plot(df_plot.index, df_plot["frobenius_loss"], **kw)
        ax2.plot(df_plot.index, df_plot["min_var_vol_ratio"], **kw)

    ax2.axhline(
        1.0, color="black", linewidth=0.8, linestyle="--", label="Perfect (1.0)"
    )
    _style_ax(ax1, ylabel="‖Σ_pred − Σ_realised‖_F", title="Frobenius loss")
    _style_ax(
        ax2,
        ylabel="Realised vol / Predicted vol",
        title="Min-variance portfolio vol ratio",
    )
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def fig_risk_decomposition(
    fcr: FactorCovResult,
    weights: pd.Series,
) -> plt.Figure:
    """Systematic vs idiosyncratic vol bars + per-factor variance contributions.

    Parameters
    ----------
    fcr : FactorCovResult
        Factor model snapshot for the portfolio date of interest.
    weights : pd.Series
        Portfolio weights (index = asset names).
    """
    rd = decompose_risk(weights, fcr)
    total_vol = np.sqrt(rd.total_var) * 100
    sys_vol = np.sqrt(rd.systematic_var) * 100
    idio_vol = np.sqrt(rd.idiosyncratic_var) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Risk decomposition — equal-weight portfolio  (total σ = {total_vol:.2f}% p.a.)",
        fontsize=13,
        fontweight="bold",
    )

    bars = ax1.bar(
        ["Systematic", "Idiosyncratic"],
        [sys_vol, idio_vol],
        color=["#2166ac", "#d6604d"],
        alpha=0.85,
        width=0.5,
    )
    for bar, val in zip(bars, [sys_vol, idio_vol]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(sys_vol, idio_vol) * 1.25)
    _style_ax(
        ax1, ylabel="Annualised volatility (%)", title="Systematic vs Idiosyncratic"
    )

    fc_pct = rd.factor_contributions / rd.total_var * 100
    colors_bar = ["#d6604d" if v < 0 else "#2166ac" for v in fc_pct.values]
    bars2 = ax2.barh(
        fc_pct.index.tolist()[::-1],
        fc_pct.values[::-1],
        color=colors_bar[::-1],
        alpha=0.85,
    )
    ax2.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars2, fc_pct.values[::-1]):
        offset = 0.3 if val >= 0 else -0.3
        ax2.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=8,
        )
    _style_ax(ax2, title="Factor contributions to total portfolio variance (%)")
    ax2.set_xlabel("Contribution (%)", fontsize=9)

    fig.tight_layout()
    return fig


def fig_risk_contribution_matrix(
    fcr: FactorCovResult,
    weights: pd.Series,
) -> plt.Figure:
    """N×K risk contribution matrix heatmap with row/column/grand total annotations.

    Parameters
    ----------
    fcr : FactorCovResult
        Factor model snapshot.
    weights : pd.Series
        Portfolio weights (index = asset names).
    """
    rc_df = risk_contribution_matrix_df(weights, fcr)

    row_totals = rc_df.sum(axis=1).rename("Total")
    rc_sorted = rc_df.loc[row_totals.sort_values(ascending=False).index]
    row_totals_sorted = row_totals.loc[rc_sorted.index]

    col_totals = rc_df.sum(axis=0)
    grand_total = float(col_totals.sum())

    display = rc_sorted.copy()
    display["Total"] = row_totals_sorted
    totals_row = col_totals.to_frame().T
    totals_row.index = pd.Index(["Total"])
    totals_row["Total"] = grand_total
    display = pd.concat([display, totals_row])

    n_rows, n_cols = display.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.2), n_rows * 0.45 + 2.5))

    vals = display.values.astype(float)
    vmax = np.nanmax(np.abs(vals[:-1, :-1]))

    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(display.columns.tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(display.index.tolist(), fontsize=8)

    ax.axhline(n_rows - 1.5, color="black", linewidth=1.5)
    ax.axvline(n_cols - 1.5, color="black", linewidth=1.5)

    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            bold = i == n_rows - 1 or j == n_cols - 1
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=7 if not bold else 8,
                fontweight="bold" if bold else "normal",
                color="black" if abs(v) < vmax * 0.6 else "white",
            )

    fig.colorbar(im, ax=ax, shrink=0.6, label="Risk contribution")
    ax.set_title(
        "Risk contribution matrix (N×K) — asset × factor  "
        "|  Total column = per-asset sum  |  Total row = per-factor sum",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def fig_loading_snapshots(
    method: str,
    results: dict[pd.Timestamp, object],
    burn_date: pd.Timestamp,
    n_snapshots: int = 4,
    method_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """N×K factor loading heatmaps at evenly-spaced dates for one beta method.

    Exact-zero loadings (produced by sparse estimators such as group lasso
    or LASSO) are overlaid in grey so that active loadings stand out.
    Dense methods (OLS, EWMA) will have no grey cells.

    Parameters
    ----------
    method : str
        Beta method key — used to look up the display label and to decide
        whether the sparse-zero overlay note is shown.
    results : dict[pd.Timestamp, FactorCovResult]
        Rolling results for one beta method.
    burn_date : pd.Timestamp
        Earliest snapshot date to include (skip warm-up period).
    n_snapshots : int
        Number of evenly-spaced snapshot dates to render.
    method_labels : dict[str, str] | None
        Human-readable method name mapping.  Falls back to *method* key.
    """
    post_burn = {d: fcr for d, fcr in results.items() if d >= burn_date}
    if not post_burn:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No snapshots after burn date", ha="center", va="center")
        return fig

    sorted_dates = sorted(post_burn.keys())
    if len(sorted_dates) <= n_snapshots:
        snap_dates = sorted_dates
    else:
        indices = np.linspace(0, len(sorted_dates) - 1, n_snapshots, dtype=int)
        snap_dates = [sorted_dates[i] for i in indices]

    ref_fcr = post_burn[snap_dates[0]]
    all_factors = list(ref_fcr.loadings.columns)
    all_assets = list(ref_fcr.loadings.index)

    n = len(snap_dates)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(max(5 * n, 12), max(len(all_assets) * 0.45 + 2, 6)),
    )
    if n == 1:
        axes = [axes]

    label = _label(method_labels, method)
    is_sparse = method in ("group_lasso", "lasso", "elastic_net")
    sparse_note = "  (grey = shrunk to zero)" if is_sparse else ""
    fig.suptitle(
        f"Factor loadings over time — {label}{sparse_note}",
        fontsize=12,
        fontweight="bold",
    )

    vmax = max(
        np.nanmax(
            np.abs(
                post_burn[d]
                .loadings.reindex(index=all_assets, columns=all_factors)
                .values
            )
        )
        for d in snap_dates
    )
    if vmax < 1e-10:
        vmax = 1.0

    for ax, date in zip(axes, snap_dates):
        fcr = post_burn[date]
        B = fcr.loadings.reindex(index=all_assets, columns=all_factors).fillna(0.0)
        vals = B.values.astype(float)

        im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        grey_arr = np.full_like(vals, np.nan)
        grey_arr[vals == 0.0] = 0.5
        ax.imshow(grey_arr, cmap="Greys", vmin=0, vmax=1, aspect="auto", alpha=0.5)

        ax.set_xticks(range(len(all_factors)))
        ax.set_xticklabels(all_factors, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_assets)))
        ax.set_yticklabels(all_assets, fontsize=7)
        ax.set_title(date.strftime("%b %Y"), fontweight="bold", fontsize=10)

        for i in range(len(all_assets)):
            for j in range(len(all_factors)):
                v = vals[i, j]
                if abs(v) > 1e-6:
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if abs(v) < vmax * 0.6 else "white",
                    )

        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    fig.tight_layout()
    return fig


def fig_risk_contribution_bars(
    weights_by_label: dict[str, pd.Series],
    cov: pd.DataFrame,
    relative: bool = False,
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """Grouped horizontal bars of risk contribution per asset.

    One bar group per portfolio, so the concentration of risk can be
    compared at a glance (e.g. equal-weight vs risk-parity).

    Parameters
    ----------
    weights_by_label : dict[str, pd.Series]
        Mapping ``{portfolio_label: weights}``.  All weight vectors are
        reindexed to ``cov.columns``.
    cov : pd.DataFrame
        N×N covariance matrix shared by all portfolios.
    relative : bool
        Plot fractional contributions (sum to 1) instead of absolute
        volatility contributions.
    colors : dict[str, str] | None
        Optional per-label colour overrides.
    """
    labels = list(weights_by_label)
    assets = list(cov.columns)
    n_assets, n_port = len(assets), len(labels)

    rc = {
        lbl: risk_contributions(w, cov, relative=relative).reindex(assets)
        for lbl, w in weights_by_label.items()
    }

    y = np.arange(n_assets)
    height = 0.8 / n_port

    fig, ax = plt.subplots(figsize=(9, max(4, n_assets * 0.5)))
    for i, lbl in enumerate(labels):
        offset = (i - (n_port - 1) / 2) * height
        ax.barh(
            y + offset,
            rc[lbl].values,
            height=height,
            label=lbl,
            color=_color(colors, lbl, i),
            alpha=0.85,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(assets, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    unit = "Fraction of total risk" if relative else "Vol contribution (annualised)"
    _style_ax(ax, title="Risk contribution by asset")
    ax.set_xlabel(unit, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_comparison_over_time(
    navs: dict[str, pd.Series],
    weights_by_label: dict[str, pd.DataFrame],
    rc_by_label: dict[str, pd.DataFrame],
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """NAV / weights / risk-contribution comparison across portfolios.

    Layout: row 0 overlays every portfolio's NAV (log scale); row 1 shows
    one stacked weight-area panel per portfolio; row 2 shows one
    risk-contribution line panel per portfolio (with a 1/N reference).

    Reused by the notebook and
    ``saa/examples/run_risk_budget_backtest_example.py`` so the plotting
    logic lives in one place.

    Parameters
    ----------
    navs : dict[str, pd.Series]
        Mapping ``{label: nav_series}``.
    weights_by_label : dict[str, pd.DataFrame]
        Mapping ``{label: weights_df}`` (index = dates, columns = assets).
    rc_by_label : dict[str, pd.DataFrame]
        Mapping ``{label: rc_df}`` of **relative** risk contributions
        (each row sums to 1), aligned with the weight dates.
    colors : dict[str, str] | None
        Optional per-label colour overrides for the NAV panel.
    """
    labels = list(navs)
    n = len(labels)

    fig = plt.figure(figsize=(6.5 * n, 11))
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.1, 1.1])

    ax_nav = fig.add_subplot(gs[0, :])
    for i, lbl in enumerate(labels):
        nav = navs[lbl]
        ax_nav.plot(
            nav.index,
            nav.values,
            label=lbl,
            color=_color(colors, lbl, i),
            linewidth=1.5,
        )
    ax_nav.set_yscale("log")
    ax_nav.legend(fontsize=9, loc="upper left")
    _style_ax(ax_nav, ylabel="NAV (log)", title="Net asset value")

    for i, lbl in enumerate(labels):
        ax_w = fig.add_subplot(gs[1, i])
        weights_by_label[lbl].plot.area(
            ax=ax_w, linewidth=0, alpha=0.85, legend=False
        )
        ax_w.set_ylim(0, 1)
        _style_ax(
            ax_w,
            ylabel="Weight" if i == 0 else "",
            title=f"{lbl} — weights",
        )
        if i == n - 1:
            ax_w.legend(loc="upper left", fontsize=6, ncol=2)

        ax_rc = fig.add_subplot(gs[2, i])
        rc_df = rc_by_label[lbl]
        for col in rc_df.columns:
            ax_rc.plot(rc_df.index, rc_df[col], linewidth=0.8, label=col)
        ax_rc.axhline(
            1.0 / rc_df.shape[1],
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="1/N",
        )
        _style_ax(
            ax_rc,
            ylabel="Risk contribution" if i == 0 else "",
            title=f"{lbl} — risk contributions",
        )

    fig.tight_layout()
    return fig
