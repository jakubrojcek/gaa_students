"""
robust.py
---------
Worst-case (robust) mean-variance optimizer with elliptical uncertainty
sets for **both** the expected-return vector and the covariance matrix.

This is the native GAA analogue of riskfolio-lib's
``wc_optimization(Umu="ellip", Ucov="ellip")``.  The estimated inputs
``(μ̂, Σ̂)`` are treated as the centres of ellipsoidal uncertainty sets,
and the optimizer hedges against the worst realisation inside them.

Formulation
-----------
**Mean ellipsoid** ``Uμ = {μ : (μ−μ̂)ᵀ Σμ⁻¹ (μ−μ̂) ≤ kμ²}``.  The
worst-case (minimum) expected return is

    min_{μ∈Uμ}  μᵀw  =  μ̂ᵀw − kμ · ‖Σμ^{1/2} w‖₂          (a SOC term)

**Covariance ellipsoid** ``UΣ = {Σ : Σ_{ij} estimation-error ellipsoid}``.
The worst-case (maximum) variance is ``max_Σ wᵀΣw``.  Because ``wᵀΣw``
is quadratic in ``w`` we lift it with an auxiliary matrix ``W ⪰ wwᵀ``
(a Schur complement ``[[W, w], [wᵀ, 1]] ⪰ 0`` — a PSD cone) and bound

    max_Σ wᵀΣw  ≤  ⟨Σ̂, W⟩ + kΣ · ‖√V ⊙ W‖_F                (a SOC term)

where ``V_{ij} = Var(Σ̂_{ij})`` is the entrywise estimation variance of
the sample covariance (independent-entry / diagonal-``svec`` ellipsoid).
Minimising over ``W`` drives ``W → wwᵀ`` at the optimum, so the bound is
tight.  Together with the mean term this gives a small SDP solvable by
CLARABEL or SCS (both bundled with cvxpy).

Objectives
----------
- ``"min_variance"`` — minimise the worst-case variance.
- ``"max_utility"`` — maximise ``worst-case return − (λ/2)·worst-case variance``.

Robust **max-Sharpe** is intentionally not provided: the fractional
objective does not compose with the PSD lift without a substantially
larger formulation.  Use ``"max_utility"`` (or a shared objective) when
comparing against classic mean-variance and risk-parity portfolios.

References
----------
- Goldfarb, D. & Iyengar, G. (2003).  *Robust Portfolio Selection
  Problems*.  Mathematics of Operations Research.
- Lobo, M. & Boyd, S. (2000).  *The Worst-Case Risk of a Portfolio*.
- Ben-Tal, A. & Nemirovski, A. (1998).  *Robust Convex Optimization*.
- Tütüncü, R. & Koenig, M. (2004).  *Robust Asset Allocation*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import chi2

from optimization.constraints import (
    PortfolioConstraint,
    build_cvxpy_constraints,
    build_scipy_bounds,
    build_scipy_constraints,
)
from optimization.optimizer import BaseOptimizer, SolverBackend

RobustObjective = Literal["min_variance", "max_utility"]
"""Supported robust objectives."""


# ---------------------------------------------------------------------------
# Uncertainty-set parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorstCaseStats:
    """Estimated centres and sizes of the elliptical uncertainty sets.

    Attributes
    ----------
    mu : pd.Series
        Centre of the mean ellipsoid — annualised expected returns in
        **percentage points** (consistent with ``MeanRiskOptimizer``).
    cov : pd.DataFrame
        Centre of the covariance ellipsoid — annualised covariance in
        **decimal** units.
    cov_mean : pd.DataFrame
        ``Σμ`` — covariance of the mean estimate (decimal, annualised
        return units), used to shape the mean ellipsoid.
    cov_var : pd.DataFrame
        ``V`` — entrywise estimation variance of the sample covariance
        (``V_{ij} = Var(Σ̂_{ij})``), used to shape the covariance ellipsoid.
    k_mu : float
        Radius of the mean ellipsoid (χ² quantile, ``n`` dof).
    k_sigma : float
        Radius of the covariance ellipsoid (χ² quantile, ``n(n+1)/2`` dof).
    """

    mu: pd.Series
    cov: pd.DataFrame
    cov_mean: pd.DataFrame
    cov_var: pd.DataFrame
    k_mu: float
    k_sigma: float


def worst_case_stats(
    returns: pd.DataFrame,
    q: float = 0.05,
    periods_per_year: int = 12,
) -> WorstCaseStats:
    r"""Estimate elliptical uncertainty-set parameters under normality.

    The native analogue of riskfolio-lib's ``wc_stats`` with the
    normal-distribution assumption (``box='n', ellip='n'``):

    - ``Σμ = Σ̂ / T``  (the sampling covariance of the mean vector).
    - ``V_{ij} = (Σ̂_{ii} Σ̂_{jj} + Σ̂_{ij}²) / T``  (the Wishart variance
      of each sample-covariance entry).
    - ``kμ = √χ²_{1−q}(n)``,  ``kΣ = √χ²_{1−q}(n(n+1)/2)``.

    Means and covariances are annualised by *periods_per_year* so they
    are consistent with the rest of the optimization module.

    Parameters
    ----------
    returns : pd.DataFrame
        T×N simple returns at the base frequency (e.g. monthly).
        NaN rows are dropped.
    q : float
        Tail probability for the χ² confidence radii (default 0.05).
    periods_per_year : int
        Annualisation factor for means and covariances (12 for monthly).

    Returns
    -------
    WorstCaseStats
    """
    clean = returns.dropna()
    n_obs = len(clean)
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")

    assets = clean.columns
    n = len(assets)

    mu_ann = clean.mean().values * periods_per_year  # decimal
    cov_ann = clean.cov().values * periods_per_year  # decimal

    cov_mean = cov_ann / n_obs
    diag = np.diag(cov_ann)
    cov_var = (np.outer(diag, diag) + cov_ann**2) / n_obs

    k_mu = float(np.sqrt(chi2.ppf(1.0 - q, df=n)))
    k_sigma = float(np.sqrt(chi2.ppf(1.0 - q, df=n * (n + 1) // 2)))

    return WorstCaseStats(
        mu=pd.Series(mu_ann * 100.0, index=assets, name="expected_return"),
        cov=pd.DataFrame(cov_ann, index=assets, columns=assets),
        cov_mean=pd.DataFrame(cov_mean, index=assets, columns=assets),
        cov_var=pd.DataFrame(cov_var, index=assets, columns=assets),
        k_mu=k_mu,
        k_sigma=k_sigma,
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class RobustMeanVarianceOptimizer(BaseOptimizer):
    """Worst-case mean-variance optimizer (elliptical μ and Σ uncertainty).

    Two solver backends, mirroring the rest of the optimization module:

    - ``"cvxpy"`` (default) — exact convex formulation.  The worst-case
      covariance term is made convex with the ``W ⪰ wwᵀ`` Schur-complement
      lift, giving a semidefinite program (CLARABEL/SCS).
    - ``"scipy"`` — the *same* worst-case objective written in closed form
      (no lift), ``wᵀΣ̂w + kΣ‖√V ⊙ wwᵀ‖_F`` for the variance and
      ``μ̂ᵀw − kμ√(wᵀΣμw)`` for the return, solved by SLSQP.  Convenient
      when an SDP solver is unavailable; cross-validates the cvxpy result.

    Comparison with riskfolio-lib
    -----------------------------
    The uncertainty-set definitions match riskfolio's ``wc_optimization``
    (``Umu='ellip', Ucov='ellip'``): the mean ellipsoid
    ``(μ−μ̂)ᵀΣμ⁻¹(μ−μ̂) ≤ kμ²`` and the covariance ellipsoid on
    ``vec(Σ)``, with χ² radii (``n`` and ``n(n+1)/2`` dof) and
    ``Σμ = Σ̂/T`` under normality.  The one deliberate simplification: we
    shape the covariance ellipsoid with a **diagonal** ``Σ_Σ`` (independent
    entries, ``Var(Σ̂_{ij}) = (Σ̂_{ii}Σ̂_{jj}+Σ̂_{ij}²)/T``) rather than
    riskfolio's full vectorised covariance — this keeps the program a clean
    Frobenius/SOC term and is plenty for teaching.

    Parameters
    ----------
    solver : SolverBackend
        ``"cvxpy"`` (default) or ``"scipy"``.
    long_only : bool
        Constrain weights ≥ 0 (default ``True``).
    transaction_cost_bps : float
        Proportional turnover penalty in basis points (applied to the
        worst-case objective).

    Examples
    --------
    >>> import pandas as pd
    >>> from optimization.robust import RobustMeanVarianceOptimizer, worst_case_stats
    >>> stats = worst_case_stats(returns)              # doctest: +SKIP
    >>> opt = RobustMeanVarianceOptimizer()            # doctest: +SKIP
    >>> w = opt.optimize(stats, objective="max_utility", risk_aversion=2.0)  # doctest: +SKIP
    """

    def __init__(
        self,
        solver: SolverBackend = "cvxpy",
        long_only: bool = True,
        transaction_cost_bps: float = 0.0,
    ) -> None:
        super().__init__(
            solver=solver,
            long_only=long_only,
            transaction_cost_bps=transaction_cost_bps,
        )

    def optimize(
        self,
        stats: WorstCaseStats,
        objective: RobustObjective = "max_utility",
        risk_aversion: float = 1.0,
        constraints: list[PortfolioConstraint] | None = None,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Compute worst-case optimal weights.

        Parameters
        ----------
        stats : WorstCaseStats
            Uncertainty-set parameters (see :func:`worst_case_stats`).
        objective : RobustObjective
            ``"min_variance"`` or ``"max_utility"``.
        risk_aversion : float
            Trade-off coefficient λ for ``"max_utility"`` (ignored for
            ``"min_variance"``).
        constraints : list[PortfolioConstraint] | None
            Additional portfolio constraints.
        current_weights : pd.Series | None
            Current weights for the transaction-cost penalty.

        Returns
        -------
        pd.Series
            Optimal portfolio weights (index = asset names).
        """
        if objective not in ("min_variance", "max_utility"):
            raise ValueError(
                f"objective must be 'min_variance' or 'max_utility', "
                f"got {objective!r}"
            )

        assets = stats.cov.columns
        n = len(assets)
        self._validate_covariance(stats.cov)

        mu = stats.mu.reindex(assets).values / 100.0  # → decimal
        cov = stats.cov.values
        V = stats.cov_var.reindex(index=assets, columns=assets).values
        cov_mean = stats.cov_mean.reindex(index=assets, columns=assets).values
        w_cur = self._align_weights(current_weights, assets, "current_weights")
        cons_in = constraints or []

        args = (
            mu, cov, V, cov_mean, stats.k_mu, stats.k_sigma,
            objective, risk_aversion, assets, cons_in, w_cur,
        )
        if self.solver == "scipy":
            raw = self._solve_scipy(*args)
        elif self.solver == "cvxpy":
            raw = self._solve_cvxpy(*args)
        else:
            raise ValueError(f"Unknown solver {self.solver!r}")
        return pd.Series(raw, index=assets, name="weights")

    # ------------------------------------------------------------------
    # Cvxpy SDP backend
    # ------------------------------------------------------------------

    def _solve_cvxpy(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        V: np.ndarray,
        cov_mean: np.ndarray,
        k_mu: float,
        k_sigma: float,
        objective: RobustObjective,
        risk_aversion: float,
        assets,
        user_constraints: list[PortfolioConstraint],
        current_weights: np.ndarray | None,
    ) -> np.ndarray:
        n = len(mu)
        L_mu = _psd_sqrt(cov_mean)
        w = cp.Variable(n)
        W = cp.Variable((n, n), symmetric=True)

        # Worst-case variance via the W ⪰ wwᵀ lift (Schur complement)
        sqrt_V = np.sqrt(np.maximum(V, 0.0))
        wc_var = cp.trace(cov @ W) + k_sigma * cp.norm(cp.multiply(sqrt_V, W), "fro")

        schur = cp.bmat(
            [
                [W, cp.reshape(w, (n, 1), order="F")],
                [cp.reshape(w, (1, n), order="F"), np.array([[1.0]])],
            ]
        )

        cons: list = [cp.sum(w) == 1, schur >> 0]
        cons.extend(
            build_cvxpy_constraints(user_constraints, w, assets, self.long_only)
        )

        if objective == "min_variance":
            expr = wc_var
        else:  # max_utility
            wc_ret = mu @ w - k_mu * cp.norm(L_mu @ w, 2)
            expr = -wc_ret + (risk_aversion / 2.0) * wc_var

        expr += self._turnover_penalty(w, current_weights)

        prob = cp.Problem(cp.Minimize(expr), cons)
        _solve_sdp(prob)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Robust optimisation failed: {prob.status}")

        return np.asarray(w.value).flatten()

    # ------------------------------------------------------------------
    # Scipy backend (closed-form worst case, no SDP lift)
    # ------------------------------------------------------------------

    def _solve_scipy(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        V: np.ndarray,
        cov_mean: np.ndarray,
        k_mu: float,
        k_sigma: float,
        objective: RobustObjective,
        risk_aversion: float,
        assets,
        user_constraints: list[PortfolioConstraint],
        current_weights: np.ndarray | None,
    ) -> np.ndarray:
        n = len(mu)
        sqrt_V = np.sqrt(np.maximum(V, 0.0))

        def wc_variance(w: np.ndarray) -> float:
            # max_Σ wᵀΣw over the ellipsoid, with W = wwᵀ in closed form
            outer = np.outer(w, w)
            spread = np.sqrt(np.sum((sqrt_V * outer) ** 2))
            return float(w @ cov @ w) + k_sigma * spread

        def wc_return(w: np.ndarray) -> float:
            return float(w @ mu) - k_mu * float(np.sqrt(max(w @ cov_mean @ w, 0.0)))

        def obj(w: np.ndarray) -> float:
            if objective == "min_variance":
                base = wc_variance(w)
            else:  # max_utility
                base = -wc_return(w) + (risk_aversion / 2.0) * wc_variance(w)
            return base + self._turnover_penalty(w, current_weights)

        bounds = build_scipy_bounds(user_constraints, assets, self.long_only)
        constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        constraints.extend(build_scipy_constraints(user_constraints, assets))

        result = scipy_minimize(
            obj,
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"Robust optimisation failed: {result.message}")

        return result.x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition (clips negatives)."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _solve_sdp(prob: cp.Problem) -> None:
    """Solve a PSD-cone problem, trying CLARABEL then falling back to SCS."""
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver)
        except (cp.error.SolverError, cp.error.DCPError):
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            return
