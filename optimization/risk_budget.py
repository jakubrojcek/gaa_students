"""
risk_budget.py
--------------
Risk parity and risk budgeting portfolio optimizer.

Uses the Spinu (2013) log-barrier formulation:

    minimize  (1/2) wᵀΣw  −  Σ bᵢ ln(wᵢ)

where *b* is the target risk-budget vector (sums to 1).  At the
optimum the marginal risk contribution of each asset is proportional
to its budget:

    wᵢ (Σw)ᵢ  ∝  bᵢ

When no user constraints or penalties are active the problem is solved
without a fully-invested constraint and normalised afterwards — convex,
fast, and giving exact risk contributions equal to the target budgets.
Both ``"scipy"`` (L-BFGS-B) and ``"cvxpy"`` (CLARABEL) backends solve
this frictionless path.

When user constraints **or** penalty terms (transaction costs,
tracking error) are present, the optimizer instead minimises the
risk-budgeting *error* directly,

    minimize  Σᵢ (RCᵢ(w) − bᵢ)²  +  tc·‖w − w_cur‖₁  +  λ_TE·(w−bench)ᵀΣ(w−bench)
    s.t.      Σ wᵢ = 1,  w ≥ 0  (+ user constraints)

where ``RCᵢ(w) = wᵢ(Σw)ᵢ / (wᵀΣw)`` is the relative risk contribution.
Optimising the *definition* of risk budgeting needs no convexity
reformulation: the RC-error term is dimensionless, so the turnover and
tracking-error penalties trade off on a stable O(1) scale and the
frictionless problem recovers exact budgets.  The objective is
non-convex, so it is solved with SLSQP regardless of the chosen backend;
the surface is benign for PSD Σ, so a 1/N start converges reliably.

References
----------
- Spinu, F. (2013).  *An Algorithm for Computing Risk Parity Weights*.
- Maillard, S., Roncalli, T. & Teïletche, J. (2010).  *The Properties of
  Equally Weighted Risk Contribution Portfolios*.
- Roncalli, T. (2013).  *Introduction to Risk Parity and Budgeting*.
"""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

from optimization.constraints import (
    PortfolioConstraint,
    build_scipy_bounds,
    build_scipy_constraints,
)
from optimization.optimizer import BaseOptimizer, SolverBackend


class RiskBudgetOptimizer(BaseOptimizer):
    """Risk parity / risk budgeting portfolio optimizer.

    Finds weights whose marginal risk contributions match a target
    budget vector *b*.  Equal budgets (``b = 1/n``) give the classical
    *risk parity* portfolio.

    Parameters
    ----------
    solver : SolverBackend
        ``"scipy"`` or ``"cvxpy"``.
    long_only : bool
        Constrain weights ≥ 0 (default ``True``).
    transaction_cost_bps : float
        Proportional turnover penalty in basis points.
    tracking_error_aversion : float
        Quadratic TE penalty coefficient.

    Examples
    --------
    >>> from optimization.mock_inputs import mock_covariance
    >>> opt = RiskBudgetOptimizer(solver="scipy")
    >>> w = opt.optimize(mock_covariance())
    >>> RiskBudgetOptimizer.risk_contributions(w, mock_covariance())
    """

    def __init__(
        self,
        solver: SolverBackend = "scipy",
        long_only: bool = True,
        transaction_cost_bps: float = 0.0,
        tracking_error_aversion: float = 0.0,
    ) -> None:
        super().__init__(
            solver, long_only, transaction_cost_bps, tracking_error_aversion
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        covariance: pd.DataFrame,
        risk_budgets: pd.Series | None = None,
        constraints: list[PortfolioConstraint] | None = None,
        benchmark: pd.Series | None = None,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Compute risk-budgeted portfolio weights.

        Parameters
        ----------
        covariance : pd.DataFrame
            N×N annualised covariance matrix (decimal).
        risk_budgets : pd.Series | None
            Target risk budget per asset (must sum to 1).
            ``None`` defaults to equal risk budgets (1/N).
        constraints : list[PortfolioConstraint] | None
            Additional portfolio constraints.
        benchmark : pd.Series | None
            Benchmark weights (for tracking-error penalty).
        current_weights : pd.Series | None
            Current portfolio weights (for transaction-cost penalty).

        Returns
        -------
        pd.Series
            Optimal portfolio weights.
        """
        assets = covariance.columns
        n = len(assets)
        self._validate_covariance(covariance)

        # Default to equal risk budgets
        if risk_budgets is None:
            budgets = np.ones(n) / n
        else:
            budgets = risk_budgets.reindex(assets).values
            if np.any(budgets <= 0):
                raise ValueError("All risk budgets must be strictly positive")
            budgets = budgets / budgets.sum()  # normalise to sum to 1

        cov = covariance.values
        bench = self._align_weights(benchmark, assets, "benchmark")
        w_cur = self._align_weights(current_weights, assets, "current_weights")
        cons = constraints or []

        raw = self._dispatch(cov, budgets, bench, w_cur, assets, cons)
        return pd.Series(raw, index=assets, name="weights")

    # ------------------------------------------------------------------
    # Risk contribution diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def risk_contributions(
        weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """Compute relative marginal risk contributions.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights.
        covariance : pd.DataFrame
            N×N covariance matrix.

        Returns
        -------
        pd.Series
            Fractional risk contribution per asset (sums to 1).
            ``rc_i = w_i · (Σw)_i / wᵀΣw``.
        """
        w = weights.reindex(covariance.columns).values
        cov = covariance.values
        sigma_w = cov @ w
        marginal = w * sigma_w
        total = float(w @ sigma_w)
        if total < 1e-16:
            return pd.Series(
                np.nan, index=covariance.columns, name="risk_contributions"
            )
        return pd.Series(
            marginal / total,
            index=covariance.columns,
            name="risk_contributions",
        )

    # ------------------------------------------------------------------
    # Solver dispatch
    # ------------------------------------------------------------------

    def _use_constrained(
        self,
        user_constraints: list[PortfolioConstraint],
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
    ) -> bool:
        """Whether to use the constrained formulation.

        The unconstrained Spinu + normalise path gives exact risk
        budgets but cannot accommodate user constraints or penalty
        terms.  Fall back to the constrained path when needed.
        """
        if user_constraints:
            return True
        if current_weights is not None and self.transaction_cost_bps > 0:
            return True
        if benchmark is not None and self.tracking_error_aversion > 0:
            return True
        return False

    def _dispatch(
        self,
        cov: np.ndarray,
        budgets: np.ndarray,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
    ) -> np.ndarray:
        """Route to the selected solver backend."""
        if self.solver == "scipy":
            return self._solve_scipy(
                cov,
                budgets,
                benchmark,
                current_weights,
                assets,
                user_constraints,
            )
        if self.solver == "cvxpy":
            return self._solve_cvxpy(
                cov,
                budgets,
                benchmark,
                current_weights,
                assets,
                user_constraints,
            )
        raise ValueError(f"Unknown solver {self.solver!r}")

    # ------------------------------------------------------------------
    # Scipy backend
    # ------------------------------------------------------------------

    def _solve_scipy(
        self,
        cov: np.ndarray,
        budgets: np.ndarray,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
    ) -> np.ndarray:
        n = len(budgets)
        constrained = self._use_constrained(
            user_constraints, benchmark, current_weights
        )

        if constrained:
            return self._solve_constrained_rc_error(
                cov,
                budgets,
                benchmark,
                current_weights,
                assets,
                user_constraints,
            )

        # --- Unconstrained Spinu: exact risk budgeting -----------------
        return self._unconstrained_spinu(cov, budgets)

    @staticmethod
    def _unconstrained_spinu(cov: np.ndarray, budgets: np.ndarray) -> np.ndarray:
        """Solve the unconstrained Spinu log-barrier problem and normalise.

        ``min (1/2) wᵀΣw − Σ bᵢ ln wᵢ`` then ``w / Σw``.  This gives the
        exact risk-budgeting portfolio (each ``RC_i ∝ b_i``).  Uses
        L-BFGS-B (box bounds only), which handles extreme budget ratios
        more robustly than SLSQP.
        """
        n = len(budgets)
        eps = 1e-10

        def obj(w: np.ndarray) -> float:
            return 0.5 * float(w @ cov @ w) - float(
                budgets @ np.log(np.maximum(w, eps))
            )

        def grad(w: np.ndarray) -> np.ndarray:
            return cov @ w - budgets / np.maximum(w, eps)

        # Initial guess: diagonal solution w_i = sqrt(b_i / Σ_ii)
        x0 = np.maximum(np.sqrt(budgets / np.diag(cov)), eps)

        result = scipy_minimize(
            obj,
            x0,
            jac=grad,
            method="L-BFGS-B",
            bounds=[(eps, None)] * n,
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(
                f"Scipy risk-budget optimisation failed: {result.message}"
            )

        w_opt = result.x
        return w_opt / w_opt.sum()

    def _solve_constrained_rc_error(
        self,
        cov: np.ndarray,
        budgets: np.ndarray,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
    ) -> np.ndarray:
        """Constrained / penalised risk budgeting via a direct RC-error NLP.

        Minimises ``Σᵢ (RCᵢ(w) − bᵢ)²`` plus the turnover and tracking-error
        penalties, subject to ``Σw = 1``, weight bounds, and user
        constraints, where ``RCᵢ(w) = wᵢ(Σw)ᵢ / (wᵀΣw)``.

        Optimising the definition of risk budgeting directly avoids any
        convexity reformulation (and the associated coefficient tuning):
        the RC-error term is dimensionless, so a basis-point turnover cost
        trades off on a stable scale and the frictionless problem recovers
        exact budgets.  The objective is non-convex, but the surface is
        benign for PSD Σ, so SLSQP from a 1/N start converges reliably —
        this path is therefore used for both the ``"scipy"`` and
        ``"cvxpy"`` backends.
        """
        n = len(budgets)
        eps = 1e-12

        # Bounds: long-only / user bounds (non-negativity ensures RCᵢ valid)
        weight_bounds = build_scipy_bounds(user_constraints, assets, self.long_only)
        bounds = [
            (max(lo if lo is not None else 0.0, 0.0), hi) for lo, hi in weight_bounds
        ]

        # Constraints: fully-invested + user
        constraints: list[dict] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        constraints.extend(build_scipy_constraints(user_constraints, assets))

        def obj(w: np.ndarray) -> float:
            sigma_w = cov @ w
            var = float(w @ sigma_w)
            rc = w * sigma_w / max(var, eps)  # relative risk contributions
            rc_error = float(np.sum((rc - budgets) ** 2))
            tp = self._turnover_penalty(w, current_weights)
            tep = self._tracking_error_penalty(w, benchmark, cov)
            return rc_error + tp + tep

        result = scipy_minimize(
            obj,
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 2000},
        )
        if not result.success:
            raise RuntimeError(
                f"Risk-budget optimisation failed: {result.message}"
            )

        return result.x

    # ------------------------------------------------------------------
    # Cvxpy backend
    # ------------------------------------------------------------------

    def _solve_cvxpy(
        self,
        cov: np.ndarray,
        budgets: np.ndarray,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
    ) -> np.ndarray:
        n = len(budgets)
        constrained = self._use_constrained(
            user_constraints, benchmark, current_weights
        )

        if constrained:
            # The constrained / penalised risk-budgeting objective is
            # non-convex, so it is solved by the SLSQP NLP regardless of the
            # selected backend.
            return self._solve_constrained_rc_error(
                cov,
                budgets,
                benchmark,
                current_weights,
                assets,
                user_constraints,
            )

        # --- Unconstrained Spinu: exact risk budgeting -----------------
        w = cp.Variable(n, pos=True)

        risk = 0.5 * cp.quad_form(w, cp.psd_wrap(cov))
        log_barrier = budgets @ cp.log(w)

        prob = cp.Problem(cp.Minimize(risk - log_barrier))
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Cvxpy risk-budget optimisation failed: {prob.status}")

        w_val = np.array(w.value).flatten()
        return w_val / w_val.sum()
