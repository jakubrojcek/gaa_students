"""
mean_risk.py
------------
Formula-based mean-risk portfolio optimizer.

Operates on expected returns (μ), a covariance matrix (Σ), and
optionally a scenario returns matrix for empirical risk measures.

Objectives are passed as :class:`~optimization.objectives.Objective`
dataclasses (e.g. ``MinRisk``, ``MaxSharpe``, ``MaxUtility``), each
composing a pluggable :class:`~optimization.risk_measures.RiskMeasure`.

Both ``"scipy"`` (SLSQP) and ``"cvxpy"`` (CLARABEL) backends are
supported.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

from optimization.constraints import (
    PortfolioConstraint,
    WeightBounds,
    build_cvxpy_constraints,
    build_scipy_bounds,
    build_scipy_constraints,
)
from optimization.objectives import (
    MaxReturn,
    MaxSharpe,
    MaxUtility,
    MinRisk,
    Objective,
    TargetReturn,
)
from optimization.optimizer import BaseOptimizer
from optimization.risk_measures import RiskMeasure

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EfficientFrontierObjective = Literal[
    "target_return",
    "target_risk",
    "max_utility",
]
"""Parameterisation for :meth:`MeanRiskOptimizer.efficient_frontier`."""


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class MeanRiskOptimizer(BaseOptimizer):
    """Formula-based mean-risk portfolio optimizer.

    Uses expected returns (percentage points) and an annualised
    covariance matrix.  Transaction-cost and tracking-error penalties
    are inherited from :class:`BaseOptimizer`.

    Examples
    --------
    >>> from optimization.mock_inputs import mock_expected_returns, mock_covariance
    >>> from optimization.risk_measures import Variance
    >>> from optimization.objectives import MinRisk
    >>> opt = MeanRiskOptimizer(solver="scipy")
    >>> w = opt.optimize(mock_expected_returns(), mock_covariance(), MinRisk(Variance()))
    """

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        objective: Objective,
        constraints: list[PortfolioConstraint] | None = None,
        benchmark: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        scenario_returns: np.ndarray | pd.DataFrame | None = None,
    ) -> pd.Series:
        """Compute optimal weights.

        Parameters
        ----------
        expected_returns : pd.Series
            Annualised expected returns in **percentage points**.
        covariance : pd.DataFrame
            N×N annualised covariance matrix (decimal).
        objective : Objective
            Optimisation objective dataclass (``MinRisk``, ``MaxSharpe``,
            ``MaxUtility``, ``TargetReturn``, ``MaxReturn``).
        constraints : list[PortfolioConstraint] | None
            Additional portfolio constraints.
        benchmark : pd.Series | None
            Benchmark weights (for tracking-error penalty).
        current_weights : pd.Series | None
            Current portfolio weights (for transaction-cost penalty).
        scenario_returns : np.ndarray | pd.DataFrame | None
            T×N matrix of historical or simulated returns.  Required
            when the objective's risk measure needs scenarios (e.g. CVaR).

        Returns
        -------
        pd.Series
            Optimal portfolio weights.
        """
        assets = covariance.columns
        self._validate_covariance(covariance)
        self._validate_expected_returns(expected_returns, assets)

        mu = expected_returns.reindex(assets).values / 100.0
        cov = covariance.values
        bench = self._align_weights(benchmark, assets, "benchmark")
        w_cur = self._align_weights(current_weights, assets, "current_weights")
        cons = constraints or []
        sc_ret = np.asarray(scenario_returns) if scenario_returns is not None else None

        # Validate scenario_returns requirement
        rm = getattr(objective, "risk_measure", None)
        if rm is not None and rm.requires_scenarios and sc_ret is None:
            raise ValueError(
                f"{type(rm).__name__} requires scenario_returns"
            )

        raw = self._dispatch(mu, cov, objective, bench, w_cur, assets, cons, sc_ret)
        return pd.Series(raw, index=assets, name="weights")

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        risk_measure: RiskMeasure,
        efficient_frontier_objective: EfficientFrontierObjective = "target_return",
        n_points: int = 20,
        constraints: list[PortfolioConstraint] | None = None,
        benchmark: pd.Series | None = None,
        current_weights: pd.Series | None = None,
        scenario_returns: np.ndarray | pd.DataFrame | None = None,
    ) -> dict[int, pd.Series]:
        """Trace the efficient frontier by sweeping an objective parameter.

        Parameters
        ----------
        expected_returns : pd.Series
            Annualised expected returns in **percentage points**.
        covariance : pd.DataFrame
            N×N annualised covariance matrix (decimal).
        risk_measure : RiskMeasure
            Risk measure used for all frontier portfolios.
        efficient_frontier_objective : EfficientFrontierObjective
            Which parameterisation to sweep:

            - ``"target_return"`` — vary return from min-risk to
              max-return (minimise risk at each level).
            - ``"target_risk"`` — vary risk ceiling from min-risk to
              max-risk (maximise return at each level).
            - ``"max_utility"`` — vary risk aversion from high
              (≈ min-risk) to low (≈ max-return).
        n_points : int
            Number of frontier portfolios (default 20).
        constraints : list[PortfolioConstraint] | None
            Portfolio constraints applied to every frontier portfolio.
        benchmark, current_weights
            Forwarded to :meth:`optimize`.
        scenario_returns : np.ndarray | pd.DataFrame | None
            Required when *risk_measure* needs scenarios (e.g. CVaR).

        Returns
        -------
        dict[int, pd.Series]
            ``{0: weights, 1: weights, ...}`` ordered from low-risk
            to high-risk end of the frontier.
        """
        if efficient_frontier_objective not in (
            "target_return", "target_risk", "max_utility",
        ):
            raise ValueError(
                f"efficient_frontier supports 'target_return', 'target_risk', "
                f"or 'max_utility', got {efficient_frontier_objective!r}"
            )

        common: dict = dict(
            expected_returns=expected_returns,
            covariance=covariance,
            constraints=constraints,
            benchmark=benchmark,
            current_weights=current_weights,
            scenario_returns=scenario_returns,
        )

        # --- Anchor: min-risk portfolio (under same constraints) ----------
        w_minrisk = self.optimize(
            **common, objective=MinRisk(risk_measure=risk_measure),
        )
        mu_dec = expected_returns.reindex(covariance.columns).values / 100.0
        cov_arr = covariance.values
        data: dict = {"cov": cov_arr}
        if scenario_returns is not None:
            data["scenario_returns"] = np.asarray(scenario_returns)

        minrisk_ret = float(w_minrisk.values @ mu_dec) * 100.0
        minrisk_risk = risk_measure.evaluate(w_minrisk.values, **data)

        # --- Anchor: max-return portfolio (generous risk ceiling) ---------
        n = len(mu_dec)
        max_risk_ceiling = max(
            risk_measure.evaluate(np.eye(n)[i], **data) for i in range(n)
        ) * 1.01
        w_maxret = self.optimize(
            **common,
            objective=MaxReturn(risk_measure=risk_measure, max_risk=max_risk_ceiling),
        )
        maxret_ret = float(w_maxret.values @ mu_dec) * 100.0
        maxret_risk = risk_measure.evaluate(w_maxret.values, **data)

        # --- Sweep --------------------------------------------------------
        if efficient_frontier_objective == "target_return":
            eps = (maxret_ret - minrisk_ret) * 0.01
            targets = np.linspace(minrisk_ret + eps, maxret_ret, n_points)
            return self._sweep(common, [
                TargetReturn(risk_measure=risk_measure, target_return=t)
                for t in targets
            ])

        if efficient_frontier_objective == "target_risk":
            eps = (maxret_risk - minrisk_risk) * 0.01
            targets = np.linspace(minrisk_risk + eps, maxret_risk, n_points)
            return self._sweep(common, [
                MaxReturn(risk_measure=risk_measure, max_risk=t)
                for t in targets
            ])

        # max_utility
        lam_values = np.linspace(10, 1e-3, n_points)
        return self._sweep(common, [
            MaxUtility(risk_measure=risk_measure, risk_aversion=lam)
            for lam in lam_values
        ])

    def _sweep(
        self,
        common: dict,
        objectives: list[Objective],
    ) -> dict[int, pd.Series]:
        """Solve a sequence of objectives, skipping failures."""
        results: dict[int, pd.Series] = {}
        idx = 0
        for obj in objectives:
            try:
                w = self.optimize(**common, objective=obj)
                results[idx] = w
                idx += 1
            except RuntimeError:
                warnings.warn(
                    f"Solver failed for {obj!r}, skipping.",
                    stacklevel=3,
                )
        return results

    # ------------------------------------------------------------------
    # Solver dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        objective: Objective,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
        scenario_returns: np.ndarray | None,
    ) -> np.ndarray:
        """Route to the selected solver backend."""
        if self.solver == "scipy":
            return self._solve_scipy(
                mu, cov, objective, benchmark, current_weights,
                assets, user_constraints, scenario_returns,
            )
        if self.solver == "cvxpy":
            return self._solve_cvxpy(
                mu, cov, objective, benchmark, current_weights,
                assets, user_constraints, scenario_returns,
            )
        raise ValueError(f"Unknown solver {self.solver!r}")

    # ------------------------------------------------------------------
    # Scipy backend
    # ------------------------------------------------------------------

    def _solve_scipy(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        objective: Objective,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
        scenario_returns: np.ndarray | None,
    ) -> np.ndarray:
        n = len(mu)
        data: dict = {"cov": cov}
        if scenario_returns is not None:
            data["scenario_returns"] = scenario_returns

        # --- Auxiliary variables from risk measure ------------------------
        rm = getattr(objective, "risk_measure", None)
        n_scenarios = scenario_returns.shape[0] if scenario_returns is not None else 0
        n_aux = rm.scipy_n_auxiliary(n_scenarios) if rm else 0

        # --- Bounds -------------------------------------------------------
        weight_bounds = build_scipy_bounds(user_constraints, assets, self.long_only)
        aux_bounds = rm.scipy_auxiliary_bounds(n_scenarios) if n_aux > 0 else []
        all_bounds = weight_bounds + aux_bounds

        # --- Constraints: fully-invested + user + objective ---------------
        constraints: list[dict] = [
            {"type": "eq", "fun": lambda x, _n=n: np.sum(x[:_n]) - 1.0},
        ]

        # User constraints (wrap for augmented vector when needed)
        user_cons = build_scipy_constraints(user_constraints, assets)
        if n_aux > 0:
            for c in user_cons:
                orig = c["fun"]
                constraints.append({
                    "type": c["type"],
                    "fun": lambda x, _f=orig, _n=n: _f(x[:_n]),
                })
        else:
            constraints.extend(user_cons)

        # Objective function + objective-specific constraints
        obj_fn_base, obj_extra_cons = objective.build_scipy(mu, n, **data)
        constraints.extend(obj_extra_cons)

        # --- Wrap with penalty terms --------------------------------------
        def obj_fn(x: np.ndarray) -> float:
            w = x[:n]
            base = obj_fn_base(x)
            tp = self._turnover_penalty(w, current_weights)
            tep = self._tracking_error_penalty(w, benchmark, cov)
            penalty = tp + tep

            # MaxSharpe: penalties enter the numerator of -ret/σ
            if isinstance(objective, MaxSharpe) and penalty != 0.0:
                var = float(w @ cov @ w)
                if var > 1e-14:
                    return base + penalty / np.sqrt(var)
                return base

            return base + penalty

        # --- Initial guess ------------------------------------------------
        x0 = np.ones(n) / n
        if n_aux > 0:
            x0 = np.concatenate([x0, rm.scipy_auxiliary_x0(n_scenarios)])

        # --- Solve --------------------------------------------------------
        result = scipy_minimize(
            obj_fn, x0, method="SLSQP",
            bounds=all_bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"Scipy optimisation failed: {result.message}")

        return result.x[:n]

    # ------------------------------------------------------------------
    # Cvxpy backend
    # ------------------------------------------------------------------

    def _solve_cvxpy(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        objective: Objective,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
        scenario_returns: np.ndarray | None,
    ) -> np.ndarray:
        # MaxSharpe uses a dedicated variance-normalised transform
        if isinstance(objective, MaxSharpe):
            return self._solve_max_sharpe_cvxpy(
                mu, cov, benchmark, current_weights,
                assets, user_constraints,
            )

        n = len(mu)
        w = cp.Variable(n)
        data: dict = {"cov": cov}
        if scenario_returns is not None:
            data["scenario_returns"] = scenario_returns

        # Constraints: fully-invested + long-only/user bounds + objective
        cons: list = [cp.sum(w) == 1]
        cons.extend(build_cvxpy_constraints(user_constraints, w, assets, self.long_only))

        obj_expr, obj_extra_cons = objective.build_cvxpy(w, mu, **data)
        cons.extend(obj_extra_cons)

        # Penalties
        expr = obj_expr
        expr += self._turnover_penalty(w, current_weights)
        expr += self._tracking_error_penalty(w, benchmark, cov)

        prob = cp.Problem(cp.Minimize(expr), cons)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Cvxpy optimisation failed: {prob.status}")

        return np.array(w.value).flatten()

    def _solve_max_sharpe_cvxpy(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        benchmark: np.ndarray | None,
        current_weights: np.ndarray | None,
        assets: Sequence[str],
        user_constraints: list[PortfolioConstraint],
    ) -> np.ndarray:
        """Max TC-adjusted Sharpe via a variance-normalised transform.

        Introduces ``y = w · t`` where ``t ≥ 0`` is a scaling factor.
        Constraining ``yᵀΣy ≤ 1`` makes ``t = 1/√(wᵀΣw)`` at the
        optimum, so the objective

            ``μᵀy − tc · ‖y − t·w_cur‖₁ − λ_TE · (y − t·b)ᵀΣ(y − t·b)``

        equals ``(μᵀw − tc·‖w − w_cur‖₁ − λ_TE·(w−b)ᵀΣ(w−b)) / σ``,
        i.e. the penalty-adjusted Sharpe ratio.

        User constraints are mapped into y/t space:

        - **Bounds** ``lo ≤ w ≤ hi`` become ``lo·t ≤ y ≤ hi·t``.
        - **Linear inequalities** ``aᵀw ≤ b`` become ``aᵀy ≤ b·t``.
        """
        n = len(mu)
        y = cp.Variable(n)
        t = cp.Variable(nonneg=True)

        # Objective: maximise return − penalties (all in y/t space)
        obj_expr = mu @ y
        obj_expr -= self._turnover_penalty(y, t * current_weights
                                           if current_weights is not None
                                           else None)
        obj_expr -= self._tracking_error_penalty(y, t * benchmark
                                                 if benchmark is not None
                                                 else None, cov)

        cons: list = [
            cp.quad_form(y, cp.psd_wrap(cov)) <= 1,
            cp.sum(y) == t,
        ]

        # Map user constraints into y/t space
        has_bounds = any(isinstance(c, WeightBounds) for c in user_constraints)
        if not has_bounds and self.long_only:
            cons.append(y >= 0)

        for c in user_constraints:
            cons.extend(c.to_cvxpy_constraints_homogeneous(y, t, assets))

        prob = cp.Problem(cp.Maximize(obj_expr), cons)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Max-Sharpe sub-problem failed: {prob.status}")

        y_val = np.array(y.value).flatten()
        t_val = float(t.value)
        return y_val / t_val
