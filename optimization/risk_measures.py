"""
risk_measures.py
----------------
Pluggable risk measure classes for portfolio optimisation.

Each risk measure implements both scipy and cvxpy interfaces:

- **Scipy**: ``scipy_risk()`` computes the risk value from a (possibly
  augmented) decision vector.  Auxiliary variables and constraints are
  provided for formulations that require them (e.g. Rockafellar-Uryasev
  for CVaR).
- **Cvxpy**: ``cvxpy_risk()`` returns a convex expression and any
  extra constraints.

Available measures:

- ``Variance``  — portfolio variance ``wᵀΣw``
- ``CVaR``      — Conditional Value-at-Risk (Expected Shortfall)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RiskMeasure(ABC):
    """Abstract base for portfolio risk measures.

    Each risk measure implements both scipy and cvxpy interfaces so it
    can be used with either solver backend.
    """

    @abstractmethod
    def evaluate(self, w: np.ndarray, **data) -> float:
        """Evaluate the risk measure for given weights (pure numpy).

        Parameters
        ----------
        w : np.ndarray
            Portfolio weights (n_assets).
        **data
            Risk-measure-specific data (``cov``, ``scenario_returns``).
        """
        ...

    # --- Scipy interface ---------------------------------------------------

    @abstractmethod
    def scipy_risk(self, x: np.ndarray, n_assets: int, **data) -> float:
        """Risk measure value from the (possibly augmented) decision vector.

        Parameters
        ----------
        x : np.ndarray
            Decision vector ``[weights | auxiliary_variables]``.
        n_assets : int
            Number of weight variables (``x[:n_assets]`` are weights).
        **data
            ``cov``, ``scenario_returns``, etc.
        """
        ...

    def scipy_n_auxiliary(self, n_scenarios: int = 0) -> int:
        """Number of auxiliary variables beyond the weight vector."""
        return 0

    def scipy_auxiliary_x0(self, n_scenarios: int = 0) -> np.ndarray:
        """Initial values for auxiliary variables."""
        return np.array([])

    def scipy_auxiliary_bounds(
        self,
        n_scenarios: int = 0,
    ) -> list[tuple[float | None, float | None]]:
        """Bounds for auxiliary variables."""
        return []

    def scipy_auxiliary_constraints(self, n_assets: int, **data) -> list[dict]:
        """Extra scipy constraint dicts for auxiliary variables."""
        return []

    # --- Cvxpy interface ---------------------------------------------------

    @abstractmethod
    def cvxpy_risk(
        self,
        w: cp.Variable,
        **data,
    ) -> tuple[cp.Expression, list]:
        """Risk measure as a cvxpy minimisation expression.

        Returns
        -------
        tuple[cp.Expression, list]
            ``(risk_expression, extra_constraints)``
        """
        ...

    # --- Metadata ----------------------------------------------------------

    @property
    def requires_scenarios(self) -> bool:
        """Whether this risk measure needs ``scenario_returns`` data."""
        return False


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variance(RiskMeasure):
    """Portfolio variance: ``wᵀΣw``.

    Requires ``cov`` (N×N covariance matrix) in the data dict.
    """

    def evaluate(self, w: np.ndarray, **data) -> float:
        return float(w @ data["cov"] @ w)

    def scipy_risk(self, x: np.ndarray, n_assets: int, **data) -> float:
        w = x[:n_assets]
        return float(w @ data["cov"] @ w)

    def cvxpy_risk(
        self,
        w: cp.Variable,
        **data,
    ) -> tuple[cp.Expression, list]:
        return cp.quad_form(w, cp.psd_wrap(data["cov"])), []


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVaR(RiskMeasure):
    r"""Conditional Value-at-Risk (Expected Shortfall).

    Uses the Rockafellar-Uryasev LP relaxation for scipy (SLSQP) and
    ``cp.pos`` for cvxpy.

    Requires ``scenario_returns`` (T×N matrix of historical or
    simulated per-period returns) in the data dict.

    Parameters
    ----------
    alpha : float
        Tail probability (default 0.05 = 5 % CVaR).
    """

    alpha: float = 0.05

    @property
    def requires_scenarios(self) -> bool:
        return True

    # --- Evaluate (pure numpy) ---------------------------------------------

    def evaluate(self, w: np.ndarray, **data) -> float:
        """Empirical CVaR from scenario returns.

        Returns a positive number: higher = riskier.
        """
        R = data["scenario_returns"]  # T × N
        portfolio_returns = R @ w
        sorted_ret = np.sort(portfolio_returns)
        cutoff = max(1, int(np.ceil(self.alpha * len(sorted_ret))))
        return -float(sorted_ret[:cutoff].mean())

    # --- Scipy: Rockafellar-Uryasev ----------------------------------------
    #
    # Decision vector:  x = [w (n), ζ (1), u (T)]
    #
    # CVaR = ζ + 1/(T·(1−α)) · Σ uᵢ
    #
    # Subject to:  uᵢ ≥ Lᵢ − ζ   where Lᵢ = −Rᵢᵀw  (portfolio loss)
    #              uᵢ ≥ 0          (enforced via bounds)
    #
    # At optimum ζ = VaR and uᵢ = max(Lᵢ − VaR, 0).

    def scipy_n_auxiliary(self, n_scenarios: int = 0) -> int:
        return 1 + n_scenarios  # ζ + u₁..uₜ

    def scipy_auxiliary_x0(self, n_scenarios: int = 0) -> np.ndarray:
        return np.zeros(1 + n_scenarios)

    def scipy_auxiliary_bounds(
        self,
        n_scenarios: int = 0,
    ) -> list[tuple[float | None, float | None]]:
        # ζ unbounded, u ≥ 0
        return [(None, None)] + [(0, None)] * n_scenarios

    def scipy_risk(self, x: np.ndarray, n_assets: int, **data) -> float:
        T = data["scenario_returns"].shape[0]
        zeta = x[n_assets]
        u = x[n_assets + 1 :]
        return float(zeta + (1 / (T * (1 - self.alpha))) * np.sum(u))

    def scipy_auxiliary_constraints(self, n_assets: int, **data) -> list[dict]:
        R = data["scenario_returns"]  # T × N
        n = n_assets
        # uᵢ ≥ −Rᵢᵀw − ζ  ⟺  uᵢ + Rᵢᵀw + ζ ≥ 0
        return [
            {
                "type": "ineq",
                "fun": lambda x, _R=R, _n=n: x[_n + 1 :] + _R @ x[:_n] + x[_n],
            },
        ]

    # --- Cvxpy --------------------------------------------------------------

    def cvxpy_risk(
        self,
        w: cp.Variable,
        **data,
    ) -> tuple[cp.Expression, list]:
        R = data["scenario_returns"]  # T × N
        T = R.shape[0]
        losses = -R @ w  # T-vector of portfolio losses
        z = cp.Variable(name="cvar_z")
        cvar = z + (1 / (T * (1 - self.alpha))) * cp.sum(cp.pos(losses - z))
        return cvar, []
