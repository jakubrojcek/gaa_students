"""
objectives.py
-------------
Objective dataclasses for portfolio optimisation.

Each objective defines the problem structure (what to minimise/maximise)
and delegates risk computation to a pluggable
:class:`~optimization.risk_measures.RiskMeasure`.

Available objectives:

- ``MinRisk``       — minimise risk
- ``MaxReturn``     — maximise return subject to risk ceiling
- ``MaxUtility``    — maximise return − (λ/2)·risk
- ``TargetReturn``  — minimise risk subject to return floor
- ``MaxSharpe``     — maximise Sharpe ratio (variance-only)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

from optimization.risk_measures import RiskMeasure

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Objective(ABC):
    """Abstract base for optimisation objectives.

    Subclasses implement ``build_scipy`` and ``build_cvxpy`` which
    return the objective function/expression and any extra constraints
    specific to the objective (e.g. a return floor or risk ceiling).
    """

    @abstractmethod
    def build_scipy(
        self,
        mu: np.ndarray,
        n_assets: int,
        **data: Any,
    ) -> tuple[Callable[[np.ndarray], float], list[dict]]:
        """Build scipy objective function and extra constraints.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns in **decimal** (not percentage points).
        n_assets : int
            Number of assets.  ``x[:n_assets]`` are weights in the
            (possibly augmented) decision vector.
        **data
            Risk-measure-specific data (``cov``, ``scenario_returns``).

        Returns
        -------
        tuple[Callable, list[dict]]
            ``(objective_fn(x) -> float, extra_scipy_constraints)``
        """
        ...

    @abstractmethod
    def build_cvxpy(
        self,
        w: cp.Variable,
        mu: np.ndarray,
        **data: Any,
    ) -> tuple[cp.Expression, list]:
        """Build cvxpy minimisation expression and extra constraints.

        Returns
        -------
        tuple[cp.Expression, list]
            ``(minimise_expression, extra_cvxpy_constraints)``
        """
        ...


# ---------------------------------------------------------------------------
# MinRisk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinRisk(Objective):
    """Minimise portfolio risk.

    Parameters
    ----------
    risk_measure : RiskMeasure
        Risk measure to minimise (e.g. ``Variance()``, ``CVaR()``).
    """

    risk_measure: RiskMeasure

    def build_scipy(self, mu: np.ndarray, n_assets: int, **data: Any):
        rm = self.risk_measure

        def obj(x: np.ndarray) -> float:
            return rm.scipy_risk(x, n_assets, **data)

        return obj, rm.scipy_auxiliary_constraints(n_assets, **data)

    def build_cvxpy(self, w: cp.Variable, mu: np.ndarray, **data: Any):
        return self.risk_measure.cvxpy_risk(w, **data)


# ---------------------------------------------------------------------------
# MaxReturn
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxReturn(Objective):
    """Maximise expected return subject to a risk ceiling.

    Parameters
    ----------
    risk_measure : RiskMeasure
        Risk measure whose value is capped.
    max_risk : float
        Upper bound on the risk measure value.
    """

    risk_measure: RiskMeasure
    max_risk: float

    def build_scipy(self, mu: np.ndarray, n_assets: int, **data: Any):
        rm = self.risk_measure
        mr = self.max_risk

        def obj(x: np.ndarray) -> float:
            return -float(x[:n_assets] @ mu)

        cons = rm.scipy_auxiliary_constraints(n_assets, **data)
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, _rm=rm, _n=n_assets, _mr=mr, _d=data: (
                    _mr - _rm.scipy_risk(x, _n, **_d)
                ),
            }
        )
        return obj, cons

    def build_cvxpy(self, w: cp.Variable, mu: np.ndarray, **data: Any):
        risk_expr, risk_cons = self.risk_measure.cvxpy_risk(w, **data)
        return -mu @ w, risk_cons + [risk_expr <= self.max_risk]


# ---------------------------------------------------------------------------
# MaxUtility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxUtility(Objective):
    r"""Maximise expected return minus risk penalty.

    ``wᵀμ − (λ/2) · risk(w)``

    Parameters
    ----------
    risk_measure : RiskMeasure
        Risk measure.
    risk_aversion : float
        Trade-off coefficient λ (higher = more risk-averse).
    """

    risk_measure: RiskMeasure
    risk_aversion: float = 1.0

    def build_scipy(self, mu: np.ndarray, n_assets: int, **data: Any):
        rm = self.risk_measure
        lam = self.risk_aversion

        def obj(x: np.ndarray) -> float:
            w = x[:n_assets]
            return -float(w @ mu) + (lam / 2) * rm.scipy_risk(x, n_assets, **data)

        return obj, rm.scipy_auxiliary_constraints(n_assets, **data)

    def build_cvxpy(self, w: cp.Variable, mu: np.ndarray, **data: Any):
        risk_expr, risk_cons = self.risk_measure.cvxpy_risk(w, **data)
        return -mu @ w + (self.risk_aversion / 2) * risk_expr, risk_cons


# ---------------------------------------------------------------------------
# TargetReturn
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetReturn(Objective):
    """Minimise risk subject to a return floor.

    Parameters
    ----------
    risk_measure : RiskMeasure
        Risk measure to minimise.
    target_return : float
        Minimum expected return in **percentage points**.
    """

    risk_measure: RiskMeasure
    target_return: float

    def build_scipy(self, mu: np.ndarray, n_assets: int, **data: Any):
        rm = self.risk_measure
        target_dec = self.target_return / 100.0

        def obj(x: np.ndarray) -> float:
            return rm.scipy_risk(x, n_assets, **data)

        cons = rm.scipy_auxiliary_constraints(n_assets, **data)
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, _mu=mu, _t=target_dec, _n=n_assets: (
                    float(x[:_n] @ _mu) - _t
                ),
            }
        )
        return obj, cons

    def build_cvxpy(self, w: cp.Variable, mu: np.ndarray, **data: Any):
        risk_expr, risk_cons = self.risk_measure.cvxpy_risk(w, **data)
        return risk_expr, risk_cons + [mu @ w >= self.target_return / 100.0]


# ---------------------------------------------------------------------------
# MaxSharpe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxSharpe(Objective):
    r"""Maximise the Sharpe ratio ``wᵀμ / √(wᵀΣw)``.

    Inherently variance-based — the cvxpy backend uses a dedicated
    variance-normalised ``y = w·t`` transform (SOCP).  Not compatible
    with non-variance risk measures.
    """

    def build_scipy(self, mu: np.ndarray, n_assets: int, **data: Any):
        cov = data["cov"]

        def obj(x: np.ndarray) -> float:
            w = x[:n_assets]
            var = float(w @ cov @ w)
            if var < 1e-14:
                return 0.0
            return -float(w @ mu) / np.sqrt(var)

        return obj, []

    def build_cvxpy(self, w: cp.Variable, mu: np.ndarray, **data: Any):
        raise NotImplementedError(
            "MaxSharpe uses a dedicated variance-normalised y/t transform "
            "for cvxpy.  The optimizer handles this internally."
        )
