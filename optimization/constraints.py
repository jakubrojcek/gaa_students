"""
constraints.py
--------------
Composable portfolio constraint dataclasses for both scipy and cvxpy backends.

Each constraint implements two interfaces:

- **Scipy**: ``to_scipy_bounds()`` for per-variable box constraints, and
  ``to_scipy_constraints()`` for general inequality/equality constraints
  (returned as dicts compatible with ``scipy.optimize.minimize``).
- **Cvxpy**: ``to_cvxpy()`` returns a list of cvxpy constraint expressions.

Usage
-----
>>> from optimization.constraints import WeightBounds, GroupConstraint
>>> constraints = [
...     WeightBounds(lower=0.0, upper=0.40),
...     GroupConstraint(assets=["EQ_US", "EQ_EU"], lower=0.20, upper=0.60),
... ]
>>> opt = MeanRiskOptimizer(solver="scipy")
>>> w = opt.optimize(mu, cov, "min_variance", constraints=constraints)
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import cvxpy as cp
import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PortfolioConstraint(ABC):
    """Abstract base for portfolio constraints.

    Subclasses implement backend-specific builders that the optimizer
    calls when assembling the problem.

    Parameters passed by the optimizer at build time:

    - ``assets`` — sequence of asset names (defines position mapping).
    """

    def to_scipy_bounds(
        self, assets: Sequence[str],
    ) -> list[tuple[float | None, float | None]] | None:
        """Return per-variable bounds, or ``None`` if not applicable.

        When multiple constraints return bounds, the optimizer intersects
        them (tightest lower, tightest upper per asset).
        """
        return None

    def to_scipy_constraints(self, assets: Sequence[str]) -> list[dict]:
        """Return scipy constraint dicts (``type`` + ``fun``)."""
        return []

    def to_cvxpy_constraints(self, w: cp.Variable, assets: Sequence[str]) -> list:
        """Return a list of cvxpy constraint expressions."""
        return []

    def to_cvxpy_constraints_homogeneous(
        self, y: cp.Variable, t: cp.Variable, assets: Sequence[str],
    ) -> list:
        """Return cvxpy constraints in the ``y = w·t`` space.

        Used by the max-Sharpe variance-normalised transform.  A
        constraint ``f(w) ≤ 0`` becomes ``f(y/t) ≤ 0``; for linear
        constraints this simplifies to a form linear in ``(y, t)``.
        """
        return []


# ---------------------------------------------------------------------------
# Weight bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightBounds(PortfolioConstraint):
    """Per-asset box constraints on individual weights.

    Parameters
    ----------
    lower : float | dict[str, float]
        Minimum weight.  Scalar applies uniformly; dict maps asset names
        to individual lower bounds (missing assets default to 0).
    upper : float | dict[str, float]
        Maximum weight.  Scalar applies uniformly; dict maps asset names
        to individual upper bounds (missing assets default to 1).
    """

    lower: float | dict[str, float] = 0.0
    upper: float | dict[str, float] = 1.0

    def _resolve(
        self, assets: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand lower/upper to arrays aligned with *assets*.

        When *lower* or *upper* is a dict, assets not mentioned get
        ``-inf`` / ``+inf`` (i.e. no bound).  The optimizer's
        ``build_scipy_bounds`` then intersects these with the
        ``long_only`` default, so unmentioned assets inherit the
        correct sign constraint without ``WeightBounds`` needing to
        know about ``long_only``.
        """
        n = len(assets)
        if isinstance(self.lower, dict):
            lo = np.array([self.lower.get(a, -np.inf) for a in assets])
        else:
            lo = np.full(n, self.lower)

        if isinstance(self.upper, dict):
            hi = np.array([self.upper.get(a, np.inf) for a in assets])
        else:
            hi = np.full(n, self.upper)

        return lo, hi

    def to_scipy_bounds(
        self, assets: Sequence[str],
    ) -> list[tuple[float | None, float | None]]:
        lo, hi = self._resolve(assets)
        return [
            (None if np.isneginf(lb) else float(lb), None if np.isposinf(ub) else float(ub))
            for lb, ub in zip(lo, hi)
        ]

    def to_cvxpy_constraints(self, w: cp.Variable, assets: Sequence[str]) -> list:
        lo, hi = self._resolve(assets)
        cons: list = []
        if not np.all(np.isneginf(lo)):
            cons.append(w >= lo)
        if not np.all(np.isposinf(hi)):
            cons.append(w <= hi)
        return cons

    def to_cvxpy_constraints_homogeneous(
        self, y: cp.Variable, t: cp.Variable, assets: Sequence[str],
    ) -> list:
        """``lo ≤ w ≤ hi`` → ``lo·t ≤ y ≤ hi·t``."""
        lo, hi = self._resolve(assets)
        cons: list = []
        if not np.all(np.isneginf(lo)):
            cons.append(y >= lo * t)
        if not np.all(np.isposinf(hi)):
            cons.append(y <= hi * t)
        return cons


# ---------------------------------------------------------------------------
# Group constraint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupConstraint(PortfolioConstraint):
    """Constrain the total weight of a group of assets.

    ``lower <= sum(w[group]) <= upper``

    Delegates to :class:`LinearInequality` with unit coefficients on
    group members.

    Parameters
    ----------
    assets : list[str]
        Asset names in the group.
    lower : float
        Minimum aggregate weight.
    upper : float
        Maximum aggregate weight.
    """

    assets: list[str] = field(default_factory=list)
    lower: float = field(default=0.0, repr=True)
    upper: float = field(default=0.0, repr=True)

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("GroupConstraint requires at least one asset")
        if self.lower == 0.0 and self.upper == 0.0:
            raise ValueError(
                "GroupConstraint requires explicit lower and/or upper bounds"
            )
        if self.lower > self.upper:
            raise ValueError(
                f"lower ({self.lower}) must be <= upper ({self.upper})"
            )

    def _to_linear(self, universe: Sequence[str]) -> list[LinearInequality]:
        """Build LinearInequality pair for the group sum bounds."""
        coeffs = {a: 1.0 for a in self.assets}
        if not any(a in universe for a in self.assets):
            raise ValueError(
                f"None of {self.assets} found in asset universe {list(universe)}"
            )
        return [
            LinearInequality(coefficients=coeffs, bound=self.upper, sense="<="),
            LinearInequality(coefficients=coeffs, bound=self.lower, sense=">="),
        ]

    def to_scipy_constraints(self, assets: Sequence[str]) -> list[dict]:
        result: list[dict] = []
        for li in self._to_linear(assets):
            result.extend(li.to_scipy_constraints(assets))
        return result

    def to_cvxpy_constraints(self, w: cp.Variable, assets: Sequence[str]) -> list:
        result: list = []
        for li in self._to_linear(assets):
            result.extend(li.to_cvxpy_constraints(w, assets))
        return result

    def to_cvxpy_constraints_homogeneous(
        self, y: cp.Variable, t: cp.Variable, assets: Sequence[str],
    ) -> list:
        result: list = []
        for li in self._to_linear(assets):
            result.extend(li.to_cvxpy_constraints_homogeneous(y, t, assets))
        return result


# ---------------------------------------------------------------------------
# Pair constraint (relative weight)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairConstraint(PortfolioConstraint):
    """Constrain the weight of one asset relative to another.

    ``lower <= w[asset_a] - w[asset_b] <= upper``

    Delegates to :class:`LinearInequality` with ``+1`` for *asset_a*
    and ``-1`` for *asset_b*.

    Parameters
    ----------
    asset_a, asset_b : str
        Asset names.
    lower : float | None
        Minimum difference ``w[a] - w[b]``.  ``None`` = unbounded below.
    upper : float | None
        Maximum difference ``w[a] - w[b]``.  ``None`` = unbounded above.
    """

    asset_a: str = ""
    asset_b: str = ""
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.asset_a or not self.asset_b:
            raise ValueError("PairConstraint requires both asset_a and asset_b")
        if self.lower is None and self.upper is None:
            raise ValueError("At least one of lower/upper must be specified")

    def _to_linear(self, universe: Sequence[str]) -> list[LinearInequality]:
        """Build LinearInequality pair for the difference bounds."""
        assets_list = list(universe)
        if self.asset_a not in assets_list:
            raise ValueError(f"{self.asset_a!r} not in asset universe")
        if self.asset_b not in assets_list:
            raise ValueError(f"{self.asset_b!r} not in asset universe")

        coeffs = {self.asset_a: 1.0, self.asset_b: -1.0}
        result: list[LinearInequality] = []
        if self.upper is not None:
            result.append(LinearInequality(coefficients=coeffs, bound=self.upper, sense="<="))
        if self.lower is not None:
            result.append(LinearInequality(coefficients=coeffs, bound=self.lower, sense=">="))
        return result

    def to_scipy_constraints(self, assets: Sequence[str]) -> list[dict]:
        result: list[dict] = []
        for li in self._to_linear(assets):
            result.extend(li.to_scipy_constraints(assets))
        return result

    def to_cvxpy_constraints(self, w: cp.Variable, assets: Sequence[str]) -> list:
        result: list = []
        for li in self._to_linear(assets):
            result.extend(li.to_cvxpy_constraints(w, assets))
        return result

    def to_cvxpy_constraints_homogeneous(
        self, y: cp.Variable, t: cp.Variable, assets: Sequence[str],
    ) -> list:
        result: list = []
        for li in self._to_linear(assets):
            result.extend(li.to_cvxpy_constraints_homogeneous(y, t, assets))
        return result


# ---------------------------------------------------------------------------
# Linear inequality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearInequality(PortfolioConstraint):
    r"""General linear inequality constraint.

    ``a ᵀ w  ≤  b``   (or ``≥``, depending on *sense*).

    Parameters
    ----------
    coefficients : dict[str, float] | np.ndarray
        If *dict*, maps asset names to coefficients (missing assets get 0).
        If *ndarray*, must have length ``n_assets`` (order matches the
        covariance columns).
    bound : float
        Right-hand side scalar.
    sense : ``"<="`` | ``">="``
        Direction of the inequality (default ``"<="``).
    """

    coefficients: dict[str, float] | np.ndarray = field(default_factory=dict)
    bound: float = 0.0
    sense: Literal["<=", ">="] = "<="

    def _to_array(self, assets: Sequence[str]) -> np.ndarray:
        if isinstance(self.coefficients, np.ndarray):
            if len(self.coefficients) != len(assets):
                raise ValueError(
                    f"coefficients length ({len(self.coefficients)}) "
                    f"!= number of assets ({len(assets)})"
                )
            return self.coefficients
        return np.array([self.coefficients.get(a, 0.0) for a in assets])

    def to_scipy_constraints(self, assets: Sequence[str]) -> list[dict]:
        a = self._to_array(assets)
        if self.sense == "<=":
            # b - a'w >= 0
            return [{"type": "ineq", "fun": lambda w, a=a, b=self.bound: b - float(a @ w)}]
        # a'w - b >= 0
        return [{"type": "ineq", "fun": lambda w, a=a, b=self.bound: float(a @ w) - b}]

    def to_cvxpy_constraints(self, w: cp.Variable, assets: Sequence[str]) -> list:
        a = self._to_array(assets)
        if self.sense == "<=":
            return [a @ w <= self.bound]
        return [a @ w >= self.bound]

    def to_cvxpy_constraints_homogeneous(
        self, y: cp.Variable, t: cp.Variable, assets: Sequence[str],
    ) -> list:
        """``aᵀw ≤ b`` → ``aᵀy ≤ b·t``."""
        a = self._to_array(assets)
        if self.sense == "<=":
            return [a @ y <= self.bound * t]
        return [a @ y >= self.bound * t]


# ---------------------------------------------------------------------------
# Helpers for optimizer integration
# ---------------------------------------------------------------------------


def build_scipy_bounds(
    constraints: list[PortfolioConstraint],
    assets: Sequence[str],
    long_only: bool,
) -> list[tuple[float | None, float | None]]:
    """Merge bounds from multiple constraints (tightest wins).

    Starts from the ``long_only`` default (``[0, 1]`` or ``[-∞, +∞]``)
    and intersects with any :class:`WeightBounds` in the list.
    """
    n = len(assets)
    if long_only:
        lo = np.zeros(n)
        hi = np.ones(n)
    else:
        lo = np.full(n, -np.inf)
        hi = np.full(n, np.inf)

    for c in constraints:
        b = c.to_scipy_bounds(assets)
        if b is not None:
            for i, (bl, bh) in enumerate(b):
                if bl is not None:
                    lo[i] = max(lo[i], bl)
                if bh is not None:
                    hi[i] = min(hi[i], bh)

    return [
        (None if np.isneginf(l) else float(l), None if np.isposinf(h) else float(h))
        for l, h in zip(lo, hi)
    ]


def build_scipy_constraints(
    constraints: list[PortfolioConstraint],
    assets: Sequence[str],
) -> list[dict]:
    """Collect all scipy constraint dicts from a constraint list."""
    result: list[dict] = []
    for c in constraints:
        result.extend(c.to_scipy_constraints(assets))
    return result


def build_cvxpy_constraints(
    constraints: list[PortfolioConstraint],
    w: cp.Variable,
    assets: Sequence[str],
    long_only: bool,
) -> list:
    """Collect all cvxpy constraints from a constraint list.

    When *constraints* contains :class:`WeightBounds`, the ``long_only``
    flag is ignored (the bounds subsume it).  Otherwise the default
    ``w >= 0`` is applied when ``long_only=True``.
    """
    result: list = []
    has_bounds = any(isinstance(c, WeightBounds) for c in constraints)

    if not has_bounds and long_only:
        result.append(w >= 0)

    for c in constraints:
        result.extend(c.to_cvxpy_constraints(w, assets))

    return result
