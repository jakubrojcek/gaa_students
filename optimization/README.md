# optimization

Portfolio optimization: mean–variance, risk budgeting / risk parity, and robust (worst-case)
optimisation. Convex programs are solved with `cvxpy`; some objectives also have a `scipy` backend.

## Contents

- `mean_risk.py` — `MeanRiskOptimizer` (`.optimize(mu, S, objective, constraints=...)`,
  `.efficient_frontier(...)`). `long_only` toggle on the constructor.
- `objectives.py` — `MinRisk`, `MaxReturn`, `MaxSharpe`, `MaxUtility`, `TargetReturn`.
- `risk_measures.py` — `Variance`, `CVaR`.
- `risk_budget.py` — `RiskBudgetOptimizer` (Spinu log-barrier risk parity / risk budgeting) plus
  the static `RiskBudgetOptimizer.risk_contributions(w, cov)`.
- `robust.py` — `RobustMeanVarianceOptimizer` + `worst_case_stats` (elliptical uncertainty sets
  for μ and Σ).
- `constraints.py` — `WeightBounds`, `GroupConstraint`, `PairConstraint`, `LinearInequality`.
- `optimizer.py` — `BaseOptimizer`, `SolverBackend` (shared plumbing).

> Expected returns are passed in **percentage points** to `MeanRiskOptimizer`; the covariance is
> in decimal² units. `MaxSharpe` maximises μ/σ, so feed it **excess** returns to get the true
> tangency portfolio.

## Examples

```python
from optimization.mean_risk import MeanRiskOptimizer
from optimization.objectives import MinRisk, MaxSharpe, MaxUtility
from optimization.risk_measures import Variance

opt = MeanRiskOptimizer(solver="cvxpy")          # long-only, fully invested
V = Variance()

w_minvar = opt.optimize(mu_pct, S, MinRisk(V))
w_tangency = opt.optimize(mu_excess_pct, S, MaxSharpe())
frontier = opt.efficient_frontier(mu_pct, S, V, "target_return", n_points=40)
```

```python
from optimization.risk_budget import RiskBudgetOptimizer

w_rp = RiskBudgetOptimizer(solver="scipy").optimize(cov)      # equal risk contribution
rc = RiskBudgetOptimizer.risk_contributions(w_rp, cov)
```

```python
from optimization.robust import RobustMeanVarianceOptimizer, worst_case_stats

stats = worst_case_stats(returns, q=0.05)                     # k_mu, k_sigma, mu, cov
w_robust = RobustMeanVarianceOptimizer().optimize(stats, objective="max_utility", risk_aversion=5.0)
```

See `AssetAllocation2_Optimization.ipynb` and `AssetAllocation4_Robust.ipynb`.
