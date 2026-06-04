# risk_models

Covariance estimation and risk-model analytics.

## Contents

- `covariance.py` — `compute_covariance` and `compute_rolling_covariance`. Methods (`CovMethod`):
  `sample`, `ledoit_wolf_cc`, `ledoit_wolf_sf`, `ledoit_wolf_id`, `ewma`, `semicovariance`.
  Also `nearest_psd`.
- `factor_covariance.py` — `compute_rolling_asset_factor_cov` builds a linear factor model
  Σ = B·F·Bᵀ + D; returns `FactorCovResult` (`.asset_cov`, `.loadings`, `.idiosyncratic_var`).
- `risk_analytics.py` — risk contributions and decomposition: `marginal_risk_contributions`,
  `risk_contributions`, `risk_contribution_matrix(_df)`, `decompose_risk`, out-of-sample
  evaluation (`evaluate_oos_covariance`, `oos_results_to_df`), and ready-made figures
  (`fig_risk_contribution_bars`, `fig_comparison_over_time`).
- `utils/` — small helpers (`cov_to_corr`, `extract_rolling_corr`, `map_dates_to_index`).

Monthly covariances are annualised by `× 12` at the call site (as in the notebooks).

## Examples

```python
from risk_models.covariance import compute_covariance, compute_rolling_covariance
from risk_models.utils import cov_to_corr, extract_rolling_corr

S = compute_covariance(returns, method="ledoit_wolf_sf") * 12
corr = cov_to_corr(S)

roll = compute_rolling_covariance(returns, freq="QE", method="ewma", halflife=36)
eq_tsy = extract_rolling_corr(roll, "S&P 500", "U.S. Treasuries")
```

```python
from risk_models.factor_covariance import compute_rolling_asset_factor_cov

fcr = next(iter(compute_rolling_asset_factor_cov(
    returns, factor_returns, dates=[returns.index[-1]],
    beta_method="static", cov_method="ledoit_wolf_sf", annualize=True,
).values()))
loadings, asset_cov = fcr.loadings, fcr.asset_cov
```

```python
from risk_models.risk_analytics import risk_contributions, fig_risk_contribution_bars

shares = risk_contributions(weights, S, relative=True)        # sums to 1
fig = fig_risk_contribution_bars({"1/N": weights}, S, relative=True)
```

See `AssetAllocation3_Covariance.ipynb` and `AssetAllocation4_Robust.ipynb`.
