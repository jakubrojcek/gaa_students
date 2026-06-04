# risk_models.utils

Small shared helpers used by the covariance and risk-analytics modules.

## Contents

- `cov_to_corr(cov)` — covariance matrix → correlation matrix (same index/columns).
- `extract_rolling_corr(rolling_cov, asset_a, asset_b)` — pull a pairwise correlation time series
  out of a `{date: covariance}` dict (the output of `compute_rolling_covariance`).
- `map_dates_to_index(dates, index)` — map requested dates to the nearest prior observation date.

## Example

```python
from risk_models.utils import cov_to_corr, extract_rolling_corr

corr = cov_to_corr(sample_cov)
series = extract_rolling_corr(rolling_cov, "S&P 500", "U.S. Treasuries")
```
