# data

Minimal data helpers for the notebooks. (The market-data panels themselves —
`perf_M.parquet`, `perf_Q.parquet` — live at the repo root.)

## Contents

- `generate_data.py`
  - `unsmooth_Geltner(returns, max_beta=None)` — invert an AR(1) appraisal-smoothing filter
    (Geltner et al. 1994) to recover the volatility of illiquid / appraisal-based series.
    Returns `(unsmoothed_series, rho)`; the first observation is `NaN` (the inversion needs a lag).
    Cap `rho` with `max_beta` to limit how aggressively a series is unsmoothed.

## Example

```python
from data.generate_data import unsmooth_Geltner

unsmoothed, rho = unsmooth_Geltner(real_estate_returns)
print(f"AR(1) rho = {rho:.3f}")
# vol is understated when reported; unsmoothing restores it
```

See Part A of `AssetAllocation6_Cases_GAA.ipynb` (Yale illiquid assets).
