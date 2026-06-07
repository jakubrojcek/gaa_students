# data

Minimal data helpers for the notebooks. (The market-data panels themselves —
`perf_M.parquet`, `perf_Q.parquet` — live at the repo root.)

## Contents

- `perf_loader.py` — convenience loaders for the bundled performance panels:
  - `load_returns(assets, frequency="M", balanced_panel=True)` — simple-return (`-r`) panel;
    `balanced_panel=True` keeps only fully-populated rows.
  - `load_single_returns(asset, frequency="M")` — one asset's return series.
  - `load_prices(assets, frequency="M")` — price-level (`-i`) panel fed to `backtesting.Backtester`.
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
