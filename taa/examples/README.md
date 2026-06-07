# taa.examples

Asset universes and small reporting helpers for the momentum / validation
notebook. (The perf-parquet data loaders live in `data.perf_loader`.)

## Contents

- `_validation_data.py`
  - Universes: `MACRO_ASSETS` (10 multi-asset), `US_INDUSTRIES` (10 MSCI USA sectors),
    `EQUITY_STYLES` (9 style tilts).
  - `sharpe(returns)` — annualised, zero-rf Sharpe (frequency inferred from the index).
  - `describe_distribution(values, label)` — one-line mean / std / quantile summary
    of a distribution (e.g. CPCV path Sharpes).

## Example

```python
from data.perf_loader import load_returns, load_single_returns
from taa.examples._validation_data import MACRO_ASSETS, sharpe

eq = load_single_returns("EQ_US", frequency="M")
macro = load_returns(MACRO_ASSETS, frequency="M", balanced_panel=True)
print(sharpe(eq))
```

See `AssetAllocation5_TAA.ipynb`.
