# taa.examples

Shared data loaders and asset universes for the momentum / validation notebook.

## Contents

- `_validation_data.py`
  - Universes: `MACRO_ASSETS` (10 multi-asset), `US_INDUSTRIES` (10 MSCI USA sectors),
    `EQUITY_STYLES` (9 style tilts).
  - `load_returns(assets, frequency="M", balanced=True)` — simple-return (`-r`) panel for a list
    of asset tickers; `balanced=True` keeps only fully-populated rows.
  - `load_single_returns(asset, frequency="M")` — one asset's return series.
  - `sharpe(returns)` — annualised, zero-rf Sharpe (frequency inferred from the index).

  Data is read from `perf_M.parquet` at the repo root.

## Example

```python
from taa.examples._validation_data import load_returns, load_single_returns, MACRO_ASSETS, sharpe

eq = load_single_returns("EQ_US", frequency="M")
macro = load_returns(MACRO_ASSETS, frequency="M", balanced=True)
print(sharpe(eq))
```

See `AssetAllocation5_TAA.ipynb`.
