# signals

Trading signals and the machinery to validate them. This teaching edition ships two subpackages:

- **`stcma/`** — Short-Term Capital Market Assumptions: momentum signals (time-series and
  cross-sectional).
- **`ml_pipeline/`** — backtest validation, driven through `backtesting.Backtester`:
  a parameter-grid harness, walk-forward analysis, combinatorial purged
  cross-validation (CPCV), and the stationary-bootstrap reality check / SPA test.

See each subfolder's README for details and examples, and
`AssetAllocation5_TAA.ipynb` for the full walk-through.
