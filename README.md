# Asset Allocation — Course Edition

A minimal, self-contained slice of the Global Asset Allocation (GAA) framework:
the six teaching notebooks and the market data they run on, with only the
framework code the notebooks need.

## Getting started (uv)

This project uses [**uv**](https://docs.astral.sh/uv/) to manage the Python
environment. You only need uv — it installs the right Python and all packages
for you.

1. **Install uv** (once, per machine):

   - **Windows (PowerShell):**
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **macOS / Linux:**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. **Create the environment** (run this in the project folder):

   ```bash
   uv sync
   ```

   This creates a `.venv/` with Python 3.12 and every dependency
   (numpy, pandas, matplotlib, scipy, cvxpy, statsmodels, JupyterLab).

3. **Open the notebooks:**

   ```bash
   uv run jupyter lab
   ```

   Or open this folder in VS Code and select the `.venv` interpreter as the
   notebook kernel.

That's it — every notebook should run top to bottom.

## Notebooks

| Notebook | Topic |
|---|---|
| `AssetAllocation1_BacktestReturns.ipynb` | Returns, averages, constant-weight backtests |
| `AssetAllocation2_Optimization.ipynb` | Mean–variance optimization & the efficient frontier |
| `AssetAllocation3_Covariance.ipynb` | Covariance estimation: sample, shrinkage, factor model |
| `AssetAllocation4_Robust.ipynb` | Risk parity, risk budgeting, robust optimization |
| `AssetAllocation5_TAA.ipynb` | Momentum, walk-forward, CPCV, reality check |
| `AssetAllocation6_Cases_GAA.ipynb` | Case-study prep: Yale & UBS risk parity |

## Data

Two market-data panels sit at the repo root and are loaded directly by the
notebooks:

- `perf_M.parquet` — monthly performance panel
- `perf_Q.parquet` — quarterly performance panel (illiquid series)

Columns are named `{ASSET}-r` (simple return) and `{ASSET}-i` (price index).
