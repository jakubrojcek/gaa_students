# signals.ml_pipeline

Honest out-of-sample validation for parameterised strategies, **driven by the
backtest engine itself**. A `ParameterGrid` plus a *strategy factory* wires any
rule (the momentum strategies in `signals.stcma`, or anything implementing the
`Strategy` protocol) into `backtesting.Backtester`. Each grid cell is backtested
once over the full sample and cached; because backtested returns are causal, that
one cache feeds every scheme below (in-sample pick, walk-forward, CPCV,
data-snooping).

## Contents

- `signal_research_pipeline.py` — the grid harness: `ParameterGrid`, `CellParams`,
  `backtest_param_cells` (full-sample per-cell return cache), `select_best_param_cell`
  (best-Sharpe pick), `backtested_trial_matrix` (the data-snooping trial matrix),
  and `backtest_window`. (`SignalResearchPipeline` writes per-cell PDF reports in
  the full framework; the notebook does not use it.)
- `walk_forward.py` — `WalkForwardStrategy`: a self-refitting `Strategy` that
  re-selects the best cell on an expanding / rolling train window inside
  `compute_weights`, so a **single backtest** produces the whole out-of-sample
  path. Also `make_train_test_windows` and `WalkForwardWindow`.
- `combinatorial_purged_cv.py` — `run_cpcv(...)` / `CombinatorialPurgedCV`: split
  the timeline into groups, test every combination of `k`, purge + embargo, and
  recombine into many full-length OOS paths (`CPCVResult.path_sharpes`).
- `reality_check.py` — multiple-testing controls: `whites_reality_check` and
  `hansens_spa` (stationary-bootstrap p-values that the best trial beats a
  benchmark), plus `relative_performance` and `stationary_bootstrap_indices`.

## Example

```python
from backtesting import BacktestConfig
from signals.ml_pipeline import (
    ParameterGrid, WalkForwardStrategy, backtest_param_cells,
    select_best_param_cell, run_cpcv, backtested_trial_matrix, whites_reality_check,
)
from signals.stcma import time_series_momentum_factory

grid = ParameterGrid(strategy={"lookback": [3, 6, 9, 12, 18, 24]})
config = BacktestConfig(long_short=True, initial_weights_from_strategy=True)

# one full-sample backtest per cell, reused by every scheme
cells = backtest_param_cells(grid, time_series_momentum_factory, prices, config)
best, _ = select_best_param_cell(grid, time_series_momentum_factory, prices, config,
                                 periods_per_year=12, cell_returns=cells)

# a single backtest with this strategy IS the walk-forward OOS path
wfa = WalkForwardStrategy(time_series_momentum_factory, grid, config,
                          min_train_size=60, retrain_every_rebalances=12,
                          periods_per_year=12, cell_returns=cells)

cpcv = run_cpcv(time_series_momentum_factory, grid, prices, config,
                n_groups=6, n_test_groups=2, embargo_pct=0.02,
                label_horizon=24, periods_per_year=12)

trials = backtested_trial_matrix(grid, time_series_momentum_factory, prices, config,
                                 cell_returns=cells)
rc = whites_reality_check(trials, n_resamples=2000, seed=7)
print(rc.p_value, rc.best_trial)
```

See `AssetAllocation5_TAA.ipynb`.
