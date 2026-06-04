# signals.ml_pipeline

Honest out-of-sample validation for parameterised strategies (the momentum rules in
`signals.stcma`, but anything with the same `fit`/`param_grid`/`strategy_returns` interface works).

## Contents

- `walk_forward.py` — `walk_forward_strategy_returns(...)` refits parameters on an expanding (or
  rolling) train window and collects returns on the next disjoint block, giving one OOS path
  (`WalkForwardResult`: `.oos_returns`, `.windows`, `.selected_params`). Also `FitPredictStrategy`
  (the structural protocol) and `make_train_test_windows`.
- `combinatorial_purged_cv.py` — `run_cpcv(...)` / `CombinatorialPurgedCV`: split the timeline into
  groups, test every combination, purge + embargo, and recombine into many full-length OOS paths
  (`CPCVResult`: `.n_splits`, `.n_paths`, `.path_sharpes`).
- `reality_check.py` — multiple-testing controls: `whites_reality_check` and `hansens_spa`
  (stationary-bootstrap p-values that the best trial beats a benchmark), plus
  `relative_performance` and `stationary_bootstrap_indices`.

## Example

```python
from signals.ml_pipeline import walk_forward_strategy_returns, run_cpcv, whites_reality_check
from signals.stcma import TimeSeriesMomentum

lookbacks = [3, 6, 9, 12, 18, 24]

wfa = walk_forward_strategy_returns(
    TimeSeriesMomentum(lookbacks), eq_returns,
    mode="expanding", min_train_size=60, retrain_every=12, test_horizon=12,
)
wfa.oos_returns                       # single honest OOS path

cpcv = run_cpcv(
    TimeSeriesMomentum(lookbacks), eq_returns,
    n_groups=6, n_test_groups=2, embargo_pct=0.02,
    label_horizon=max(lookbacks), periods_per_year=12,
)
cpcv.path_sharpes                     # distribution of OOS Sharpes

import pandas as pd
trials = pd.DataFrame({f"lb{p['lookback']}": TimeSeriesMomentum(lookbacks).strategy_returns(eq_returns, **p)
                       for p in TimeSeriesMomentum(lookbacks).param_grid()}).dropna()
rc = whites_reality_check(trials, n_resamples=2000, seed=7)
print(rc.p_value, rc.best_trial)
```

See `AssetAllocation5_TAA.ipynb`.
