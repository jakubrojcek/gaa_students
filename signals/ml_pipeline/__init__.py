"""signals.ml_pipeline — backtest validation schemes used by the course notebooks.

Walk-forward analysis, combinatorial purged cross-validation (CPCV), and the
stationary-bootstrap reality check / SPA test.
"""

from signals.ml_pipeline.combinatorial_purged_cv import (
    CombinatorialPurgedCV,
    CPCVResult,
    CPCVSplit,
    reconstruct_paths,
    run_cpcv,
)
from signals.ml_pipeline.reality_check import (
    RealityCheckResult,
    SPAResult,
    hansens_spa,
    relative_performance,
    stationary_bootstrap_indices,
    whites_reality_check,
)
from signals.ml_pipeline.walk_forward import (
    FitPredictStrategy,
    WalkForwardResult,
    WalkForwardWindow,
    make_train_test_windows,
    walk_forward_strategy_returns,
)

__all__ = [
    "CPCVResult",
    "CPCVSplit",
    "CombinatorialPurgedCV",
    "FitPredictStrategy",
    "RealityCheckResult",
    "SPAResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "hansens_spa",
    "make_train_test_windows",
    "reconstruct_paths",
    "relative_performance",
    "run_cpcv",
    "stationary_bootstrap_indices",
    "walk_forward_strategy_returns",
    "whites_reality_check",
]
