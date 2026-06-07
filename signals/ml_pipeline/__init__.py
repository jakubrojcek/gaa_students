"""signals.ml_pipeline — backtest validation, Backtester-driven.

A strategy-agnostic parameter-grid research harness (:class:`ParameterGrid` plus
a strategy factory), the self-refitting :class:`WalkForwardStrategy`,
combinatorial purged cross-validation (CPCV), and the stationary-bootstrap
reality check / SPA test.  A single full-sample backtest per grid cell is cached
and reused (sliced) by every scheme, so each reported number flows through
:class:`backtesting.Backtester`.
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
from signals.ml_pipeline.signal_research_pipeline import (
    CellParams,
    ParameterGrid,
    ResearchResult,
    SignalResearchPipeline,
    backtest_param_cells,
    backtest_window,
    backtested_trial_matrix,
    select_best_param_cell,
)
from signals.ml_pipeline.walk_forward import (
    WalkForwardStrategy,
    WalkForwardWindow,
    make_train_test_windows,
)

__all__ = [
    "CPCVResult",
    "CPCVSplit",
    "CellParams",
    "CombinatorialPurgedCV",
    "ParameterGrid",
    "RealityCheckResult",
    "ResearchResult",
    "SPAResult",
    "SignalResearchPipeline",
    "WalkForwardStrategy",
    "WalkForwardWindow",
    "backtest_param_cells",
    "backtest_window",
    "backtested_trial_matrix",
    "hansens_spa",
    "make_train_test_windows",
    "reconstruct_paths",
    "relative_performance",
    "run_cpcv",
    "select_best_param_cell",
    "stationary_bootstrap_indices",
    "whites_reality_check",
]
