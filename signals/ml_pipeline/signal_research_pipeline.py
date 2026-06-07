"""
signal_research_pipeline.py
---------------------------
Strategy-agnostic parameter-grid research harness.

Usage
~~~~~

1. Build a :class:`ParameterGrid` of lists: backtest-config overrides,
   strategy kwargs, signal kwargs.
2. Provide a ``strategy_factory(CellParams) -> Strategy`` that knows how
   to wire up a strategy from one cell's parameters.
3. Call :meth:`SignalResearchPipeline.run` and receive a
   :class:`ResearchResult` with per-cell NAVs, weights, metrics, and PDF
   paths; a cross-sectional comparison PDF is written separately by
   :mod:`analytics.cross_section_report`.

Results are persisted under ``output_dir/runs/cell_XXXX/`` and
``output_dir/summary/`` so rerunning the same grid after widening only
executes the newly-added cells.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.strategy_analyzer import StrategyAnalyzer
from analytics.timeseries_analyzer import TimeseriesAnalyzer, infer_periods_per_year
from backtesting.backtest_engine import BacktestConfig, Backtester
from backtesting.backtest_result import BacktestResult
from backtesting.strategy_protocols import Strategy

# ---------------------------------------------------------------------------
# Grid types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellParams:
    """Concrete parameter slice — one cell of the grid.

    Each sub-dict holds scalar values (one chosen value per swept key).
    ``cell_id`` is a stable identifier assigned by :class:`ParameterGrid`.
    """

    cell_id: str
    backtest: dict[str, object] = field(default_factory=dict)
    strategy: dict[str, object] = field(default_factory=dict)
    signal: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            **{f"backtest.{k}": v for k, v in self.backtest.items()},
            **{f"strategy.{k}": v for k, v in self.strategy.items()},
            **{f"signal.{k}": v for k, v in self.signal.items()},
        }

    def fingerprint(self) -> str:
        """Stable string fingerprint used for caching."""
        payload = {
            "backtest": self.backtest,
            "strategy": self.strategy,
            "signal": self.signal,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()


@dataclass
class ParameterGrid:
    """Cartesian product of parameter lists across three sections.

    Each dict maps a parameter name to a *list* of candidate values.  The
    cartesian product includes every value from every list across every
    section.
    """

    backtest: dict[str, list] = field(default_factory=dict)
    strategy: dict[str, list] = field(default_factory=dict)
    signal: dict[str, list] = field(default_factory=dict)

    def expand(self) -> list[CellParams]:
        """Flatten the grid into a list of :class:`CellParams`."""
        flat_keys: list[tuple[str, str]] = []
        flat_value_lists: list[list] = []
        for section_name, section_dict in (
            ("backtest", self.backtest),
            ("strategy", self.strategy),
            ("signal", self.signal),
        ):
            for key, values in section_dict.items():
                if not values:
                    raise ValueError(
                        f"Empty value list for {section_name}.{key} "
                        "(at least one value required)."
                    )
                flat_keys.append((section_name, key))
                flat_value_lists.append(list(values))

        if not flat_keys:
            return [CellParams(cell_id=self._cell_id(0))]

        cells: list[CellParams] = []
        for idx, combination in enumerate(itertools.product(*flat_value_lists)):
            per_section: dict[str, dict[str, object]] = {
                "backtest": {},
                "strategy": {},
                "signal": {},
            }
            for (section_name, key), value in zip(flat_keys, combination, strict=True):
                per_section[section_name][key] = value
            cells.append(
                CellParams(
                    cell_id=self._cell_id(idx),
                    backtest=per_section["backtest"],
                    strategy=per_section["strategy"],
                    signal=per_section["signal"],
                )
            )
        return cells

    @staticmethod
    def _cell_id(idx: int) -> str:
        return f"cell_{idx:04d}"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ResearchResult:
    """Aggregated output of a :class:`SignalResearchPipeline` run."""

    parameters: pd.DataFrame
    metrics: pd.DataFrame
    nav_matrix: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    report_paths: dict[str, Path]

    def summary_table(self) -> pd.DataFrame:
        """Parameters + metrics joined on cell_id — handy for reporting."""
        return self.parameters.join(self.metrics, how="inner")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SignalResearchPipeline:
    """Run one backtest per grid cell, persist artefacts, aggregate results.

    Parameters
    ----------
    strategy_factory : Callable[[CellParams], Strategy]
        Builds a concrete ``Strategy`` from one cell's parameters.  The
        factory owns the mapping from ``CellParams.strategy`` /
        ``.signal`` dicts onto constructor kwargs.
    prices : pd.DataFrame
        Price-level DataFrame (``-i`` columns) used for every cell.
    benchmark_nav : pd.Series | None
        Benchmark NAV — passed through to per-cell PDFs.  When ``None``,
        the strategy NAV itself is used as a trivial benchmark, which
        leaves IR/TE undefined.
    base_backtest_config : BacktestConfig
        Default backtester config; every cell starts from this and
        applies the overrides in ``CellParams.backtest``.
    output_dir : Path
        Root directory for cell artefacts and summary outputs.
    generate_individual_reports : bool
        When ``True`` (default), a per-cell PDF is written via
        :func:`analytics.single_strategy_report.build_single_strategy_report`.
    parallel : bool
        Reserved for future multiprocessing support (Phase 2).  Must be
        ``False`` in this implementation.
    signal_history_fn : Callable[[pd.DataFrame, CellParams], pd.Series] | None
        Optional callback to compute a signal history for the per-cell
        PDF report (drives the IC panel).  When ``None``, the IC and
        signal-vs-weights pages are skipped.
    """

    def __init__(
        self,
        strategy_factory: Callable[[CellParams], Strategy],
        prices: pd.DataFrame,
        benchmark_nav: pd.Series | None,
        base_backtest_config: BacktestConfig,
        output_dir: Path,
        generate_individual_reports: bool = True,
        parallel: bool = False,
        signal_history_fn: Callable[[pd.DataFrame, CellParams], pd.Series]
        | None = None,
    ) -> None:
        if parallel:
            raise NotImplementedError(
                "parallel=True is reserved for a future multiprocessing phase."
            )

        self.strategy_factory = strategy_factory
        self.prices = prices
        self.benchmark_nav = benchmark_nav
        self.base_backtest_config = base_backtest_config
        self.output_dir = Path(output_dir)
        self.generate_individual_reports = generate_individual_reports
        self.signal_history_fn = signal_history_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, grid: ParameterGrid) -> ResearchResult:
        cells = grid.expand()
        if not cells:
            raise ValueError("ParameterGrid.expand() returned zero cells.")

        runs_dir = self.output_dir / "runs"
        summary_dir = self.output_dir / "summary"
        runs_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        per_cell_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        nav_frames: dict[str, pd.Series] = {}
        weights: dict[str, pd.DataFrame] = {}
        report_paths: dict[str, Path] = {}

        for cell in cells:
            cell_dir = runs_dir / cell.cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)

            nav_path = cell_dir / "nav.parquet"
            weights_path = cell_dir / "weights.parquet"
            params_path = cell_dir / "params.json"
            report_path = cell_dir / "report.pdf"

            cached = (
                nav_path.exists()
                and weights_path.exists()
                and params_path.exists()
                and self._params_match(params_path, cell)
            )
            if cached:
                nav_frame = pd.read_parquet(nav_path)
                weights_frame = pd.read_parquet(weights_path)
                nav = nav_frame["nav"]
                turnover = nav_frame["turnover"]
                returns = nav_frame["returns"]
                result = BacktestResult(
                    nav=nav,
                    returns=returns,
                    weights_history=weights_frame,
                    turnover=turnover,
                    rebalance_dates=list(
                        pd.to_datetime(nav_frame.index[turnover > 0.0])
                    ),
                )
            else:
                strategy = self.strategy_factory(cell)
                config = self._override_config(cell)
                result = Backtester(config).run(prices=self.prices, strategy=strategy)
                self._persist_cell(cell, result, params_path, nav_path, weights_path)

            cell_benchmark = self._aligned_benchmark(result.nav)
            metrics = self._compute_metrics(
                result=result,
                benchmark_nav=cell_benchmark,
                cost_bps=float(self._resolved_cost_bps(cell)),
            )
            metric_rows.append({"cell_id": cell.cell_id, **metrics})
            per_cell_rows.append(cell.to_dict())
            nav_frames[cell.cell_id] = result.nav.rename(cell.cell_id)
            weights[cell.cell_id] = result.weights_history

            if self.generate_individual_reports:
                if not report_path.exists() or not cached:
                    self._write_cell_pdf(
                        cell=cell,
                        result=result,
                        benchmark_nav=cell_benchmark,
                        output_path=report_path,
                    )
                report_paths[cell.cell_id] = report_path

        parameters = pd.DataFrame(per_cell_rows).set_index("cell_id")
        metrics_df = pd.DataFrame(metric_rows).set_index("cell_id")
        nav_matrix = pd.concat(nav_frames.values(), axis=1)

        parameters.to_parquet(summary_dir / "parameters.parquet")
        metrics_df.to_parquet(summary_dir / "metrics.parquet")
        nav_matrix.to_parquet(summary_dir / "nav_matrix.parquet")

        return ResearchResult(
            parameters=parameters,
            metrics=metrics_df,
            nav_matrix=nav_matrix,
            weights=weights,
            report_paths=report_paths,
        )

    # ------------------------------------------------------------------
    # Cell execution helpers
    # ------------------------------------------------------------------

    def _override_config(self, cell: CellParams) -> BacktestConfig:
        return override_backtest_config(self.base_backtest_config, cell)

    def _resolved_cost_bps(self, cell: CellParams) -> float:
        cfg = self._override_config(cell)
        return float(cfg.transaction_cost_bps)

    def _persist_cell(
        self,
        cell: CellParams,
        result: BacktestResult,
        params_path: Path,
        nav_path: Path,
        weights_path: Path,
    ) -> None:
        params_path.write_text(
            json.dumps(
                {
                    "cell_id": cell.cell_id,
                    "fingerprint": cell.fingerprint(),
                    "backtest": {k: _json_safe(v) for k, v in cell.backtest.items()},
                    "strategy": {k: _json_safe(v) for k, v in cell.strategy.items()},
                    "signal": {k: _json_safe(v) for k, v in cell.signal.items()},
                },
                indent=2,
            )
        )

        nav_frame = pd.concat(
            {
                "nav": result.nav,
                "returns": result.returns,
                "turnover": result.turnover,
            },
            axis=1,
        )
        nav_frame.to_parquet(nav_path)
        result.weights_history.to_parquet(weights_path)

    @staticmethod
    def _params_match(params_path: Path, cell: CellParams) -> bool:
        try:
            payload = json.loads(params_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("fingerprint") == cell.fingerprint()

    def _aligned_benchmark(self, nav: pd.Series) -> pd.Series:
        if self.benchmark_nav is None:
            # Use the strategy NAV itself → trivial benchmark.  IR/TE
            # become NaN, which downstream tooling handles.
            return nav.copy()
        return self.benchmark_nav.reindex(nav.index).ffill()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        result: BacktestResult,
        benchmark_nav: pd.Series,
        cost_bps: float,
    ) -> dict[str, float]:
        """Collapse one cell's ``BacktestResult`` into a flat metric dict."""
        ts = TimeseriesAnalyzer(result.nav.to_frame("strategy"))
        ts_metrics = ts.apply_standard_functions()
        out: dict[str, float] = {}
        for metric, row in ts_metrics.iterrows():
            value = row["strategy"]
            try:
                out[str(metric)] = float(value)
            except (TypeError, ValueError):
                continue  # skip timestamps / NaT from max_drawdown helpers

        analyzer = StrategyAnalyzer(
            nav=result.nav,
            benchmark_nav=benchmark_nav,
            weights_history=result.weights_history,
            turnover=result.turnover,
            signal_history=None,
            cost_bps=cost_bps,
        )
        active = analyzer.apply_standard_functions()
        for metric in active.index:
            out[str(metric)] = float(active.loc[metric, "Strategy vs Benchmark"])

        out["turnover_mean"] = float(result.turnover.mean())
        tc = analyzer.accumulated_transaction_costs()
        out["accumulated_transaction_cost"] = (
            float(tc.iloc[-1]) if not tc.empty else 0.0
        )
        return out

    # ------------------------------------------------------------------
    # PDF helpers
    # ------------------------------------------------------------------

    def _write_cell_pdf(
        self,
        cell: CellParams,
        result: BacktestResult,
        benchmark_nav: pd.Series,
        output_path: Path,
    ) -> None:
        from analytics.single_strategy_report import build_single_strategy_report

        signal_history = None
        if self.signal_history_fn is not None:
            signal_history = self.signal_history_fn(self.prices, cell)
            if signal_history is not None:
                signal_history = signal_history.reindex(result.nav.index)

        build_single_strategy_report(
            result=result,
            benchmark_nav=benchmark_nav,
            cost_bps=float(self._resolved_cost_bps(cell)),
            strategy_name=cell.cell_id,
            output_path=output_path,
            signal_history=signal_history,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_safe(value: object) -> object:
    """Convert numpy / pandas scalars into plain Python for JSON dumps."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Backtester-driven parameter search (used by walk-forward / CPCV validation)
# ---------------------------------------------------------------------------


def override_backtest_config(
    base_config: BacktestConfig,
    cell: CellParams,
) -> BacktestConfig:
    """Apply a cell's ``backtest`` overrides onto ``base_config``."""
    kwargs: dict[str, object] = {}
    for key, value in cell.backtest.items():
        if not hasattr(base_config, key):
            raise ValueError(
                f"BacktestConfig has no attribute '{key}' (cell {cell.cell_id})."
            )
        kwargs[key] = value
    return dataclasses.replace(base_config, **kwargs)


def cell_param_values(cell: CellParams) -> dict[str, object]:
    """Flatten a cell's swept parameters (strategy + signal + backtest) into one
    dict — used to record the *selected* parameters of a walk-forward / CPCV run.
    """
    return {**cell.strategy, **cell.signal, **cell.backtest}


def max_grid_lookback(grid: ParameterGrid) -> int:
    """Largest ``lookback`` swept by the grid (``1`` when none is swept).

    Validation schemes use this to require a train window long enough for the
    strategy to form a signal.
    """
    lookbacks = grid.strategy.get("lookback")
    return int(max(lookbacks)) if lookbacks else 1


def backtest_param_cells(
    param_grid: ParameterGrid,
    strategy_factory: Callable[[CellParams], Strategy],
    prices: pd.DataFrame,
    base_config: BacktestConfig,
) -> dict[str, pd.Series]:
    """Full-sample per-period strategy returns for every grid cell.

    Each cell is backtested once over the *whole* ``prices`` history; the
    resulting return series is causal (the return at ``t`` depends only on
    prices up to ``t``), so it can later be reindexed to any train or test
    sub-index.  Results are keyed by ``cell_id``.
    """
    out: dict[str, pd.Series] = {}
    for param_cell in param_grid.expand():
        config = override_backtest_config(base_config, param_cell)
        result = Backtester(config).run(
            prices=prices, strategy=strategy_factory(param_cell)
        )
        out[param_cell.cell_id] = result.returns
    return out


def select_best_param_cell(
    param_grid: ParameterGrid,
    strategy_factory: Callable[[CellParams], Strategy],
    prices: pd.DataFrame,
    base_config: BacktestConfig,
    metric_index: pd.DatetimeIndex | None = None,
    periods_per_year: float | None = None,
    cell_returns: dict[str, pd.Series] | None = None,
) -> tuple[CellParams, float]:
    """Grid cell maximising the annualised (rf=0) Sharpe over ``metric_index``.

    ``cell_returns`` lets the caller pass a precomputed full-sample backtest
    cache (see :func:`backtest_cells`) so a parameter sweep is not re-run for
    every train window.  ``metric_index`` restricts the Sharpe to a train
    sub-index (causal slicing of the cached returns); ``None`` uses the whole
    sample.  Ties keep the first cell in grid order.
    """
    param_cells = param_grid.expand()
    if cell_returns is None:
        cell_returns = backtest_param_cells(
            param_grid, strategy_factory, prices, base_config
        )

    best_param_cell = param_cells[0]
    best_sharpe = float("-inf")
    for cell in param_cells:
        series = cell_returns[cell.cell_id]
        if metric_index is not None:
            series = series.reindex(metric_index)
        sharpe = _selection_sharpe(series, periods_per_year)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_param_cell = cell
    return best_param_cell, best_sharpe


def backtest_window(
    strategy: Strategy,
    prices: pd.DataFrame,
    window_index: pd.DatetimeIndex,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest ``strategy`` over a date window with full-history warm-up.

    The engine sees the *full* ``prices`` (so ``compute_weights`` warms up on
    real prior history) but the simulated period is restricted to
    ``[window_index[0], window_index[-1]]`` via ``start_date`` / ``end_date``;
    the starting weights are taken from the strategy at the window's first bar.
    """
    cfg = dataclasses.replace(
        config,
        start_date=window_index[0],
        end_date=window_index[-1],
        initial_weights_from_strategy=True,
    )
    return Backtester(cfg).run(prices=prices, strategy=strategy)


def backtested_trial_matrix(
    grid: ParameterGrid,
    strategy_factory: Callable[[CellParams], Strategy],
    prices: pd.DataFrame,
    base_config: BacktestConfig,
    cell_returns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Full-sample backtested returns for every grid cell, as a trial matrix.

    Columns are labelled by the cell's swept parameters (falling back to the
    ``cell_id``); rows are dates.  Feeds the data-snooping tests
    (:func:`whites_reality_check`, :func:`hansens_spa`,
    :func:`deflated_sharpe_ratio_from_returns`).  Pass ``cell_returns`` (from
    :func:`backtest_param_cells`) to reuse a cache instead of re-backtesting.
    """
    if cell_returns is None:
        cell_returns = backtest_param_cells(grid, strategy_factory, prices, base_config)
    columns = {_cell_label(cell): cell_returns[cell.cell_id] for cell in grid.expand()}
    return pd.DataFrame(columns).dropna(how="any")


def _selection_sharpe(returns: pd.Series, periods_per_year: float | None) -> float:
    """Annualised (rf=0) Sharpe for parameter selection (``-inf`` if degenerate).

    Returns ``-inf`` (not ``NaN``) on a degenerate slice so it never wins an
    ``argmax`` over the grid.
    """
    series = returns.dropna()
    if len(series) < 2:
        return float("-inf")
    std = float(series.std(ddof=1))
    if std == 0.0:
        return float("-inf")
    ppy = periods_per_year or infer_periods_per_year(series.index)
    return float(series.mean()) / std * sqrt(ppy)


def _cell_label(cell: CellParams) -> str:
    """Compact ``key=value`` label for a cell's swept params (``cell_id`` if none)."""
    values = cell_param_values(cell)
    if not values:
        return cell.cell_id
    return "_".join(f"{key}{value}" for key, value in values.items())
