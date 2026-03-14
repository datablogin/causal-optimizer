"""CLI for the causal optimizer."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.storage.sqlite import ExperimentStore
from causal_optimizer.types import ExperimentStatus

if TYPE_CHECKING:
    from causal_optimizer.domain_adapters.base import DomainAdapter


class _AdapterRunner:
    """Wraps a DomainAdapter into the ExperimentRunner protocol."""

    def __init__(self, adapter: DomainAdapter) -> None:
        self._adapter = adapter

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        return self._adapter.run_experiment(parameters)


def _load_adapter(spec: str) -> DomainAdapter:
    """Load a DomainAdapter from a 'module:ClassName' string."""
    if ":" not in spec:
        print(f"Error: adapter spec must be 'module:ClassName', got {spec!r}", file=sys.stderr)
        sys.exit(1)
    module_path, class_name = spec.rsplit(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        print(f"Error: could not import module {module_path!r}: {exc}", file=sys.stderr)
        sys.exit(1)
    cls = getattr(mod, class_name, None)
    if cls is None:
        print(f"Error: {module_path!r} has no attribute {class_name!r}", file=sys.stderr)
        sys.exit(1)
    return cls()  # type: ignore[no-any-return]


def _adapter_engine_kwargs(adapter: DomainAdapter) -> dict[str, Any]:
    """Build common engine kwargs from a domain adapter."""
    kwargs: dict[str, Any] = {"search_space": adapter.get_search_space()}
    graph = adapter.get_prior_graph()
    if graph is not None:
        kwargs["causal_graph"] = graph
    descriptors = adapter.get_descriptor_names()
    if descriptors:
        kwargs["descriptor_names"] = descriptors
    return kwargs


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a new experiment."""
    adapter = _load_adapter(args.adapter)
    store = ExperimentStore(args.db)
    experiment_id: str = args.id or str(uuid.uuid4())

    engine_kwargs = _adapter_engine_kwargs(adapter)
    engine_kwargs.update(
        runner=_AdapterRunner(adapter),
        store=store,
        experiment_id=experiment_id,
    )
    if args.seed is not None:
        engine_kwargs["seed"] = args.seed

    engine = ExperimentEngine(**engine_kwargs)
    engine.run_loop(args.budget)

    best = engine.log.best_result()
    if best is not None:
        print(f"Best objective: {best.metrics.get('objective', 'N/A')}")
    print(f"Experiment ID: {experiment_id}")


def _cmd_resume(args: argparse.Namespace) -> None:
    """Resume an interrupted experiment."""
    adapter = _load_adapter(args.adapter)
    store = ExperimentStore(args.db)

    engine_kwargs = _adapter_engine_kwargs(adapter)
    engine = ExperimentEngine.resume(
        store=store,
        experiment_id=args.id,
        runner=_AdapterRunner(adapter),
        **engine_kwargs,
    )
    engine.run_loop(args.budget)

    best = engine.log.best_result()
    if best is not None:
        print(f"Best objective: {best.metrics.get('objective', 'N/A')}")


def _cmd_report(args: argparse.Namespace) -> None:
    """Print a report for an experiment."""
    store = ExperimentStore(args.db)
    try:
        log = store.load_log(args.id)
    except KeyError:
        print(f"Error: experiment {args.id!r} not found", file=sys.stderr)
        sys.exit(1)

    n_kept = sum(1 for r in log.results if r.status == ExperimentStatus.KEEP)
    n_discarded = sum(1 for r in log.results if r.status == ExperimentStatus.DISCARD)
    n_crash = sum(1 for r in log.results if r.status == ExperimentStatus.CRASH)
    phases = sorted({r.metadata.get("phase", "unknown") for r in log.results})
    best = log.best_result()

    if args.format == "json":
        data: dict[str, Any] = {
            "experiment_id": args.id,
            "total_steps": len(log.results),
            "n_kept": n_kept,
            "n_discarded": n_discarded,
            "n_crash": n_crash,
            "phases": phases,
            "best_result": best.model_dump() if best is not None else None,
        }
        print(json.dumps(data, indent=2))
    else:
        # Table format
        print(f"Experiment: {args.id}")
        print(f"Total steps: {len(log.results)}")
        print(f"Kept: {n_kept}  Discarded: {n_discarded}  Crashed: {n_crash}")
        print(f"Phases: {', '.join(phases)}")
        if best is not None:
            print(f"Best result: {best.metrics}")
            print(f"  Parameters: {best.parameters}")
        else:
            print("No kept results.")


def _cmd_list(args: argparse.Namespace) -> None:
    """List all experiments in the DB."""
    store = ExperimentStore(args.db)
    experiments = store.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    # Print table
    print(f"{'ID':<20} {'Created':<28} {'Steps':>6}")
    print("-" * 56)
    for exp in experiments:
        print(f"{exp['id']:<20} {exp['created_at']:<28} {exp['step_count']:>6}")


def main() -> None:
    """Entry point for the causal-optimizer CLI."""
    parser = argparse.ArgumentParser(
        prog="causal-optimizer",
        description="Causally-informed experiment optimization engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Run a new experiment")
    run_parser.add_argument("--adapter", required=True, help="module:ClassName for DomainAdapter")
    run_parser.add_argument("--budget", type=int, required=True, help="Number of steps to run")
    run_parser.add_argument("--db", required=True, help="Path to SQLite database")
    run_parser.add_argument("--id", default=None, help="Experiment ID (auto-generated if omitted)")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # resume
    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted experiment")
    resume_parser.add_argument(
        "--adapter", required=True, help="module:ClassName for DomainAdapter"
    )
    resume_parser.add_argument("--id", required=True, help="Experiment ID to resume")
    resume_parser.add_argument("--db", required=True, help="Path to SQLite database")
    resume_parser.add_argument("--budget", type=int, default=10, help="Additional steps to run")

    # report
    report_parser = subparsers.add_parser("report", help="Print experiment report")
    report_parser.add_argument("--id", required=True, help="Experiment ID")
    report_parser.add_argument("--db", required=True, help="Path to SQLite database")
    report_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # list
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.add_argument("--db", required=True, help="Path to SQLite database")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run": _cmd_run,
        "resume": _cmd_resume,
        "report": _cmd_report,
        "list": _cmd_list,
    }
    commands[args.command](args)
