"""Local-only A/B harness for Sprint 21 reranking comparison.

.. deprecated:: Sprint 22
   The ``CAUSAL_OPT_RERANKING_MODE`` env-var toggle was removed in Sprint 22
   after the Sprint 21 A/B comparison showed alignment-only re-ranking is
   equal or better.  This script is retained for historical reference but is
   **no longer functional** — running it will use alignment-only on both sides,
   producing identical A and B outputs.

Runs the counterfactual or null benchmark twice in the same interpreter
session — once with balanced re-ranking (default) and once with
alignment-only re-ranking (via env var) — to eliminate environment drift.

The toggle uses ``CAUSAL_OPT_RERANKING_MODE=alignment_only`` which is
checked inside ``_suggest_bayesian()``.  When the env var is unset
(normal production use), balanced re-ranking is always used.

This is NOT production code.  It exists solely for the Sprint 21
controlled A/B comparison.

Usage::

    uv run python scripts/ab_reranking_harness.py \\
        --data-path data/ercot_north_c_dfw_2022_2024.parquet \\
        --variant base \\
        --budgets 20,40,80 \\
        --seeds 0,1,2,3,4,5,6,7,8,9 \\
        --strategies random,surrogate_only,causal \\
        --output-a artifacts/ab_base_balanced.json \\
        --output-b artifacts/ab_base_alignment_only.json

    # Null control:
    uv run python scripts/ab_reranking_harness.py \\
        --data-path data/ercot_north_c_dfw_2022_2024.parquet \\
        --null \\
        --budgets 20,40 \\
        --seeds 0,1,2 \\
        --strategies random,surrogate_only,causal \\
        --output-a artifacts/ab_null_balanced.json \\
        --output-b artifacts/ab_null_alignment_only.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run_benchmark(
    *,
    data_path: str,
    variant: str | None,
    budgets: str,
    seeds: str,
    strategies: str,
    output: str,
    null: bool,
    env_override: dict[str, str] | None = None,
) -> None:
    """Run a single benchmark via subprocess with optional env overrides."""
    if null:
        cmd = [
            sys.executable,
            "scripts/null_energy_benchmark.py",
            "--data-path",
            data_path,
            "--budgets",
            budgets,
            "--seeds",
            seeds,
            "--strategies",
            strategies,
            "--output",
            output,
        ]
    else:
        cmd = [
            sys.executable,
            "scripts/counterfactual_benchmark.py",
            "--data-path",
            data_path,
            "--variant",
            variant or "base",
            "--budgets",
            budgets,
            "--seeds",
            seeds,
            "--strategies",
            strategies,
            "--output",
            output,
        ]

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    print(f"[A/B harness] Running: {' '.join(cmd)}")
    if env_override:
        print(f"[A/B harness] Env overrides: {env_override}")
    subprocess.run(cmd, check=True, env=env)  # noqa: S603


def main() -> None:
    """Run A-side (balanced) and B-side (alignment-only) benchmarks back-to-back."""
    parser = argparse.ArgumentParser(
        description="Sprint 21 A/B reranking harness (local-only, not production code).",
    )
    parser.add_argument("--data-path", required=True, help="Path to ERCOT parquet file.")
    parser.add_argument("--variant", default="base", help="Counterfactual variant.")
    parser.add_argument("--null", action="store_true", help="Run null-control benchmark.")
    parser.add_argument("--budgets", default="20,40,80", help="Comma-separated budgets.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds.")
    parser.add_argument("--strategies", default="random,surrogate_only,causal")
    parser.add_argument("--output-a", required=True, help="Output for A-side (balanced).")
    parser.add_argument("--output-b", required=True, help="Output for B-side (alignment-only).")
    args = parser.parse_args()

    Path(args.output_a).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_b).parent.mkdir(parents=True, exist_ok=True)

    # A-side: balanced re-ranking (default, no env override)
    print("\n=== A-SIDE: Balanced re-ranking (Sprint 20 default) ===\n")
    _run_benchmark(
        data_path=args.data_path,
        variant=args.variant,
        budgets=args.budgets,
        seeds=args.seeds,
        strategies=args.strategies,
        output=args.output_a,
        null=args.null,
    )

    # B-side: alignment-only re-ranking (env var toggle)
    print("\n=== B-SIDE: Alignment-only re-ranking (Sprint 19, env toggle) ===\n")
    _run_benchmark(
        data_path=args.data_path,
        variant=args.variant,
        budgets=args.budgets,
        seeds=args.seeds,
        strategies=args.strategies,
        output=args.output_b,
        null=args.null,
        env_override={"CAUSAL_OPT_RERANKING_MODE": "alignment_only"},
    )

    print("\n=== A/B comparison complete ===")
    print(f"  A-side (balanced):       {args.output_a}")
    print(f"  B-side (alignment-only): {args.output_b}")


if __name__ == "__main__":
    main()
