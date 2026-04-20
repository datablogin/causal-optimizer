"""Helper to turn the Sprint 35.C benchmark JSON artifact into report Markdown.

Reads the JSON written by ``scripts/open_bandit_benchmark.py`` and
prints Markdown tables ready for ``sprint-35-open-bandit-benchmark-report.md``.

Not part of the CLI surface of the benchmark; kept as a separate helper
because it is only used at report-drafting time.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu


def _load(artifact_path: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(Path(artifact_path).read_text())
    return payload


def _group(results: list[dict[str, Any]]) -> dict[tuple[bool, str, int], list[dict[str, Any]]]:
    out: dict[tuple[bool, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        out[(bool(r["is_null_control"]), r["strategy"], int(r["budget"]))].append(r)
    return dict(out)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _mwu_p(a: list[float], b: list[float]) -> float:
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def _classify(p: float) -> str:
    if np.isnan(p):
        return "not-applicable"
    if p <= 0.05:
        return "certified"
    if p <= 0.15:
        return "trending"
    return "not-significant"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _open_bandit_report_helper.py <artifact.json>", file=sys.stderr)
        sys.exit(2)
    payload = _load(sys.argv[1])
    results = payload["results"]
    groups = _group(results)

    real_by_sb: dict[tuple[str, int], list[float]] = defaultdict(list)
    null_by_sb: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_seed_dm: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_seed_dr: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (is_null, strat, bud), bucket in groups.items():
        for r in bucket:
            pv = r["policy_value_snipw"]
            if pv is None:
                continue
            if is_null:
                null_by_sb[(strat, bud)].append(float(pv))
            else:
                real_by_sb[(strat, bud)].append(float(pv))
                if r["policy_value_dm"] is not None:
                    per_seed_dm[(strat, bud)].append(float(r["policy_value_dm"]))
                if r["policy_value_dr"] is not None:
                    per_seed_dr[(strat, bud)].append(float(r["policy_value_dr"]))

    print("## Per-Budget SNIPW Summary")
    print()
    print("| Strategy | Budget | n | Mean | Std (ddof=0) | Min | Max |")
    print("|----------|--------|---|------|--------------|-----|-----|")
    for (strat, bud), vals in sorted(real_by_sb.items()):
        m, s = _mean_std(vals)
        lo = min(vals)
        hi = max(vals)
        print(f"| {strat} | {bud} | {len(vals)} | {m:.6f} | {s:.6f} | {lo:.6f} | {hi:.6f} |")

    print()
    print("## DM and DR Secondary Means (real data only)")
    print()
    print("| Strategy | Budget | DM mean | DR mean |")
    print("|----------|--------|---------|---------|")
    for (strat, bud), _vals in sorted(real_by_sb.items()):
        dm = per_seed_dm.get((strat, bud), [])
        dr = per_seed_dr.get((strat, bud), [])
        dm_m = float(np.mean(dm)) if dm else float("nan")
        dr_m = float(np.mean(dr)) if dr else float("nan")
        print(f"| {strat} | {bud} | {dm_m:.6f} | {dr_m:.6f} |")

    # Pairwise comparisons (causal vs surrogate_only, causal vs random, surrogate_only vs random)
    print()
    print("## Pairwise Comparisons (two-sided MWU on SNIPW)")
    print()
    print("| Comparison | Budget | p-value | Verdict |")
    print("|------------|--------|---------|---------|")
    budgets_set = {bud for (_, bud) in real_by_sb}
    for cmp_a, cmp_b in (
        ("causal", "surrogate_only"),
        ("causal", "random"),
        ("surrogate_only", "random"),
    ):
        for bud in sorted(budgets_set):
            a = real_by_sb.get((cmp_a, bud), [])
            b = real_by_sb.get((cmp_b, bud), [])
            if not a or not b:
                continue
            p = _mwu_p(a, b)
            verdict = _classify(p)
            print(f"| {cmp_a} vs {cmp_b} | {bud} | {p:.4f} | {verdict} |")

    # Null control
    print()
    print("## Null Control (Section 7a)")
    print()
    print("| Strategy | Budget | Mean Policy Value | μ_null | Ratio | Within 5% band |")
    print("|----------|--------|-------------------|--------|-------|----------------|")
    null_vals = np.asarray([r["policy_value_snipw"] for r in results if r["is_null_control"]])
    mu_null = float(null_vals.mean())
    # mu_null here is the mean over permuted policy values; the gates payload
    # carries the data-level raw reward mean which is what the report quotes.
    gates = payload.get("gates", {}).get("gates", {})
    null_info = gates.get("null_control", {})
    mu_null_gate = float(null_info.get("mu_null", mu_null))
    threshold = float(null_info.get("threshold", 1.05 * mu_null_gate))
    for (strat, bud), vals in sorted(null_by_sb.items()):
        m, _ = _mean_std(vals)
        within = m <= threshold
        print(
            f"| {strat} | {bud} | {m:.6f} | {mu_null_gate:.6f} | "
            f"{(m / mu_null_gate if mu_null_gate > 0 else float('nan')):.4f} | "
            f"{'yes' if within else 'NO'} |"
        )
    print(f"\n**μ_null = {mu_null_gate:.6f}**, threshold = 1.05 × μ_null = {threshold:.6f}.")

    # Gates
    print()
    print("## Section 7 Gate Details")
    print()
    for gate_name, gate_data in gates.items():
        letter = gate_data.get("gate_letter", "?")
        passed = gate_data.get("passed", False)
        print(f"### 7{letter} {gate_name} ({'PASS' if passed else 'FAIL'})")
        print()
        for k, v in gate_data.items():
            if k in {"gate_letter", "passed", "per_cell_values", "per_cell_ratios", "per_seed"}:
                continue
            print(f"- `{k}`: {v}")
        print()

    # Per-seed B80 table
    print()
    print("## Per-Seed Detail (B80, SNIPW)")
    print()
    b80 = [r for r in results if r["budget"] == 80 and not r["is_null_control"]]
    seeds = sorted({int(r["seed"]) for r in b80})
    strategies = ["random", "surrogate_only", "causal"]
    print(f"| Seed | {' | '.join(strategies)} |")
    print(f"|------|{'|'.join(['-----'] * len(strategies))}|")
    for s in seeds:
        row = [str(s)]
        for strat in strategies:
            match = [r for r in b80 if r["strategy"] == strat and int(r["seed"]) == s]
            if match:
                row.append(f"{match[0]['policy_value_snipw']:.6f}")
            else:
                row.append("—")
        print(f"| {' | '.join(row)} |")

    # Provenance summary
    print()
    print("## Provenance Summary")
    dp = payload.get("data_provenance", {})
    prov = payload.get("provenance", {})
    print()
    for k in (
        "data_path",
        "men_csv_path",
        "men_csv_sha256",
        "n_rounds",
        "n_actions",
        "n_positions",
        "pscore_mean",
        "click_mean",
        "propensity_schema",
        "position_handling_default",
        "obp_version",
        "min_propensity_clip",
    ):
        print(f"- `{k}`: {dp.get(k)}")
    print(f"- `optimizer_path`: {prov.get('optimizer_path')}")
    print(f"- `git_sha`: {prov.get('git_sha')}")
    print(f"- `python_version`: {prov.get('python_version')}")
    print(f"- `timestamp`: {prov.get('timestamp')}")


if __name__ == "__main__":
    main()
