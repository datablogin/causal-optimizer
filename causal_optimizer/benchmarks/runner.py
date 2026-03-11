"""Benchmark runner: compare causal vs. naive optimization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import SearchSpace, VariableType


@dataclass
class BenchmarkResult:
    """Result of running one strategy on one benchmark with one seed.

    Attributes:
        convergence_curve: Best-so-far objective at each step (monotonically non-increasing
            for minimization).
        final_best: Best objective value found across all experiments.
        experiments_to_threshold: Number of experiments to reach within ``threshold_pct``
            of the known optimum, or None if the threshold was never reached.
        strategy: Name of the strategy used.
        benchmark_name: Name of the benchmark class.
        seed: Random seed used for this run.
    """

    convergence_curve: list[float]
    final_best: float
    experiments_to_threshold: int | None
    strategy: str
    benchmark_name: str
    seed: int


class BenchmarkRunner:
    """Run benchmark SCMs with different optimization strategies.

    Supports three strategies:
    - ``"causal"``: ExperimentEngine with the benchmark's causal graph.
    - ``"random"``: Uniform random sampling from the search space.
    - ``"surrogate_only"``: ExperimentEngine without a causal graph (RF surrogate only).

    Parameters:
        benchmark: A benchmark SCM instance satisfying the ``BenchmarkSCM`` protocol.
        threshold_pct: Percentage tolerance for ``experiments_to_threshold``.
            A result is "within threshold" when ``|objective - optimum| <= threshold_pct
            * |optimum|``. Defaults to 10%.
    """

    _STRATEGIES = frozenset({"causal", "random", "surrogate_only"})

    def __init__(
        self,
        benchmark: Any,
        threshold_pct: float = 0.10,
    ) -> None:
        self.benchmark = benchmark
        self.threshold_pct = threshold_pct

    def run(
        self,
        strategy: str,
        budget: int,
        seed: int = 0,
        known_optimum: float | None = None,
    ) -> BenchmarkResult:
        """Run a single strategy on the benchmark.

        Args:
            strategy: One of ``"causal"``, ``"random"``, ``"surrogate_only"``.
            budget: Number of experiments to run.
            seed: Random seed for reproducibility.
            known_optimum: Known optimal objective value for threshold calculation.
                If None, ``experiments_to_threshold`` will be None.

        Returns:
            A ``BenchmarkResult`` with convergence curve and summary statistics.

        Raises:
            ValueError: If ``strategy`` is not one of the supported strategies.
        """
        if strategy not in self._STRATEGIES:
            msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(self._STRATEGIES)}."
            raise ValueError(msg)

        benchmark_name = type(self.benchmark).__name__
        rng = np.random.default_rng(seed)

        if strategy == "random":
            return self._run_random(budget, seed, rng, benchmark_name, known_optimum)

        return self._run_engine(strategy, budget, seed, rng, benchmark_name, known_optimum)

    def compare(
        self,
        strategies: list[str],
        budget: int,
        n_seeds: int = 3,
        known_optimum: float | None = None,
    ) -> list[BenchmarkResult]:
        """Run multiple strategies across multiple seeds and collect results.

        Args:
            strategies: List of strategy names to compare.
            budget: Number of experiments per run.
            n_seeds: Number of random seeds to use per strategy.
            known_optimum: Known optimal objective value for threshold calculation.

        Returns:
            List of ``BenchmarkResult``, one per (strategy, seed) pair.
        """
        results: list[BenchmarkResult] = []
        for strategy in strategies:
            for seed in range(n_seeds):
                result = self.run(strategy, budget, seed=seed, known_optimum=known_optimum)
                results.append(result)
        return results

    def _run_engine(
        self,
        strategy: str,
        budget: int,
        seed: int,
        rng: np.random.Generator,
        benchmark_name: str,
        known_optimum: float | None,
    ) -> BenchmarkResult:
        """Run ExperimentEngine with or without a causal graph."""
        # Create a fresh benchmark instance with the given seed for the runner
        benchmark_type = type(self.benchmark)
        bench = benchmark_type(
            noise_scale=self.benchmark.noise_scale,
            rng=np.random.default_rng(seed),
        )
        space = benchmark_type.search_space()
        graph = benchmark_type.causal_graph() if strategy == "causal" else None

        engine = ExperimentEngine(
            search_space=space,
            runner=bench,
            causal_graph=graph,
        )

        convergence_curve: list[float] = []
        best_so_far = float("inf")

        for _ in range(budget):
            result = engine.step()
            obj = result.metrics.get("objective", float("inf"))
            best_so_far = min(best_so_far, obj)
            convergence_curve.append(best_so_far)

        experiments_to_threshold = self._compute_threshold_step(convergence_curve, known_optimum)

        return BenchmarkResult(
            convergence_curve=convergence_curve,
            final_best=best_so_far,
            experiments_to_threshold=experiments_to_threshold,
            strategy=strategy,
            benchmark_name=benchmark_name,
            seed=seed,
        )

    def _run_random(
        self,
        budget: int,
        seed: int,
        rng: np.random.Generator,
        benchmark_name: str,
        known_optimum: float | None,
    ) -> BenchmarkResult:
        """Run uniform random sampling from the search space."""
        benchmark_type = type(self.benchmark)
        bench = benchmark_type(
            noise_scale=self.benchmark.noise_scale,
            rng=np.random.default_rng(seed + 1_000_000),
        )
        space = benchmark_type.search_space()

        convergence_curve: list[float] = []
        best_so_far = float("inf")

        for _ in range(budget):
            params = _sample_random_params(space, rng)
            metrics = bench.run(params)
            obj = metrics.get("objective", float("inf"))
            best_so_far = min(best_so_far, obj)
            convergence_curve.append(best_so_far)

        experiments_to_threshold = self._compute_threshold_step(convergence_curve, known_optimum)

        return BenchmarkResult(
            convergence_curve=convergence_curve,
            final_best=best_so_far,
            experiments_to_threshold=experiments_to_threshold,
            strategy="random",
            benchmark_name=benchmark_name,
            seed=seed,
        )

    def _compute_threshold_step(
        self,
        convergence_curve: list[float],
        known_optimum: float | None,
    ) -> int | None:
        """Find the first step where the objective is within threshold of optimum."""
        if known_optimum is None:
            return None

        abs_optimum = abs(known_optimum) if known_optimum != 0.0 else 1.0
        threshold = self.threshold_pct * abs_optimum

        for i, val in enumerate(convergence_curve):
            if abs(val - known_optimum) <= threshold:
                return i + 1  # 1-indexed step count
        return None


def _sample_random_params(space: SearchSpace, rng: np.random.Generator) -> dict[str, Any]:
    """Sample uniformly random parameters from a search space."""
    params: dict[str, Any] = {}
    for var in space.variables:
        if var.variable_type == VariableType.CONTINUOUS:
            lower = var.lower if var.lower is not None else -1.0
            upper = var.upper if var.upper is not None else 1.0
            params[var.name] = float(rng.uniform(lower, upper))
        elif var.variable_type == VariableType.INTEGER:
            lower = int(var.lower) if var.lower is not None else 0
            upper = int(var.upper) if var.upper is not None else 10
            params[var.name] = int(rng.integers(lower, upper + 1))
        elif var.variable_type == VariableType.BOOLEAN:
            params[var.name] = bool(rng.choice([True, False]))
        elif var.variable_type == VariableType.CATEGORICAL:
            choices = var.choices or []
            if choices:
                params[var.name] = choices[int(rng.integers(0, len(choices)))]
    return params
