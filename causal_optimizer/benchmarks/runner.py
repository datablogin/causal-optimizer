"""Benchmark runner: compare causal vs. naive optimization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import SearchSpace, VariableType

if TYPE_CHECKING:
    from causal_optimizer.benchmarks import BenchmarkSCM


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

    All current benchmarks minimize ``"objective"``; the runner hardcodes
    ``min()`` and ``"objective"`` accordingly. See known issues in CLAUDE.md.

    Note on seed handling: the ``"random"`` strategy uses ``SeedSequence.spawn()``
    to create two independent RNG streams (one for sampling, one for benchmark
    noise), ensuring proper decorrelation. The ``"causal"``/``"surrogate_only"``
    strategies delegate RNG to the engine.

    Parameters:
        benchmark: A benchmark SCM instance satisfying the ``BenchmarkSCM`` protocol.
            Must accept ``noise_scale`` and ``rng`` keyword arguments in ``__init__``
            (all existing benchmarks do, but the Protocol cannot enforce constructors).
        threshold_pct: Percentage tolerance for ``experiments_to_threshold``.
            A result is "within threshold" when ``objective <= known_optimum +
            threshold_pct * |optimum|`` (one-sided: at-or-better than the
            optimum counts, even if the measured value overshoots). Defaults
            to 10%.
    """

    _STRATEGIES = frozenset({"causal", "random", "surrogate_only"})

    def __init__(
        self,
        benchmark: BenchmarkSCM,
        threshold_pct: float = 0.10,
    ) -> None:
        self.benchmark = benchmark
        self.threshold_pct = threshold_pct

    def _validate_strategy(self, strategy: str) -> None:
        """Raise ValueError if strategy is not supported."""
        if strategy not in self._STRATEGIES:
            msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(self._STRATEGIES)}."
            raise ValueError(msg)

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
            seed: Random seed for reproducibility. For the ``"random"`` strategy,
                both sampling and noise RNGs are fully controlled. For engine-based
                strategies (``"causal"``, ``"surrogate_only"``), only the benchmark
                noise is seeded; the engine's internal RNG (suggestions, bootstrap)
                is not controlled, so results may vary across runs.
            known_optimum: Known optimal objective value for threshold calculation.
                If None, ``experiments_to_threshold`` will be None.

        Returns:
            A ``BenchmarkResult`` with convergence curve and summary statistics.

        Raises:
            ValueError: If ``strategy`` is not one of the supported strategies.
        """
        self._validate_strategy(strategy)
        if budget <= 0:
            msg = f"budget must be a positive integer, got {budget!r}."
            raise ValueError(msg)

        benchmark_name = type(self.benchmark).__name__

        if strategy == "random":
            return self._run_random(budget, seed, benchmark_name, known_optimum)

        return self._run_engine(strategy, budget, seed, benchmark_name, known_optimum)

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

        Raises:
            ValueError: If any strategy name is not supported, or if ``n_seeds``
                is not a positive integer.
        """
        if n_seeds <= 0:
            msg = f"n_seeds must be a positive integer, got {n_seeds!r}."
            raise ValueError(msg)

        for strategy in strategies:
            self._validate_strategy(strategy)

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
        benchmark_name: str,
        known_optimum: float | None,
    ) -> BenchmarkResult:
        """Run ExperimentEngine with or without a causal graph."""
        # Create a fresh benchmark instance with the given seed for the runner.
        # Cast to Any because Protocol cannot enforce constructor signatures;
        # all existing benchmarks accept (noise_scale, rng) kwargs.
        benchmark_type: Any = type(self.benchmark)
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
            # TODO: hardcodes "objective" key and min(); needs parameterization
            # when ExperimentLog.best_result bug is fixed (see CLAUDE.md known issues)
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
        benchmark_name: str,
        known_optimum: float | None,
    ) -> BenchmarkResult:
        """Run uniform random sampling from the search space."""
        # Use SeedSequence.spawn() for proper RNG stream splitting:
        # stream 0 = benchmark structural noise, stream 1 = parameter sampling.
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(2)
        noise_rng = np.random.default_rng(child_seeds[0])
        sample_rng = np.random.default_rng(child_seeds[1])
        # Cast to Any because Protocol cannot enforce constructor signatures
        benchmark_type: Any = type(self.benchmark)
        bench = benchmark_type(
            noise_scale=self.benchmark.noise_scale,
            rng=noise_rng,
        )
        space = benchmark_type.search_space()

        convergence_curve: list[float] = []
        best_so_far = float("inf")

        for _ in range(budget):
            params = sample_random_params(space, sample_rng)
            metrics = bench.run(params)
            # TODO: hardcodes "objective" key and min(); see engine TODO above
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

        # Use |optimum| as the scale factor; fall back to 1.0 when optimum is zero
        # so that threshold_pct is interpreted as an absolute tolerance.
        abs_optimum = abs(known_optimum) if known_optimum != 0.0 else 1.0
        threshold = self.threshold_pct * abs_optimum

        for i, val in enumerate(convergence_curve):
            # One-sided check: value at-or-better than (optimum + threshold) counts
            # as converged. This correctly handles noisy overshoot where the
            # objective goes below the optimum (for minimization).
            if val <= known_optimum + threshold:
                return i + 1  # 1-indexed step count
        return None


def sample_random_params(space: SearchSpace, rng: np.random.Generator) -> dict[str, Any]:
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
            if not choices:
                msg = f"Variable {var.name!r} is CATEGORICAL but has no choices defined."
                raise ValueError(msg)
            params[var.name] = choices[int(rng.integers(0, len(choices)))]
        else:
            msg = f"Unsupported variable type {var.variable_type!r} for variable {var.name!r}."
            raise ValueError(msg)
    return params
