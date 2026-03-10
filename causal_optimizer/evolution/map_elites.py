"""MAP-Elites algorithm for maintaining diverse solution populations.

Instead of converging to a single optimum (which may be a local minimum),
MAP-Elites maintains a grid of solutions indexed by behavioral descriptors.
This ensures diversity and enables discovering solutions that combine
different strategies — inspired by AlphaEvolve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_optimizer.types import ExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class EliteCell:
    """A single cell in the MAP-Elites archive."""

    descriptor: tuple[int, ...]
    result: ExperimentResult
    fitness: float


class MAPElites:
    """MAP-Elites archive for maintaining diverse high-quality solutions.

    Solutions are indexed by behavioral descriptors — discrete bins that
    characterize *how* a solution achieves its result (not just how good it is).

    Example descriptors for ML optimization:
    - (model_size_bin, memory_usage_bin): diverse solutions across size/memory tradeoffs
    - (architecture_type, optimizer_type): diverse solutions across design choices
    """

    def __init__(
        self,
        descriptor_names: list[str],
        n_bins: int = 10,
        descriptor_ranges: dict[str, tuple[float, float]] | None = None,
        minimize: bool = True,
    ) -> None:
        self.descriptor_names = descriptor_names
        self.n_bins = n_bins
        self.descriptor_ranges = descriptor_ranges or {}
        self.minimize = minimize
        self.archive: dict[tuple[int, ...], EliteCell] = {}
        self._observed_ranges: dict[str, tuple[float, float]] = {}

    def add(
        self,
        result: ExperimentResult,
        fitness: float,
        descriptors: dict[str, float],
    ) -> bool:
        """Add a solution to the archive. Returns True if it was accepted."""
        # Update observed ranges
        for name, value in descriptors.items():
            if name in self._observed_ranges:
                lo, hi = self._observed_ranges[name]
                self._observed_ranges[name] = (min(lo, value), max(hi, value))
            else:
                self._observed_ranges[name] = (value, value)

        # Compute bin indices
        bin_indices = self._compute_bins(descriptors)

        # Check if this cell is empty or if the new solution is better
        existing = self.archive.get(bin_indices)
        if existing is None:
            self.archive[bin_indices] = EliteCell(
                descriptor=bin_indices, result=result, fitness=fitness
            )
            logger.debug(f"New elite in cell {bin_indices}: fitness={fitness:.6f}")
            return True

        is_better = (fitness < existing.fitness) if self.minimize else (fitness > existing.fitness)
        if is_better:
            self.archive[bin_indices] = EliteCell(
                descriptor=bin_indices, result=result, fitness=fitness
            )
            logger.debug(
                f"Improved elite in cell {bin_indices}: {existing.fitness:.6f} -> {fitness:.6f}"
            )
            return True

        return False

    def sample_elite(self) -> ExperimentResult | None:
        """Sample a random elite from the archive for mutation."""
        if not self.archive:
            return None
        rng = np.random.default_rng()
        cells = list(self.archive.values())
        idx = rng.integers(0, len(cells))
        return cells[idx].result

    def sample_diverse(self, n: int = 5) -> list[ExperimentResult]:
        """Sample diverse elites from different regions of the archive."""
        if not self.archive:
            return []
        cells = list(self.archive.values())
        n = min(n, len(cells))
        rng = np.random.default_rng()
        indices = rng.choice(len(cells), size=n, replace=False)
        return [cells[i].result for i in indices]

    @property
    def best(self) -> EliteCell | None:
        """Return the best elite across all cells."""
        if not self.archive:
            return None
        if self.minimize:
            return min(self.archive.values(), key=lambda c: c.fitness)
        return max(self.archive.values(), key=lambda c: c.fitness)

    @property
    def coverage(self) -> float:
        """Fraction of the archive that is filled."""
        total_cells = self.n_bins ** len(self.descriptor_names)
        return len(self.archive) / total_cells if total_cells > 0 else 0.0

    def _compute_bins(self, descriptors: dict[str, float]) -> tuple[int, ...]:
        """Map continuous descriptors to discrete bin indices."""
        bins = []
        for name in self.descriptor_names:
            value = descriptors.get(name, 0.0)
            # Use configured or observed range
            if name in self.descriptor_ranges:
                lo, hi = self.descriptor_ranges[name]
            elif name in self._observed_ranges:
                lo, hi = self._observed_ranges[name]
            else:
                lo, hi = 0.0, 1.0

            if hi <= lo:
                bins.append(0)
            else:
                normalized = (value - lo) / (hi - lo)
                bin_idx = int(np.clip(normalized * self.n_bins, 0, self.n_bins - 1))
                bins.append(bin_idx)

        return tuple(bins)
