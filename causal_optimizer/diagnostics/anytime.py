"""Anytime metrics — learning curve checkpoints during optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnytimeMetrics:
    """Learning curve at checkpoints during optimization."""

    checkpoints: list[int]  # budget values where metrics were sampled
    best_objective_at: list[float]  # best objective at each checkpoint
    n_evaluated_at: list[int]  # experiments evaluated at each checkpoint
    n_skipped_at: list[int]  # experiments skipped at each checkpoint
