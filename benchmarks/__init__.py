"""Benchmark structural causal models for testing the optimization engine."""

from __future__ import annotations

from benchmarks.complete_graph import CompleteGraphBenchmark
from benchmarks.interaction import InteractionBenchmark
from benchmarks.toy_graph import ToyGraphBenchmark

__all__ = ["CompleteGraphBenchmark", "InteractionBenchmark", "ToyGraphBenchmark"]
