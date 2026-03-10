"""ML training optimization adapter.

Similar to autoresearch but with causal structure — optimizes model
architecture, hyperparameters, and training configuration.
"""

from __future__ import annotations

from typing import Any

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class MLTrainingAdapter(DomainAdapter):
    """Adapter for ML model training optimization.

    Optimizes hyperparameters, architecture, and training configuration
    to minimize validation loss within a fixed compute budget.
    """

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(
                    name="learning_rate",
                    variable_type=VariableType.CONTINUOUS,
                    lower=1e-5,
                    upper=1e-1,
                ),
                Variable(name="batch_size", variable_type=VariableType.INTEGER, lower=8, upper=512),
                Variable(name="n_layers", variable_type=VariableType.INTEGER, lower=2, upper=24),
                Variable(name="n_heads", variable_type=VariableType.INTEGER, lower=1, upper=16),
                Variable(
                    name="hidden_dim", variable_type=VariableType.INTEGER, lower=128, upper=2048
                ),
                Variable(
                    name="dropout", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=0.5
                ),
                Variable(
                    name="weight_decay",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=0.5,
                ),
                Variable(
                    name="optimizer",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["adamw", "muon", "sgd", "lion"],
                ),
                Variable(
                    name="activation",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["gelu", "swiglu", "relu"],
                ),
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run an ML training experiment."""
        raise NotImplementedError(
            "MLTrainingAdapter.run_experiment requires a training script. "
            "Override this method to modify and run your training code."
        )

    def get_prior_graph(self) -> CausalGraph:
        """ML training causal graph based on known relationships."""
        return CausalGraph(
            edges=[
                ("learning_rate", "gradient_scale"),
                ("gradient_scale", "training_stability"),
                ("training_stability", "val_loss"),
                ("batch_size", "gradient_noise"),
                ("gradient_noise", "training_stability"),
                ("batch_size", "throughput"),
                ("batch_size", "memory_usage"),
                ("n_layers", "model_capacity"),
                ("n_heads", "model_capacity"),
                ("hidden_dim", "model_capacity"),
                ("hidden_dim", "memory_usage"),
                ("model_capacity", "val_loss"),
                ("memory_usage", "max_batch_size"),
                ("dropout", "regularization"),
                ("weight_decay", "regularization"),
                ("regularization", "val_loss"),
                ("optimizer", "gradient_scale"),
                ("optimizer", "convergence_speed"),
                ("convergence_speed", "val_loss"),
                ("activation", "gradient_flow"),
                ("gradient_flow", "training_stability"),
                ("throughput", "tokens_seen"),
                ("tokens_seen", "val_loss"),
            ],
            bidirected_edges=[
                # Hardware/platform confounds both throughput and memory usage
                ("throughput", "memory_usage"),
                # Data distribution confounds both model capacity needs and val_loss
                ("model_capacity", "val_loss"),
            ],
        )

    def get_descriptor_names(self) -> list[str]:
        return ["memory_usage", "model_capacity"]
