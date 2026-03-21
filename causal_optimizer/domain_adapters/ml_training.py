"""ML training optimization adapter.

Simulates realistic ML model training dynamics with causal structure:
hyperparameters and architecture choices affect intermediate quantities
(gradient scale, model capacity, regularization) which determine val_loss.

Structural equations follow the prior causal graph:
  learning_rate -> gradient_scale -> training_stability -> val_loss
  batch_size -> gradient_noise, throughput
  n_layers, n_heads, hidden_dim -> model_capacity -> val_loss
  dropout, weight_decay -> regularization -> val_loss
  optimizer -> gradient_scale
  activation -> model_capacity
  Bidirected: U_hardware <-> (throughput, memory_usage)
  Bidirected: U_data_distribution <-> (model_capacity, val_loss)

Realistic failure modes:
  - High LR (>0.05): gradient_scale explodes -> training_stability collapses -> divergence
  - Large model + no regularization: overfitting penalty increases val_loss
  - Tiny model: insufficient capacity -> underfitting (high irreducible loss)

Categorical mappings:
  optimizer: adamw=1.0, sgd=0.8, muon=1.1, lion=0.95
  activation: gelu=1.0, swiglu=1.1, relu=0.9

Approximate known optimum:
  learning_rate ~= 3e-4, batch_size ~= 128, n_layers ~= 12, n_heads ~= 8,
  hidden_dim ~= 1024, dropout ~= 0.15, weight_decay ~= 0.05,
  optimizer = "muon", activation = "swiglu"
  -> val_loss ~= 0.15
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class MLTrainingAdapter(DomainAdapter):
    """Adapter for ML model training optimization.

    Optimizes hyperparameters, architecture, and training configuration
    to minimize validation loss within a fixed compute budget.

    Args:
        seed: Random seed for reproducibility.
        noise_scale: Standard deviation of Gaussian noise added to each
            intermediate variable. Default 0.1.
    """

    _OPTIMIZER_FACTOR: dict[str, float] = {
        "adamw": 1.0,
        "sgd": 0.8,
        "muon": 1.1,
        "lion": 0.95,
    }
    _ACTIVATION_FACTOR: dict[str, float] = {
        "gelu": 1.0,
        "swiglu": 1.1,
        "relu": 0.9,
    }

    def __init__(self, seed: int | None = None, noise_scale: float = 0.1) -> None:
        self._seed = seed
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """ExperimentRunner protocol: delegates to run_experiment."""
        return self.run_experiment(parameters)

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(
                    name="learning_rate",
                    variable_type=VariableType.CONTINUOUS,
                    lower=1e-5,
                    upper=1e-1,
                ),
                Variable(
                    name="batch_size",
                    variable_type=VariableType.INTEGER,
                    lower=8,
                    upper=512,
                ),
                Variable(
                    name="n_layers",
                    variable_type=VariableType.INTEGER,
                    lower=2,
                    upper=24,
                ),
                Variable(
                    name="n_heads",
                    variable_type=VariableType.INTEGER,
                    lower=1,
                    upper=16,
                ),
                Variable(
                    name="hidden_dim",
                    variable_type=VariableType.INTEGER,
                    lower=128,
                    upper=2048,
                ),
                Variable(
                    name="dropout",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=0.5,
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
        """Run a simulated ML training experiment with realistic dynamics.

        Each intermediate variable is computed from its causal parents plus
        Gaussian noise. Two latent confounders (U_hardware and
        U_data_distribution) inject correlated variation.

        Failure modes:
        - High learning_rate (>0.05): gradient explosion -> training instability
        - Large model + no regularization: overfitting penalty
        - Tiny model: underfitting (high irreducible error)
        """
        sigma = self._noise_scale

        # Extract parameters
        lr = float(parameters.get("learning_rate", 1e-3))
        batch_size = int(parameters.get("batch_size", 64))
        n_layers = int(parameters.get("n_layers", 6))
        n_heads = int(parameters.get("n_heads", 8))
        hidden_dim = int(parameters.get("hidden_dim", 512))
        dropout = float(parameters.get("dropout", 0.1))
        weight_decay = float(parameters.get("weight_decay", 0.01))
        optimizer_name = parameters.get("optimizer", "adamw")
        activation = parameters.get("activation", "gelu")

        # --- Latent confounders ---
        u_hardware = self._rng.normal(0, 1)
        u_data_dist = self._rng.normal(0, 1)

        # --- Categorical encodings ---
        opt_factor = self._OPTIMIZER_FACTOR.get(optimizer_name, 1.0)
        act_factor = self._ACTIVATION_FACTOR.get(activation, 1.0)

        # --- Structural equations ---

        # learning_rate + optimizer -> gradient_scale
        # Optimal gradient_scale is near 1.0; too high = explosion
        gradient_scale = lr * 1000 * opt_factor + self._rng.normal(0, sigma * 0.1)

        # gradient_scale -> training_stability
        # Stability is highest when gradient_scale is near 1.0
        # Very high gradient_scale (>10) causes near-total instability
        stability_penalty = (gradient_scale - 1.0) ** 2
        training_stability = 1.0 / (1.0 + stability_penalty * 0.5) + self._rng.normal(
            0, sigma * 0.05
        )
        training_stability = max(0.01, min(1.0, training_stability))

        # batch_size -> gradient_noise
        gradient_noise = 1.0 / np.sqrt(max(batch_size, 1))
        # Gradient noise reduces stability moderately
        training_stability *= max(0.3, 1.0 - gradient_noise * 0.3)

        # activation -> gradient_flow -> training_stability
        gradient_flow = act_factor
        training_stability *= gradient_flow
        # Final clamp after all multiplicative adjustments
        training_stability = max(0.01, min(1.0, training_stability))

        # batch_size -> throughput (+ hardware confounder)
        throughput = batch_size * 0.8 + u_hardware * 10 + self._rng.normal(0, sigma)
        throughput = max(1.0, throughput)

        # n_layers, n_heads, hidden_dim -> model_capacity
        # Log scale so capacity grows sublinearly with size
        raw_capacity = n_layers * n_heads * hidden_dim
        model_capacity = (
            np.log(raw_capacity + 1)
            + u_data_dist * 0.2  # data distribution affects effective capacity
            + self._rng.normal(0, sigma * 0.1)
        )
        model_capacity = max(0.1, model_capacity)

        # hidden_dim, n_layers, batch_size -> memory_usage (+ hardware confounder)
        memory_usage = (
            hidden_dim * n_layers * 4 / 1e6  # parameters in MB
            + batch_size * hidden_dim * n_layers / 1e7  # activations
            + u_hardware * 0.5  # hardware overhead variance
            + self._rng.normal(0, sigma * 0.5)
        )
        memory_usage = max(0.01, memory_usage)

        # dropout, weight_decay -> regularization
        regularization = dropout * 0.5 + weight_decay * 5.0 + self._rng.normal(0, sigma * 0.05)
        regularization = max(0.0, regularization)

        # throughput -> tokens_seen
        tokens_seen = throughput * 100

        # --- Val loss computation ---
        # Base loss: starts high, reduced by capacity, stability, regularization, tokens
        base_loss = 2.0

        # Model capacity reduces loss (more capacity = lower loss, up to a point)
        capacity_benefit = min(model_capacity * 0.12, 1.2)

        # Training stability reduces loss
        stability_benefit = training_stability * 0.4

        # Optimizer convergence speed
        convergence_benefit = opt_factor * 0.1

        # Tokens seen benefit (log scale, diminishing returns)
        tokens_benefit = np.log(tokens_seen + 1) * 0.04

        # Regularization benefit (moderate reg is good, too much hurts)
        # Optimal around 0.3-0.5
        reg_benefit = min(regularization, 0.5) * 0.3 - max(0.0, regularization - 0.8) * 0.5

        # --- Failure mode penalties ---

        # High LR divergence: exponential penalty when gradient_scale > 5
        divergence_penalty = 0.0
        if gradient_scale > 5.0:
            divergence_penalty = (gradient_scale - 5.0) ** 1.5 * 0.1

        # Overfitting: large model + low regularization
        # Ratio of capacity to regularization — high ratio = overfit risk
        overfit_risk = model_capacity / (regularization + 0.1)
        overfit_penalty = 0.0
        if overfit_risk > 30.0:
            overfit_penalty = (overfit_risk - 30.0) * 0.005

        # Underfitting: tiny model has irreducible error
        underfit_penalty = 0.0
        if model_capacity < 5.0:
            underfit_penalty = (5.0 - model_capacity) * 0.08

        # Data distribution confounder directly affects val_loss
        data_dist_effect = u_data_dist * 0.05

        val_loss = (
            base_loss
            - capacity_benefit
            - stability_benefit
            - convergence_benefit
            - tokens_benefit
            - reg_benefit
            + divergence_penalty
            + overfit_penalty
            + underfit_penalty
            + data_dist_effect
            + self._rng.normal(0, sigma)
        )
        val_loss = max(0.01, val_loss)

        return {
            "val_loss": float(val_loss),
            "memory_usage": float(memory_usage),
            "model_capacity": float(model_capacity),
        }

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

    def get_objective_name(self) -> str:
        return "val_loss"

    def get_minimize(self) -> bool:
        return True  # minimize validation loss
