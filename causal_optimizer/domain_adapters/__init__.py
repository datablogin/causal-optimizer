"""Domain adapters — plug in different experiment domains."""

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.domain_adapters.marketing import MarketingAdapter
from causal_optimizer.domain_adapters.ml_training import MLTrainingAdapter

__all__ = ["DomainAdapter", "MarketingAdapter", "MLTrainingAdapter"]
