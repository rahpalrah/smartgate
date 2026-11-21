"""smartgate — semantic cache & model routing gateway SDK."""
from .analysis import PromptAnalysis, PromptAnalyzer, detect_pii, estimate_tokens, redact_pii
from .cache import CacheConfig, InMemoryVectorStore, SemanticCache, SQLiteVectorStore
from .config import (DEFAULT_MODELS, PRESETS, gateway_from_config, gateway_from_env,
                     gateway_from_file, gateway_from_preset)
from .cost import Budget, CostController
from .embeddings import CallableEmbedder, HashingEmbedder
from .errors import (AllModelsFailedError, BudgetExceededError, ConfigError, GatewayError,
                     PolicyViolationError, ProviderError)
from .gateway import Gateway, PolicyConfig, ResilienceConfig
from .providers import CallableProvider, MockProvider, Provider
from .resilience import CircuitBreaker, RetryPolicy
from .routing import ModelRouter, RoutingConfig, RoutingRule
from .types import GatewayRequest, GatewayResponse, Message, ModelSpec, ProviderResponse, Usage

__version__ = "0.1.0"

__all__ = [n for n in dir() if not n.startswith("_")]
