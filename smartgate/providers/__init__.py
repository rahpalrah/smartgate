from .base import CallableProvider, Provider
from .mock import MockProvider


def __getattr__(name):  # lazy: only import the anthropic SDK when used
    if name == "AnthropicProvider":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider
    raise AttributeError(name)


__all__ = ["Provider", "CallableProvider", "MockProvider", "AnthropicProvider"]
