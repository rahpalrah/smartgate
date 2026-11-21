import pytest

from smartgate import Gateway, MockProvider, ModelSpec, RetryPolicy, ResilienceConfig

MODELS = [
    ModelSpec("small-a", "mock", "small", 0.1, 0.5),
    ModelSpec("medium-a", "mock", "medium", 2.0, 10.0),
    ModelSpec("large-a", "mock", "large", 4.0, 20.0),
]


@pytest.fixture
def provider():
    return MockProvider()


@pytest.fixture
def make_gateway(provider):
    def _make(**kwargs):
        kwargs.setdefault("resilience", ResilienceConfig(retry=RetryPolicy(max_attempts=2, base_delay=0)))
        return Gateway(models=kwargs.pop("models", MODELS), providers={"mock": kwargs.pop("provider", provider)},
                       sleep=lambda s: None, **kwargs)
    return _make
