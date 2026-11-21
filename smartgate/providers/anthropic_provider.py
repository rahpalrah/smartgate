"""Claude provider using the official `anthropic` Python SDK (pip install anthropic)."""
from __future__ import annotations

from typing import Any, Optional

from ..errors import ProviderError
from ..types import GatewayRequest, ModelSpec, ProviderResponse, Usage
from .base import Provider


class AnthropicProvider(Provider):
    """Calls Claude via `client.messages.create`.

    Credentials resolve the standard way (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    or an `ant auth login` profile). Pass `client=` to inject a pre-configured
    client (e.g. `AnthropicBedrockMantle`, `AnthropicVertex`, `AnthropicFoundry`).

    SDK retries are disabled by default (max_retries=0) so the gateway's own
    retry + fallback policy is the single source of truth.

    Per-model extra params come from `ModelSpec.params`, e.g.
    `{"output_config": {"effort": "low"}}` for a fast small tier.
    """

    name = "anthropic"

    def __init__(self, client: Any = None, timeout: float = 120.0, max_retries: int = 0) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError("AnthropicProvider requires `pip install anthropic`") from exc
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.client = client

    def complete(self, model: ModelSpec, request: GatewayRequest) -> ProviderResponse:
        import anthropic

        kwargs = dict(
            model=model.name,
            max_tokens=min(request.max_tokens, model.max_output_tokens),
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            **model.params,
        )
        if request.system:
            kwargs["system"] = request.system

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            raise ProviderError(f"bad request: {e.message}", fallback=False, kind="bad_request", status_code=400)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            # Credentials for this provider are broken; another provider may work.
            raise ProviderError(f"auth: {e.message}", kind="auth", status_code=e.status_code)
        except anthropic.NotFoundError as e:
            raise ProviderError(f"model not found: {e.message}", kind="not_found", status_code=404)
        except anthropic.RateLimitError as e:
            raise ProviderError("rate limited", retryable=True, kind="rate_limit", status_code=429,
                                retry_after=_retry_after(e))
        except anthropic.APITimeoutError:
            raise ProviderError("timeout", retryable=True, kind="timeout")
        except anthropic.APIStatusError as e:
            retryable = e.status_code >= 500 or e.status_code in (408, 409)
            raise ProviderError(f"api error {e.status_code}: {e.message}", retryable=retryable,
                                kind="server" if retryable else "error", status_code=e.status_code)
        except anthropic.APIConnectionError:
            raise ProviderError("connection error", retryable=True, kind="connection")

        if response.stop_reason == "refusal":
            # Not retryable on the same model; a different model may answer.
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise ProviderError(f"refusal ({category})", kind="refusal", fallback=True)

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        return ProviderResponse(
            text=text,
            model=response.model,
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            stop_reason=response.stop_reason,
            raw=response,
        )


def _retry_after(err: Any) -> Optional[float]:
    try:
        return float(err.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
