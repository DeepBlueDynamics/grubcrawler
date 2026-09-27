"""Base provider interface and factory for LLM adapters.

Every provider adapter must implement `LLMAdapter`. The factory
function `create_provider()` reads config and returns the right one,
with optional fallback rotation on transient failures.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.agent.types import AssistantAction

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Abstract interface that every provider adapter must satisfy."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AssistantAction:
        """Send conversation + tool schemas, get back an AssistantAction.

        Args:
            messages: OpenAI-style message list (role, content, tool_calls, etc.)
            tools: List of tool schemas in OpenAI function-calling format.

        Returns:
            Respond(text) or ToolCalls([ToolCall, ...])
        """
        ...

    async def vision(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        detail: str = "low",
    ) -> str:
        """Extract text from an image using the provider's vision model.

        Default implementation raises NotImplementedError — providers that
        support vision (OpenAI gpt-4o, Anthropic claude-sonnet-4-5-20250929) override this.

        Args:
            image_bytes: JPEG/PNG image data.
            prompt: Extraction instruction.
            detail: "low" or "high" — controls token cost.

        Returns:
            Extracted text string.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support vision")


class FallbackAdapter(LLMAdapter):
    """Wraps multiple adapters; retries transient failures by rotating.

    On the first transient error, retries once with the same provider.
    On the second failure, rotates to the next provider in the list.
    """

    def __init__(self, adapters: List[LLMAdapter]):
        if not adapters:
            raise ValueError("FallbackAdapter requires at least one adapter")
        self.adapters = adapters
        self._current = 0

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AssistantAction:
        last_exc: Optional[Exception] = None

        for attempt in range(len(self.adapters) * 2):
            adapter = self.adapters[self._current]
            try:
                return await adapter.complete(messages, tools)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Provider %s failed (attempt %d): %s",
                    adapter.__class__.__name__,
                    attempt + 1,
                    exc,
                )
                # Rotate to next provider
                self._current = (self._current + 1) % len(self.adapters)

        raise last_exc  # type: ignore[misc]

    async def vision(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        detail: str = "low",
    ) -> str:
        last_exc: Optional[Exception] = None

        for attempt in range(len(self.adapters)):
            adapter = self.adapters[self._current]
            try:
                return await adapter.vision(image_bytes, prompt, detail=detail)
            except NotImplementedError:
                # This provider doesn't support vision — rotate
                self._current = (self._current + 1) % len(self.adapters)
                continue
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Vision provider %s failed (attempt %d): %s",
                    adapter.__class__.__name__,
                    attempt + 1,
                    exc,
                )
                self._current = (self._current + 1) % len(self.adapters)

        if last_exc:
            raise last_exc
        raise NotImplementedError("No adapter in the fallback chain supports vision")


def create_provider(
    provider_name: str = "openai",
    **kwargs,
) -> LLMAdapter:
    """Factory: create a provider adapter by name.

    Args:
        provider_name: "openai", "anthropic", or "ollama"
        **kwargs: Forwarded to the adapter constructor (api_key, model, etc.)

    Returns:
        An LLMAdapter instance.
    """
    name = provider_name.lower().strip()

    if name == "openai":
        from app.agent.providers.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(**kwargs)
    elif name == "anthropic":
        from app.agent.providers.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(**kwargs)
    elif name == "ollama":
        from app.agent.providers.ollama_adapter import OllamaAdapter
        return OllamaAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name!r} (expected openai, anthropic, or ollama)")


def create_provider_from_config() -> LLMAdapter:
    """Build an LLMAdapter (or FallbackAdapter) from app settings."""
    from app.config import settings

    primary = create_provider(
        settings.agent_provider,
        api_key=_pick_key(settings, settings.agent_provider),
        model=_pick_model(settings, settings.agent_provider),
        base_url=_pick_base_url(settings, settings.agent_provider),
    )
    return primary


def _pick_key(settings, provider: str) -> Optional[str]:
    if provider == "openai":
        return settings.openai_api_key
    if provider == "anthropic":
        return settings.anthropic_api_key
    if provider == "ollama":
        return settings.ollama_api_key
    return None


def _pick_model(settings, provider: str) -> str:
    if provider == "openai":
        return settings.openai_model
    if provider == "anthropic":
        return settings.anthropic_model
    if provider == "ollama":
        return settings.ollama_model
    return ""


def _pick_base_url(settings, provider: str) -> Optional[str]:
    if provider == "ollama":
        return settings.ollama_base_url
    return None
