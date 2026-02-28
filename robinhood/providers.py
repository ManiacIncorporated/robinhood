"""
robinhood.providers — Multi-provider LLM client

Unified interface for calling Anthropic, OpenAI, and OpenRouter APIs.
Handles API key resolution, client construction, and response normalization
so the rest of robinhood doesn't need to care which backend is in use.

Provider detection priority:
    1. Explicit ``provider`` argument
    2. Inferred from model name (e.g. "gpt-4o" -> openai, "claude-*" -> anthropic)
    3. Inferred from which API key env var is set

API key resolution (per provider):
    anthropic:   --api-key flag > ANTHROPIC_API_KEY env var
    openai:      --api-key flag > OPENAI_API_KEY env var
    openrouter:  --api-key flag > OPENROUTER_API_KEY env var
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List


Provider = Literal["anthropic", "openai", "openrouter"]

# Env var names per provider
_ENV_KEYS: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def detect_provider(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Provider:
    """
    Determine which provider to use.

    Resolution order:
        1. Explicit ``provider`` if given
        2. Model name heuristic
        3. Whichever API key env var is set
    """
    if provider:
        p = provider.lower().strip()
        if p in ("anthropic", "claude"):
            return "anthropic"
        if p in ("openai", "gpt"):
            return "openai"
        if p in ("openrouter", "or"):
            return "openrouter"
        raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic', 'openai', or 'openrouter'.")

    model_lower = model.lower()
    # OpenRouter models use org/model format (e.g. "anthropic/claude-sonnet-4")
    if "/" in model_lower:
        return "openrouter"
    if any(t in model_lower for t in ("claude",)):
        return "anthropic"
    if any(t in model_lower for t in ("gpt-", "o1-", "o3-", "o4-")):
        return "openai"

    for prov, env_key in _ENV_KEYS.items():
        if os.getenv(env_key):
            return prov

    return "anthropic"


def resolve_api_key(
    provider: Provider,
    api_key: Optional[str] = None,
) -> str:
    """Return the API key for ``provider``, raising if none is found."""
    if api_key:
        return api_key

    env_key = _ENV_KEYS[provider]
    val = os.getenv(env_key)
    if val:
        return val

    raise ValueError(
        f"No API key found for provider '{provider}'. "
        f"Set the {env_key} environment variable or pass --api-key."
    )


# ---------------------------------------------------------------------------
# Lightweight response wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Normalized response from any provider."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking: str = ""
    raw: Any = None


# ---------------------------------------------------------------------------
# Provider-specific client factories
# ---------------------------------------------------------------------------

def _make_anthropic_client(api_key: str, async_: bool = False):
    import anthropic
    cls = anthropic.AsyncAnthropic if async_ else anthropic.Anthropic
    return cls(api_key=api_key)


def _make_openai_client(api_key: str, base_url: Optional[str] = None):
    from openai import OpenAI
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _make_openai_async_client(api_key: str, base_url: Optional[str] = None):
    from openai import AsyncOpenAI
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Unified client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client that wraps Anthropic, OpenAI, and OpenRouter.

    Usage::

        client = LLMClient(provider="anthropic")
        resp = client.complete("Hello", model="claude-sonnet-4-20250514")

        client = LLMClient(provider="openai", api_key="sk-...")
        resp = client.complete("Hello", model="gpt-4o")

        client = LLMClient(provider="openrouter")
        resp = client.complete("Hello", model="anthropic/claude-sonnet-4")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.provider: Provider = detect_provider(model, provider, api_key)
        self.api_key = resolve_api_key(self.provider, api_key)
        self.default_model = model

        self._sync_client = None
        self._async_client = None

    def _get_sync(self):
        if self._sync_client is None:
            if self.provider == "anthropic":
                self._sync_client = _make_anthropic_client(self.api_key)
            elif self.provider == "openai":
                self._sync_client = _make_openai_client(self.api_key)
            elif self.provider == "openrouter":
                self._sync_client = _make_openai_client(
                    self.api_key, base_url=_OPENROUTER_BASE_URL,
                )
        return self._sync_client

    def _get_async(self):
        if self._async_client is None:
            if self.provider == "anthropic":
                self._async_client = _make_anthropic_client(self.api_key, async_=True)
            elif self.provider == "openai":
                self._async_client = _make_openai_async_client(self.api_key)
            elif self.provider == "openrouter":
                self._async_client = _make_openai_async_client(
                    self.api_key, base_url=_OPENROUTER_BASE_URL,
                )
        return self._async_client

    # ---- Anthropic helpers ------------------------------------------------

    def _anthropic_complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        client = self._get_sync()
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if thinking:
            kwargs["thinking"] = thinking
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        resp = client.messages.create(**kwargs)

        text_parts = []
        think_parts = []
        for block in resp.content:
            if block.type == "thinking":
                think_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)

        usage = resp.usage
        return LLMResponse(
            content="".join(text_parts),
            model=resp.model,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            thinking="".join(think_parts),
            raw=resp,
        )

    async def _anthropic_complete_async(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        client = self._get_async()
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if thinking:
            kwargs["thinking"] = thinking
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        resp = await client.messages.create(**kwargs)

        text_parts = []
        think_parts = []
        for block in resp.content:
            if block.type == "thinking":
                think_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)

        usage = resp.usage
        return LLMResponse(
            content="".join(text_parts),
            model=resp.model,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            thinking="".join(think_parts),
            raw=resp,
        )

    # ---- OpenAI / OpenRouter helpers --------------------------------------

    def _openai_complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_sync()
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        resp = client.chat.completions.create(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = resp.choices[0]
        content = choice.message.content or ""

        # Some OpenRouter models return reasoning in <think> tags
        thinking = ""
        if "<think>" in content and "</think>" in content:
            start = content.index("<think>") + len("<think>")
            end = content.index("</think>")
            thinking = content[start:end].strip()
            content = content[end + len("</think>"):].strip()

        usage = resp.usage
        return LLMResponse(
            content=content,
            model=resp.model or model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            thinking=thinking,
            raw=resp,
        )

    async def _openai_complete_async(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_async()
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        resp = await client.chat.completions.create(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = resp.choices[0]
        content = choice.message.content or ""

        thinking = ""
        if "<think>" in content and "</think>" in content:
            start = content.index("<think>") + len("<think>")
            end = content.index("</think>")
            thinking = content[start:end].strip()
            content = content[end + len("</think>"):].strip()

        usage = resp.usage
        return LLMResponse(
            content=content,
            model=resp.model or model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            thinking=thinking,
            raw=resp,
        )

    # ---- Public API -------------------------------------------------------

    def complete(
        self,
        user_message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Send a single user message and return the response.

        Args:
            user_message: The user prompt.
            model: Override the default model.
            system: System prompt.
            max_tokens: Max output tokens.
            temperature: Sampling temperature (forced to 1.0 for Anthropic thinking).
            thinking: Anthropic thinking config dict, e.g.
                      ``{"type": "enabled", "budget_tokens": 10000}``.
                      Ignored for OpenAI/OpenRouter providers.
        """
        model = model or self.default_model
        messages = [{"role": "user", "content": user_message}]

        if self.provider == "anthropic":
            return self._anthropic_complete(
                messages, model, system, max_tokens, temperature, thinking,
            )
        else:
            return self._openai_complete(
                messages, model, system, max_tokens, temperature,
            )

    async def complete_async(
        self,
        user_message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Async version of :meth:`complete`."""
        model = model or self.default_model
        messages = [{"role": "user", "content": user_message}]

        if self.provider == "anthropic":
            return await self._anthropic_complete_async(
                messages, model, system, max_tokens, temperature, thinking,
            )
        else:
            return await self._openai_complete_async(
                messages, model, system, max_tokens, temperature,
            )

    @property
    def supports_thinking(self) -> bool:
        """Whether this provider supports native extended thinking blocks."""
        return self.provider == "anthropic"

    def __repr__(self) -> str:
        key_preview = self.api_key[:8] + "..." if len(self.api_key) > 8 else "***"
        return (
            f"LLMClient(provider={self.provider!r}, "
            f"model={self.default_model!r}, "
            f"key={key_preview})"
        )
