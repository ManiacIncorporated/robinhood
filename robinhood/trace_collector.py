"""
robinhood.trace_collector — Reasoning Trace Collector

Queries LLMs with extended thinking (Anthropic) or chain-of-thought
prompting (OpenAI / OpenRouter) and captures reasoning traces alongside
final outputs.

Supports all providers via ``robinhood.providers.LLMClient``:
    - Anthropic: native extended thinking blocks
    - OpenAI: ``<think>``-tag reasoning via system prompt encouragement
    - OpenRouter: same as OpenAI, with access to many models

Users are responsible for ensuring their use of model outputs complies
with the provider's terms of service.
"""

import asyncio
import json
import time
import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from robinhood.providers import LLMClient, LLMResponse, detect_provider, resolve_api_key


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking / chain-of-thought."""
    enabled: bool = True
    budget_tokens: int = 10000
    effort: Optional[str] = None  # "low", "medium", "high", "max" (Anthropic 4.6+)


@dataclass
class CollectionConfig:
    """Configuration for the trace collection process."""
    model: str = "claude-sonnet-4-20250514"
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    max_output_tokens: int = 4096
    temperature: float = 1.0
    batch_size: int = 5
    max_concurrent: int = 10
    rate_limit_rpm: int = 50
    retry_max_attempts: int = 5
    retry_base_delay: float = 2.0
    save_interval: int = 50

    # Multi-answer rejection sampling.  When > 1, N independent completions
    # are collected for each prompt so the verification stage can pick the
    # best one.
    samples_per_prompt: int = 1

    # Compliance mode — when True, retries will NOT be attempted on
    # rate-limit (429) responses.  The pipeline will instead wait and
    # respect the provider's backoff signal without aggressive retries.
    compliance_mode: bool = False

    # Provider settings — if None, auto-detected from model name / env vars
    provider: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class CollectedTrace:
    """A single collected reasoning trace with metadata."""
    trace_id: str
    model: str
    prompt_category: str
    system_prompt: Optional[str]
    user_message: str
    thinking_text: str
    output_text: str
    thinking_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    timestamp: str
    provider: str = "anthropic"
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "trace_id": self.trace_id,
            "model": self.model,
            "provider": self.provider,
            "prompt_category": self.prompt_category,
            "system_prompt": self.system_prompt,
            "user_message": self.user_message,
            "thinking_text": self.thinking_text,
            "output_text": self.output_text,
            "thinking_tokens": self.thinking_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "timestamp": self.timestamp,
        }
        if self.raw_response:
            d["raw_response"] = self.raw_response
        return d


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._last_request + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


# System prompt appended for non-Anthropic providers to encourage
# chain-of-thought in <think> tags (since they lack native thinking blocks).
_COT_SYSTEM_SUFFIX = (
    "\n\nBefore answering, reason step-by-step inside <think>...</think> tags. "
    "Then give your final answer after the closing </think> tag."
)


class TraceCollector:
    """
    Collects reasoning traces from any supported LLM provider.

    For Anthropic models this uses native extended thinking blocks.
    For OpenAI / OpenRouter models it prompts the model to reason in
    ``<think>`` tags and parses them from the response.

    Usage::

        collector = TraceCollector(config=CollectionConfig(
            model="claude-sonnet-4-20250514",
        ))
        traces = collector.collect_traces_sync(prompts)

        # Or with OpenAI:
        collector = TraceCollector(config=CollectionConfig(
            model="gpt-4o",
            provider="openai",
        ))
    """

    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        self._client = LLMClient(
            provider=self.config.provider,
            api_key=self.config.api_key,
            model=self.config.model,
        )
        self._rate_limiter = RateLimiter(self.config.rate_limit_rpm)
        self._collected: List[CollectedTrace] = []

        print(f"[COLLECTOR] Provider: {self._client.provider}, Model: {self.config.model}")

    def _generate_trace_id(
        self, user_message: str, system_prompt: str = None, sample_idx: int = 0,
    ) -> str:
        content = f"{self.config.model}:{system_prompt or ''}:{user_message}:{sample_idx}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def prompt_id(user_message: str, system_prompt: str = None) -> str:
        """Stable identifier for a prompt (independent of sample index)."""
        content = f"{system_prompt or ''}:{user_message}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _build_thinking_param(self) -> Optional[Dict[str, Any]]:
        """Build the Anthropic thinking param (None for other providers)."""
        if not self._client.supports_thinking:
            return None
        tc = self.config.thinking
        if not tc.enabled:
            return None
        return {"type": "enabled", "budget_tokens": tc.budget_tokens}

    def _effective_system_prompt(self, system_prompt: Optional[str]) -> Optional[str]:
        """
        For non-Anthropic providers, append a CoT instruction to the system
        prompt so the model outputs reasoning in <think> tags.
        """
        if self._client.supports_thinking:
            return system_prompt
        base = system_prompt or ""
        if self.config.thinking.enabled:
            return (base + _COT_SYSTEM_SUFFIX).strip()
        return system_prompt

    def _response_to_trace(
        self,
        resp: LLMResponse,
        user_message: str,
        system_prompt: Optional[str],
        category: str,
        latency: float,
        sample_idx: int = 0,
    ) -> CollectedTrace:
        return CollectedTrace(
            trace_id=self._generate_trace_id(user_message, system_prompt, sample_idx),
            model=resp.model,
            prompt_category=category,
            system_prompt=system_prompt,
            user_message=user_message,
            thinking_text=resp.thinking,
            output_text=resp.content,
            thinking_tokens=0,
            output_tokens=resp.output_tokens,
            total_tokens=resp.input_tokens + resp.output_tokens,
            latency_seconds=latency,
            timestamp=datetime.utcnow().isoformat(),
            provider=self._client.provider,
        )

    async def _collect_single(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        category: str = "general",
        sample_idx: int = 0,
    ) -> Optional[CollectedTrace]:
        """Collect a single reasoning trace with retry logic."""
        for attempt in range(self.config.retry_max_attempts):
            try:
                await self._rate_limiter.acquire()

                effective_system = self._effective_system_prompt(system_prompt)
                thinking_param = self._build_thinking_param()

                start = time.monotonic()
                resp = await self._client.complete_async(
                    user_message=user_message,
                    model=self.config.model,
                    system=effective_system,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    thinking=thinking_param,
                )
                latency = time.monotonic() - start

                return self._response_to_trace(
                    resp, user_message, system_prompt, category, latency,
                    sample_idx=sample_idx,
                )

            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()

                if is_rate_limit and self.config.compliance_mode:
                    print(
                        f"[COLLECTOR] Rate-limited ({type(e).__name__}). "
                        f"Compliance mode: waiting without aggressive retry."
                    )
                    await asyncio.sleep(60.0)
                    return None

                delay = self.config.retry_base_delay * (2 ** attempt)
                print(
                    f"[COLLECTOR] Attempt {attempt + 1}/{self.config.retry_max_attempts} "
                    f"failed ({type(e).__name__}: {e}). Retrying in {delay:.1f}s..."
                )
                if attempt < self.config.retry_max_attempts - 1:
                    await asyncio.sleep(delay)
                else:
                    print(f"[COLLECTOR] All attempts exhausted for prompt: {user_message[:80]}...")
                    return None

    async def collect_multi(
        self,
        prompts: List[Dict[str, Any]],
        samples_per_prompt: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[CollectedTrace]]:
        """
        Collect N independent traces per prompt for rejection sampling.

        Returns a dict keyed by prompt_id, where each value is a list of
        up to ``samples_per_prompt`` CollectedTrace objects.  The caller
        (typically the verification stage) picks the best one.
        """
        n = samples_per_prompt or self.config.samples_per_prompt
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        results: Dict[str, List[CollectedTrace]] = {}
        total_tasks = len(prompts) * n
        completed = 0

        async def _bounded(prompt_dict: Dict[str, Any], sample_idx: int):
            nonlocal completed
            async with semaphore:
                trace = await self._collect_single(
                    user_message=prompt_dict["user_message"],
                    system_prompt=prompt_dict.get("system_prompt"),
                    category=prompt_dict.get("category", "general"),
                    sample_idx=sample_idx,
                )
                pid = self.prompt_id(
                    prompt_dict["user_message"],
                    prompt_dict.get("system_prompt"),
                )
                if trace:
                    results.setdefault(pid, []).append(trace)
                completed += 1
                if completed % 20 == 0 or completed == total_tasks:
                    print(
                        f"[COLLECTOR] Multi-sample progress: {completed}/{total_tasks} "
                        f"({len(results)} prompts with traces)"
                    )

        tasks = []
        for prompt_dict in prompts:
            for si in range(n):
                tasks.append(_bounded(prompt_dict, si))

        await asyncio.gather(*tasks)

        if save_path:
            flat = [t.to_dict() for traces in results.values() for t in traces]
            self._save_traces_raw(flat, save_path)

        total_traces = sum(len(v) for v in results.values())
        self._collected.extend(
            t for traces in results.values() for t in traces
        )
        print(
            f"[COLLECTOR] Multi-sample collection complete: "
            f"{total_traces} traces across {len(results)} prompts "
            f"({n} samples/prompt)"
        )
        return results

    def collect_multi_sync(
        self,
        prompts: List[Dict[str, Any]],
        samples_per_prompt: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[CollectedTrace]]:
        """Synchronous wrapper for :meth:`collect_multi`."""
        return asyncio.run(
            self.collect_multi(prompts, samples_per_prompt, save_path)
        )

    async def collect_traces(
        self,
        prompts: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> List[CollectedTrace]:
        """
        Collect reasoning traces for a batch of prompts.

        Args:
            prompts: List of dicts with keys:
                - ``user_message`` (required)
                - ``system_prompt`` (optional)
                - ``category`` (optional)
            save_path: Incrementally save traces to this JSON file.

        Returns:
            List of CollectedTrace objects.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        traces: List[CollectedTrace] = []

        async def _bounded_collect(prompt_dict: Dict[str, Any], idx: int):
            async with semaphore:
                trace = await self._collect_single(
                    user_message=prompt_dict["user_message"],
                    system_prompt=prompt_dict.get("system_prompt"),
                    category=prompt_dict.get("category", "general"),
                )
                if trace:
                    traces.append(trace)
                    if idx % 10 == 0:
                        print(
                            f"[COLLECTOR] Progress: {idx + 1}/{len(prompts)} "
                            f"({len(traces)} successful)"
                        )
                    if save_path and len(traces) % self.config.save_interval == 0:
                        self._save_traces(traces, save_path)

        tasks = [_bounded_collect(p, i) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)

        if save_path:
            self._save_traces(traces, save_path)

        self._collected.extend(traces)
        print(
            f"[COLLECTOR] Collection complete: {len(traces)}/{len(prompts)} "
            f"traces collected successfully"
        )
        return traces

    def collect_traces_sync(
        self,
        prompts: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> List[CollectedTrace]:
        """Synchronous wrapper for collect_traces."""
        return asyncio.run(self.collect_traces(prompts, save_path))

    def _save_traces(self, traces: List[CollectedTrace], path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([t.to_dict() for t in traces], f, indent=2)
        print(f"[COLLECTOR] Saved {len(traces)} traces to {path}")

    def _save_traces_raw(self, trace_dicts: List[Dict[str, Any]], path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(trace_dicts, f, indent=2)
        print(f"[COLLECTOR] Saved {len(trace_dicts)} traces to {path}")

    @staticmethod
    def load_traces(path: str) -> List[Dict[str, Any]]:
        """Load traces from a JSON file."""
        with open(path) as f:
            return json.load(f)

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self._collected:
            return {"total_traces": 0}

        thinking_lengths = [len(t.thinking_text) for t in self._collected]
        output_lengths = [len(t.output_text) for t in self._collected]
        latencies = [t.latency_seconds for t in self._collected]
        categories: Dict[str, int] = {}
        for t in self._collected:
            categories[t.prompt_category] = categories.get(t.prompt_category, 0) + 1

        return {
            "total_traces": len(self._collected),
            "provider": self._client.provider,
            "categories": categories,
            "avg_thinking_chars": sum(thinking_lengths) / len(thinking_lengths),
            "avg_output_chars": sum(output_lengths) / len(output_lengths),
            "avg_latency_seconds": sum(latencies) / len(latencies),
            "total_output_tokens": sum(t.output_tokens for t in self._collected),
        }


# Backward-compatible alias
ClaudeTraceCollector = TraceCollector
