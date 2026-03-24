"""LLM client wrapper around the Anthropic SDK."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Tracks LLM usage and costs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost based on Claude Sonnet pricing."""
        # Sonnet: $3/M input, $15/M output
        input_cost = self.total_input_tokens * 3.0 / 1_000_000
        output_cost = self.total_output_tokens * 15.0 / 1_000_000
        return input_cost + output_cost

    def format(self) -> str:
        return (
            f"Calls: {self.total_calls} | "
            f"Cache: {self.cache_hits} | "
            f"Tokens: {self.total_input_tokens}in/{self.total_output_tokens}out | "
            f"Cost: ${self.estimated_cost_usd:.3f} | "
            f"Errors: {self.errors}"
        )


class LLMClient:
    """Async wrapper around the Anthropic API with rate limiting and caching."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.8,
        max_concurrent: int = 5,
        cost_limit_usd: float = 10.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cost_limit_usd = cost_limit_usd
        self.stats = UsageStats()

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[str, str] = {}
        self._client = None

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def _cache_key(self, system: str, user: str) -> str:
        """Generate a cache key from truncated prompts."""
        content = f"{system[:200]}|{user[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    async def complete(self, system: str, user: str, use_cache: bool = True) -> str:
        """Send a completion request with rate limiting and caching.

        Returns the assistant's text response.
        """
        # Check cost limit
        if self.stats.estimated_cost_usd >= self.cost_limit_usd:
            logger.warning("Cost limit reached: $%.3f", self.stats.estimated_cost_usd)
            return '{"action": "idle", "content": "", "opinion_change": 0, "emotional_reaction": "neutral", "reasoning": "Cost limit reached"}'

        # Check cache
        if use_cache:
            key = self._cache_key(system, user)
            if key in self._cache:
                self.stats.cache_hits += 1
                return self._cache[key]

        async with self._semaphore:
            try:
                client = self._get_client()
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

                text = response.content[0].text
                self.stats.total_calls += 1
                self.stats.total_input_tokens += response.usage.input_tokens
                self.stats.total_output_tokens += response.usage.output_tokens

                # Cache the response
                if use_cache:
                    self._cache[key] = text

                return text

            except Exception as e:
                self.stats.errors += 1
                logger.error("LLM call failed: %s", e)
                return '{"action": "idle", "content": "", "opinion_change": 0, "emotional_reaction": "neutral", "reasoning": "API error"}'

    def complete_sync(self, system: str, user: str, use_cache: bool = True) -> str:
        """Synchronous wrapper for complete()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.complete(system, user, use_cache))
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(self.complete(system, user, use_cache))
