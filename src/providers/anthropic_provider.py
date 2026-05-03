"""Anthropic (Claude) provider implementation."""

import os
from anthropic import AsyncAnthropic
from .base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model: str):
        self.model = model
        self._client = None  # lazy init

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    async def query(self, prompt: str, prompt_id: str, trial: int) -> ProviderResponse:
        response = await self._get_client().messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            prompt_id=prompt_id,
            trial=trial,
            prompt_text=prompt,
            response_text=content,
            usage=usage,
        )

    def estimate_cost(self, n_prompts: int, trials: int) -> float:
        # Haiku 4.5: $0.80/M input, $4.00/M output — avg ~600 tokens total
        total_calls = n_prompts * trials
        return total_calls * (300 * 0.80 + 300 * 4.00) / 1_000_000
