"""OpenAI (GPT) provider implementation."""

import os
from openai import AsyncOpenAI
from .base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str):
        self.model = model
        self._client = None  # lazy init

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    async def query(self, prompt: str, prompt_id: str, trial: int) -> ProviderResponse:
        response = await self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
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
        # gpt-4o-mini: $0.15/M input, $0.60/M output — avg ~600 tokens total
        total_calls = n_prompts * trials
        return total_calls * (300 * 0.15 + 300 * 0.60) / 1_000_000
