"""Google Gemini provider implementation (google-genai SDK)."""

import asyncio
import os
from .base import BaseProvider, ProviderResponse


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str):
        self.model = model
        self._client = None  # lazy init — requires GEMINI_API_KEY at query time

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    async def query(self, prompt: str, prompt_id: str, trial: int) -> ProviderResponse:
        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=self.model, contents=prompt),
        )
        content = response.text if hasattr(response, "text") else ""
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
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
        # Gemini 1.5 Flash: effectively free within generous API limits for this volume
        return 0.0
