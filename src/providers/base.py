"""Base provider interface for AI search platforms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    provider: str
    model: str
    prompt_id: str
    trial: int
    prompt_text: str
    response_text: str
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base for all AI search providers."""

    name: str
    model: str

    @abstractmethod
    async def query(self, prompt: str, prompt_id: str, trial: int) -> ProviderResponse:
        """Send a single prompt and return the structured response."""
        ...

    @abstractmethod
    def estimate_cost(self, n_prompts: int, trials: int) -> float:
        """Rough cost estimate in USD for a given call volume."""
        ...
