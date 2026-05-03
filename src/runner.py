"""
Async runner — orchestrates prompt × provider × trial sweep with persistence.

Persists raw responses to data/raw/{provider}/{run_id}/{prompt_id}_trial{n}.json
Skips calls where the output file already exists (idempotent reruns).
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pathlib import Path as _Path
from rich.console import Console

# Load from project root regardless of working directory
load_dotenv(_Path(__file__).parent.parent / ".env", override=True)
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .prompts import ALL_PROMPTS, Prompt
from .providers import AnthropicProvider, OpenAIProvider, GeminiProvider, BaseProvider

console = Console()

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_providers(config: dict, single_provider: Optional[str] = None) -> dict[str, BaseProvider]:
    """Instantiate requested providers from config."""
    provider_names = [single_provider] if single_provider else config["providers"]
    models = config["models"]
    providers: dict[str, BaseProvider] = {}

    for name in provider_names:
        if name == "anthropic":
            providers[name] = AnthropicProvider(models["anthropic"])
        elif name == "openai":
            providers[name] = OpenAIProvider(models["openai"])
        elif name == "gemini":
            providers[name] = GeminiProvider(models["gemini"])
        else:
            raise ValueError(f"Unknown provider: {name}")

    return providers


def output_path(run_id: str, provider: str, prompt_id: str, trial: int) -> Path:
    path = DATA_DIR / provider / run_id / f"{prompt_id}_trial{trial}.json"
    return path


async def run_single(
    provider: BaseProvider,
    prompt: Prompt,
    trial: int,
    run_id: str,
    semaphore: asyncio.Semaphore,
    delay: float = 0.0,
) -> Optional[dict]:
    """Run one prompt/trial, persisting the result. Returns None if already cached."""
    out = output_path(run_id, provider.name, prompt.id, trial)

    if out.exists():
        return None  # cached

    async with semaphore:
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            response = await provider.query(prompt.text, prompt.id, trial)
            out.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "run_id": run_id,
                "provider": response.provider,
                "model": response.model,
                "prompt_id": response.prompt_id,
                "trial": response.trial,
                "prompt_text": response.prompt_text,
                "response_text": response.response_text,
                "usage": response.usage,
                "metadata": response.metadata,
                "timestamp": time.time(),
            }
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            return record
        except Exception as e:
            console.print(f"[red]Error {provider.name}/{prompt.id}/trial{trial}: {e}[/red]")
            return None


async def run_sweep(
    run_id: str,
    config: dict,
    single_provider: Optional[str] = None,
    company_filter: Optional[str] = None,
    dry_run: bool = False,
) -> list[dict]:
    """Full async sweep — returns list of result records."""
    providers = build_providers(config, single_provider)
    trials = config["run"]["trials_per_combination"]
    concurrency = config["run"]["max_concurrent_per_provider"]

    prompts = ALL_PROMPTS
    if company_filter:
        prompts = [p for p in prompts if p.company == company_filter]

    total = len(prompts) * len(providers) * trials
    console.print(f"\n[bold cyan]Run ID:[/bold cyan] {run_id}")
    console.print(f"[bold]Providers:[/bold] {list(providers.keys())}")
    console.print(f"[bold]Prompts:[/bold] {len(prompts)} × [bold]Trials:[/bold] {trials} × [bold]Providers:[/bold] {len(providers)} = [bold yellow]{total}[/bold yellow] calls\n")

    # Cost estimate
    for name, provider in providers.items():
        est = provider.estimate_cost(len(prompts), trials)
        console.print(f"  {name}: ~${est:.3f}")

    if dry_run:
        console.print("\n[yellow]--dry-run: no API calls made.[/yellow]")
        for prompt in prompts:
            for pname in providers:
                for t in range(1, trials + 1):
                    console.print(f"  would call: {pname}/{prompt.id}/trial{t}  |  {prompt.text[:80]}")
        return []

    # Per-provider rate limit config
    rate_limits = config.get("rate_limits", {})
    per_provider_concurrency = {
        name: rate_limits.get(name, {}).get("concurrency", concurrency)
        for name in providers
    }
    per_provider_delay = {
        name: rate_limits.get(name, {}).get("delay_between_calls", 0.0)
        for name in providers
    }

    semaphores = {name: asyncio.Semaphore(per_provider_concurrency[name]) for name in providers}
    tasks = []
    for prompt in prompts:
        for pname, provider in providers.items():
            for trial in range(1, trials + 1):
                tasks.append((provider, prompt, trial, semaphores[pname], per_provider_delay[pname]))

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Running sweep...", total=len(tasks))
        coros = [run_single(p, prompt, trial, run_id, sem, delay) for p, prompt, trial, sem, delay in tasks]

        for coro in asyncio.as_completed(coros):
            result = await coro
            if result:
                results.append(result)
            progress.advance(task_id)

    console.print(f"\n[green]Done.[/green] {len(results)} new responses saved to data/raw/")
    return results


def load_all_raw(run_id: str) -> list[dict]:
    """Load every persisted response for a given run_id."""
    records = []
    run_dir = DATA_DIR
    for provider_dir in run_dir.iterdir():
        if not provider_dir.is_dir():
            continue
        run_subdir = provider_dir / run_id
        if not run_subdir.exists():
            continue
        for f in run_subdir.glob("*.json"):
            records.append(json.loads(f.read_text()))
    return records
