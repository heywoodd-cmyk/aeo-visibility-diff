"""CLI entry point for the AEO visibility sweep."""

import asyncio
import time
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


def _load_config() -> dict:
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """AEO visibility sweep — measure how AI search represents your brand."""


@cli.command()
@click.option("--run-id", default=None, help="Custom run ID. Defaults to timestamp.")
@click.option("--provider", "single_provider", default=None, type=click.Choice(["anthropic", "openai", "gemini"]), help="Run a single provider only.")
@click.option("--company", default=None, type=click.Choice(["ramp", "linear", "rippling"]), help="Run prompts for one company only.")
@click.option("--dry-run", is_flag=True, help="Print calls without executing them.")
def sweep(run_id: Optional[str], single_provider: Optional[str], company: Optional[str], dry_run: bool):
    """Run the full prompt sweep across providers."""
    from .runner import run_sweep

    config = _load_config()
    rid = run_id or f"run_{int(time.time())}"

    asyncio.run(run_sweep(
        run_id=rid,
        config=config,
        single_provider=single_provider,
        company_filter=company,
        dry_run=dry_run,
    ))

    if not dry_run:
        console.print(f"\n[bold]Run ID:[/bold] [cyan]{rid}[/cyan]")
        console.print("Next step: [bold]aeo score --run-id {rid}[/bold]")


@cli.command()
@click.option("--run-id", required=True, help="Run ID to score (from the sweep step).")
@click.option("--concurrency", default=5, help="Concurrent scoring calls to Claude.")
def score(run_id: str, concurrency: int):
    """Run the scoring pass over raw sweep data."""
    from .scorer import score_run

    config = _load_config()
    df = asyncio.run(score_run(run_id=run_id, config=config, concurrency=concurrency))

    if df.empty:
        console.print("[red]No data scored.[/red]")
        return

    console.print(f"\n[bold]Scored {len(df)} records.[/bold]")
    console.print("Next step: [bold]aeo report --run-id {run_id}[/bold]")


@cli.command()
@click.option("--run-id", default=None, help="Run ID to report on. Defaults to most recent.")
def report(run_id: Optional[str]):
    """Generate markdown reports from scored data."""
    from .scorer import load_scored, latest_run_id
    from .reporter import generate_reports

    rid = run_id or latest_run_id()
    if not rid:
        console.print("[red]No scored data found. Run 'aeo score' first.[/red]")
        return

    console.print(f"[bold cyan]Generating reports for run_id={rid}[/bold cyan]")
    df = load_scored(rid)
    generate_reports(df)
    console.print("\n[green]Reports written to reports/[/green]")


@cli.command()
@click.option("--run-id", required=True, help="Run ID to show prompts for.")
def show_prompts(run_id: str):
    """Print the full prompt list."""
    from .prompts import ALL_PROMPTS

    for p in ALL_PROMPTS:
        console.print(f"[cyan]{p.id}[/cyan]  [dim]({p.category})[/dim]  {p.text}")


@cli.command()
def estimate():
    """Estimate total API cost for a full sweep."""
    from .runner import build_providers
    from .prompts import ALL_PROMPTS

    config = _load_config()
    providers = build_providers(config)
    trials = config["run"]["trials_per_combination"]
    n_prompts = len(ALL_PROMPTS)

    console.print(f"\n[bold]Cost estimate — {n_prompts} prompts × {trials} trials[/bold]\n")
    total = 0.0
    for name, provider in providers.items():
        cost = provider.estimate_cost(n_prompts, trials)
        console.print(f"  {name:12s} ${cost:.3f}")
        total += cost
    console.print(f"\n  {'TOTAL':12s} [bold yellow]${total:.3f}[/bold yellow]")


if __name__ == "__main__":
    cli()
