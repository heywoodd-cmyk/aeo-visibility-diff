"""
Scoring pass — reads raw responses, extracts structured signals via Claude Haiku.

Output: data/scored/{run_id}.csv (or .parquet if pyarrow available)
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from rich.console import Console

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from .prompts import COMPANY_CONFIGS, ALL_PROMPTS

console = Console()

SCORED_DIR = Path(__file__).parent.parent / "data" / "scored"

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All company/brand names mentioned in the response",
        },
        "target_mentioned": {
            "type": "boolean",
            "description": "Whether the target company is named anywhere in the response",
        },
        "target_sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "not_mentioned"],
            "description": "Overall sentiment toward the target company in the response",
        },
        "target_position": {
            "type": "string",
            "enum": ["first", "middle", "late", "not_mentioned"],
            "description": "Where in the response the target company first appears",
        },
        "kpcs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key purchase criteria or attributes AI associates with the target company (e.g. 'easy to use', 'expensive', 'best for enterprise'). Empty if not mentioned.",
        },
        "recommendation_rank": {
            "type": "integer",
            "description": "If the response ranks or recommends tools, what position is the target? 0 if not recommended, -1 if not applicable.",
        },
    },
    "required": ["mentions", "target_mentioned", "target_sentiment", "target_position", "kpcs", "recommendation_rank"],
}


def _competitor_list(company: str, config: dict) -> list[str]:
    return config["competitor_sets"].get(company, [])


def _target_name(prompt_id: str) -> str:
    company_key = prompt_id.split("_")[0]
    return COMPANY_CONFIGS[company_key]["name"]


def _company_from_prompt_id(prompt_id: str) -> str:
    return prompt_id.split("_")[0]


SCORING_SYSTEM = """\
You are an AI brand analyst extracting structured signals from AI assistant responses.
Given a prompt and response, identify which brands are mentioned and how the target brand is treated.
Return only valid JSON that matches the provided schema. Be precise — do not hallucinate mentions.
"""


async def score_one(
    record: dict,
    client: AsyncAnthropic,
    model: str,
    config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Score a single raw response record. Returns enriched record dict."""
    company = _company_from_prompt_id(record["prompt_id"])
    target_name = _target_name(record["prompt_id"])
    competitors = _competitor_list(company, config)

    scoring_prompt = f"""\
Target company: {target_name}
Known competitor set: {', '.join(competitors)}

Original prompt asked to the AI:
{record['prompt_text']}

AI response to score:
{record['response_text']}

Extract the structured signals as JSON matching this schema:
{json.dumps(SCORE_SCHEMA, indent=2)}
"""

    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                system=SCORING_SYSTEM,
                messages=[{"role": "user", "content": scoring_prompt}],
            )
            raw_json = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]
            score = json.loads(raw_json)
        except Exception as e:
            console.print(f"[red]Scoring error for {record['prompt_id']}: {e}[/red]")
            score = {
                "mentions": [],
                "target_mentioned": False,
                "target_sentiment": "not_mentioned",
                "target_position": "not_mentioned",
                "kpcs": [],
                "recommendation_rank": -1,
            }

    prompt_obj = next((p for p in ALL_PROMPTS if p.id == record["prompt_id"]), None)

    return {
        **record,
        "company": company,
        "target_name": target_name,
        "prompt_category": prompt_obj.category if prompt_obj else "?",
        **{f"score_{k}": v for k, v in score.items()},
    }


async def score_run(run_id: str, config: dict, concurrency: int = 5) -> pd.DataFrame:
    """Score all raw records for a run_id, return as DataFrame."""
    from .runner import load_all_raw

    records = load_all_raw(run_id)
    if not records:
        console.print(f"[red]No raw data found for run_id={run_id}[/red]")
        return pd.DataFrame()

    console.print(f"\n[bold cyan]Scoring {len(records)} records...[/bold cyan]")

    scoring_model = config["scoring"]["model"]
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    semaphore = asyncio.Semaphore(concurrency)

    scored = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task_id = progress.add_task("Scoring...", total=len(records))
        coros = [score_one(r, client, scoring_model, config, semaphore) for r in records]
        for coro in asyncio.as_completed(coros):
            result = await coro
            scored.append(result)
            progress.advance(task_id)

    df = pd.DataFrame(scored)
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    output_format = config["run"].get("output_format", "csv")

    try:
        if output_format == "parquet":
            out_path = SCORED_DIR / f"{run_id}.parquet"
            df.to_parquet(out_path, index=False)
        else:
            raise ImportError("prefer csv")
    except Exception:
        out_path = SCORED_DIR / f"{run_id}.csv"
        df.to_csv(out_path, index=False)

    console.print(f"[green]Scored data saved to {out_path}[/green]")
    return df


def load_scored(run_id: str) -> pd.DataFrame:
    """Load a previously-scored run as a DataFrame."""
    csv_path = SCORED_DIR / f"{run_id}.csv"
    parquet_path = SCORED_DIR / f"{run_id}.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Parse list columns stored as Python list strings (single quotes) or JSON
        import ast

        def _safe_parse(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                x = x.strip()
                if not x or x in ("nan", "[]"):
                    return []
                try:
                    result = ast.literal_eval(x)
                    return result if isinstance(result, list) else []
                except Exception:
                    try:
                        return json.loads(x)
                    except Exception:
                        return []
            return []

        for col in ["score_mentions", "score_kpcs"]:
            if col in df.columns:
                df[col] = df[col].apply(_safe_parse)
        return df
    elif parquet_path.exists():
        return pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(f"No scored data for run_id={run_id}")


def latest_run_id() -> Optional[str]:
    """Return the most recently modified run_id in data/scored/."""
    scored_dir = SCORED_DIR
    if not scored_dir.exists():
        return None
    files = list(scored_dir.glob("*.csv")) + list(scored_dir.glob("*.parquet"))
    if not files:
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return latest.stem
