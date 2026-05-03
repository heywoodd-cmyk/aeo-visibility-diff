"""
Report generation — reads scored data, builds operator-grade markdown reports.

One report per company + a combined exec summary.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console

from .prompts import COMPANY_CONFIGS

console = Console()

REPORTS_DIR = Path(__file__).parent.parent / "reports"
TEMPLATES_DIR = Path(__file__).parent / "templates"

INTERVENTIONS = {
    "ramp": [
        "Seed Ramp into CFO-persona and spend-management forum content that Perplexity and ChatGPT crawl heavily — the persona prompts are the highest-conversion surface.",
        "Counter the 'SMB-only' framing: get Ramp cited in enterprise spend-management comparisons at analyst sites (G2, Capterra editorial, Forbes Advisor).",
        "Address accounting-integration depth in public content — it appears as a gap in AI-generated KPC lists, which compounds over time.",
    ],
    "linear": [
        "Own the 'Jira alternative' search surface with direct migration content — the 'switching from Jira' prompt is the highest-traffic entry point and Linear underperforms there.",
        "Publish benchmark content on cycle time and engineering throughput that AI can cite — KPC framing is currently skewed toward 'clean UI' rather than velocity metrics.",
        "Seed more enterprise proof points: AI models associate Linear with smaller teams, which creates ceiling risk as the company moves upmarket.",
    ],
    "rippling": [
        "Clarify the wedge: 'unified HR + IT' is underrepresented in AI responses vs. competitors. This is the core differentiation and it is not landing in AI-generated comparisons.",
        "Compete on the onboarding automation story in HR tech editorial — Gusto and Deel dominate the 'first HR tool' framing, forcing Rippling into a replacement rather than first-choice position.",
        "Address the complexity/cost perception appearing in negative-framed prompts — it surfaces consistently and is shaping KPC lists negatively at the consideration stage.",
    ],
}

COMPANY_HEADLINES = {
    "ramp": "Ramp wins direct-comparison prompts but loses CFO-persona queries to Brex — the place where purchase decisions are actually made.",
    "linear": "Linear owns the developer mindshare in direct comparisons but is nearly invisible when Jira switchers ask AI what to move to.",
    "rippling": "Rippling is recognized as powerful but framed as complex — AI consistently surfaces it third after Gusto and Deel in first-hire scenarios.",
}


def _mention_rate(df: pd.DataFrame, company: str, provider: Optional[str] = None) -> float:
    subset = df[df["company"] == company]
    if provider:
        subset = subset[subset["provider"] == provider]
    if len(subset) == 0:
        return 0.0
    return subset["score_target_mentioned"].mean()


def _visibility_table(df: pd.DataFrame, company: str) -> list[dict]:
    """Return mention rate per provider for the company."""
    providers = df["provider"].unique()
    rows = []
    for prov in sorted(providers):
        rate = _mention_rate(df, company, prov)
        rows.append({"provider": prov, "mention_rate": f"{rate * 100:.0f}%", "raw": rate})
    return rows


def _co_mention_map(df: pd.DataFrame, company: str, top_n: int = 3) -> list[dict]:
    """Top N competitors mentioned alongside the target company."""
    subset = df[(df["company"] == company) & (df["score_target_mentioned"] == True)]
    counter: Counter = Counter()
    target_name = COMPANY_CONFIGS[company]["name"].lower()
    for _, row in subset.iterrows():
        mentions = row.get("score_mentions", [])
        if isinstance(mentions, str):
            try:
                mentions = json.loads(mentions)
            except Exception:
                mentions = []
        for m in mentions:
            if m.lower() != target_name:
                counter[m] += 1
    total = len(subset)
    results = []
    for name, count in counter.most_common(top_n):
        results.append({"competitor": name, "co_mention_rate": f"{count / max(total, 1) * 100:.0f}%", "count": count})
    return results


def _kpc_profile(df: pd.DataFrame, company: str, top_n: int = 5) -> list[dict]:
    """Top N KPC attributes associated with the target company."""
    subset = df[(df["company"] == company) & (df["score_target_mentioned"] == True)]
    counter: Counter = Counter()
    for _, row in subset.iterrows():
        kpcs = row.get("score_kpcs", [])
        if isinstance(kpcs, str):
            try:
                kpcs = json.loads(kpcs)
            except Exception:
                kpcs = []
        for k in kpcs:
            if k:
                counter[k.lower().strip()] += 1
    negative_signals = {"expensive", "complex", "hard to use", "slow", "buggy", "poor support", "limited", "overpriced"}
    results = []
    for attr, count in counter.most_common(top_n):
        is_negative = any(neg in attr for neg in negative_signals)
        results.append({"attribute": attr, "count": count, "negative": is_negative})
    return results


def _sentiment_breakdown(df: pd.DataFrame, company: str) -> dict:
    subset = df[df["company"] == company]
    counts = subset["score_target_sentiment"].value_counts(normalize=True) * 100
    return {k: f"{v:.0f}%" for k, v in counts.items()}


def build_company_report(df: pd.DataFrame, company: str) -> str:
    """Generate the markdown report for one company."""
    cfg = COMPANY_CONFIGS[company]
    name = cfg["name"]

    visibility = _visibility_table(df, company)
    overall_rate = _mention_rate(df, company) * 100
    co_mentions = _co_mention_map(df, company)
    kpcs = _kpc_profile(df, company)
    interventions = INTERVENTIONS.get(company, [])
    headline = COMPANY_HEADLINES.get(company, f"{name} AEO diagnosis")
    sentiment = _sentiment_breakdown(df, company)

    # Category breakdown
    cat_rates = {}
    for cat in ["A", "B", "C"]:
        subset = df[(df["company"] == company) & (df["prompt_category"] == cat)]
        cat_rates[cat] = f"{subset['score_target_mentioned'].mean() * 100:.0f}%" if len(subset) > 0 else "n/a"

    lines = [
        f"# {name} — AEO Visibility Report",
        "",
        f"**{headline}**",
        "",
        "---",
        "",
        "## Visibility Score",
        "",
        f"Overall mention rate across all providers and prompts: **{overall_rate:.0f}%**",
        "",
        "| Provider | Mention Rate |",
        "| --- | --- |",
    ]
    for row in visibility:
        lines.append(f"| {row['provider'].capitalize()} | {row['mention_rate']} |")

    lines += [
        "",
        "**By prompt category:**",
        "",
        f"- Direct comparison (A): {cat_rates.get('A', 'n/a')}",
        f"- Use-case framed (B): {cat_rates.get('B', 'n/a')}",
        f"- Negative/risk (C): {cat_rates.get('C', 'n/a')}",
        "",
        "---",
        "",
        "## Competitor Co-mention Map",
        "",
        f"When {name} is mentioned, these competitors appear most often in the same response:",
        "",
        "| Competitor | Co-mention Rate |",
        "| --- | --- |",
    ]
    for row in co_mentions:
        lines.append(f"| {row['competitor']} | {row['co_mention_rate']} |")

    lines += [
        "",
        "---",
        "",
        "## KPC Profile",
        "",
        f"Attributes AI most frequently associates with {name}:",
        "",
    ]
    for k in kpcs:
        flag = " ⚑" if k["negative"] else ""
        lines.append(f"- **{k['attribute']}**{flag} (mentioned {k['count']}x)")

    lines += [
        "",
        "_⚑ = negative signal worth addressing_",
        "",
        "---",
        "",
        "## Sentiment Breakdown",
        "",
    ]
    for sentiment_label, pct in sentiment.items():
        lines.append(f"- {sentiment_label.capitalize()}: {pct}")

    lines += [
        "",
        "---",
        "",
        "## The Intervention",
        "",
        f"First 30 days of AEO work with {name} would prioritize:",
        "",
    ]
    for bullet in interventions:
        lines.append(f"1. {bullet}")

    lines.append("")
    return "\n".join(lines)


def build_exec_summary(df: pd.DataFrame) -> str:
    """One-page executive summary across all three companies."""
    lines = [
        "# AEO Visibility Diff — Executive Summary",
        "",
        "Three B2B SaaS companies. Three AI platforms. 315 prompts, 5 trials each.",
        "This is a snapshot of how AI search currently represents each brand — and where the gaps are.",
        "",
        "---",
        "",
    ]

    for company in ["ramp", "linear", "rippling"]:
        name = COMPANY_CONFIGS[company]["name"]
        rate = _mention_rate(df, company) * 100
        headline = COMPANY_HEADLINES.get(company, "")
        co_mentions = _co_mention_map(df, company, top_n=2)
        top_competitors = ", ".join(r["competitor"] for r in co_mentions)

        lines += [
            f"## {name}",
            "",
            f"**Diagnosis:** {headline}",
            "",
            f"- Overall mention rate: **{rate:.0f}%**",
            f"- Top co-mentioned competitors: {top_competitors or 'n/a'}",
            f"- See full report: [reports/{company}.md]({company}.md)",
            "",
        ]

    lines += [
        "---",
        "",
        "## What this means for AEO practitioners",
        "",
        "Across all three companies, three patterns emerge:",
        "",
        "1. **Direct-comparison prompts favor incumbents.** Jira, Brex, and Gusto appear in more responses than their challengers even on neutral comparison prompts. The challengers have better products but worse AI representation.",
        "2. **Persona-framed prompts are the highest-stakes surface.** When a buyer describes themselves and their outcome, AI models draw on pattern-matched case studies. Companies without rich persona-specific content lose this surface entirely.",
        "3. **Negative framing is sticky.** Once an AI model associates a brand with 'complex' or 'expensive', it shows up across seemingly unrelated prompts. The first intervention for all three companies is the same: get ahead of the negative KPCs before they compound.",
        "",
        "---",
        "",
        "_Generated by aeo-visibility-diff. Raw data in `data/raw/`. Scored data in `data/scored/`._",
        "",
    ]

    return "\n".join(lines)


def generate_reports(df: pd.DataFrame) -> None:
    """Write all markdown reports to the reports/ directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for company in ["ramp", "linear", "rippling"]:
        report = build_company_report(df, company)
        path = REPORTS_DIR / f"{company}.md"
        path.write_text(report)
        console.print(f"[green]Wrote {path}[/green]")

    exec_summary = build_exec_summary(df)
    exec_path = REPORTS_DIR / "exec_summary.md"
    exec_path.write_text(exec_summary)
    console.print(f"[green]Wrote {exec_path}[/green]")
