"""
Prompt generation for AEO visibility sweep.

21 prompts across 3 companies × 7 prompts each.
Three categories per company: Direct comparison (A), Use-case framed (B), Negative/risk (C).
"""

from dataclasses import dataclass

COMPANY_CONFIGS = {
    "ramp": {
        "name": "Ramp",
        "category": "corporate card and spend management",
        "persona": "CFO at a 100-person Series B startup",
        "outcome": "automated expense management and real-time spend visibility",
        "specific_feature": "automated receipt matching and accounting integrations",
        "incumbent": "Concur or Amex corporate card",
    },
    "linear": {
        "name": "Linear",
        "category": "software project management and issue tracking",
        "persona": "engineering lead at a fast-growing product team",
        "outcome": "fast cycle times and clean engineering workflows",
        "specific_feature": "Git integration and automated issue tracking",
        "incumbent": "Jira",
    },
    "rippling": {
        "name": "Rippling",
        "category": "HR, payroll, and IT management",
        "persona": "Head of People at a company scaling from 50 to 200 employees",
        "outcome": "unified HR, payroll, and device management in one platform",
        "specific_feature": "automated onboarding that provisions payroll, benefits, and devices simultaneously",
        "incumbent": "ADP or Gusto",
    },
}


@dataclass
class Prompt:
    id: str
    company: str
    category: str  # A, B, or C
    text: str


def build_prompts() -> list[Prompt]:
    """Generate all 21 prompts (7 per company × 3 companies)."""
    prompts: list[Prompt] = []

    for company_key, cfg in COMPANY_CONFIGS.items():
        cat = cfg["category"]
        persona = cfg["persona"]
        outcome = cfg["outcome"]
        feature = cfg["specific_feature"]
        incumbent = cfg["incumbent"]
        name = cfg["name"]

        # Category A — Direct comparison (2 prompts)
        prompts.append(Prompt(
            id=f"{company_key}_A1",
            company=company_key,
            category="A",
            text=f"What's the best {cat} tool in 2026?",
        ))
        prompts.append(Prompt(
            id=f"{company_key}_A2",
            company=company_key,
            category="A",
            text=f"Compare the top {cat} platforms for a Series B startup.",
        ))

        # Category B — Use-case framed (3 prompts)
        prompts.append(Prompt(
            id=f"{company_key}_B1",
            company=company_key,
            category="B",
            text=f"I'm a {persona} looking for {outcome}. What should I use?",
        ))
        prompts.append(Prompt(
            id=f"{company_key}_B2",
            company=company_key,
            category="B",
            text=f"Best {cat} tool for a 50-person company that needs {feature}.",
        ))
        prompts.append(Prompt(
            id=f"{company_key}_B3",
            company=company_key,
            category="B",
            text=f"I'm switching from {incumbent} — what's the modern alternative?",
        ))

        # Category C — Negative / risk framed (2 prompts)
        prompts.append(Prompt(
            id=f"{company_key}_C1",
            company=company_key,
            category="C",
            text=f"What are the downsides of using {name}?",
        ))
        prompts.append(Prompt(
            id=f"{company_key}_C2",
            company=company_key,
            category="C",
            text=f"Why are companies leaving {name}?",
        ))

    return prompts


def get_prompts_for_company(company: str) -> list[Prompt]:
    return [p for p in build_prompts() if p.company == company]


ALL_PROMPTS = build_prompts()
