"""Sanity tests for prompt generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import build_prompts, get_prompts_for_company, ALL_PROMPTS, COMPANY_CONFIGS


def test_total_prompt_count():
    prompts = build_prompts()
    assert len(prompts) == 21, f"Expected 21 prompts, got {len(prompts)}"


def test_each_company_has_7_prompts():
    for company in COMPANY_CONFIGS:
        company_prompts = get_prompts_for_company(company)
        assert len(company_prompts) == 7, f"{company} should have 7 prompts, got {len(company_prompts)}"


def test_category_distribution():
    for company in COMPANY_CONFIGS:
        company_prompts = get_prompts_for_company(company)
        categories = [p.category for p in company_prompts]
        assert categories.count("A") == 2, f"{company} should have 2 category-A prompts"
        assert categories.count("B") == 3, f"{company} should have 3 category-B prompts"
        assert categories.count("C") == 2, f"{company} should have 2 category-C prompts"


def test_prompt_ids_unique():
    ids = [p.id for p in ALL_PROMPTS]
    assert len(ids) == len(set(ids)), "Prompt IDs must be unique"


def test_prompt_texts_nonempty():
    for p in ALL_PROMPTS:
        assert p.text.strip(), f"Prompt {p.id} has empty text"


def test_negative_prompts_mention_company():
    for company, cfg in COMPANY_CONFIGS.items():
        c_prompts = get_prompts_for_company(company)
        c_prompts_cat = [p for p in c_prompts if p.category == "C"]
        for p in c_prompts_cat:
            assert cfg["name"] in p.text, f"Category C prompt {p.id} should mention {cfg['name']}"
