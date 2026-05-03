"""Sanity tests for scoring logic (no API calls)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer import load_scored, latest_run_id
from src.reporter import (
    _mention_rate,
    _co_mention_map,
    _kpc_profile,
    build_company_report,
)


def _make_fake_df() -> pd.DataFrame:
    """Minimal scored DataFrame for unit tests."""
    rows = []
    for company in ["ramp", "linear", "rippling"]:
        for i in range(10):
            rows.append({
                "run_id": "test_run",
                "provider": "anthropic" if i % 2 == 0 else "openai",
                "model": "claude-haiku-4-5-20251001",
                "prompt_id": f"{company}_A1",
                "trial": i % 5 + 1,
                "prompt_text": "What's the best tool?",
                "response_text": "Some response",
                "company": company,
                "target_name": company.capitalize(),
                "prompt_category": "A",
                "score_mentions": json.dumps(["Ramp", "Brex"] if company == "ramp" else ["Linear", "Jira"]),
                "score_target_mentioned": i % 3 != 0,
                "score_target_sentiment": "positive" if i % 2 == 0 else "neutral",
                "score_target_position": "first" if i % 2 == 0 else "middle",
                "score_kpcs": json.dumps(["easy to use", "fast"] if i % 2 == 0 else ["expensive"]),
                "score_recommendation_rank": 1 if i % 2 == 0 else -1,
            })
    return pd.DataFrame(rows)


def test_mention_rate_range():
    df = _make_fake_df()
    rate = _mention_rate(df, "ramp")
    assert 0.0 <= rate <= 1.0, "Mention rate must be between 0 and 1"


def test_mention_rate_by_provider():
    df = _make_fake_df()
    rate_anthropic = _mention_rate(df, "ramp", "anthropic")
    rate_openai = _mention_rate(df, "ramp", "openai")
    assert 0.0 <= rate_anthropic <= 1.0
    assert 0.0 <= rate_openai <= 1.0


def test_co_mention_map_returns_list():
    df = _make_fake_df()
    result = _co_mention_map(df, "ramp", top_n=3)
    assert isinstance(result, list)
    assert len(result) <= 3
    if result:
        assert "competitor" in result[0]
        assert "co_mention_rate" in result[0]


def test_kpc_profile_returns_list():
    df = _make_fake_df()
    result = _kpc_profile(df, "ramp", top_n=5)
    assert isinstance(result, list)
    assert len(result) <= 5
    if result:
        assert "attribute" in result[0]
        assert "negative" in result[0]


def test_build_company_report_produces_markdown():
    df = _make_fake_df()
    report = build_company_report(df, "ramp")
    assert "# Ramp" in report
    assert "## Visibility Score" in report
    assert "## The Intervention" in report


def test_latest_run_id_returns_none_on_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.scorer.SCORED_DIR", tmp_path)
    assert latest_run_id() is None
