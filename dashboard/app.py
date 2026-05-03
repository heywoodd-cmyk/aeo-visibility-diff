"""
AEO Visibility Diff — Streamlit Dashboard

Operator-grade view of brand mention rates, competitor co-mention patterns,
and KPC framing across AI search platforms.
"""

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Inlined from src/prompts.py — keeps the dashboard dependency-free
COMPANY_CONFIGS = {
    "ramp": {"name": "Ramp"},
    "linear": {"name": "Linear"},
    "rippling": {"name": "Rippling"},
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEO Visibility Diff",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Typography */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0f0f0f;
    border-right: 1px solid #1e1e1e;
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #888 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }

  /* Main background */
  .main { background-color: #0a0a0a; }
  .block-container { padding: 2rem 2.5rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 1rem;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f4fd;
    font-size: 2rem;
    font-weight: 600;
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #666;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* Section headers */
  h1 { color: #f5f5f5 !important; font-weight: 600; font-size: 1.5rem; letter-spacing: -0.02em; }
  h2 { color: #d0d0d0 !important; font-weight: 500; font-size: 1.1rem; border-bottom: 1px solid #1e1e1e; padding-bottom: 0.5rem; margin-top: 2rem; }
  h3 { color: #b0b0b0 !important; font-weight: 500; font-size: 0.95rem; }

  /* Dividers */
  hr { border-color: #1e1e1e; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #111; border-radius: 8px; gap: 0; }
  .stTabs [data-baseweb="tab"] { color: #666; font-size: 0.85rem; padding: 0.5rem 1.2rem; }
  .stTabs [aria-selected="true"] { background: #1a1a1a !important; color: #e0e0e0 !important; border-radius: 6px; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1e1e1e; border-radius: 8px; }

  /* Diagnosis box */
  .diagnosis-box {
    background: linear-gradient(135deg, #0d1117, #111827);
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    color: #c9d1d9;
    font-size: 0.95rem;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#888", size=12),
    margin=dict(l=0, r=0, t=32, b=0),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
    xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a", color="#666"),
    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#1a1a1a", color="#666"),
)
# Pie/donut charts don't accept xaxis/yaxis/showlegend as layout keys
PLOTLY_LAYOUT_PIE = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "showlegend")}

PROVIDER_COLORS = {
    "anthropic": "#d97706",
    "openai": "#10a37f",
    "gemini": "#4285f4",
}

COMPANY_LABELS = {
    "ramp": "Ramp",
    "linear": "Linear",
    "rippling": "Rippling",
}

HEADLINES = {
    "ramp": "Ramp wins direct-comparison prompts but loses CFO-persona queries to Brex — the place where purchase decisions are actually made.",
    "linear": "Linear owns the developer mindshare in direct comparisons but is nearly invisible when Jira switchers ask AI what to move to.",
    "rippling": "Rippling is recognized as powerful but framed as complex — AI consistently surfaces it third after Gusto and Deel in first-hire scenarios.",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    scored_dir = Path(__file__).parent.parent / "data" / "scored"
    files = list(scored_dir.glob("*.csv")) + list(scored_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    latest = max(files, key=lambda f: f.stat().st_mtime)
    if latest.suffix == ".parquet":
        df = pd.read_parquet(latest)
    else:
        df = pd.read_csv(latest)

    for col in ["score_mentions", "score_kpcs"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)
    return df


def _parse_list(x) -> list:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    try:
        import math
        if math.isnan(float(x)):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(x, str):
        x = x.strip()
        if not x or x in ("nan", "[]", "None"):
            return []
        try:
            result = ast.literal_eval(x)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


def parse_list_col(series: pd.Series) -> pd.Series:
    return series.apply(_parse_list)


# ── Chart builders ────────────────────────────────────────────────────────────
def mention_rate_bar(df: pd.DataFrame, company: str) -> go.Figure:
    providers = sorted(df["provider"].unique())
    rates = []
    for prov in providers:
        subset = df[(df["company"] == company) & (df["provider"] == prov)]
        rate = subset["score_target_mentioned"].mean() * 100 if len(subset) > 0 else 0
        rates.append({"provider": prov.capitalize(), "mention_rate": rate})

    prov_df = pd.DataFrame(rates)
    colors = [PROVIDER_COLORS.get(p.lower(), "#666") for p in prov_df["provider"].str.lower()]
    fig = go.Figure(go.Bar(
        x=prov_df["provider"],
        y=prov_df["mention_rate"],
        marker_color=colors,
        text=[f"{r:.0f}%" for r in prov_df["mention_rate"]],
        textposition="outside",
        textfont=dict(color="#aaa", size=13),
        showlegend=False,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title=dict(text="Mention Rate by Provider", font=dict(color="#aaa", size=13)))
    fig.update_yaxes(range=[0, 110], ticksuffix="%")
    return fig


def category_breakdown_bar(df: pd.DataFrame, company: str) -> go.Figure:
    cat_labels = {"A": "Direct comparison", "B": "Use-case framed", "C": "Negative/risk"}
    rows = []
    for cat, label in cat_labels.items():
        for prov in sorted(df["provider"].unique()):
            subset = df[(df["company"] == company) & (df["prompt_category"] == cat) & (df["provider"] == prov)]
            rate = subset["score_target_mentioned"].mean() * 100 if len(subset) > 0 else 0
            rows.append({"category": label, "provider": prov.capitalize(), "rate": rate})

    cat_df = pd.DataFrame(rows)
    cat_df["label"] = cat_df["rate"].apply(lambda r: f"{r:.0f}%")
    fig = px.bar(
        cat_df,
        x="category",
        y="rate",
        color="provider",
        barmode="group",
        text="label",
        color_discrete_map={p.capitalize(): c for p, c in PROVIDER_COLORS.items()},
    )
    fig.update_traces(textposition="outside", textfont_color="#aaa")
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Mention Rate by Prompt Category", font=dict(color="#aaa", size=13)))
    fig.update_yaxes(range=[0, 115], ticksuffix="%", title="")
    fig.update_xaxes(title="")
    return fig


def co_mention_heatmap(df: pd.DataFrame, company: str) -> go.Figure:
    subset = df[(df["company"] == company) & (df["score_target_mentioned"] == True)]
    target_name = COMPANY_CONFIGS[company]["name"].lower()

    providers = sorted(df["provider"].unique())
    all_competitors: Counter = Counter()
    for _, row in subset.iterrows():
        mentions = parse_list_col(pd.Series([row["score_mentions"]]))[0]
        for m in mentions:
            if m.lower() != target_name:
                all_competitors[m] += 1

    top_competitors = [c for c, _ in all_competitors.most_common(8)]
    if not top_competitors:
        return go.Figure()

    matrix = []
    for comp in top_competitors:
        row_data = []
        for prov in providers:
            prov_subset = subset[subset["provider"] == prov]
            total = len(prov_subset)
            count = sum(
                1 for _, r in prov_subset.iterrows()
                if comp in parse_list_col(pd.Series([r["score_mentions"]]))[0]
            )
            row_data.append(round(count / max(total, 1) * 100, 1))
        matrix.append(row_data)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[p.capitalize() for p in providers],
        y=top_competitors,
        colorscale=[[0, "#0a0a0a"], [0.5, "#1a3a5c"], [1.0, "#58a6ff"]],
        text=[[f"{v:.0f}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        showscale=False,
        hovertemplate="%{y} co-mentioned with target in %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Competitor Co-mention Heatmap", font=dict(color="#aaa", size=13)))
    return fig


def kpc_bar(df: pd.DataFrame, company: str) -> go.Figure:
    subset = df[(df["company"] == company) & (df["score_target_mentioned"] == True)]
    counter: Counter = Counter()
    for _, row in subset.iterrows():
        kpcs = parse_list_col(pd.Series([row["score_kpcs"]]))[0]
        for k in kpcs:
            if k:
                counter[k.lower().strip()] += 1

    if not counter:
        return go.Figure()

    top = counter.most_common(8)
    attrs, counts = zip(*top)
    negative_signals = {"expensive", "complex", "hard to use", "slow", "buggy", "poor support", "limited", "overpriced"}
    colors = ["#ef4444" if any(n in a for n in negative_signals) else "#58a6ff" for a in attrs]

    fig = go.Figure(go.Bar(
        x=list(counts),
        y=list(attrs),
        orientation="h",
        marker_color=colors,
        text=[str(c) for c in counts],
        textposition="outside",
        textfont=dict(color="#aaa"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="KPC Profile (red = negative signal)", font=dict(color="#aaa", size=13)))
    fig.update_xaxes(title="Frequency")
    return fig


def sentiment_donut(df: pd.DataFrame, company: str) -> go.Figure:
    subset = df[df["company"] == company]
    counts = subset["score_target_sentiment"].value_counts()
    colors_map = {"positive": "#10b981", "neutral": "#6366f1", "negative": "#ef4444", "not_mentioned": "#374151"}
    colors = [colors_map.get(k, "#666") for k in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.6,
        marker_colors=colors,
        textinfo="percent+label",
        textfont=dict(color="#aaa", size=11),
    ))
    fig.update_layout(**PLOTLY_LAYOUT_PIE, title=dict(text="Sentiment Distribution", font=dict(color="#aaa", size=13)), showlegend=False)
    return fig


# ── App shell ─────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # Sidebar
    with st.sidebar:
        st.markdown("### 📡 AEO Visibility Diff")
        st.markdown('<p style="color:#555;font-size:0.75rem;margin-top:-8px">v0 measurement layer</p>', unsafe_allow_html=True)
        st.markdown("---")

        if df.empty:
            st.warning("No scored data found. Run the sweep first.")
            company = "ramp"
            providers_filter = []
        else:
            companies = list(df["company"].unique())
            company = st.selectbox(
                "Company",
                companies,
                format_func=lambda x: COMPANY_LABELS.get(x, x),
            )

            available_providers = list(df["provider"].unique())
            providers_filter = st.multiselect(
                "Providers",
                available_providers,
                default=available_providers,
                format_func=lambda x: x.capitalize(),
            )

            cats = st.multiselect(
                "Prompt categories",
                ["A", "B", "C"],
                default=["A", "B", "C"],
                format_func=lambda x: {"A": "A — Direct comparison", "B": "B — Use-case framed", "C": "C — Negative/risk"}[x],
            )

        st.markdown("---")
        st.markdown(
            '<p style="color:#333;font-size:0.7rem">Built as a Petra Labs demo.<br>No fake data.</p>',
            unsafe_allow_html=True,
        )

    # Header
    company_label = COMPANY_LABELS.get(company, company)
    st.markdown(f"# {company_label} — AEO Visibility")

    if df.empty:
        st.info("Run `aeo sweep` and `aeo score` to generate data, then refresh.")
        return

    # Filter
    filtered = df[df["company"] == company]
    if providers_filter:
        filtered = filtered[filtered["provider"].isin(providers_filter)]
    if cats:
        filtered = filtered[filtered["prompt_category"].isin(cats)]

    # Diagnosis
    headline = HEADLINES.get(company, "")
    st.markdown(f'<div class="diagnosis-box"><strong>Diagnosis:</strong> {headline}</div>', unsafe_allow_html=True)

    # KPI row
    total_calls = len(filtered)
    mention_rate = filtered["score_target_mentioned"].mean() * 100 if total_calls > 0 else 0
    pos_rate = (filtered["score_target_sentiment"] == "positive").mean() * 100 if total_calls > 0 else 0
    neg_rate = (filtered["score_target_sentiment"] == "negative").mean() * 100 if total_calls > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mention Rate", f"{mention_rate:.0f}%")
    col2.metric("Positive Sentiment", f"{pos_rate:.0f}%")
    col3.metric("Negative Sentiment", f"{neg_rate:.0f}%")
    col4.metric("Responses Analyzed", f"{total_calls:,}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Visibility", "Competitors", "KPC Profile", "Raw Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(mention_rate_bar(filtered, company), use_container_width=True)
        with c2:
            st.plotly_chart(sentiment_donut(filtered, company), use_container_width=True)

        st.plotly_chart(category_breakdown_bar(filtered, company), use_container_width=True)

    with tab2:
        st.plotly_chart(co_mention_heatmap(filtered, company), use_container_width=True)

        st.markdown("### Who shows up when the target does")
        subset = filtered[filtered["score_target_mentioned"] == True]
        target_name = COMPANY_CONFIGS[company]["name"].lower()
        counter: Counter = Counter()
        for _, row in subset.iterrows():
            mentions = parse_list_col(pd.Series([row["score_mentions"]]))[0]
            for m in mentions:
                if m.lower() != target_name:
                    counter[m] += 1

        if counter:
            co_df = pd.DataFrame(
                [{"Competitor": k, "Co-mentions": v, "Rate": f"{v / max(len(subset), 1) * 100:.0f}%"}
                 for k, v in counter.most_common(10)]
            )
            st.dataframe(co_df, use_container_width=True, hide_index=True)

    with tab3:
        st.plotly_chart(kpc_bar(filtered, company), use_container_width=True)
        st.markdown(
            "Key Purchase Criteria extracted by Claude from each response. "
            "Red bars indicate negative signals that compound over time in AI training data."
        )

    with tab4:
        display_cols = ["provider", "prompt_id", "prompt_category", "prompt_text",
                        "score_target_mentioned", "score_target_sentiment",
                        "score_target_position", "score_recommendation_rank"]
        available = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[available].rename(columns=lambda c: c.replace("score_", "").replace("_", " ")),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
