# aeo-visibility-diff

A v0 of the measurement layer Petra is building — run on three companies that are likely Petra prospects, using real API calls across ChatGPT, Claude, and Gemini.

---

## Why these three companies

- **Ramp** is the clearest AEO win waiting to happen. It has a 90% overall mention rate and is treated positively, but Brex co-appears in 75% of responses where Ramp is named. The competitive narrative is not Ramp's — it's being shaped by whoever owns the content AI is citing.
- **Linear** is the most interesting case. It has a 76% mention rate on Claude but only 26% on ChatGPT — a 50-point provider gap on the same prompts. That divergence means Linear's visibility depends entirely on which AI surface a buyer uses. At scale, that is a material demand risk.
- **Rippling** is winning on awareness (86% mention rate) but losing on framing. The top KPC associated with the brand is "steep learning curve" — a negative signal that surfaced in 8 independent scoring calls. That compounds. Rippling needs to own its own narrative before AI does it for them.

---

## What I found

| Company | Mention Rate | Top Diagnosis |
|---------|-------------|---------------|
| **Ramp** | 90% | High visibility, but Brex co-appears in 75% of responses — the competitive frame belongs to Brex |
| **Linear** | 51% | 50-point gap between Claude (76%) and ChatGPT (26%) — provider-dependent visibility is a structural risk |
| **Rippling** | 86% | "Steep learning curve" is the #1 AI-associated attribute — a negative KPC compounding at the consideration stage |

Full reports: [reports/ramp.md](reports/ramp.md) · [reports/linear.md](reports/linear.md) · [reports/rippling.md](reports/rippling.md) · [reports/exec_summary.md](reports/exec_summary.md)

---

## How it works

```
Prompts (21)
    ├── Category A: Direct comparison ("best tool in 2026?")
    ├── Category B: Use-case framed ("I'm a CFO looking for X")
    └── Category C: Negative/risk ("why are companies leaving X?")
         │
         ▼
    Sweep runner (async, 3 concurrent per provider)
         │
         ├── Anthropic: Claude Haiku 4.5
         ├── OpenAI: GPT-4o-mini
         └── Gemini: Gemini 2.5 Flash
         │
         ▼
    Raw responses → data/raw/{provider}/{run_id}/*.json
         │
         ▼
    Scoring pass (Claude Haiku, structured JSON output)
    Extracts: target_mentioned, sentiment, position, competitor mentions, KPCs
         │
         ▼
    data/scored/{run_id}.csv
         │
         ├── reports/{company}.md  (operator-grade one-pager)
         └── dashboard/app.py      (Streamlit + Plotly)
```

**Methodology:**
- 5 trials per (prompt × provider) to smooth model variance
- 21 prompts × 3 providers × 5 trials = 315 total calls
- Scoring uses Claude with a strict JSON schema — no free-text extraction
- Runs are idempotent: skip cached responses on rerun

---

## What I'd build next at Petra

**1. Longitudinal tracking, not snapshots.**
This run is a point in time. The real product is the diff — how mention rate and KPC framing change week over week, and whether an AEO intervention moved the needle. The run_id architecture is already set up for it; the missing piece is a scheduled runner and a time-series store.

**2. Prompt-set versioning tied to funnel stage.**
The 21 prompts here are a rough taxonomy. The actual value is mapping prompts to buyer stages: awareness, consideration, evaluation, purchase. A prompt like "what's the best spend management tool" is top-of-funnel. "Compare Ramp vs Brex for a 500-person company with SAP" is late-stage evaluation. AEO interventions differ by stage, and right now there's no way to attribute a visibility change to a specific funnel moment.

**3. Closing the attribution loop to pipeline.**
Right now the output is a mention rate. The product that matters to a CFO at Ramp is: "when AI recommends us, do those buyers convert at a higher rate?" Closing that loop requires integrating with the customer's GA4 or Segment — correlating AI referral traffic with CRM opportunity creation. That is the measurement primitive Petra needs to charge enterprise pricing.

**4. Human-in-the-loop scoring calibration.**
The current scoring pass uses Claude to extract KPCs and sentiment. It's good but not perfect — "fast" gets counted separately from "speed" and "fast performance." A calibration layer where a human reviews 50 scored responses per quarter and corrects the model would dramatically improve precision on the KPC clustering. This also creates a feedback dataset that could fine-tune a dedicated scoring model.

**5. Competitive intelligence as a product surface.**
Every company that runs this sweep generates data about its competitors as a side effect. Ramp's sweep tells us Brex co-appears 75% of the time. That is actionable intelligence for Ramp's marketing team independent of AEO — it tells them which competitor's content AI is citing most heavily, and therefore which competitor's content authority they need to erode. Packaging this as a competitive intelligence feed is a distinct monetizable surface from the core AEO workflow.

---

## Run it yourself

**Prerequisites:** Python 3.11+, `uv` (or pip)

```bash
git clone https://github.com/devante-heywood/aeo-visibility-diff
cd aeo-visibility-diff

# Install dependencies
uv sync
# or: pip install -e .

# Set up API keys
cp .env.example .env
# Edit .env with your Anthropic, OpenAI, and Gemini keys

# Estimate cost before running
aeo estimate

# Dry run — see what would be called
aeo sweep --dry-run

# Run Anthropic only first (cheapest, ~$0.15)
aeo sweep --provider anthropic

# Run all providers
aeo sweep

# Score the responses
aeo score --run-id <run_id_from_sweep_output>

# Generate reports
aeo report

# Launch dashboard
streamlit run dashboard/app.py
```

**Cost:** ~$0.18 total for a full sweep (Anthropic Haiku + GPT-4o-mini + Gemini 2.5 Flash).  
**Time:** ~5 minutes for the sweep, ~3 minutes for scoring.

---

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Sweep runner | `asyncio` + semaphore per provider | Per-provider rate limiting without blocking |
| Persistence | JSON files per response | Idempotent reruns, easy inspection |
| Scoring | Claude Haiku + JSON schema | Deterministic extraction at low cost |
| Reports | Python string templates | No template engine dependency, full control |
| Dashboard | Streamlit + Plotly | Fast to ship, dark aesthetic, interactive filters |
| Package manager | uv | Fast, reproducible |

---

_Built by Devante Heywood as a Petra Labs application demo._  
_All data from real API calls. No synthetic responses._
