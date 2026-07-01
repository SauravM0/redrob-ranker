# Intelligent Candidate Discovery & Ranking — Track 1 (Team "AI is it")

Ranks the **top 100 of 100,000** candidates for the Redrob *Senior AI Engineer* JD,
**beyond keyword matching**. The dataset is adversarial by design — keyword stuffers,
plain-language strong candidates, behavioral twins, and ~80 honeypots. This system
reasons about the **gap between what a profile says and what it means**.

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./team_ai_is_it.csv
```
Runs in **< 3 min on a 16 GB CPU machine, no GPU, no network**. Output passes the
organizers' `validate_submission.py`.

## Streamlit demo

Try the hosted sandbox demo: https://sauravm0-redrob-ranker-app-koqeha.streamlit.app

The demo accepts a small JSONL or JSON candidate sample and returns a ranked CSV.

## Why this wins: rank evidence, not keywords

| Capability | Naive matcher | This system |
|---|---|---|
| Core signal | keyword / embedding similarity | **evidence-backed** skill verification (skill must appear in a real job description or have a strong Redrob assessment score) |
| Reads | the skills list | **career descriptions — what they actually shipped** |
| Availability | ignored | **23 Redrob signals** as a reachability multiplier |
| Traps | fooled by stuffers & honeypots | consistency checks + **honeypot floor** (can never enter top 100) |
| Output | a number | a number **+ specific, honest, non-templated reasoning** |

## Results (against our JD-grounded relevance rubric)

We have no access to the hidden ground truth, so we built a **proxy relevance labeler**
(`eval/relevance.py`) strictly from the JD's explicit "read between the lines" definition
(tier 0 = honeypot/irrelevant … tier 5 = ideal; tier ≥ 3 = "relevant", per the spec) and
**tuned our weights to maximize the official composite** against it. Ablation
(`python eval/evaluate.py ./candidates.jsonl`):

| Ranker | Trap titles in top 100 | NDCG@10 | NDCG@50 | MAP |
|---|---|---|---|---|
| Naive keyword-count (sample style) | **86** | 0.813 | 0.321 | 0.094 |
| Ablated (no evidence/shipping) | 0 | 0.967 | 0.861 | 0.829 |
| **Ours (evidence-grounded, tuned)** | **0** | **1.000** | **0.992** | **0.985** |

- **0 honeypots** in the top 100 (DQ threshold is 10%).
- Top 10 are all genuine Senior AI / ML / NLP / Search / Recsys engineers, 5–9 yrs.

> These metrics are against *our* rubric, not the organizers' hidden labels — they
> demonstrate internal consistency and the value of each component, not a leaderboard score.

## How it works
1. **Decode the JD** → `jd_spec.yaml`: must-haves, disqualifiers, ideal profile.
2. **Semantic + lexical fit** — dense `bge-small-en` if `artifacts/` exists (see below),
   else **TF-IDF** (fully offline, no download). Fallback is automatic.
3. **Engineered features** (`src/features.py`): evidence-backed skill match (full-phrase,
   not leaky substrings), shipping evidence, seniority fit, product-vs-services ratio,
   domain fit, **platform traction** (recruiter saves / search appearances), location.
4. **Trap & honeypot defense** (`src/traps.py`): disqualifier penalties + consistency
   checks (tenure vs dates, expert-with-zero-usage); honeypots floored out of the top 100.
5. **Behavioral availability multiplier** (`src/signals.py`), bounded (0.5, 1.0].
6. **Composite score** (`src/score.py`, weights in `src/weights.json`) → top 100 →
   **code-generated reasoning** (`src/reasoning.py`) quoting each candidate's real achievements.

## Design trade-offs (why no per-candidate LLM)
The spec is explicit: a system calling a hosted LLM per candidate cannot scale to a 200K
pool in production and won't fit the 5-min CPU budget. We **precompute** heavy embeddings
offline; the **ranking step is a fast scorer over precomputed features** — exactly the
latency/quality trade-off the JD asks for. Reasoning is generated in code from real
profile fields, so it is reproducible and cannot hallucinate.

## Optional: dense embeddings (run once, with internet)
```bash
pip install sentence-transformers torch
python precompute.py --candidates ./candidates.jsonl   # writes artifacts/ (float16, ~77 MB)
python eval/pipeline.py ./candidates.jsonl artifacts/feature_cache.npz   # re-extract for dense
python tune.py artifacts/feature_cache.npz             # re-tune weights for the dense path
python rank.py --candidates ./candidates.jsonl --out ./team_ai_is_it.csv
```
`rank.py` auto-detects `artifacts/` and runs offline. The committed float16 index keeps a
fresh clone reproducible without Git LFS.

## Compute compliance
CPU-only · ≤ 5 min ranking · ≤ 16 GB RAM · no network during ranking · one-command reproduce.

## Repo map
`rank.py` (entry) · `precompute.py` (optional dense) · `tune.py` (weight tuning) ·
`src/` (features, traps, signals, score, reasoning, load) ·
`eval/` (relevance rubric, metrics, pipeline, ablation) ·
`jd_spec.yaml` · `requirements.txt` · `submission_metadata.yaml` · `app.py` (sandbox).
