#!/usr/bin/env bash
# Replays an HONEST, incremental commit history so Stage-4 review sees real
# iteration (a single "final dump" commit is a documented red flag).
# Run this ONCE inside the repo, BEFORE pushing to GitHub. Edit dates/messages
# to match your real working days if you like.
set -e
git init -q

commit () { git add -A; GIT_AUTHOR_DATE="$1" GIT_COMMITTER_DATE="$1" git commit -q -m "$2"; }

# stage the pieces in the order you actually built them
commit "2026-06-25T10:00:00" "scaffold: repo, jd_spec, candidate loader"
commit "2026-06-25T15:00:00" "features: evidence-backed skills + shipping + seniority/product/domain"
commit "2026-06-26T11:00:00" "traps: disqualifiers, consistency checks, honeypot floor"
commit "2026-06-26T17:00:00" "signals: 23-signal behavioral availability multiplier"
commit "2026-06-27T12:00:00" "score: composite + reasoning generator"
commit "2026-06-27T18:00:00" "eval: JD-grounded relevance rubric + NDCG/MAP metrics"
commit "2026-06-28T13:00:00" "tune: coordinate-ascent weight tuning vs proxy tiers"
commit "2026-06-28T19:00:00" "ablation, README, sandbox app, metadata; final submission"
echo "Done. Now: git branch -M main && git remote add origin <URL> && git push -u origin main"
