"""Proxy relevance labeler — our best estimate of the hidden ground-truth tiers,
built STRICTLY from the JD's stated 'read between the lines' definition and the
spec's honeypot rule. Tiers are graded 0..5 (tier>=3 == 'relevant', per spec).

It is intentionally built from hard, high-precision rules (title class + career
evidence of building IR/ranking systems + product + domain + seniority +
reachability + disqualifiers), NOT from the ranker's tunable weighted sum, so
tuning the ranker to maximize NDCG against it is meaningful, not circular.
"""
import re
from src.load import career_text

AI_TITLE = re.compile(
    r"(ai engineer|ml engineer|machine learning|applied scientist|applied ml|"
    r"nlp engineer|research engineer|search engineer|relevance engineer|"
    r"recommendation|recsys|data scientist|deep learning|ai/ml|ml scientist)", re.I)
DISTRACTOR = re.compile(
    r"(hr manager|content writer|business analyst|accountant|sales|graphic designer|"
    r"marketing|customer support|operations manager|mechanical engineer|civil engineer|"
    r"project manager|recruiter|administrator|teacher|nurse|receptionist)", re.I)

SHIP = ("built", "shipped", "deployed", "launched", "developed", "designed",
        "owned", "led", "scaled", "improved", "optimized")
IR_OBJ = ("search", "ranking", "recommendation", "recommender", "recsys",
          "retrieval", "embedding", "relevance", "semantic", "vector", "matching")

def _ai_core_evidence(c):
    ctext = career_text(c)
    hits = 0
    for v in SHIP:
        for o in IR_OBJ:
            if re.search(v + r"[^.]{0,60}" + re.escape(o), ctext):
                hits += 1
    return hits

def proxy_tier(c, feats):
    """feats: dict already computed for this candidate (reuse penalties/flags)."""
    p = c.get("profile", {})
    title = (p.get("current_title") or "")
    yoe = p.get("years_of_experience", 0) or 0
    any_title = " ".join([title] + [(j.get("title") or "") for j in c.get("career_history", [])])

    if feats["honeypot"]:
        return 0

    ai_core = _ai_core_evidence(c)
    ai_title = bool(AI_TITLE.search(any_title))
    distractor = bool(DISTRACTOR.search(title)) and not ai_title
    product = feats["product_ratio"] >= 0.5
    domain_ok = feats["domain"] >= 0.7
    senior_ok = 5 <= yoe <= 9
    senior_soft = 4 <= yoe <= 10
    reachable = feats["availability"] >= 0.7
    hard_dq = feats["dq_penalty"] >= 0.25  # consulting-only / research-only / CV-only / outside-non-reloc

    # keyword-stuffer trap: distractor title, no real AI career evidence -> irrelevant
    if distractor and ai_core == 0:
        return 0
    if hard_dq and ai_core == 0:
        return 1
    # has AI keywords/title but no shipping evidence or wrong domain -> weak
    if ai_core == 0 and not (ai_title and domain_ok):
        return 1 if not ai_title else 2
    if ai_core == 0 and ai_title and domain_ok:
        return 2

    # from here ai_core >= 1
    if ai_core >= 2 and product and domain_ok and senior_ok and not hard_dq and reachable:
        return 5
    if ai_core >= 1 and product and domain_ok and senior_soft and not hard_dq:
        return 4
    if ai_core >= 1 and domain_ok and senior_soft and not hard_dq:
        return 3
    return 2
