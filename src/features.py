"""Engineered features that beat keyword matching. Each returns a value in [0,1]."""
import math, re
from src.load import career_text

WORD = re.compile(r"[a-z0-9\+\#\.\-]+")

def _phrase_in(text, phrase):
    """True only if the FULL phrase appears (multi-word phrases must match as a
    contiguous span, not via the leaky 'any token in text' test)."""
    phrase = phrase.strip()
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    # single token: require a word-boundary hit, not a substring of another word
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None

def skill_evidence(c, spec):
    """Must-have skills BACKED by evidence, not just listed.
    Backed = (advanced/expert AND duration>0 AND the skill phrase actually appears
    in a career description) OR a strong Redrob assessment score for that skill.
    This is the primary defense against keyword stuffers."""
    must = spec["must_have_skills"]
    ctext = career_text(c) + " " + (c.get("profile", {}).get("summary", "") or "").lower()
    assess = (c.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    assess_l = {k.lower(): v for k, v in assess.items()}

    backed = 0
    listed = 0
    for s in c.get("skills", []):
        name = (s.get("name", "") or "").lower()
        if not name:
            continue
        is_must = any(m in name or name in m for m in must)
        if not is_must:
            continue
        listed += 1
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0) or 0
        in_desc = _phrase_in(ctext, name)
        assessed = assess_l.get(name, -1)
        strong = (prof in ("advanced", "expert") and dur > 0 and in_desc)
        verified = assessed >= 70
        if strong or verified:
            backed += 1
    # must-have phrases appearing directly in career descriptions (full-phrase match)
    desc_hits = sum(1 for m in must if _phrase_in(ctext, m))
    raw = backed * 1.0 + 0.25 * desc_hits
    return min(raw / 6.0, 1.0), backed, listed

def shipping_evidence(c, spec):
    """Did they BUILD/SHIP a search/ranking/recsys system to real users?
    Catches plain-language Tier-5s who never say 'RAG'."""
    ctext = career_text(c)
    hits = 0
    for verb in ("built", "shipped", "deployed", "launched", "developed",
                 "designed", "owned", "led", "scaled", "improved", "optimized"):
        for obj in spec["shipping_terms"]:
            if re.search(verb + r"[^.]{0,60}" + re.escape(obj), ctext):
                hits += 1
    ctx = sum(1 for t in spec["shipping_context"] if t in ctext)
    raw = hits + 0.3 * ctx
    return min(raw / 4.0, 1.0)

def seniority_fit(c, spec):
    yoe = c.get("profile", {}).get("years_of_experience", 0) or 0
    lo, hi = spec["ideal_years"]
    peak = spec["ideal_years_peak"]
    if lo <= yoe <= hi:
        base = 1.0
    else:
        base = math.exp(-((yoe - peak) ** 2) / (2 * 3.0 ** 2))
    title = (c.get("profile", {}).get("current_title", "") or "").lower()
    bonus = 0.12 if any(t in title for t in spec["senior_titles"]) else 0.0
    return min(base + bonus, 1.0)

def product_ratio(c, spec):
    """Share of career at product companies vs services/consulting."""
    svc_comp = spec["services_companies"]; svc_ind = spec["services_industries"]
    total = 0; svc = 0
    for j in c.get("career_history", []):
        dur = j.get("duration_months", 0) or 0
        total += dur
        comp = (j.get("company", "") or "").lower()
        ind = (j.get("industry", "") or "").lower()
        if any(s in comp for s in svc_comp) or any(s in ind for s in svc_ind):
            svc += dur
    if total == 0:
        return 0.5
    return 1.0 - (svc / total)

def domain_fit(c, spec):
    text = career_text(c) + " " + (c.get("profile", {}).get("summary", "") or "").lower()
    nlp_ir = any(_phrase_in(text, t) for t in spec["nlp_ir_markers"])
    cv = any(_phrase_in(text, t) for t in spec["cv_speech_robotics"])
    if nlp_ir and not cv:
        return 1.0
    if nlp_ir and cv:
        return 0.7
    if cv and not nlp_ir:
        return 0.2
    return 0.5

def location_fit(c, spec):
    p = c.get("profile", {})
    country = (p.get("country") or "").lower()
    loc = (p.get("location") or "").lower()
    if country == "india":
        if any(city in loc for city in spec["india_tier1_cities"]):
            return 1.0
        return 0.8
    willing = (c.get("redrob_signals", {}) or {}).get("willing_to_relocate", False)
    return 0.45 if willing else 0.1

def platform_traction(c):
    """Market validation from Redrob signals recruiters actually act on:
    saved_by_recruiters, search_appearance, profile_views, endorsements.
    Distinct from availability (which is about reachability). In [0,1]."""
    s = c.get("redrob_signals", {}) or {}
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    appear = s.get("search_appearance_30d", 0) or 0
    views = s.get("profile_views_received_30d", 0) or 0
    endo = s.get("endorsements_received", 0) or 0
    # squashing each to ~[0,1] with gentle saturation
    f = lambda x, k: 1.0 - math.exp(-max(x, 0) / k)
    score = (0.40 * f(saved, 6) + 0.25 * f(appear, 40) +
             0.20 * f(views, 30) + 0.15 * f(endo, 40))
    return min(score, 1.0)
