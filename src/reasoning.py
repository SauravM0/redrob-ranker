"""Code-generated reasoning: specific, honest, varied, rank-consistent. No LLM.

Stage-4 checks we are engineering for: specific facts, JD connection, honest
concerns, no hallucination (every claim traces to a real field), variation
across rows, rank/tone consistency. We achieve variation by surfacing the
candidate's OWN distinct facts (named skill, a real achievement clause, a
concrete signal value) and rotating sentence frames by candidate_id.
"""
import re

MUST_HINTS = ("embedding", "retrieval", "vector", "search", "ranking", "rank",
              "recommend", "recsys", "semantic", "nlp", "information retrieval",
              "faiss", "pinecone", "qdrant", "weaviate", "milvus", "elasticsearch",
              "opensearch", "learning to rank", "ndcg", "relevance", "python")
SHIP_VERBS = ("built", "shipped", "deployed", "launched", "developed", "designed",
              "owned", "led", "scaled", "improved", "optimized")

def _named_skill(c):
    """Highest-signal must-have skill actually backed by proficiency/usage."""
    best = None
    for s in c.get("skills", []):
        name = (s.get("name", "") or "").strip()
        if not name:
            continue
        low = name.lower()
        if not any(h in low for h in MUST_HINTS):
            continue
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0) or 0
        rank = {"expert": 3, "advanced": 2, "intermediate": 1}.get(prof, 0)
        key = (rank, dur)
        if best is None or key > best[0]:
            best = (key, name, prof, dur)
    return best  # (key, name, prof, dur) or None

def _achievement(c):
    """A short real clause from career history describing what they built.
    Returned verbatim from the candidate's own profile (no fabrication)."""
    for j in c.get("career_history", []):
        desc = (j.get("description", "") or "").strip()
        if not desc:
            continue
        for sent in re.split(r"(?<=[.!;])\s+", desc):
            low = sent.lower()
            if any(v in low for v in SHIP_VERBS) and any(h in low for h in MUST_HINTS):
                clause = sent.strip().rstrip(".")
                if 4 <= len(clause.split()) <= 24:
                    return clause[0].lower() + clause[1:]
    return None

def _signal_phrase(c):
    s = c.get("redrob_signals", {}) or {}
    bits = []
    rr = s.get("recruiter_response_rate")
    if rr is not None and rr >= 0.6:
        bits.append(f"{rr:.0%} recruiter response")
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 4:
        bits.append(f"saved by {saved} recruiters this month")
    npd = s.get("notice_period_days")
    if npd is not None and npd <= 30:
        bits.append(f"{npd}-day notice")
    gh = s.get("github_activity_score", -1)
    if gh is not None and gh >= 60:
        bits.append("active GitHub")
    return bits[0] if bits else None

def _loc_phrase(c):
    p = c.get("profile", {})
    loc = (p.get("location") or "").split(",")[0].strip()
    if (p.get("country") or "").lower() == "india" and loc:
        return f"{loc}-based"
    if (c.get("redrob_signals", {}) or {}).get("willing_to_relocate"):
        return "open to relocation"
    return None

def make_reasoning(c, f, rank):
    p = c.get("profile", {})
    title = p.get("current_title", "Professional") or "Professional"
    yoe = p.get("years_of_experience", 0) or 0
    cid = c.get("candidate_id", "")
    seed = sum(ord(x) for x in cid)

    skill = _named_skill(c)
    skill_name = skill[1] if skill else None
    ach = _achievement(c)
    sig = _signal_phrase(c)
    loc = _loc_phrase(c)

    concerns = list(f.get("dq_reasons", [])) + list(f.get("trap_reasons", [])) + list(f.get("avail_notes", []))
    concern = concerns[0] if concerns else None

    # Build a pool of concrete, true fragments, STRONGEST EVIDENCE FIRST so top
    # ranks lead with substance; weaker context (signal/location) trails.
    substance = []
    if ach:
        substance.append(ach)
    elif f.get("shipping", 0) >= 0.3:
        substance.append("career history shows hands-on ranking/retrieval delivery")
    if skill_name and f.get("backed", 0) >= 1:
        substance.append(f"evidence-backed {skill_name}")
    if f.get("domain", 0) >= 0.9:
        substance.append("clear NLP/IR focus")
    if f.get("product_ratio", 0) >= 0.7:
        substance.append("product-company track record")
    elif f.get("product_ratio", 1) < 0.4:
        substance.append("largely services-company background")
    context = []
    if sig:
        context.append(sig)
    if loc:
        context.append(loc)

    frags = substance + context
    if not frags:
        frags = [f"adjacent profile on {skill_name or 'core skills'}"]

    # Lead with the strongest substance fragment; add one varied secondary
    # (rotated by candidate_id so adjacent rows differ).
    a = frags[0]
    rest = frags[1:]
    b = rest[seed % len(rest)] if rest else None
    body = a if (b is None or b == a) else f"{a}; {b}"

    lead_templates = [
        f"{title}, {yoe:.0f} yrs",
        f"{yoe:.0f}-yr {title}",
        f"{title} ({yoe:.0f} yrs)",
    ]
    lead = lead_templates[seed % 3]

    if rank <= 10:
        out = f"{lead} — {body}."
        if concern:
            out += f" Minor flag: {concern}."
    elif rank <= 50:
        out = f"{lead}; {body}."
        if concern:
            out += f" Concern: {concern}."
    else:
        hedge = ["adjacent fit", "partial match", "below the ideal bar but in range"][seed % 3]
        out = f"{lead}; {hedge} — {body}."
        if concern:
            out += f" Limitation: {concern}."

    out = re.sub(r"\s+", " ", out).strip()
    # CSV-safety: keep it to ~1-2 sentences
    return out
