"""Disqualifiers, consistency checks, and honeypot detection."""
from src.load import career_text
from src.config import SNAPSHOT_DATE

def disqualifier_penalty(c, spec):
    """Heavy penalties for JD-stated disqualifiers. Returns (penalty, reasons[])."""
    pen = 0.0
    reasons = []
    p = c.get("profile", {})
    title = (p.get("current_title") or "").lower()
    ctext = career_text(c)
    hist = c.get("career_history", [])

    # consulting-only entire career (currently services AND no product company ever)
    svc_comp = spec["services_companies"]; svc_ind = spec["services_industries"]
    def is_services(j):
        return (any(s in (j.get("company") or "").lower() for s in svc_comp) or
                any(s in (j.get("industry") or "").lower() for s in svc_ind))
    if hist and all(is_services(j) for j in hist):
        pen += 0.45; reasons.append("consulting/services-only career")

    # research-only, no production language
    if any(m in ctext for m in spec["research_only_markers"]) and not any(
            t in ctext for t in ("production", "deployed", "shipped", "users", "at scale")):
        pen += 0.30; reasons.append("research-only, no production")

    # CV/speech/robotics primary with no NLP/IR
    has_cv = any(m in ctext for m in spec["cv_speech_robotics"])
    has_nlp = any(m in ctext for m in spec["nlp_ir_markers"])
    if has_cv and not has_nlp:
        pen += 0.25; reasons.append("CV/speech/robotics-only, no NLP/IR")

    # title-chaser: many short stints climbing seniority
    short = [j for j in hist if (j.get("duration_months", 0) or 0) < 20]
    if len(hist) >= 4 and len(short) >= 3:
        pen += 0.15; reasons.append("frequent <1.7y job changes")

    # outside India, not relocatable
    if (p.get("country") or "").lower() != "india" and not c.get("redrob_signals", {}).get("willing_to_relocate", False):
        pen += 0.25; reasons.append("outside India, not relocatable")

    return min(pen, 0.8), reasons

def consistency_and_honeypot(c):
    """Returns (consistency_penalty, honeypot_flag, reasons[])."""
    pen = 0.0
    reasons = []
    flag = False
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0) or 0
    hist = c.get("career_history", [])
    skills = c.get("skills", [])

    # tenure sum wildly exceeds stated experience
    tenure_months = sum((j.get("duration_months", 0) or 0) for j in hist)
    if tenure_months > yoe * 12 + 24:
        pen += 0.4
        reasons.append("career tenure exceeds stated experience")
        if tenure_months > yoe * 12 + 60:
            flag = True

    # expert/advanced skills claimed with zero usage duration
    zero_expert = [s for s in skills
                   if s.get("proficiency") in ("expert", "advanced") and (s.get("duration_months", 0) or 0) == 0]
    if len(zero_expert) >= 5:
        flag = True
        reasons.append("many expert skills with zero usage (impossible profile)")
    elif len(zero_expert) >= 2:
        pen += 0.15
        reasons.append("expert skills with zero usage")

    # impossible single stint (duration longer than dates allow + slack)
    for j in hist:
        dur = j.get("duration_months", 0) or 0
        sd = j.get("start_date"); ed = j.get("end_date")
        try:
            sy, sm = int(sd[:4]), int(sd[5:7])
            if ed:
                ey, em = int(ed[:4]), int(ed[5:7])
            else:
                ey, em = SNAPSHOT_DATE.year, SNAPSHOT_DATE.month
            span = (ey - sy) * 12 + (em - sm)
            if dur > span + 2:
                pen += 0.2
                reasons.append("stint duration impossible vs dates")
                break
        except Exception:
            pass

    return min(pen, 0.7), flag, reasons
