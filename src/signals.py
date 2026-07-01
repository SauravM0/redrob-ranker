"""Behavioral availability multiplier from the 23 Redrob signals.
Bounded to (0.5, 1.0]: a fully-available candidate is NOT penalized (1.0); an
unreachable one is scaled down. It never inflates a score above its fit base."""
from datetime import date
from src.config import SNAPSHOT_DATE

def _recency_months(d):
    try:
        y, m = int(d[:4]), int(d[5:7])
        return (SNAPSHOT_DATE.year - y) * 12 + (SNAPSHOT_DATE.month - m)
    except Exception:
        return 12

def availability_multiplier(c):
    """Returns (mult in (0.5,1.0], notes[]). Built as a product of small
    penalties so the dominant effect is reachability, not paper quality."""
    s = c.get("redrob_signals", {}) or {}
    notes = []
    m = 1.0

    rr = s.get("recruiter_response_rate")
    if rr is not None:
        m *= (0.7 + 0.3 * rr)            # 0.0 -> 0.70 , 1.0 -> 1.00
        if rr < 0.2: notes.append(f"low recruiter response rate {rr:.0%}")

    rec = _recency_months(s.get("last_active_date", ""))
    if rec >= 6: m *= 0.8; notes.append("inactive 6+ months")
    elif rec >= 3: m *= 0.92

    if s.get("open_to_work_flag") is False: m *= 0.9
    # open_to_work True is the norm; no boost (we only penalize)

    npd = s.get("notice_period_days")
    if npd is not None:
        if npd > 90: m *= 0.9; notes.append(f"notice period {npd}d")
        elif npd > 60: m *= 0.96

    icr = s.get("interview_completion_rate")
    if icr is not None and icr >= 0: m *= (0.9 + 0.1 * icr)

    oar = s.get("offer_acceptance_rate")
    if oar is not None and oar >= 0: m *= (0.95 + 0.05 * oar)

    pcs = s.get("profile_completeness_score")
    if pcs is not None: m *= (0.95 + 0.05 * (pcs / 100.0))

    return max(0.5, min(m, 1.0)), notes
