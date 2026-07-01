"""Composite scoring with tunable weights. Final score is bounded to [0,1] so
the submission never emits scores > 1 (validator wants a float, non-increasing;
[0,1] is cleaner and availability is applied INSIDE the cap)."""
import json, os

# Default weights (overridden by src/weights.json if present, produced by tune.py).
DEFAULT_WEIGHTS = {
    "skill_evidence": 0.24,
    "shipping":       0.20,
    "sem_fit":        0.16,
    "seniority":      0.11,
    "product_ratio":  0.09,
    "domain":         0.07,
    "traction":       0.05,
    "lexical":        0.05,
    "location":       0.03,
}
HONEYPOT_FLOOR = -1.0

def load_weights():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")
    if os.path.exists(path):
        try:
            w = json.load(open(path))
            # only keep known keys, fall back for any missing
            return {k: float(w.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)

def composite(f, weights=None):
    w = weights or load_weights()
    base = sum(w[k] * f[k] for k in w)
    base -= f["dq_penalty"]
    base -= f["consistency_penalty"]
    base = max(0.0, min(base, 1.0))
    # availability modulates fit but stays within [0,1]: a fully-available
    # candidate keeps base; an unreachable one is scaled down.
    final = base * f["availability"]
    final = max(0.0, min(final, 1.0))
    if f["honeypot"]:
        final = HONEYPOT_FLOOR
    return final
