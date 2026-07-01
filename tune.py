#!/usr/bin/env python3
"""OFFLINE: tune composite weights to maximize the challenge metric
(0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10) against the proxy tiers.
Coordinate ascent with mild L2-to-prior regularization to avoid overfitting any
one rule. Writes src/weights.json (committed) + prints a before/after table.

    python tune.py                      # uses artifacts/feature_cache.npz
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.score import DEFAULT_WEIGHTS
from eval.metrics import composite_score

COMP_KEYS = list(DEFAULT_WEIGHTS.keys())

def load_cache(path):
    z = np.load(path, allow_pickle=True)
    comp = np.stack([z[k] for k in COMP_KEYS], axis=1).astype(np.float64)  # (N,K)
    return comp, z["dq"], z["cons"], z["avail"], z["hp"], z["tiers"].astype(np.int64), z["ids"]

def score_all(comp, dq, cons, avail, hp, w):
    base = comp @ w
    base = base - dq - cons
    base = np.clip(base, 0.0, 1.0)
    final = base * avail
    final = np.clip(final, 0.0, 1.0)
    final = np.where(hp, -1.0, final)
    return final

def evaluate(final, ids, tiers, k=100):
    order = np.lexsort((ids, -final))[:max(k,50)]  # -final desc, tie-break id asc
    comp, parts = composite_score(order, tiers, k_map=k)
    return comp, parts

def main():
    cache = sys.argv[1] if len(sys.argv)>1 else "artifacts/feature_cache.npz"
    comp, dq, cons, avail, hp, tiers, ids = load_cache(cache)
    # integer-encode ids for a stable ascending tie-break
    id_rank = np.argsort(np.argsort(ids))

    prior = np.array([DEFAULT_WEIGHTS[k] for k in COMP_KEYS], dtype=np.float64)
    w = prior.copy()
    lam = float(sys.argv[2]) if len(sys.argv)>2 else 0.07  # reg toward prior

    def objective(w):
        final = score_all(comp, dq, cons, avail, hp, w)
        c,_ = evaluate(final, id_rank, tiers)
        return c - lam*np.sum((w-prior)**2)

    base_final = score_all(comp, dq, cons, avail, hp, prior)
    base_c, base_parts = evaluate(base_final, id_rank, tiers)
    print(f"[prior]  composite={base_c:.4f}  {fmt(base_parts)}")

    steps = [0.03, 0.015, 0.0075]
    best = objective(w)
    for st in steps:
        improved = True
        while improved:
            improved = False
            for j in range(len(w)):
                for delta in (st, -st):
                    cand = w.copy(); cand[j]=max(0.0, cand[j]+delta)
                    cand = cand/ cand.sum()  # keep sum=1
                    o = objective(cand)
                    if o > best + 1e-6:
                        w, best = cand, o; improved=True
    w = w/w.sum()
    tuned_final = score_all(comp, dq, cons, avail, hp, w)
    tuned_c, tuned_parts = evaluate(tuned_final, id_rank, tiers)

    out = {k: round(float(w[i]),4) for i,k in enumerate(COMP_KEYS)}
    json.dump(out, open(os.path.join("src","weights.json"),"w"), indent=2)
    print(f"[tuned]  composite={tuned_c:.4f}  {fmt(tuned_parts)}")
    print("[weights]", json.dumps(out))
    print(f"[delta]  +{(tuned_c-base_c):.4f} composite vs prior")
    # honeypots in top-100 sanity
    order = np.lexsort((id_rank, -tuned_final))[:100]
    print(f"[honeypots in top100] {int(hp[order].sum())}")

def fmt(p):
    return " ".join(f"{k}={v:.3f}" for k,v in p.items())

if __name__=="__main__":
    main()
