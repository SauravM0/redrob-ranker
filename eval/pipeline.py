#!/usr/bin/env python3
"""OFFLINE: extract all features + proxy tiers once, cache to artifacts/feature_cache.npz.
Used only for local tuning/evaluation. The submission itself is produced by rank.py."""
import os, sys, time, yaml
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.load import load_candidates, evidence_doc
from src import features as F, traps as T, signals as S
from rank import semantic_scores
from eval.relevance import proxy_tier

COMP_KEYS = ["skill_evidence","shipping","sem_fit","seniority","product_ratio",
             "domain","traction","lexical","location"]

def extract(data_path, cache_path):
    t0=time.time()
    cands = load_candidates(data_path)
    print(f"[load] {len(cands)} in {time.time()-t0:.1f}s")
    docs = [evidence_doc(c) for c in cands]
    spec = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"jd_spec.yaml")))
    sem, lex = semantic_scores(cands, docs, spec)
    print(f"[semantic] {time.time()-t0:.1f}s")

    N=len(cands)
    comp = {k: np.zeros(N, dtype=np.float32) for k in COMP_KEYS}
    dq = np.zeros(N, np.float32); cons = np.zeros(N, np.float32)
    avail = np.zeros(N, np.float32); hp = np.zeros(N, bool)
    tiers = np.zeros(N, np.int8)
    ids = []
    for i,c in enumerate(cands):
        se,backed,_ = F.skill_evidence(c,spec)
        dqp,dqr = T.disqualifier_penalty(c,spec)
        cp,h,tr = T.consistency_and_honeypot(c)
        av,_ = S.availability_multiplier(c)
        f = {
            "skill_evidence":se, "shipping":F.shipping_evidence(c,spec),
            "sem_fit":float(sem[i]), "seniority":F.seniority_fit(c,spec),
            "product_ratio":F.product_ratio(c,spec), "domain":F.domain_fit(c,spec),
            "traction":F.platform_traction(c), "lexical":float(lex[i]),
            "location":F.location_fit(c,spec),
            "dq_penalty":dqp, "consistency_penalty":cp, "availability":av, "honeypot":h,
        }
        for k in COMP_KEYS: comp[k][i]=f[k]
        dq[i]=dqp; cons[i]=cp; avail[i]=av; hp[i]=h
        tiers[i]=proxy_tier(c,f)
        ids.append(c["candidate_id"])
        if i and i%20000==0: print(f"  {i} … {time.time()-t0:.1f}s")
    np.savez(cache_path,
             ids=np.array(ids),
             tiers=tiers, dq=dq, cons=cons, avail=avail, hp=hp,
             **{k:comp[k] for k in COMP_KEYS})
    print(f"[cache] wrote {cache_path} in {time.time()-t0:.1f}s")
    # quick tier histogram
    import collections
    print("[proxy tiers]", dict(sorted(collections.Counter(tiers.tolist()).items())))

if __name__=="__main__":
    data=sys.argv[1]
    out=sys.argv[2] if len(sys.argv)>2 else "artifacts/feature_cache.npz"
    extract(data,out)
