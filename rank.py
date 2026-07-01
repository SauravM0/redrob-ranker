#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranking — ranking step.
Reproduce:  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
Runs CPU-only, no network. Heavy dense embeddings (optional) are precomputed
offline by precompute.py and loaded from artifacts/; without them, semantic
similarity falls back to in-memory TF-IDF (no download, no network).
"""
import argparse, csv, os, sys, time, yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.load import load_candidates, evidence_doc
from src import features as F
from src import traps as T
from src import signals as S
from src.score import composite
from src.reasoning import make_reasoning

def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def semantic_scores(cands, docs, spec):
    """Return sem_fit and lexical arrays. Uses precomputed dense embeddings if
    available in artifacts/, else TF-IDF (offline, no network)."""
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    emb_path = os.path.join(art, "embeddings.npy")
    if os.path.exists(emb_path) and os.path.exists(os.path.join(art, "jd_vectors.npy")):
        emb = np.load(emb_path).astype(np.float32)   # (N,d) float16 on disk -> float32 here
        jdv = np.load(os.path.join(art, "jd_vectors.npy")).astype(np.float32)  # (3,d)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        jdv = jdv / (np.linalg.norm(jdv, axis=1, keepdims=True) + 1e-9)
        sims = emb @ jdv.T
        sem = sims[:, :2].max(axis=1)
        lex = sims[:, 2]
        return minmax(sem), minmax(lex)
    # Fallback: TF-IDF (fit at ranking time, fully offline)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2),
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(docs)
    q = vec.transform([spec["jd_query"], spec["ideal_query"], spec["must_query"]])
    sims = linear_kernel(X, q)                          # (N, 3)
    sem = sims[:, :2].max(axis=1)
    lex = sims[:, 2]
    return minmax(sem), minmax(lex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "jd_spec.yaml"))
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()
    rank_candidates(args.candidates, args.out, args.spec, args.topk, verbose=True)

def rank_candidates(candidates_path, out_path, spec_path=None, topk=100, verbose=False):
    """Rank candidates and write the submission CSV. Importable (used by app.py)
    so the sandbox runs in-process — no fragile subprocess. Returns the rows."""
    if spec_path is None:
        spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jd_spec.yaml")
    log = (lambda m: print(m)) if verbose else (lambda m: None)

    t0 = time.time()
    spec = yaml.safe_load(open(spec_path))
    cands = load_candidates(candidates_path)
    log(f"[load] {len(cands)} candidates in {time.time()-t0:.1f}s")

    docs = [evidence_doc(c) for c in cands]
    sem, lex = semantic_scores(cands, docs, spec)
    log(f"[semantic] computed in {time.time()-t0:.1f}s")

    feats = []
    for i, c in enumerate(cands):
        se, backed, listed = F.skill_evidence(c, spec)
        dq_pen, dq_reasons = T.disqualifier_penalty(c, spec)
        cons_pen, hp, trap_reasons = T.consistency_and_honeypot(c)
        avail, avail_notes = S.availability_multiplier(c)
        f = {
            "skill_evidence": se,
            "shipping": F.shipping_evidence(c, spec),
            "sem_fit": float(sem[i]),
            "seniority": F.seniority_fit(c, spec),
            "product_ratio": F.product_ratio(c, spec),
            "domain": F.domain_fit(c, spec),
            "traction": F.platform_traction(c),
            "lexical": float(lex[i]),
            "location": F.location_fit(c, spec),
            "dq_penalty": dq_pen, "dq_reasons": dq_reasons,
            "consistency_penalty": cons_pen, "honeypot": hp, "trap_reasons": trap_reasons,
            "availability": avail, "avail_notes": avail_notes,
            "backed": backed,
        }
        f["final"] = composite(f)
        feats.append(f)
    log(f"[features+score] done in {time.time()-t0:.1f}s")

    topk = min(topk, len(cands))
    order = sorted(range(len(cands)),
                   key=lambda i: (-feats[i]["final"], cands[i]["candidate_id"]))
    top = order[:topk]

    # honeypot safety report
    hp_in_top = sum(1 for i in top if feats[i]["honeypot"])
    log(f"[honeypots] in top {topk}: {hp_in_top} ({hp_in_top/max(topk,1)*100:.1f}%)  (DQ if >10%)")

    # build rows; enforce strictly decreasing printed score (NDCG uses order, not magnitude)
    rows = []
    prev = None
    for rank, i in enumerate(top, start=1):
        raw = feats[i]["final"]
        score = round(max(raw, 0.0), 6)
        if prev is not None and score >= prev:
            score = round(prev - 1e-6, 6)
        prev = score
        reasoning = make_reasoning(cands[i], feats[i], rank)
        rows.append((cands[i]["candidate_id"], rank, f"{score:.6f}", reasoning))

    with open(out_path, "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow(r)

    log(f"[done] wrote {out_path} in {time.time()-t0:.1f}s")
    if verbose:
        print("\nTop 10 preview:")
        for cid, rank, score, reason in rows[:10]:
            print(f"  {rank:>3}. {cid}  {score}  {reason[:90]}")
    return rows

if __name__ == "__main__":
    main()
