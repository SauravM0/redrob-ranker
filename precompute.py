#!/usr/bin/env python3
"""
OPTIONAL offline precompute (run ONCE, WITH internet). Upgrades semantic fit from
TF-IDF to dense embeddings (bge-small-en-v1.5). Stored as float16 (~77 MB for 100K)
so the index fits under GitHub's 100 MB file limit with NO Git LFS needed.
Precompute may exceed the 5-min budget — that is allowed. rank.py then loads
artifacts/ and ranks offline within budget.

    pip install sentence-transformers torch
    python precompute.py --candidates ./candidates.jsonl

Produces: artifacts/embeddings.npy (float16), artifacts/id_index.json, artifacts/jd_vectors.npy
After precompute, RE-TUNE for the dense path (one command each):
    python eval/pipeline.py ./candidates.jsonl artifacts/feature_cache.npz
    python tune.py artifacts/feature_cache.npz
"""
import argparse, json, os, yaml
import numpy as np
from src.load import load_candidates, evidence_doc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--spec", default="jd_spec.yaml")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    os.makedirs("artifacts", exist_ok=True)
    spec = yaml.safe_load(open(args.spec))
    cands = load_candidates(args.candidates)
    docs = [evidence_doc(c) for c in cands]
    model = SentenceTransformer(args.model)  # CPU is fine
    emb = model.encode(docs, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True).astype("float16")   # float16 → small file
    np.save("artifacts/embeddings.npy", emb)
    json.dump([c["candidate_id"] for c in cands], open("artifacts/id_index.json", "w"))
    qprefix = "Represent this sentence for searching relevant passages: "
    qv = model.encode([qprefix + spec["jd_query"], qprefix + spec["ideal_query"],
                       qprefix + spec["must_query"]], normalize_embeddings=True).astype("float16")
    np.save("artifacts/jd_vectors.npy", qv)
    mb = emb.nbytes / 1e6
    print(f"Wrote artifacts/embeddings.npy ({mb:.0f} MB, float16). rank.py now uses dense embeddings.")

if __name__ == "__main__":
    main()
