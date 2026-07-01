#!/usr/bin/env python3
"""Ablation: prove evidence-grounding beats naive keyword ranking, on BOTH
trap-resistance and NDCG. Uses the cached features (fast) + one streaming pass
for titles / naive keyword counts.

    python eval/evaluate.py <candidates.jsonl> [artifacts/feature_cache.npz]
"""
import os, sys, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.score import DEFAULT_WEIGHTS, load_weights
from eval.metrics import composite_score

COMP_KEYS = list(DEFAULT_WEIGHTS.keys())
TRAP_TITLES = ["hr manager","content writer","business analyst","accountant","sales executive",
               "graphic designer","marketing manager","customer support","operations manager",
               "mechanical engineer","civil engineer","project manager"]
MUST = ["embeddings","retrieval","vector","faiss","pinecone","weaviate","qdrant","milvus",
        "elasticsearch","opensearch","sentence-transformers","semantic search","hybrid search",
        "ranking","recommendation","recsys","learning to rank","information retrieval","nlp","ndcg","python"]

def stream_aux(path, n):
    trap = np.zeros(n, bool); naive = np.zeros(n, np.int32); i=0
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            d=json.loads(line)
            t=(d.get("profile",{}).get("current_title","") or "").lower()
            trap[i]=any(x in t for x in TRAP_TITLES)
            naive[i]=sum(1 for s in d.get("skills",[]) if any(m in (s.get("name","") or "").lower() for m in MUST))
            i+=1
    return trap, naive

def order_of(scores, ids):
    return np.lexsort((ids, -scores))

def report(name, order, tiers, trap, hp):
    top=order[:100]
    comp,parts=composite_score(order, tiers)
    print(f"  {name:<34} traps={int(trap[top].sum()):>3}  honeypots={int(hp[top].sum()):>2}  "
          f"NDCG@10={parts['NDCG@10']:.3f} NDCG@50={parts['NDCG@50']:.3f} MAP={parts['MAP']:.3f} P@10={parts['P@10']:.3f}")

def main():
    data=sys.argv[1]
    cache=sys.argv[2] if len(sys.argv)>2 else "artifacts/feature_cache.npz"
    z=np.load(cache, allow_pickle=True)
    comp=np.stack([z[k] for k in COMP_KEYS],axis=1).astype(np.float64)
    dq,cons,avail,hp,tiers,ids=z["dq"],z["cons"],z["avail"],z["hp"],z["tiers"].astype(np.int64),z["ids"]
    idr=np.argsort(np.argsort(ids))
    trap,naive=stream_aux(data,len(ids))

    w=np.array([load_weights()[k] for k in COMP_KEYS])
    def sc(weights):
        base=np.clip(comp@weights - dq - cons,0,1)*avail
        return np.where(hp,-1.0,np.clip(base,0,1))
    tuned=sc(w)
    # ablated: drop skill_evidence + shipping, renormalize remaining
    wa=w.copy(); wa[0]=0; wa[1]=0; wa=wa/wa.sum()
    ablated=np.clip(comp@wa - dq - cons,0,1)*avail
    ablated=np.where(hp,-1.0,np.clip(ablated,0,1))
    naive_f=naive.astype(np.float64)  # pure keyword-count, like sample_submission

    print("Top-100 comparison (relevant = proxy tier >= 3):")
    report("naive keyword-count (sample-style)", order_of(naive_f,idr), tiers, trap, hp)
    report("ablated (no evidence/shipping)",     order_of(ablated,idr),  tiers, trap, hp)
    report("OURS (evidence-grounded, tuned)",    order_of(tuned,idr),    tiers, trap, hp)

if __name__=="__main__":
    main()
