"""Ranking metrics: NDCG@k (graded), MAP and P@k (binary, relevant = tier>=3).
Matches the challenge's composite: 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10."""
import numpy as np

def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2 ** rels - 1) * discounts))

def ndcg_at_k(order, tiers, k):
    top = order[:k]
    gains = tiers[top]
    ideal = np.sort(tiers)[::-1][:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

def average_precision(order, rel_binary, k=None):
    if k is not None:
        order = order[:k]
    hits = 0; s = 0.0
    total_rel = int(rel_binary.sum())
    if total_rel == 0:
        return 0.0
    for i, idx in enumerate(order, start=1):
        if rel_binary[idx]:
            hits += 1
            s += hits / i
    return s / min(total_rel, len(order))

def precision_at_k(order, rel_binary, k):
    top = order[:k]
    return float(rel_binary[top].sum()) / k

def composite_score(order, tiers, k_map=100):
    rel = (tiers >= 3).astype(int)
    n10 = ndcg_at_k(order, tiers, 10)
    n50 = ndcg_at_k(order, tiers, 50)
    mp  = average_precision(order, rel, k=k_map)
    p10 = precision_at_k(order, rel, 10)
    comp = 0.50 * n10 + 0.30 * n50 + 0.15 * mp + 0.05 * p10
    return comp, {"NDCG@10": n10, "NDCG@50": n50, "MAP": mp, "P@10": p10}
