"""
backend/evaluation/evaluate_humanized.py

End-to-end evaluation of the Neuro-Symbolic Clinical Reasoning Engine
and RAG Retrieval Pipeline against 30 humanized clinical vignettes.

Usage:
    python backend/evaluation/evaluate_humanized.py

Outputs:
    evaluation_results/humanized/evaluation_humanized.json
    evaluation_results/humanized/per_case_metrics.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ─── Path bootstrap: allow running from repo root or backend/ ────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
_BACKEND = _ROOT / "backend"

for p in (_BACKEND, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(_BACKEND)

# ─── Imports ─────────────────────────────────────────────────────────────────
from app.reasoning.engine import SymbolicRuleEngine
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embeddings_service

# ─── Constants ───────────────────────────────────────────────────────────────
CASES_FILE = Path(__file__).parent / "humanized_cases.json"
OUT_DIR = _ROOT / "evaluation_results" / "humanized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAG_TOP_K = 5
RAG_MIN_SCORE = 0.0


# ─── Helper: Precision / Recall / F1 ─────────────────────────────────────────
def prf1(predicted: Set[str], expected: Set[str]):
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if len(expected) == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


# ─── Helper: RAG relevance scoring ───────────────────────────────────────────
def compute_rag_relevance(chunks, keywords: List[str]) -> Dict[str, Any]:
    """Score retrieved RAG chunks for keyword coverage."""
    if not chunks:
        return {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0,
                "mrr": 0.0, "avg_score": 0.0, "keyword_coverage": 0.0,
                "retrieved_count": 0}

    kw_lower = [k.lower() for k in keywords]
    scores = [c.score for c in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Check keyword coverage
    all_text = " ".join(c.content.lower() for c in chunks)
    covered = sum(1 for kw in kw_lower if kw in all_text)
    kw_coverage = covered / len(kw_lower) if kw_lower else 1.0

    # Hit@k: at least one chunk with any keyword match
    def has_hit(chunk_list):
        combined = " ".join(c.content.lower() for c in chunk_list)
        return any(kw in combined for kw in kw_lower)

    hit1 = 1 if has_hit(chunks[:1]) else 0
    hit3 = 1 if has_hit(chunks[:3]) else 0
    hit5 = 1 if has_hit(chunks[:5]) else 0

    # MRR: rank of first relevant chunk
    mrr = 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if any(kw in chunk.content.lower() for kw in kw_lower):
            mrr = 1.0 / rank
            break

    return {
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "mrr": round(mrr, 4),
        "avg_score": round(avg_score, 4),
        "keyword_coverage": round(kw_coverage, 4),
        "retrieved_count": len(chunks),
    }


# ─── Main Evaluation ──────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 70)
    print("  Neuro-Symbolic Humanized Evaluation Pipeline")
    print("  Rules Engine + RAG Retrieval | 30 Clinical Vignettes")
    print("=" * 70)

    # Load humanized cases
    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"\n[INFO] Loaded {len(cases)} humanized test cases from {CASES_FILE.name}")

    # Initialise engine
    print("[INFO] Initialising Symbolic Rule Engine...")
    t_engine_start = time.monotonic()
    engine = SymbolicRuleEngine()
    print(f"[INFO] Rule engine ready with {len(engine.rules)} rules "
          f"(init: {(time.monotonic()-t_engine_start)*1000:.0f}ms)")

    # Initialise RAG retriever
    print("[INFO] Connecting to ChromaDB vector store...")
    try:
        vs = get_vector_store()
        doc_count = vs.count()
        rag_available = doc_count > 0
        if rag_available:
            retriever = RAGRetriever(vector_store=vs)
            print(f"[INFO] ChromaDB online — {doc_count:,} chunks available for retrieval")
        else:
            retriever = None
            print("[WARN] ChromaDB is empty — RAG evaluation will be skipped")
    except Exception as exc:
        retriever = None
        rag_available = False
        print(f"[WARN] RAG initialisation failed: {exc}")

    # Per-case results storage
    per_case = []
    all_rule_tp = all_rule_fp = all_rule_fn = 0
    all_fact_tp = all_fact_fp = all_fact_fn = 0
    alert_hits = alert_total = 0
    rag_hits1 = rag_hits3 = rag_hits5 = 0
    rag_mrr_sum = 0.0
    rag_avg_score_sum = 0.0
    rag_kw_cov_sum = 0.0
    rag_case_count = 0

    domain_stats: Dict[str, Dict] = {}
    case_type_stats: Dict[str, Dict] = {}

    print(f"\n{'='*70}")
    print(f"  {'Case':10} {'Domain':20} {'Rule P':7} {'Rule R':7} {'Rule F1':8} {'Alerts':7} {'RAG Hit@5':9}")
    print(f"{'='*70}")

    for case in cases:
        cid = case["case_id"]
        narrative = case["narrative"]
        patient_facts = case["patient_facts"]
        exp_rules = set(r.strip() for r in case["expected_rules"] if r)
        exp_facts = set(f.strip() for f in case["expected_facts"] if f)
        exp_alerts = set(a.strip() for a in case["expected_alerts"] if a)
        exp_rag_kw = case.get("expected_rag_keywords", [])
        domain = case.get("domain", "unknown")
        case_type = case.get("case_type", "unknown")

        # ── 1. Symbolic Rule Engine Evaluation ──────────────────────────────
        try:
            result = engine.evaluate(patient_facts)
            pred_rules = set(r.rule_id for r in result.triggered_rules)
            pred_facts = set(f.name for f in result.derived_facts)
            pred_alerts = set(a.name for a in result.critical_alerts)
        except Exception as exc:
            print(f"  [ERROR] {cid}: Engine error — {exc}")
            pred_rules = set()
            pred_facts = set()
            pred_alerts = set()

        rp, rr, rf1, rtp, rfp, rfn = prf1(pred_rules, exp_rules)
        fp2, fr2, ff1, ftp, ffp, ffn = prf1(pred_facts, exp_facts)
        alert_matched = len(pred_alerts & exp_alerts)
        alert_hits += alert_matched
        alert_total += len(exp_alerts)

        all_rule_tp += rtp
        all_rule_fp += rfp
        all_rule_fn += rfn
        all_fact_tp += ftp
        all_fact_fp += ffp
        all_fact_fn += ffn

        # ── 2. RAG Retrieval Evaluation ──────────────────────────────────────
        rag_metrics: Dict[str, Any] = {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0,
                                        "mrr": 0.0, "avg_score": 0.0,
                                        "keyword_coverage": 0.0, "retrieved_count": 0}
        if rag_available and retriever and exp_rag_kw:
            try:
                # Build a query from narrative + expected keywords
                query = " ".join(exp_rag_kw[:5])
                rag_res = retriever.retrieve(query=query, top_k=RAG_TOP_K)
                rag_metrics = compute_rag_relevance(rag_res.results, exp_rag_kw)
                rag_hits1 += rag_metrics["hit_at_1"]
                rag_hits3 += rag_metrics["hit_at_3"]
                rag_hits5 += rag_metrics["hit_at_5"]
                rag_mrr_sum += rag_metrics["mrr"]
                rag_avg_score_sum += rag_metrics["avg_score"]
                rag_kw_cov_sum += rag_metrics["keyword_coverage"]
                rag_case_count += 1
            except Exception as exc:
                print(f"  [WARN] {cid}: RAG retrieval failed — {exc}")

        # ── Domain & Case-type accumulation ─────────────────────────────────
        for bucket, key in [(domain_stats, domain), (case_type_stats, case_type)]:
            if key not in bucket:
                bucket[key] = {"rule_tp": 0, "rule_fp": 0, "rule_fn": 0,
                                "fact_tp": 0, "fact_fp": 0, "fact_fn": 0,
                                "cases": 0, "exact_match": 0}
            bucket[key]["rule_tp"] += rtp
            bucket[key]["rule_fp"] += rfp
            bucket[key]["rule_fn"] += rfn
            bucket[key]["fact_tp"] += ftp
            bucket[key]["fact_fp"] += ffp
            bucket[key]["fact_fn"] += ffn
            bucket[key]["cases"] += 1
            if pred_rules == exp_rules and pred_facts == exp_facts:
                bucket[key]["exact_match"] += 1

        # ── Print row ────────────────────────────────────────────────────────
        h5 = rag_metrics["hit_at_5"]
        alert_ok = "OK" if alert_matched == len(exp_alerts) else "MISS"
        print(f"  {cid:10} {domain:20} {rp:.3f}   {rr:.3f}   {rf1:.3f}    "
              f"{alert_ok:{6}}  {h5}")

        per_case.append({
            "case_id": cid,
            "domain": domain,
            "case_type": case_type,
            "narrative_excerpt": narrative[:80].replace("\n", " ") + "...",
            "expected_rules": sorted(exp_rules),
            "predicted_rules": sorted(pred_rules),
            "rule_precision": round(rp, 4),
            "rule_recall": round(rr, 4),
            "rule_f1": round(rf1, 4),
            "rule_tp": rtp, "rule_fp": rfp, "rule_fn": rfn,
            "expected_facts": sorted(exp_facts),
            "predicted_facts": sorted(pred_facts),
            "fact_precision": round(fp2, 4),
            "fact_recall": round(fr2, 4),
            "fact_f1": round(ff1, 4),
            "expected_alerts": sorted(exp_alerts),
            "predicted_alerts": sorted(pred_alerts),
            "alert_detected": alert_matched == len(exp_alerts),
            "exact_match": pred_rules == exp_rules and pred_facts == exp_facts,
            **{f"rag_{k}": v for k, v in rag_metrics.items()},
        })

    print(f"{'='*70}\n")

    # ─── Aggregate Metrics ────────────────────────────────────────────────────
    n = len(cases)

    def micro_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    rule_micro_p, rule_micro_r, rule_micro_f1 = micro_f1(all_rule_tp, all_rule_fp, all_rule_fn)
    fact_micro_p, fact_micro_r, fact_micro_f1 = micro_f1(all_fact_tp, all_fact_fp, all_fact_fn)

    macro_rule_p = round(sum(c["rule_precision"] for c in per_case) / n, 4)
    macro_rule_r = round(sum(c["rule_recall"] for c in per_case) / n, 4)
    macro_rule_f1 = round(sum(c["rule_f1"] for c in per_case) / n, 4)
    macro_fact_f1 = round(sum(c["fact_f1"] for c in per_case) / n, 4)

    exact_match_count = sum(1 for c in per_case if c["exact_match"])
    exact_match_rate = round(exact_match_count / n, 4)

    alert_recall = round(alert_hits / alert_total, 4) if alert_total > 0 else 1.0

    rag_hit1_rate = round(rag_hits1 / rag_case_count, 4) if rag_case_count > 0 else 0.0
    rag_hit3_rate = round(rag_hits3 / rag_case_count, 4) if rag_case_count > 0 else 0.0
    rag_hit5_rate = round(rag_hits5 / rag_case_count, 4) if rag_case_count > 0 else 0.0
    rag_mean_mrr = round(rag_mrr_sum / rag_case_count, 4) if rag_case_count > 0 else 0.0
    rag_mean_score = round(rag_avg_score_sum / rag_case_count, 4) if rag_case_count > 0 else 0.0
    rag_mean_kw_cov = round(rag_kw_cov_sum / rag_case_count, 4) if rag_case_count > 0 else 0.0

    # Domain summaries
    domain_summary = {}
    for d, s in domain_stats.items():
        dp, dr, df1 = micro_f1(s["rule_tp"], s["rule_fp"], s["rule_fn"])
        domain_summary[d] = {
            "cases": s["cases"],
            "exact_match_rate": round(s["exact_match"] / s["cases"], 4),
            "rule_precision": dp, "rule_recall": dr, "rule_f1": df1,
        }

    # Case type summaries
    type_summary = {}
    for t, s in case_type_stats.items():
        tp2, tr2, tf1 = micro_f1(s["rule_tp"], s["rule_fp"], s["rule_fn"])
        type_summary[t] = {
            "cases": s["cases"],
            "exact_match_rate": round(s["exact_match"] / s["cases"], 4),
            "rule_precision": tp2, "rule_recall": tr2, "rule_f1": tf1,
        }

    # ─── Print summary ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SYMBOLIC RULE ENGINE RESULTS")
    print("=" * 70)
    print(f"  Total Cases Evaluated    : {n}")
    print(f"  Rule Micro Precision     : {rule_micro_p:.4f}  ({rule_micro_p*100:.1f}%)")
    print(f"  Rule Micro Recall        : {rule_micro_r:.4f}  ({rule_micro_r*100:.1f}%)")
    print(f"  Rule Micro F1            : {rule_micro_f1:.4f}  ({rule_micro_f1*100:.1f}%)")
    print(f"  Rule Macro Precision     : {macro_rule_p:.4f}  ({macro_rule_p*100:.1f}%)")
    print(f"  Rule Macro Recall        : {macro_rule_r:.4f}  ({macro_rule_r*100:.1f}%)")
    print(f"  Rule Macro F1            : {macro_rule_f1:.4f}  ({macro_rule_f1*100:.1f}%)")
    print(f"  Derived Fact Macro F1    : {macro_fact_f1:.4f}  ({macro_fact_f1*100:.1f}%)")
    print(f"  Subset Exact Match       : {exact_match_count}/{n} ({exact_match_rate*100:.1f}%)")
    print(f"  Alert Recall             : {alert_recall:.4f}  ({alert_recall*100:.1f}%)")
    print(f"\n  RAG RETRIEVAL RESULTS ({rag_case_count} cases evaluated)")
    print(f"  Hit@1                    : {rag_hit1_rate:.4f}  ({rag_hit1_rate*100:.1f}%)")
    print(f"  Hit@3                    : {rag_hit3_rate:.4f}  ({rag_hit3_rate*100:.1f}%)")
    print(f"  Hit@5                    : {rag_hit5_rate:.4f}  ({rag_hit5_rate*100:.1f}%)")
    print(f"  Mean Reciprocal Rank     : {rag_mean_mrr:.4f}")
    print(f"  Mean Similarity Score    : {rag_mean_score:.4f}")
    print(f"  Keyword Coverage         : {rag_mean_kw_cov:.4f}  ({rag_mean_kw_cov*100:.1f}%)")
    print("=" * 70)

    # ─── Save evaluation_humanized.json ──────────────────────────────────────
    results_json = {
        "evaluation_type": "humanized_vignettes",
        "total_cases": n,
        "rag_available": rag_available,
        "rag_cases_evaluated": rag_case_count,
        "symbolic_rule_engine": {
            "micro_precision": rule_micro_p,
            "micro_recall": rule_micro_r,
            "micro_f1": rule_micro_f1,
            "macro_precision": macro_rule_p,
            "macro_recall": macro_rule_r,
            "macro_f1": macro_rule_f1,
            "derived_fact_macro_f1": macro_fact_f1,
            "subset_exact_match": exact_match_rate,
            "exact_match_count": exact_match_count,
            "alert_recall": alert_recall,
            "rule_tp": all_rule_tp,
            "rule_fp": all_rule_fp,
            "rule_fn": all_rule_fn,
        },
        "rag_retrieval": {
            "hit_at_1": rag_hit1_rate,
            "hit_at_3": rag_hit3_rate,
            "hit_at_5": rag_hit5_rate,
            "mean_reciprocal_rank": rag_mean_mrr,
            "mean_similarity_score": rag_mean_score,
            "mean_keyword_coverage": rag_mean_kw_cov,
        },
        "by_domain": domain_summary,
        "by_case_type": type_summary,
        "per_case": per_case,
    }

    out_json = OUT_DIR / "evaluation_humanized.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n[SAVED] {out_json}")

    # ─── Save per_case_metrics.csv ────────────────────────────────────────────
    out_csv = OUT_DIR / "per_case_metrics.csv"
    if per_case:
        fieldnames = list(per_case[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_case:
                writer.writerow(row)
        print(f"[SAVED] {out_csv}")

    print(f"\n[DONE] Results saved to {OUT_DIR}")
    return results_json


if __name__ == "__main__":
    run_evaluation()
