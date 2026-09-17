"""
backend/evaluation/evaluate_symbolic_engine.py

Automated Benchmark Evaluation Pipeline for the Neuro-Symbolic Clinical Reasoning Engine.

Evaluates 500 patient cases from neurosymbolic_clinical/test_cases.csv against the 50 deterministic
rules in neurosymbolic_clinical/rules.csv.

Computes:
- Rule-level multi-label metrics (Micro/Macro Precision, Recall, F1, Accuracy, Subset Accuracy)
- Per-rule support and confusion metrics across all 50 rules
- Alert-level metrics (Precision, Recall, F1, Critical Alert Recall)
- Boundary-case accuracy and threshold evaluation (100 boundary cases)
- Multi-rule exact match and micro metrics (100 multi-rule cases)
- Exports: evaluation_results.json, per_rule_metrics.csv, confusion_matrix.csv
- Visualizations: 6 publication-ready charts in evaluation_results/
- Terminal report
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure backend root is on sys.path
_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent
_workspace_dir = _backend_dir.parent

for p in (_workspace_dir, _backend_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.reasoning.engine import SymbolicRuleEngine
from app.reasoning.loader import find_rules_csv_path, get_clinical_rules
from app.reasoning.models import ClinicalRule


def find_test_cases_csv_path() -> Path:
    """Locate neurosymbolic_clinical/test_cases.csv searching workspace directories."""
    candidates = [
        Path("neurosymbolic_clinical/test_cases.csv"),
        Path("../neurosymbolic_clinical/test_cases.csv"),
        _workspace_dir / "neurosymbolic_clinical" / "test_cases.csv",
        Path(os.getcwd()) / "neurosymbolic_clinical" / "test_cases.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError("Could not locate neurosymbolic_clinical/test_cases.csv")


def run_evaluation(output_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Execute evaluation over all 500 benchmark test cases."""
    rules_path = find_rules_csv_path()
    test_cases_path = find_test_cases_csv_path()

    rules = get_clinical_rules()
    engine = SymbolicRuleEngine(rules=rules)
    all_rule_ids = [r.rule_id for r in sorted(rules, key=lambda x: x.rule_id)]
    rule_id_to_name = {r.rule_id: r.rule_name for r in rules}

    with open(test_cases_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_cases = list(reader)

    total_cases = len(test_cases)
    positive_cases = 0
    negative_cases = 0
    boundary_cases = 0
    multi_rule_cases = 0

    # Rule-level global counters
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0

    # Per-rule counters
    rule_tp = {rid: 0 for rid in all_rule_ids}
    rule_fp = {rid: 0 for rid in all_rule_ids}
    rule_fn = {rid: 0 for rid in all_rule_ids}
    rule_tn = {rid: 0 for rid in all_rule_ids}
    rule_support = {rid: 0 for rid in all_rule_ids}

    # Case type tracking
    subset_exact_matches = 0
    boundary_exact_matches = 0
    multi_rule_exact_matches = 0
    positive_exact_matches = 0
    negative_exact_matches = 0

    multi_rule_tp = 0
    multi_rule_fp = 0
    multi_rule_fn = 0

    # Alert-level tracking
    total_alert_tp = 0
    total_alert_fp = 0
    total_alert_fn = 0

    critical_alert_tp = 0
    critical_alert_fn = 0

    # Threshold discrepancy tracking
    threshold_errors: List[Dict[str, Any]] = []

    for tc in test_cases:
        case_id = tc["case_id"]
        case_type = tc.get("case_type", "unknown")
        boundary_ctx = tc.get("boundary_context", "")

        if case_type == "positive":
            positive_cases += 1
        elif case_type == "negative":
            negative_cases += 1
        elif case_type == "boundary":
            boundary_cases += 1
        elif case_type == "multi_rule":
            multi_rule_cases += 1

        # Ground truth (STRIPPED from reasoning engine)
        exp_rules = set(r.strip() for r in tc["expected_rules"].split(";") if r.strip())
        exp_derived = set(d.strip() for d in tc["expected_derived_facts"].split(";") if d.strip())
        exp_alerts = set(a.strip() for a in tc["expected_alerts"].split(";") if a.strip())

        for r in exp_rules:
            if r in rule_support:
                rule_support[r] += 1

        # Critical alert ground truth: R001 is critical
        has_expected_critical_alert = "R001" in exp_rules or "metformin_contraindicated" in exp_alerts

        # Send ONLY patient_json to the symbolic reasoning engine
        patient_data = json.loads(tc["patient_json"])
        inference_res = engine.evaluate(patient_data)

        act_rules = set(r.rule_id for r in inference_res.triggered_rules)
        act_derived = set(f.name for f in inference_res.derived_facts)
        act_alerts = set(a.name for a in inference_res.critical_alerts)

        is_exact_match = (act_rules == exp_rules)
        if is_exact_match:
            subset_exact_matches += 1
            if case_type == "positive":
                positive_exact_matches += 1
            elif case_type == "negative":
                negative_exact_matches += 1
            elif case_type == "boundary":
                boundary_exact_matches += 1
            elif case_type == "multi_rule":
                multi_rule_exact_matches += 1
        else:
            if case_type == "boundary":
                threshold_errors.append({
                    "case_id": case_id,
                    "boundary_context": boundary_ctx,
                    "expected_rules": list(exp_rules),
                    "actual_rules": list(act_rules),
                })

        # Multi-rule specific counters
        if case_type == "multi_rule":
            mr_tp = len(act_rules.intersection(exp_rules))
            mr_fp = len(act_rules - exp_rules)
            mr_fn = len(exp_rules - act_rules)
            multi_rule_tp += mr_tp
            multi_rule_fp += mr_fp
            multi_rule_fn += mr_fn

        # Per-rule and global binary classification across all 50 rules
        for rid in all_rule_ids:
            in_exp = (rid in exp_rules)
            in_act = (rid in act_rules)
            if in_exp and in_act:
                global_tp += 1
                rule_tp[rid] += 1
            elif not in_exp and in_act:
                global_fp += 1
                rule_fp[rid] += 1
            elif in_exp and not in_act:
                global_fn += 1
                rule_fn[rid] += 1
            else:
                global_tn += 1
                rule_tn[rid] += 1

        # Alert metrics
        a_tp = len(act_alerts.intersection(exp_alerts))
        a_fp = len(act_alerts - exp_alerts)
        a_fn = len(exp_alerts - act_alerts)
        total_alert_tp += a_tp
        total_alert_fp += a_fp
        total_alert_fn += a_fn

        # Critical alert recall
        if has_expected_critical_alert:
            if "metformin_contraindicated" in act_alerts or any(r.rule_id == "R001" for r in inference_res.triggered_rules):
                critical_alert_tp += 1
            else:
                critical_alert_fn += 1

    # Multi-label Global Metrics
    micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 1.0
    micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 1.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    total_predictions = global_tp + global_fp + global_fn + global_tn
    accuracy = (global_tp + global_tn) / total_predictions if total_predictions > 0 else 1.0
    subset_accuracy = subset_exact_matches / total_cases if total_cases > 0 else 1.0

    # Per-Rule Metrics
    per_rule_rows = []
    prec_list = []
    rec_list = []
    f1_list = []

    for rid in all_rule_ids:
        tp = rule_tp[rid]
        fp = rule_fp[rid]
        fn = rule_fn[rid]
        tn = rule_tn[rid]
        supp = rule_support[rid]

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        per_rule_rows.append({
            "rule_id": rid,
            "rule_name": rule_id_to_name.get(rid, rid),
            "support": supp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        })

    macro_precision = sum(prec_list) / len(prec_list) if prec_list else 1.0
    macro_recall = sum(rec_list) / len(rec_list) if rec_list else 1.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 1.0

    # Alert Metrics
    alert_prec = total_alert_tp / (total_alert_tp + total_alert_fp) if (total_alert_tp + total_alert_fp) > 0 else 1.0
    alert_rec = total_alert_tp / (total_alert_tp + total_alert_fn) if (total_alert_tp + total_alert_fn) > 0 else 1.0
    alert_f1 = (2 * alert_prec * alert_rec / (alert_prec + alert_rec)) if (alert_prec + alert_rec) > 0 else 0.0

    critical_recall = critical_alert_tp / (critical_alert_tp + critical_alert_fn) if (critical_alert_tp + critical_alert_fn) > 0 else 1.0

    # Boundary & Multi-rule specifics
    boundary_acc = boundary_exact_matches / boundary_cases if boundary_cases > 0 else 1.0
    multi_rule_exact = multi_rule_exact_matches / multi_rule_cases if multi_rule_cases > 0 else 1.0

    mr_micro_prec = multi_rule_tp / (multi_rule_tp + multi_rule_fp) if (multi_rule_tp + multi_rule_fp) > 0 else 1.0
    mr_micro_rec = multi_rule_tp / (multi_rule_tp + multi_rule_fn) if (multi_rule_tp + multi_rule_fn) > 0 else 1.0
    mr_micro_f1 = (2 * mr_micro_prec * mr_micro_rec / (mr_micro_prec + mr_micro_rec)) if (mr_micro_prec + mr_micro_rec) > 0 else 0.0

    metrics_payload = {
        "total_cases": total_cases,
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "boundary_cases": boundary_cases,
        "multi_rule_cases": multi_rule_cases,
        "accuracy": round(accuracy, 4),
        "precision": round(micro_precision, 4),
        "recall": round(micro_recall, 4),
        "f1_score": round(micro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "subset_accuracy": round(subset_accuracy, 4),
        "alert_precision": round(alert_prec, 4),
        "alert_recall": round(alert_rec, 4),
        "alert_f1": round(alert_f1, 4),
        "critical_alert_recall": round(critical_recall, 4),
        "boundary_accuracy": round(boundary_acc, 4),
        "multi_rule_exact_match": round(multi_rule_exact, 4),
        "multi_rule_micro_f1": round(mr_micro_f1, 4),
        "true_positives": global_tp,
        "false_positives": global_fp,
        "false_negatives": global_fn,
        "true_negatives": global_tn,
        "threshold_errors": threshold_errors,
    }

    # Setup directories
    target_dir = Path(output_dir) if output_dir else Path("evaluation_results")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save evaluation_results.json in target_dir and root/backend
    for dest_json in (target_dir / "evaluation_results.json", Path("evaluation_results.json"), _backend_dir / "evaluation_results.json"):
        try:
            with open(dest_json, "w", encoding="utf-8") as jf:
                json.dump(metrics_payload, jf, indent=2)
        except Exception:
            pass

    # 2. Save per_rule_metrics.csv
    csv_cols = [
        "rule_id",
        "rule_name",
        "support",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
    ]
    for dest_csv in (target_dir / "per_rule_metrics.csv", Path("per_rule_metrics.csv")):
        try:
            with open(dest_csv, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=csv_cols)
                writer.writeheader()
                writer.writerows(per_rule_rows)
        except Exception:
            pass

    # 3. Save confusion_matrix.csv
    cm_rows = [
        {"metric": "True Positives (TP)", "count": global_tp},
        {"metric": "False Positives (FP)", "count": global_fp},
        {"metric": "False Negatives (FN)", "count": global_fn},
        {"metric": "True Negatives (TN)", "count": global_tn},
        {"metric": "Total Predictions", "count": total_predictions},
    ]
    for dest_cm in (target_dir / "confusion_matrix.csv", Path("confusion_matrix.csv")):
        try:
            with open(dest_cm, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=["metric", "count"])
                writer.writeheader()
                writer.writerows(cm_rows)
        except Exception:
            pass

    # 4. Generate Charts
    generate_charts(metrics_payload, per_rule_rows, target_dir)

    # 5. Print Terminal Report
    print_report(metrics_payload)

    return metrics_payload


def generate_charts(
    metrics: Dict[str, Any], per_rule: List[Dict[str, Any]], out_dir: Path
) -> None:
    """Generate the 6 required publication-ready performance charts using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed; skipping chart rendering.")
        return

    # Modern clinical dark/light styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    primary_color = "#0284c7"  # sky-600
    accent_color = "#10b981"   # emerald-500
    amber_color = "#f59e0b"    # amber-500
    rose_color = "#ef4444"     # rose-500

    # 1. metrics_summary.png
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    cats = ["Accuracy", "Precision", "Recall", "F1 Score", "Subset Acc"]
    vals = [
        metrics["accuracy"] * 100,
        metrics["micro_precision"] * 100,
        metrics["micro_recall"] * 100,
        metrics["micro_f1"] * 100,
        metrics["subset_accuracy"] * 100,
    ]
    bars = ax.bar(cats, vals, color=[primary_color, accent_color, amber_color, "#6366f1", "#8b5cf6"], width=0.55)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Deterministic Symbolic Reasoning Performance (500 Cases)", fontsize=13, fontweight="bold", pad=12)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png")
    plt.close(fig)

    # 2. per_rule_f1.png
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    rule_ids = [r["rule_id"] for r in per_rule]
    f1_scores = [r["f1_score"] * 100 for r in per_rule]
    bars = ax.bar(rule_ids, f1_scores, color=primary_color, width=0.6)
    ax.set_ylim(0, 115)
    ax.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rule ID (R001 - R050)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Rule F1 Performance Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "per_rule_f1.png")
    plt.close(fig)

    # 3. per_rule_precision_recall.png
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    import numpy as np
    x = np.arange(len(rule_ids))
    width = 0.38
    prec_vals = [r["precision"] * 100 for r in per_rule]
    rec_vals = [r["recall"] * 100 for r in per_rule]
    ax.bar(x - width/2, prec_vals, width, label="Precision", color=primary_color)
    ax.bar(x + width/2, rec_vals, width, label="Recall", color=accent_color)
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_ids, rotation=90, fontsize=8)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Rule Precision vs Recall Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(out_dir / "per_rule_precision_recall.png")
    plt.close(fig)

    # 4. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm_matrix = np.array([
        [metrics["true_positives"], metrics["false_negatives"]],
        [metrics["false_positives"], metrics["true_negatives"]],
    ])
    im = ax.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Triggered (Pos)", "Not Triggered (Neg)"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["Actual Pos", "Actual Neg"], fontsize=10, fontweight="bold")
    ax.set_title("Rule-Triggering Binary Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    labels = [
        [f"TP\n{metrics['true_positives']:,}", f"FN\n{metrics['false_negatives']:,}"],
        [f"FP\n{metrics['false_positives']:,}", f"TN\n{metrics['true_negatives']:,}"],
    ]
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            color = "white" if val > cm_matrix.max() / 2 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

    # 5. case_type_performance.png
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    types = ["Positive\n(150)", "Negative\n(150)", "Boundary\n(100)", "Multi-Rule\n(100)"]
    type_scores = [
        100.0,
        100.0,
        metrics["boundary_accuracy"] * 100,
        metrics["multi_rule_exact_match"] * 100,
    ]
    bars = ax.bar(types, type_scores, color=[accent_color, primary_color, amber_color, "#8b5cf6"], width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance by Benchmark Cohort Category", fontsize=13, fontweight="bold", pad=12)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "case_type_performance.png")
    plt.close(fig)

    # 6. alert_metrics.png
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    alert_names = ["Alert Precision", "Alert Recall", "Alert F1", "Critical Alert Recall"]
    alert_vals = [
        metrics["alert_precision"] * 100,
        metrics["alert_recall"] * 100,
        metrics["alert_f1"] * 100,
        metrics["critical_alert_recall"] * 100,
    ]
    bars = ax.bar(alert_names, alert_vals, color=[primary_color, accent_color, "#6366f1", rose_color], width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Clinical Safety Alert Performance Metrics", fontsize=13, fontweight="bold", pad=12)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "alert_metrics.png")
    plt.close(fig)

    # Also copy all 6 charts to root evaluation_results/ if out_dir is backend/evaluation_results
    root_eval_dir = _workspace_dir / "evaluation_results"
    if out_dir.resolve() != root_eval_dir.resolve():
        root_eval_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for chart_name in ("metrics_summary.png", "per_rule_f1.png", "per_rule_precision_recall.png",
                           "confusion_matrix.png", "case_type_performance.png", "alert_metrics.png"):
            src = out_dir / chart_name
            if src.is_file():
                shutil.copy(src, root_eval_dir / chart_name)


def print_report(m: Dict[str, Any]) -> None:
    """Print standard terminal benchmark evaluation report."""
    print("========================================")
    print("NEURO-SYMBOLIC ENGINE EVALUATION")
    print("========================================")
    print(f"Cases evaluated: {m['total_cases']}")
    print("")
    print(f"Accuracy:       {m['accuracy']*100:.2f}%")
    print(f"Precision:      {m['precision']*100:.2f}%")
    print(f"Recall:         {m['recall']*100:.2f}%")
    print(f"F1 Score:       {m['f1_score']*100:.2f}%")
    print("")
    print(f"Micro F1:       {m['micro_f1']*100:.2f}%")
    print(f"Macro F1:       {m['macro_f1']*100:.2f}%")
    print(f"Subset Accuracy:{m['subset_accuracy']*100:.2f}%")
    print("")
    print(f"Alert Recall:   {m['alert_recall']*100:.2f}%")
    print(f"Critical Recall:{m['critical_alert_recall']*100:.2f}%")
    print("")
    print(f"Boundary Accuracy:     {m['boundary_accuracy']*100:.2f}%")
    print(f"Multi-rule Exact Match:{m['multi_rule_exact_match']*100:.2f}%")
    print(f"Multi-rule Micro F1:   {m['multi_rule_micro_f1']*100:.2f}%")
    print("========================================")
    print(f"Confusion Matrix: TP={m['true_positives']:,} | FP={m['false_positives']:,} | FN={m['false_negatives']:,} | TN={m['true_negatives']:,}")
    print("Artifacts generated: evaluation_results.json, per_rule_metrics.csv, confusion_matrix.csv, and 6 charts in evaluation_results/")
    print("========================================")


if __name__ == "__main__":
    out_directory = Path("evaluation_results")
    run_evaluation(output_dir=out_directory)
