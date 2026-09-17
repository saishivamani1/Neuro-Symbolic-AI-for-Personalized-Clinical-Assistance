"""
backend/evaluation/plot_metrics.py

Research-grade evaluation graphical dashboard for the Neuro-Symbolic
Humanized Clinical Evaluation.

Reads:  evaluation_results/humanized/evaluation_humanized.json
Writes: evaluation_results/humanized/charts/  (12 PNG files)
        evaluation_results/humanized/dashboard.png  (combined tile)

Usage:
    python backend/evaluation/plot_metrics.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless / file-only rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = _ROOT / "evaluation_results" / "humanized" / "evaluation_humanized.json"
CHARTS_DIR = _ROOT / "evaluation_results" / "humanized" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Design Tokens ────────────────────────────────────────────────────────────
BG_DARK   = "#0f1117"
BG_CARD   = "#1a1d27"
BG_GRID   = "#1e2230"
TEXT_PRI  = "#e8eaf0"
TEXT_SEC  = "#8b92a8"
ACCENT1   = "#6c63ff"   # violet
ACCENT2   = "#00d2c8"   # teal
ACCENT3   = "#ff6584"   # rose
ACCENT4   = "#ffd166"   # amber
ACCENT5   = "#06d6a0"   # green
ACCENT6   = "#118ab2"   # blue
PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6,
           "#c77dff", "#f77f00", "#4cc9f0", "#e9c46a"]

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor":   BG_CARD,
    "axes.edgecolor":   BG_GRID,
    "axes.labelcolor":  TEXT_PRI,
    "axes.titlecolor":  TEXT_PRI,
    "xtick.color":      TEXT_SEC,
    "ytick.color":      TEXT_SEC,
    "text.color":       TEXT_PRI,
    "grid.color":       BG_GRID,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": BG_GRID,
})


def save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    out = CHARTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Chart saved: {out.name}")
    return out



def validate_metrics(data: dict) -> None:
    """Validate required fields before producing research figures."""
    required_top = ["total_cases", "per_case", "symbolic_rule_engine", "rag_retrieval"]
    missing = [k for k in required_top if k not in data]
    if missing:
        raise ValueError(f"Missing required evaluation fields: {missing}")

    eng = data["symbolic_rule_engine"]
    for key in ["micro_precision", "micro_recall", "micro_f1",
                "macro_precision", "macro_recall", "macro_f1",
                "subset_exact_match", "alert_recall"]:
        if key not in eng:
            raise ValueError(f"Missing symbolic metric: {key}")

    rag = data["rag_retrieval"]
    for key in ["hit_at_1", "hit_at_3", "hit_at_5", "mean_reciprocal_rank"]:
        if key not in rag:
            raise ValueError(f"Missing RAG metric: {key}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Overall Rule Engine Metrics Summary (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart1_overall_rule_metrics(data: dict):
    eng = data["symbolic_rule_engine"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_DARK)

    labels = ["Micro\nPrecision", "Micro\nRecall", "Micro\nF1",
              "Macro\nPrecision", "Macro\nRecall", "Macro\nF1",
              "Fact\nMacro F1", "Exact\nMatch", "Alert\nRecall"]
    values = [
        eng["micro_precision"], eng["micro_recall"], eng["micro_f1"],
        eng["macro_precision"], eng["macro_recall"], eng["macro_f1"],
        eng["derived_fact_macro_f1"], eng["subset_exact_match"], eng["alert_recall"],
    ]
    colors = [ACCENT1, ACCENT2, ACCENT5, ACCENT1, ACCENT2, ACCENT5, ACCENT4, ACCENT3, ACCENT6]

    bars = ax.bar(labels, [v * 100 for v in values], color=colors, width=0.6,
                  edgecolor="none", linewidth=0)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9,
                color=TEXT_PRI, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", color=TEXT_PRI)
    ax.set_title("Chart 1 — Symbolic Rule Engine: Overall Performance Metrics",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axhline(y=100, color=TEXT_SEC, linestyle=":", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor(BG_CARD)
    return save(fig, "01_overall_rule_metrics")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 2 — Per-Case Rule F1 Score (Horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart2_per_case_rule_f1(data: dict):
    pc = data["per_case"]
    case_ids = [c["case_id"] for c in pc]
    f1_scores = [c["rule_f1"] * 100 for c in pc]
    domains = [c["domain"] for c in pc]

    unique_domains = sorted(set(domains))
    domain_color = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(unique_domains)}
    bar_colors = [domain_color[d] for d in domains]

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    y_pos = range(len(case_ids))
    bars = ax.barh(y_pos, f1_scores, color=bar_colors, edgecolor="none", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(case_ids, fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Rule F1 Score (%)")
    ax.set_title("Chart 2 — Per-Case Rule F1 Score\n(Coloured by Clinical Domain)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.axvline(x=100, color=TEXT_SEC, linestyle=":", alpha=0.5)
    for bar, v in zip(bars, f1_scores):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%",
                va="center", fontsize=7.5, color=TEXT_PRI)

    legend_patches = [mpatches.Patch(color=domain_color[d], label=d) for d in unique_domains]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.3)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    return save(fig, "02_per_case_rule_f1")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Confusion Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def chart3_confusion_matrix(data: dict):
    eng = data["symbolic_rule_engine"]
    tp = eng["rule_tp"]
    fp = eng["rule_fp"]
    fn = eng["rule_fn"]
    # Confusion matrix: rows = predicted, columns = actual.
    # TN must come from the total number of binary rule decisions.
    n_cases = data["total_cases"]
    n_rules = data.get("total_rules", data.get("num_rules", 50))
    total = n_cases * n_rules
    tn = total - tp - fp - fn

    matrix = np.array([[tp, fn], [fp, tn]])
    labels_row = ["Predicted\nPOSITIVE", "Predicted\nNEGATIVE"]
    labels_col = ["Actual POSITIVE", "Actual NEGATIVE"]

    cmap = LinearSegmentedColormap.from_list("neuro", [BG_CARD, ACCENT1], N=256)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SEC)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SEC)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_col, color=TEXT_PRI, fontsize=10)
    ax.set_yticklabels(labels_row, color=TEXT_PRI, fontsize=10)

    cell_labels = [["TP", "FP"], ["FN", "TN"]]
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.text(j, i, f"{cell_labels[i][j]}\n{val:,}",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=TEXT_PRI)

    ax.set_title("Chart 3 — Rule Decision Confusion Matrix\n(Rule Engine vs Ground Truth)",
                 fontsize=12, fontweight="bold", pad=12)
    return save(fig, "03_confusion_matrix")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 4 — Case Type Performance (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart4_case_type_performance(data: dict):
    by_type = data["by_case_type"]
    types = sorted(by_type.keys())
    precisions = [by_type[t]["rule_precision"] * 100 for t in types]
    recalls = [by_type[t]["rule_recall"] * 100 for t in types]
    f1s = [by_type[t]["rule_f1"] * 100 for t in types]
    exact = [by_type[t]["exact_match_rate"] * 100 for t in types]

    x = np.arange(len(types))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    b1 = ax.bar(x - 1.5*width, precisions, width, label="Precision", color=ACCENT1)
    b2 = ax.bar(x - 0.5*width, recalls, width, label="Recall", color=ACCENT2)
    b3 = ax.bar(x + 0.5*width, f1s, width, label="F1 Score", color=ACCENT5)
    b4 = ax.bar(x + 1.5*width, exact, width, label="Exact Match", color=ACCENT4)

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT_PRI)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Score (%)")
    ax.set_title("Chart 4 — Rule Engine Performance by Case Type",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return save(fig, "04_case_type_performance")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 5 — Alert Detection Metrics (Bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart5_alert_metrics(data: dict):
    eng = data["symbolic_rule_engine"]
    alert_precision = eng.get("alert_precision", None)
    alert_recall = eng.get("alert_recall", None)
    alert_f1 = eng.get("alert_f1", None)

    # Case-level diagnostics are kept separate from alert-level metrics.
    pc = data["per_case"]
    alert_cases = [c for c in pc if c.get("expected_alerts")]
    hit_cases = [c for c in alert_cases if c.get("alert_detected")]

    metric_labels = ["Precision", "Recall", "F1"]
    metric_values = [
        (alert_precision if alert_precision is not None else 0.0) * 100,
        (alert_recall if alert_recall is not None else 0.0) * 100,
        (alert_f1 if alert_f1 is not None else 0.0) * 100,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(BG_DARK)

    # Left: alert-level precision / recall / F1
    ax = axes[0]
    ax.set_facecolor(BG_CARD)
    bars = ax.bar(metric_labels, metric_values, color=[ACCENT1, ACCENT2, ACCENT5], width=0.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Alert-Level Metrics")
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 2, f"{value:.1f}%",
                ha="center", fontsize=11, fontweight="bold", color=TEXT_PRI)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=100, color=TEXT_SEC, linestyle=":", alpha=0.5)

    # Right: pie of detected vs missed
    ax2 = axes[1]
    ax2.set_facecolor(BG_DARK)
    missed = len(alert_cases) - len(hit_cases)
    wedge_data = [len(hit_cases), missed] if missed > 0 else [len(hit_cases), 0]
    labels_pie = ["Correctly\nDetected", "Missed"] if missed > 0 else ["All Detected", ""]
    colors_pie = [ACCENT5, ACCENT3]
    if len(hit_cases) > 0 or missed > 0:
        wedges, texts, autotexts = ax2.pie(
            [max(v, 0.001) for v in wedge_data],
            labels=labels_pie, autopct="%1.0f%%",
            colors=colors_pie, startangle=140,
            textprops={"color": TEXT_PRI, "fontsize": 10},
        )
        for at in autotexts:
            at.set_fontweight("bold")
    ax2.set_title("Alert Case Breakdown")

    fig.suptitle("Chart 5 — Clinical Alert Detection Performance",
                 fontsize=13, fontweight="bold", y=1.02)
    return save(fig, "05_alert_metrics")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 6 — RAG Hit Rate @1, @3, @5 (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart6_rag_hit_rate(data: dict):
    rag = data["rag_retrieval"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    ks = ["Hit@1", "Hit@3", "Hit@5"]
    vals = [rag["hit_at_1"]*100, rag["hit_at_3"]*100, rag["hit_at_5"]*100]
    bars = ax.bar(ks, vals, color=[ACCENT6, ACCENT2, ACCENT5], width=0.45, edgecolor="none")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT_PRI)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Chart 6 — RAG Retrieval Hit Rate @K\n(Humanized Clinical Queries)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=100, color=TEXT_SEC, linestyle=":", alpha=0.4)
    return save(fig, "06_rag_hit_rate")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 7 — MRR per Case Type (Bar with per-case overlay dots)
# ═══════════════════════════════════════════════════════════════════════════════
def chart7_mrr_per_case_type(data: dict):
    pc = data["per_case"]
    type_mrr: dict = {}
    for c in pc:
        ct = c["case_type"]
        mrr_val = c.get("rag_mrr", 0.0)
        type_mrr.setdefault(ct, []).append(mrr_val)

    types = sorted(type_mrr.keys())
    means = [np.mean(type_mrr[t]) for t in types]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    x = np.arange(len(types))
    bars = ax.bar(x, means, color=PALETTE[:len(types)], width=0.55, alpha=0.8, edgecolor="none")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=TEXT_PRI)

    # Scatter overlay
    for xi, t in enumerate(types):
        jitter = np.random.uniform(-0.15, 0.15, len(type_mrr[t]))
        ax.scatter(xi + jitter, type_mrr[t], alpha=0.6, s=30,
                   color=TEXT_PRI, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_title("Chart 7 — RAG Mean Reciprocal Rank (MRR) by Case Type",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    return save(fig, "07_mrr_per_case_type")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 8 — RAG Similarity Score Distribution (Histogram + KDE)
# ═══════════════════════════════════════════════════════════════════════════════
def chart8_rag_score_distribution(data: dict):
    pc = data["per_case"]
    scores = [c.get("rag_avg_score", 0.0) for c in pc if c.get("rag_retrieved_count", 0) > 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    if scores:
        n_bins = min(15, len(scores))
        ax.hist(scores, bins=n_bins, color=ACCENT2, alpha=0.75, edgecolor=BG_DARK, linewidth=0.5)
        # KDE overlay
        from scipy.stats import gaussian_kde
        try:
            if len(set(scores)) > 2:
                kde = gaussian_kde(scores)
                xs = np.linspace(min(scores)-0.05, max(scores)+0.05, 300)
                ax2 = ax.twinx()
                ax2.set_facecolor(BG_CARD)
                ax2.plot(xs, kde(xs), color=ACCENT1, linewidth=2.5, label="KDE")
                ax2.set_ylabel("Density", color=ACCENT1)
                ax2.tick_params(axis="y", colors=ACCENT1)
                ax2.yaxis.label.set_color(ACCENT1)
        except Exception:
            pass
        mean_v = np.mean(scores)
        ax.axvline(mean_v, color=ACCENT4, linewidth=2, linestyle="--",
                   label=f"Mean = {mean_v:.3f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No RAG data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color=TEXT_SEC)

    ax.set_xlabel("Average RAG Similarity Score per Case")
    ax.set_ylabel("Case Count")
    ax.set_title("Chart 8 — RAG Retrieval Similarity Score Distribution",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    return save(fig, "08_rag_score_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 9 — Neuro-Symbolic Radar Chart (Spider)
# ═══════════════════════════════════════════════════════════════════════════════
def chart9_radar_chart(data: dict):
    eng = data["symbolic_rule_engine"]
    rag = data["rag_retrieval"]

    categories = ["Rule\nPrecision", "Rule\nRecall", "Rule\nF1",
                  "Alert\nRecall", "Exact\nMatch"]
    values_sym = [
        eng["micro_precision"], eng["micro_recall"], eng["micro_f1"],
        eng["alert_recall"], eng["subset_exact_match"],
    ]
    # Clamp to [0,1]
    values_sym = [min(max(v, 0.0), 1.0) for v in values_sym]
    ideal = [1.0] * len(categories)

    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_sym += values_sym[:1]
    ideal += ideal[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    ax.plot(angles, ideal, color=TEXT_SEC, linewidth=1, linestyle=":", label="Ideal (1.0)")
    ax.fill(angles, ideal, alpha=0.05, color=TEXT_SEC)
    ax.plot(angles, values_sym, color=ACCENT1, linewidth=2.5, label="Neuro-Symbolic Engine")
    ax.fill(angles, values_sym, alpha=0.25, color=ACCENT1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, color=TEXT_PRI)
    ax.set_rlabel_position(30)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=TEXT_SEC, size=8)
    ax.set_ylim(0, 1)
    ax.tick_params(colors=TEXT_SEC)

    ax.spines["polar"].set_color(BG_GRID)
    ax.grid(color=BG_GRID, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Chart 9 — Symbolic Reasoning Performance Radar",
                 fontsize=12, fontweight="bold", pad=25)
    return save(fig, "09_radar_chart")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 10 — Per-Domain Rule Accuracy (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart10_per_domain_accuracy(data: dict):
    by_domain = data["by_domain"]
    domains = sorted(by_domain.keys())
    f1s = [by_domain[d]["rule_f1"] * 100 for d in domains]
    exact = [by_domain[d]["exact_match_rate"] * 100 for d in domains]

    fig, ax = plt.subplots(figsize=(9, max(5, len(domains) * 0.7)))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    y = np.arange(len(domains))
    h = 0.35
    bars1 = ax.barh(y + h/2, f1s, h, label="Rule F1", color=ACCENT1, edgecolor="none")
    bars2 = ax.barh(y - h/2, exact, h, label="Exact Match", color=ACCENT4, edgecolor="none")

    for bars in (bars1, bars2):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.0f}%",
                    va="center", fontsize=8, color=TEXT_PRI)

    ax.set_yticks(y)
    ax.set_yticklabels([d.replace("_", "\n") for d in domains], fontsize=9)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Score (%)")
    ax.set_title("Chart 10 — Per-Domain Rule F1 & Exact Match Rate",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=100, color=TEXT_SEC, linestyle=":", alpha=0.4)
    ax.invert_yaxis()
    return save(fig, "10_per_domain_accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 11 — Derived Facts Coverage (Stacked Bar per Case)
# ═══════════════════════════════════════════════════════════════════════════════
def chart11_rule_firing_coverage(data: dict):
    pc = data["per_case"]
    case_ids = [c["case_id"] for c in pc]
    correct = [c["rule_tp"] for c in pc]
    missed  = [c["rule_fn"] for c in pc]
    extra   = [c["rule_fp"] for c in pc]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    x = np.arange(len(case_ids))
    w = 0.65
    ax.bar(x, correct, w, label="Correctly Fired (TP)", color=ACCENT5, edgecolor="none")
    ax.bar(x, missed, w, bottom=correct, label="Missed (FN)", color=ACCENT3, edgecolor="none")
    ax.bar(x, extra, w,
           bottom=[c + m for c, m in zip(correct, missed)],
           label="Extra (FP)", color=ACCENT4, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Rule Count")
    ax.set_title("Chart 11 — Rule Firing Coverage per Humanized Case\n"
                 "(TP=Correct  FN=Missed  FP=Extra)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return save(fig, "11_derived_facts_coverage")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART 12 — Combined Dashboard Tile (4×3 mini panels)
# ═══════════════════════════════════════════════════════════════════════════════
def chart12_dashboard_tile(data: dict, chart_paths: list):
    eng = data["symbolic_rule_engine"]
    rag = data["rag_retrieval"]
    n = data["total_cases"]

    # Big KPI tiles
    kpis = [
        ("Rule Micro F1", f"{eng['micro_f1']*100:.1f}%", ACCENT1),
        ("Rule Exact Match", f"{eng['subset_exact_match']*100:.1f}%", ACCENT5),
        ("Alert Recall", f"{eng['alert_recall']*100:.1f}%", ACCENT3),
        ("RAG Hit@5", f"{rag['hit_at_5']*100:.1f}%", ACCENT2),
        ("RAG MRR", f"{rag['mean_reciprocal_rank']:.3f}", ACCENT4),
        ("Rule Macro F1", f"{eng['macro_f1']*100:.1f}%", ACCENT6),
    ]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "Neuro-Symbolic Clinical Reasoning  ·  Humanized Evaluation Dashboard",
        fontsize=16, fontweight="bold", color=TEXT_PRI, y=0.99,
    )

    # Top row: 6 KPI tiles
    gs_top = gridspec.GridSpec(1, 6, figure=fig,
                               left=0.02, right=0.98, top=0.93, bottom=0.73,
                               wspace=0.12)
    for i, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs_top[i])
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.text(0.5, 0.62, val, transform=ax.transAxes,
                ha="center", va="center", fontsize=22,
                fontweight="bold", color=color)
        ax.text(0.5, 0.18, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=TEXT_SEC)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom: 6 chart image tiles (load the saved pngs)
    selected = ["01_overall_rule_metrics", "02_per_case_rule_f1",
                 "06_rag_hit_rate", "07_mrr_per_case_type",
                 "10_per_domain_accuracy", "09_radar_chart"]
    gs_bot = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.02, right=0.98, top=0.70, bottom=0.02,
                               wspace=0.06, hspace=0.1)
    for i, chart_name in enumerate(selected):
        chart_file = CHARTS_DIR / f"{chart_name}.png"
        ax = fig.add_subplot(gs_bot[i // 3, i % 3])
        ax.set_facecolor(BG_DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if chart_file.exists():
            img = plt.imread(str(chart_file))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Chart\n{chart_name}", ha="center",
                    va="center", color=TEXT_SEC, fontsize=9,
                    transform=ax.transAxes)

    # Footer text
    fig.text(0.5, 0.005,
             f"Evaluated {n} humanized clinical vignettes  ·  "
             f"50 deterministic rules  ·  {data.get('rag_cases_evaluated', n)} RAG queries",
             ha="center", fontsize=9, color=TEXT_SEC)

    out = _ROOT / "evaluation_results" / "humanized" / "dashboard.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Dashboard tile saved: {out.name}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Neuro-Symbolic Humanized Evaluation — Visualization Dashboard")
    print("=" * 70)

    if not RESULTS_FILE.exists():
        print(f"\n[ERROR] Results file not found: {RESULTS_FILE}")
        print("  Run evaluate_humanized.py first to generate the data.\n")
        sys.exit(1)

    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    validate_metrics(data)
    print(f"\n[INFO] Loaded results for {data['total_cases']} cases")
    print(f"[INFO] Generating evaluation charts -> {CHARTS_DIR}\n")

    np.random.seed(42)

    paths = []
    paths.append(chart1_overall_rule_metrics(data))
    paths.append(chart2_per_case_rule_f1(data))
    paths.append(chart3_confusion_matrix(data))
    paths.append(chart4_case_type_performance(data))
    paths.append(chart5_alert_metrics(data))
    paths.append(chart6_rag_hit_rate(data))
    paths.append(chart7_mrr_per_case_type(data))
    paths.append(chart8_rag_score_distribution(data))
    paths.append(chart9_radar_chart(data))
    paths.append(chart10_per_domain_accuracy(data))
    paths.append(chart11_rule_firing_coverage(data))
    paths.append(chart12_dashboard_tile(data, paths))

    print(f"\n{'='*70}")
    print(f"  [OK] All evaluation charts generated successfully!")
    print(f"  [DIR] Charts directory : {CHARTS_DIR}")
    print(f"  [FILE] Dashboard       : {_ROOT / 'evaluation_results' / 'humanized' / 'dashboard.png'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
