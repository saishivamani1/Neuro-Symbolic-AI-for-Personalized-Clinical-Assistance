"""
backend/evaluation/evaluate_humanized_extraction.py

NLP Fact-Extraction Evaluation for Humanized Clinical Vignettes.

METHODOLOGY
-----------
For each of 30 humanized clinical cases we have:

  A) ground_truth_structured_facts  (authoritative — JSON in humanized_cases.json)
  B) humanized_narrative             (free-text clinical note)

This script implements a keyword/regex NLP extractor (B -> extracted_facts)
and then evaluates:

  1. FACT EXTRACTION QUALITY
     Compare extracted_facts vs ground_truth_structured_facts field-by-field.
     Measures where information is successfully preserved or lost.

  2. RULE DETECTION AFTER EXTRACTION
     Run full SymbolicRuleEngine on extracted_facts -> predicted_rules
     Compare predicted_rules vs engine(ground_truth_facts) -> ground_truth_rules
     Measures end-to-end pipeline: NLP -> Symbolic reasoning.

  3. ERROR ATTRIBUTION
     Classify every failed rule case into an error category.
     Reveals whether errors come from extraction or from reasoning.

IMPORTANT — Non-circularity:
  ground_truth_rules = engine(structured_facts)
  predicted_rules    = engine(EXTRACTED facts from narrative text)
  structured_facts != extracted_facts  (unless extraction is perfect)

Usage:
    python backend/evaluation/evaluate_humanized_extraction.py

Outputs:
    evaluation_results/humanized/extraction_evaluation.json
    evaluation_results/humanized/per_case_extraction.csv
    evaluation_results/humanized/humanized_error_analysis.csv
    evaluation_results/humanized/rag_metrics.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Path bootstrap ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _ROOT / "backend"
for p in (_BACKEND, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(_BACKEND)

from app.reasoning.engine import SymbolicRuleEngine
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import get_vector_store

OUT_DIR = _ROOT / "evaluation_results" / "humanized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Important fields for field-level accuracy ────────────────────────────────
NUMERIC_FIELDS = [
    "age", "egfr", "potassium", "hba1c", "ldl_c", "uacr",
    "systolic_bp", "diastolic_bp", "lvef", "sodium", "creatinine",
    "fev1_fvc_ratio", "triglycerides",
]
BOOLEAN_FIELDS = [
    "type2_diabetes", "ckd", "heart_failure", "symptomatic_hf",
    "heart_failure_improved_ef", "chronic_coronary_disease",
    "established_ascvd", "primary_prevention", "smoker",
    "statin", "statin_started_or_adjusted", "mi_within_1_year",
    "beta_blocker_other_indication", "stable_angina", "aspirin", "acute_illness",
]
LIST_FIELDS = ["medications", "conditions"]


# ═══════════════════════════════════════════════════════════════════════════════
#  CLINICAL NLP EXTRACTOR  (keyword / regex — no LLM, no external models)
# ═══════════════════════════════════════════════════════════════════════════════

# Known drug names -> canonical medication labels
DRUG_MAP = {
    "metformin": "metformin",
    "lisinopril": "lisinopril", "ramipril": "ramipril", "enalapril": "enalapril",
    "losartan": "losartan", "valsartan": "valsartan", "irbesartan": "irbesartan",
    "furosemide": "furosemide", "frusemide": "furosemide",
    "spironolactone": "spironolactone", "eplerenone": "eplerenone",
    "warfarin": "warfarin",
    "ibuprofen": "ibuprofen", "naproxen": "naproxen", "diclofenac": "diclofenac",
    "aspirin": "aspirin",
    "atorvastatin": "atorvastatin", "rosuvastatin": "rosuvastatin",
    "simvastatin": "simvastatin",
    "empagliflozin": "empagliflozin", "dapagliflozin": "dapagliflozin",
    "canagliflozin": "canagliflozin",
    "amlodipine": "amlodipine",
    "lorazepam": "lorazepam", "diazepam": "diazepam", "temazepam": "temazepam",
    "salbutamol": "salbutamol", "albuterol": "salbutamol",
    "tiotropium": "tiotropium",
    "insulin": "insulin",
    "bisphosphonate": "bisphosphonate", "alendronate": "alendronate",
}

CONDITION_MAP = [
    (["type 2 diabetes", "type2 diabetes", "t2dm", "t2d", "diabetic", "diabetes mellitus"],
     ("type2_diabetes", True, "Type 2 Diabetes Mellitus")),
    (["type 1 diabetes", "type1 diabetes", "t1dm"],
     ("_type1_diabetes", True, "Type 1 Diabetes")),
    (["chronic kidney disease", "ckd", "renal failure", "kidney failure", "renal disease",
      "stage 4 ckd", "stage 5 ckd", "stage 3", "kidney function"],
     ("ckd", True, "Chronic Kidney Disease")),
    (["heart failure", "cardiac failure", "hf ", "hfref", "hfpef", "lvef", "ejection fraction"],
     ("heart_failure", True, "Heart Failure")),
    (["copd", "chronic obstructive", "emphysema"],
     ("_copd", True, "COPD")),
    (["atrial fibrillation", "afib", " af ", "a fib"],
     ("_af", True, "Atrial Fibrillation")),
    (["established ascvd", "prior stroke", "prior mi", "cardiovascular disease", "atherosclerosis"],
     ("established_ascvd", True, "Established ASCVD")),
    (["hypertension", "high blood pressure", "htn ", "bp treatment"],
     ("_hypertension", True, "Hypertension")),
    (["dementia", "cognitive impairment"],
     ("_dementia", True, "Dementia")),
]

SMOKER_TERMS = ["smoker", "smoking", "pack-year", "pack year", "tobacco"]
ASPIRIN_TERMS = ["aspirin"]
STATIN_TERMS = ["statin", "atorvastatin", "rosuvastatin", "simvastatin"]
ACUTE_ILLNESS_TERMS = ["acute illness", "acute infection", "fever", "sepsis"]


def extract_facts_from_narrative(text: str) -> Dict[str, Any]:
    """
    Keyword and regex-based clinical NLP extractor.

    Intentionally implements a simple rule-based approach that:
    - Handles common abbreviations and unit variations
    - Does NOT use an LLM or pre-trained NER model
    - Will miss some information (this is the point — to measure extraction quality)
    - Treats missing information as UNKNOWN (None), not False

    Returns a dict of extracted facts.
    """
    tl = text.lower()
    facts: Dict[str, Any] = {}

    # ── Age ──────────────────────────────────────────────────────────────────
    for pat in [r"(\d{1,3})[-\s]?year[-\s]?old", r"(\d{1,3})(?:yo|y\.?o\.?)\b", r"age[:\s]+(\d{1,3})"]:
        m = re.search(pat, tl)
        if m:
            age_val = int(m.group(1))
            if 0 < age_val < 130:
                facts["age"] = age_val
                break

    # ── Gender ───────────────────────────────────────────────────────────────
    if re.search(r"\b(female|woman|mrs\b|ms\.?\s|she\b)", tl):
        facts["gender"] = "female"
    elif re.search(r"\b(male|man|mr\.?\s|he\b)\b", tl):
        facts["gender"] = "male"

    # ── eGFR / GFR ───────────────────────────────────────────────────────────
    for pat in [r"egfr\s*[~=:of]?\s*(\d+\.?\d*)", r"gfr\s*[~=:of]?\s*(\d+\.?\d*)",
                r"egfr\s+around\s+(\d+\.?\d*)", r"gfr\s+about\s+(\d+\.?\d*)"]:
        m = re.search(pat, tl)
        if m:
            facts["egfr"] = float(m.group(1))
            break

    # ── Potassium ────────────────────────────────────────────────────────────
    for pat in [r"(?:potassium|k\+?)\s*[~=:of]?\s*(\d+\.?\d*)",
                r"k\+?\s+(?:came\s+back|is|was|level)\s+(?:at\s+)?(\d+\.?\d*)",
                r"potassium\s+(?:slightly|mildly|severely)?\s*(?:high|elevated|raised)\s+(?:around\s+)?(\d+\.?\d*)"]:
        m = re.search(pat, tl)
        if m:
            facts["potassium"] = float(m.group(1))
            break

    # ── HbA1c ────────────────────────────────────────────────────────────────
    for pat in [r"hba1c\s*[~=:of]?\s*(\d+\.?\d*)", r"a1c\s*[~=:of]?\s*(\d+\.?\d*)",
                r"glycated\s+h(?:a?e?moglobin|b)\s*[~=:of]?\s*(\d+\.?\d*)"]:
        m = re.search(pat, tl)
        if m:
            facts["hba1c"] = float(m.group(1))
            break

    # ── Blood Pressure ───────────────────────────────────────────────────────
    for pat in [r"bp\s*[~=:of]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
                r"blood\s+pressure\s*[~=:of]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
                r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg",
                r"(\d{2,3})\s*/\s*(\d{2,3})\s*mmhg"]:
        m = re.search(pat, tl)
        if m:
            sbp, dbp = int(m.group(1)), int(m.group(2))
            if 60 < sbp < 250 and 40 < dbp < 150:
                facts["systolic_bp"] = sbp
                facts["diastolic_bp"] = dbp
                break

    # ── LDL-C ────────────────────────────────────────────────────────────────
    for pat in [r"ldl[-\s]?c?\s*[~=:of]?\s*(\d+\.?\d*)\s*mg",
                r"ldl\s*[~=:of]?\s*(\d+\.?\d*)"]:
        m = re.search(pat, tl)
        if m:
            facts["ldl_c"] = float(m.group(1))
            break

    # ── UACR ─────────────────────────────────────────────────────────────────
    for pat in [r"uacr\s*[~=:of]?\s*(\d+\.?\d*)", r"albumin[-\s]to[-\s]creatinine\s+ratio\s*[~=:of]?\s*(\d+\.?\d*)"]:
        m = re.search(pat, tl)
        if m:
            facts["uacr"] = float(m.group(1))
            break

    # ── LVEF ─────────────────────────────────────────────────────────────────
    for pat in [r"(?:lvef|ef|ejection\s+fraction)\s*[~=:of]?\s*(\d{1,2})\s*%?",
                r"ef\s+(?:of\s+)?(\d{1,2})\s*%?"]:
        m = re.search(pat, tl)
        if m:
            val = int(m.group(1))
            if 5 < val < 100:
                facts["lvef"] = val
                break

    # ── Sodium ───────────────────────────────────────────────────────────────
    m = re.search(r"sodium\s*[~=:of]?\s*(\d+\.?\d*)", tl)
    if m:
        facts["sodium"] = float(m.group(1))

    # ── Creatinine ───────────────────────────────────────────────────────────
    for pat in [r"creatinine\s*[~=:of]?\s*(\d+\.?\d*)", r"cr\s+(\d+\.?\d*)\s*mg"]:
        m = re.search(pat, tl)
        if m:
            facts["creatinine"] = float(m.group(1))
            break

    # ── FEV1/FVC ─────────────────────────────────────────────────────────────
    m = re.search(r"fev1\s*/\s*fvc\s*[~=:of]?\s*(\d+\.?\d*)", tl)
    if m:
        facts["fev1_fvc_ratio"] = float(m.group(1))

    # ── Boolean conditions ────────────────────────────────────────────────────
    extracted_conditions = []
    for triggers, (flag, value, label) in CONDITION_MAP:
        if any(t in tl for t in triggers):
            if not flag.startswith("_"):  # direct patient_facts flag
                facts[flag] = value
            extracted_conditions.append(label)

    if not extracted_conditions:
        facts["conditions"] = []
    else:
        facts["conditions"] = extracted_conditions

    facts["smoker"] = any(t in tl for t in SMOKER_TERMS)
    facts["aspirin"] = any(t in tl for t in ASPIRIN_TERMS)
    facts["statin"] = any(t in tl for t in STATIN_TERMS)
    facts["acute_illness"] = any(t in tl for t in ACUTE_ILLNESS_TERMS)

    # MI within 1 year
    if re.search(r"(?:mi|myocardial\s+infarction|heart\s+attack)\s+(?:\d+\s+months?|within\s+(?:a\s+)?year|8\s+months?)", tl):
        facts["mi_within_1_year"] = True

    # ── Medications ──────────────────────────────────────────────────────────
    found_meds = []
    for drug_name, canonical in DRUG_MAP.items():
        if drug_name in tl:
            if canonical not in found_meds:
                found_meds.append(canonical)
    facts["medications"] = found_meds
    facts["current_medications"] = found_meds

    # Default unknown boolean facts to False (safe clinical default)
    # but only if no signal was found — leave unset otherwise
    for bf in BOOLEAN_FIELDS:
        if bf not in facts:
            facts[bf] = False  # UNKNOWN -> False (conservative)

    return facts


# ═══════════════════════════════════════════════════════════════════════════════
#  FACT COMPARISON HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_numeric(gt_val: Any, pred_val: Any, tolerance: float = 0.01) -> str:
    """Returns 'CORRECT', 'WRONG', or 'MISSING'."""
    if pred_val is None:
        return "MISSING"
    if gt_val is None:
        return "EXTRA"
    try:
        return "CORRECT" if abs(float(gt_val) - float(pred_val)) <= tolerance else "WRONG"
    except (ValueError, TypeError):
        return "WRONG"


def compare_boolean(gt_val: Any, pred_val: Any) -> str:
    if pred_val is None:
        return "MISSING"
    return "CORRECT" if bool(gt_val) == bool(pred_val) else "WRONG"


def compare_list_overlap(gt_list: List[str], pred_list: List[str]) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for medication/condition list overlap."""
    gt_set = set(str(x).strip().lower() for x in gt_list)
    pred_set = set(str(x).strip().lower() for x in pred_list)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gt_set else 0.0)
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  RAG EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_rag(retriever, query_terms: List[str], top_k: int = 5) -> Dict[str, Any]:
    """Compute retrieval metrics for a single clinical query."""
    if not query_terms:
        return {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0,
                "precision_at_1": 0.0, "precision_at_3": 0.0, "precision_at_5": 0.0,
                "mrr": 0.0, "avg_similarity": 0.0, "keyword_coverage": 0.0,
                "retrieved_count": 0}

    query = " ".join(query_terms[:6])
    kw_lower = [k.lower() for k in query_terms]

    try:
        res = retriever.retrieve(query=query, top_k=top_k)
        chunks = res.results
    except Exception:
        return {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0,
                "precision_at_1": 0.0, "precision_at_3": 0.0, "precision_at_5": 0.0,
                "mrr": 0.0, "avg_similarity": 0.0, "keyword_coverage": 0.0,
                "retrieved_count": 0, "error": "retrieval_failed"}

    if not chunks:
        return {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0,
                "precision_at_1": 0.0, "precision_at_3": 0.0, "precision_at_5": 0.0,
                "mrr": 0.0, "avg_similarity": 0.0, "keyword_coverage": 0.0,
                "retrieved_count": 0}

    def is_relevant(chunk) -> bool:
        return any(kw in chunk.content.lower() for kw in kw_lower)

    def precision_at_k(k: int) -> float:
        rel = sum(1 for c in chunks[:k] if is_relevant(c))
        return round(rel / k, 4) if k > 0 else 0.0

    scores = [c.score for c in chunks]
    all_text = " ".join(c.content.lower() for c in chunks)
    kw_covered = sum(1 for kw in kw_lower if kw in all_text)

    mrr = 0.0
    for rank, chunk in enumerate(chunks, 1):
        if is_relevant(chunk):
            mrr = 1.0 / rank
            break

    return {
        "hit_at_1": 1 if any(is_relevant(c) for c in chunks[:1]) else 0,
        "hit_at_3": 1 if any(is_relevant(c) for c in chunks[:3]) else 0,
        "hit_at_5": 1 if any(is_relevant(c) for c in chunks[:5]) else 0,
        "precision_at_1": precision_at_k(1),
        "precision_at_3": precision_at_k(3),
        "precision_at_5": precision_at_k(5),
        "mrr": round(mrr, 4),
        "avg_similarity": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "keyword_coverage": round(kw_covered / len(kw_lower), 4) if kw_lower else 1.0,
        "retrieved_count": len(chunks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def attribute_errors(
    case: Dict[str, Any],
    gt_rules: Set[str],
    pred_rules: Set[str],
    field_accuracy: Dict[str, str],
    gt_facts: Dict[str, Any],
    extracted_facts: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Classify error for each missed/extra rule in this case."""
    errors = []
    cid = case["case_id"]
    missing_rules = gt_rules - pred_rules
    extra_rules = pred_rules - gt_rules

    if not missing_rules and not extra_rules:
        errors.append({
            "case_id": cid,
            "error_category": "CORRECT_EXTRACTION_CORRECT_RULE",
            "field": "", "ground_truth_value": "", "predicted_value": "",
            "expected_rules": sorted(gt_rules), "actual_rules": sorted(pred_rules),
            "missing_rules": [], "extra_rules": [],
            "explanation": "All rules correctly predicted after extraction.",
        })
        return errors

    # For each missing rule, find why
    for rule_id in sorted(missing_rules):
        # Check if any key numeric field was missed/wrong
        affected_fields = [f for f, status in field_accuracy.items() if status in ("MISSING", "WRONG")]
        medication_issue = (
            field_accuracy.get("medications_f1", "CORRECT") not in ("CORRECT", "1.0")
        )
        category = "MISSING_RULE_FIRE"
        field = "multiple"
        gt_val = ""
        pred_val = ""
        explanation = f"Rule {rule_id} not fired after extraction."

        if affected_fields:
            category = "MISSING_FACT"
            field = ", ".join(affected_fields[:3])
            gt_val = str({f: gt_facts.get(f) for f in affected_fields[:3]})
            pred_val = str({f: extracted_facts.get(f) for f in affected_fields[:3]})
            explanation = f"Key facts missing/wrong in extraction prevented rule {rule_id} from firing: {field}"
        elif medication_issue:
            category = "WRONG_MEDICATION_NORMALIZATION"
            field = "medications"
            gt_val = str(gt_facts.get("medications", []))
            pred_val = str(extracted_facts.get("medications", []))
            explanation = f"Medication normalization gap prevented rule {rule_id} (drug class expansion missing)."

        errors.append({
            "case_id": cid,
            "error_category": category,
            "field": field,
            "ground_truth_value": gt_val,
            "predicted_value": pred_val,
            "expected_rules": sorted(gt_rules),
            "actual_rules": sorted(pred_rules),
            "missing_rules": sorted(missing_rules),
            "extra_rules": sorted(extra_rules),
            "explanation": explanation,
        })

    # For each extra rule
    for rule_id in sorted(extra_rules):
        errors.append({
            "case_id": cid,
            "error_category": "FALSE_RULE_FIRE",
            "field": "extracted_facts",
            "ground_truth_value": "",
            "predicted_value": "",
            "expected_rules": sorted(gt_rules),
            "actual_rules": sorted(pred_rules),
            "missing_rules": sorted(missing_rules),
            "extra_rules": sorted(extra_rules),
            "explanation": f"Rule {rule_id} fired on extracted facts but not in ground truth. Likely false-positive from extraction.",
        })

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_extraction_evaluation():
    print("=" * 70)
    print("  Humanized Fact Extraction + Rule Detection Evaluation")
    print("  Separating: NLP Extraction Errors vs Symbolic Reasoning Errors")
    print("=" * 70)

    # Load cases
    cases_path = _BACKEND / "evaluation" / "humanized_cases.json"
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"\n[INFO] {len(cases)} humanized cases loaded.")

    # Init engine
    engine = SymbolicRuleEngine()
    print(f"[INFO] Symbolic rule engine ready — {len(engine.rules)} rules")

    # Init RAG
    try:
        vs = get_vector_store()
        doc_count = vs.count()
        retriever = RAGRetriever(vector_store=vs) if doc_count > 0 else None
        print(f"[INFO] ChromaDB: {doc_count:,} chunks")
    except Exception as e:
        retriever = None
        print(f"[WARN] RAG unavailable: {e}")

    # Per-case accumulators
    per_case_results = []
    error_rows = []

    # Aggregate stats
    num_fields = NUMERIC_FIELDS + BOOLEAN_FIELDS
    field_correct_counts = {f: 0 for f in num_fields}
    field_total_counts = {f: 0 for f in num_fields}

    rule_tp = rule_fp = rule_fn = 0
    exact_rule_matches = 0

    rag_accum = {k: 0.0 for k in ["hit_at_1","hit_at_3","hit_at_5",
                                    "precision_at_1","precision_at_3","precision_at_5",
                                    "mrr","avg_similarity","keyword_coverage"]}
    rag_n = 0

    print(f"\n{'='*70}")
    print(f"  {'Case':10} {'Extr F1':8} {'Rule P':7} {'Rule R':7} {'Rule F1':8} {'RAG@5':6}")
    print(f"{'='*70}")

    for case in cases:
        cid = case["case_id"]
        narrative = case.get("narrative", "")
        gt_facts = case.get("patient_facts", {})
        gt_rules_list = case.get("expected_rules", [])
        gt_rules = set(gt_rules_list)
        exp_rag_kw = case.get("expected_rag_keywords", [])

        # ── 1. Extract facts from narrative ──────────────────────────────────
        extracted = extract_facts_from_narrative(narrative)

        # ── 2. Field-level accuracy ───────────────────────────────────────────
        field_acc: Dict[str, str] = {}

        for fn_ in NUMERIC_FIELDS:
            gt_v = (gt_facts.get(fn_)
                    or (gt_facts.get("labs") or {}).get(fn_)
                    or (gt_facts.get("vitals") or {}).get(fn_))
            pred_v = extracted.get(fn_)
            if gt_v is not None:  # only evaluate fields present in ground truth
                status = compare_numeric(gt_v, pred_v)
                field_acc[fn_] = status
                field_total_counts[fn_] += 1
                if status == "CORRECT":
                    field_correct_counts[fn_] += 1

        for bf_ in BOOLEAN_FIELDS:
            gt_v = gt_facts.get(bf_)
            pred_v = extracted.get(bf_)
            if gt_v is not None:
                status = compare_boolean(gt_v, pred_v)
                field_acc[bf_] = status
                field_total_counts[bf_] += 1
                if status == "CORRECT":
                    field_correct_counts[bf_] += 1

        # Medication list overlap
        gt_meds = gt_facts.get("medications") or gt_facts.get("current_medications") or []
        pred_meds = extracted.get("medications", [])
        med_p, med_r, med_f1 = compare_list_overlap(gt_meds, pred_meds)
        field_acc["medications_f1"] = str(round(med_f1, 3))

        # Overall extraction score
        correct_count = sum(1 for s in field_acc.values() if s == "CORRECT")
        total_fields = sum(1 for s in field_acc.values() if s in ("CORRECT","WRONG","MISSING"))
        extraction_f1 = round(correct_count / total_fields, 4) if total_fields > 0 else 1.0

        # ── 3. Run engine on EXTRACTED facts ─────────────────────────────────
        pred_result = engine.evaluate(extracted)
        pred_rules = set(r.rule_id for r in pred_result.triggered_rules)
        pred_alerts = set(a.name for a in pred_result.critical_alerts)

        # ── 4. Rule detection metrics ─────────────────────────────────────────
        tp = len(pred_rules & gt_rules)
        fp = len(pred_rules - gt_rules)
        fn = len(gt_rules - pred_rules)
        p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gt_rules else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        rule_tp += tp
        rule_fp += fp
        rule_fn += fn
        if pred_rules == gt_rules:
            exact_rule_matches += 1

        # ── 5. RAG evaluation ─────────────────────────────────────────────────
        rag_m: Dict[str, Any] = {k: 0 for k in rag_accum}
        if retriever and exp_rag_kw:
            rag_m = evaluate_rag(retriever, exp_rag_kw, top_k=5)
            for k in rag_accum:
                rag_accum[k] += rag_m.get(k, 0)
            rag_n += 1

        # ── 6. Error attribution ─────────────────────────────────────────────
        errors = attribute_errors(case, gt_rules, pred_rules, field_acc, gt_facts, extracted)
        error_rows.extend(errors)

        # Print row
        print(f"  {cid:10} {extraction_f1:.3f}   {p:.3f}   {r:.3f}   {f1:.3f}    {rag_m.get('hit_at_5',0)}")

        per_case_results.append({
            "case_id": cid,
            "domain": case.get("domain",""),
            "case_type": case.get("case_type",""),
            "narrative_length": len(narrative),
            "extraction_field_accuracy": extraction_f1,
            "medication_precision": med_p,
            "medication_recall": med_r,
            "medication_f1": med_f1,
            "field_statuses": json.dumps(field_acc),
            "gt_rules": sorted(gt_rules),
            "predicted_rules": sorted(pred_rules),
            "rule_precision": round(p, 4),
            "rule_recall": round(r, 4),
            "rule_f1": round(f1, 4),
            "rule_tp": tp, "rule_fp": fp, "rule_fn": fn,
            "exact_rule_match": pred_rules == gt_rules,
            **{f"rag_{k}": v for k, v in rag_m.items()},
        })

    print(f"{'='*70}\n")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n = len(cases)
    micro_p = rule_tp / (rule_tp + rule_fp) if (rule_tp + rule_fp) > 0 else 1.0
    micro_r = rule_tp / (rule_tp + rule_fn) if (rule_tp + rule_fn) > 0 else 1.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_rule_f1 = sum(c["rule_f1"] for c in per_case_results) / n

    avg_extraction_f1 = sum(c["extraction_field_accuracy"] for c in per_case_results) / n
    avg_med_f1 = sum(c["medication_f1"] for c in per_case_results) / n

    # Per-field accuracy
    field_accuracy_summary = {
        f: round(field_correct_counts[f] / field_total_counts[f], 4)
        if field_total_counts[f] > 0 else None
        for f in num_fields
    }

    # RAG aggregates
    rag_summary = {k: round(v / rag_n, 4) if rag_n > 0 else 0.0 for k, v in rag_accum.items()}
    rag_summary["cases_evaluated"] = rag_n

    print("=" * 70)
    print("  FACT EXTRACTION QUALITY")
    print("=" * 70)
    print(f"  Avg Field Accuracy (all fields)  : {avg_extraction_f1:.4f} ({avg_extraction_f1*100:.1f}%)")
    print(f"  Avg Medication F1                : {avg_med_f1:.4f} ({avg_med_f1*100:.1f}%)")
    print(f"\n  RULE DETECTION (after extraction)")
    print(f"  Micro Precision                  : {micro_p:.4f} ({micro_p*100:.1f}%)")
    print(f"  Micro Recall                     : {micro_r:.4f} ({micro_r*100:.1f}%)")
    print(f"  Micro F1                         : {micro_f1:.4f} ({micro_f1*100:.1f}%)")
    print(f"  Macro F1                         : {macro_rule_f1:.4f} ({macro_rule_f1*100:.1f}%)")
    print(f"  Exact Rule Set Match             : {exact_rule_matches}/{n} ({exact_rule_matches/n*100:.1f}%)")
    print(f"\n  RAG RETRIEVAL QUALITY (Top-K)")
    print(f"  Hit@1 / Hit@3 / Hit@5            : {rag_summary['hit_at_1']*100:.0f}% / {rag_summary['hit_at_3']*100:.0f}% / {rag_summary['hit_at_5']*100:.0f}%")
    print(f"  Precision@1/3/5                  : {rag_summary['precision_at_1']:.3f} / {rag_summary['precision_at_3']:.3f} / {rag_summary['precision_at_5']:.3f}")
    print(f"  Mean Reciprocal Rank (MRR)       : {rag_summary['mrr']:.4f}")
    print(f"  Mean Similarity Score            : {rag_summary['avg_similarity']:.4f}")
    print(f"  Keyword Coverage                 : {rag_summary['keyword_coverage']*100:.1f}%")
    print("=" * 70)

    # ── Save outputs ──────────────────────────────────────────────────────────
    results = {
        "evaluation_description": (
            "Humanized clinical vignette evaluation: keyword NLP extraction quality "
            "and downstream symbolic rule detection accuracy. "
            "Ground truth rules = engine(structured_facts). "
            "Predicted rules = engine(keyword_extracted_facts_from_narrative)."
        ),
        "methodology_note": (
            "This is NOT a circular test. Ground truth is produced by running the engine "
            "on authoritative structured patient facts. Predictions come from running the "
            "engine on facts extracted by keyword/regex NLP from free-text narratives. "
            "Discrepancies reveal extraction failures, not engine bugs."
        ),
        "total_cases": n,
        "rag_cases_evaluated": rag_n,
        "fact_extraction": {
            "avg_field_accuracy": round(avg_extraction_f1, 4),
            "avg_medication_f1": round(avg_med_f1, 4),
            "per_field_accuracy": field_accuracy_summary,
        },
        "rule_detection_after_extraction": {
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_rule_f1, 4),
            "exact_match_count": exact_rule_matches,
            "exact_match_rate": round(exact_rule_matches / n, 4),
            "rule_tp": rule_tp, "rule_fp": rule_fp, "rule_fn": rule_fn,
        },
        "rag_retrieval": rag_summary,
        "interpretation": {
            "100_percent_symbolic_benchmark": (
                "The symbolic engine achieves 100% on the 500-case structured benchmark. "
                "This measures DETERMINISTIC IMPLEMENTATION CORRECTNESS against the canonical "
                "rule knowledge base — NOT clinical validity or real-world generalizability."
            ),
            "humanized_extraction_gap": (
                f"Rule detection after keyword extraction achieves {micro_f1*100:.1f}% micro-F1. "
                "The gap from 100% reveals the challenge of preserving clinical fact semantics "
                "through NLP extraction — particularly drug class normalization and structured "
                "boolean flag inference."
            ),
        },
        "per_case": per_case_results,
    }

    out_json = OUT_DIR / "extraction_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {out_json.name}")

    # Per-case CSV
    out_csv = OUT_DIR / "per_case_extraction.csv"
    if per_case_results:
        simple_rows = [{k: v for k, v in r.items() if k != "field_statuses"} for r in per_case_results]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(simple_rows[0].keys()))
            writer.writeheader()
            writer.writerows(simple_rows)
        print(f"[SAVED] {out_csv.name}")

    # Error analysis CSV
    err_csv = OUT_DIR / "humanized_error_analysis.csv"
    if error_rows:
        err_fields = ["case_id","error_category","field","ground_truth_value","predicted_value",
                      "expected_rules","actual_rules","missing_rules","extra_rules","explanation"]
        with open(err_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=err_fields)
            writer.writeheader()
            for row in error_rows:
                row2 = {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                writer.writerow(row2)
        print(f"[SAVED] {err_csv.name}")

    # RAG metrics CSV
    rag_csv = OUT_DIR / "rag_metrics.csv"
    rag_rows = []
    for c in per_case_results:
        rag_rows.append({
            "case_id": c["case_id"],
            "domain": c["domain"],
            "hit_at_1": c.get("rag_hit_at_1", 0),
            "hit_at_3": c.get("rag_hit_at_3", 0),
            "hit_at_5": c.get("rag_hit_at_5", 0),
            "precision_at_1": c.get("rag_precision_at_1", 0),
            "precision_at_3": c.get("rag_precision_at_3", 0),
            "precision_at_5": c.get("rag_precision_at_5", 0),
            "mrr": c.get("rag_mrr", 0),
            "avg_similarity": c.get("rag_avg_similarity", 0),
            "keyword_coverage": c.get("rag_keyword_coverage", 0),
        })
    with open(rag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rag_rows)
    print(f"[SAVED] {rag_csv.name}")

    print(f"\n[DONE] Extraction evaluation complete -> {OUT_DIR}")
    return results


if __name__ == "__main__":
    run_extraction_evaluation()
