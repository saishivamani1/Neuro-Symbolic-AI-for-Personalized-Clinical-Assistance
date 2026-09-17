"""
backend/evaluation/generate_ground_truth.py

Canonical Reference Ground-Truth Generator.

PURPOSE
-------
Provides an INDEPENDENT, SIMPLE reference implementation of clinical rule
evaluation that is used as the ground-truth oracle for humanized benchmark
evaluation. It intentionally does NOT implement:

  - Drug class expansion (lisinopril -> ace_inhibitor)
  - Forward-chaining derived fact cascades
  - Nested dictionary resolution beyond one level

This allows us to measure what the full SymbolicRuleEngine ADDS beyond
naive direct condition evaluation, and to detect any engine bugs where the
engine fires a rule that the reference would not (or vice versa).

The reference evaluator reads rules.csv DIRECTLY (same source of truth as
the engine) and applies conditions using the same logical operators but via
a minimal, obviously-correct implementation.

RESEARCH USE
------------
  reference_output = reference_evaluate(structured_facts, rules)
  engine_output    = SymbolicRuleEngine().evaluate(structured_facts)

  Rules in BOTH     -> unambiguously correct fires
  Rules in ENGINE ONLY  -> engine's advanced reasoning (drug expansion / FC)
  Rules in REFERENCE ONLY -> potential engine suppression bug
  Rules in NEITHER  -> correctly suppressed

Usage:
    python backend/evaluation/generate_ground_truth.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
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

# ─── Output paths ─────────────────────────────────────────────────────────────
OUT_DIR = _ROOT / "evaluation_results" / "humanized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALERT_CONCLUSION_TYPES = {
    "safety_alert", "risk_alert", "contraindication",
    "safety_monitoring", "safety_consideration",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  REFERENCE CONDITION EVALUATOR
#  Simple, no forward chaining, no drug expansion, no nested resolution beyond
#  one level. Uses same operator names as rules.csv.
# ═══════════════════════════════════════════════════════════════════════════════

def _ref_resolve(facts: Dict[str, Any], fact_name: str) -> Any:
    """
    Minimal fact resolver:
    1. Root-level key lookup
    2. One-level nested lookup under labs/lab_results/vitals
    No aliases. No drug class expansion. No forward chaining.
    """
    if fact_name in facts:
        return facts[fact_name]

    # One level of nesting
    for sub in ("labs", "lab_results", "vitals"):
        sub_dict = facts.get(sub)
        if isinstance(sub_dict, dict) and fact_name in sub_dict:
            return sub_dict[fact_name]

    return None  # UNKNOWN — never treated as False


def _ref_eval_condition(cond: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    """Evaluate a single condition dict against patient facts.

    Returns True if the condition is satisfied, False if not or if the
    required fact is UNKNOWN (None).  Treating UNKNOWN as False is the
    safest clinical default — we never assume absence.
    """
    fact_name = cond.get("fact") or cond.get("fact_path", "")
    op = str(cond.get("operator", cond.get("op", ""))).strip()
    expected = cond.get("value")

    actual = _ref_resolve(facts, fact_name)

    # --- emptiness operators ---
    if op == "is_empty":
        return actual is None or actual == "" or actual == [] or actual == {}
    if op == "is_not_empty":
        return actual is not None and actual != "" and actual != [] and actual != {}

    # UNKNOWN: if actual is None and not an emptiness check, condition FAILS
    if actual is None:
        return False

    # --- equality ---
    if op == "==":
        if isinstance(actual, bool) and isinstance(expected, bool):
            return actual is expected
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return float(actual) == float(expected)
        return str(actual).strip().lower() == str(expected).strip().lower()

    if op == "!=":
        if isinstance(actual, bool) and isinstance(expected, bool):
            return actual is not expected
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return float(actual) != float(expected)
        return str(actual).strip().lower() != str(expected).strip().lower()

    # --- numeric comparisons ---
    if op in ("<", "<=", ">", ">="):
        try:
            a = float(actual)
            e = float(expected)
            return {"<": a < e, "<=": a <= e, ">": a > e, ">=": a >= e}[op]
        except (ValueError, TypeError):
            return False

    # --- containment (list / string) ---
    if op == "contains":
        exp_str = str(expected).strip().lower()
        if isinstance(actual, list):
            return any(exp_str in str(item).strip().lower() for item in actual)
        if isinstance(actual, str):
            return exp_str in actual.strip().lower()
        return False

    if op == "not_contains":
        if actual is None or actual == [] or actual == "":
            return True
        exp_str = str(expected).strip().lower()
        if isinstance(actual, list):
            return not any(exp_str in str(item).strip().lower() for item in actual)
        if isinstance(actual, str):
            return exp_str not in actual.strip().lower()
        return True

    # --- between ---
    if op == "between":
        if isinstance(expected, (list, tuple)) and len(expected) == 2:
            try:
                return float(expected[0]) <= float(actual) <= float(expected[1])
            except (ValueError, TypeError):
                return False
        return False

    return False


def reference_evaluate(
    facts: Dict[str, Any],
    rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reference ground-truth evaluation — no forward chaining, no expansion.

    Returns:
        {
          "expected_rules": list of fired rule_ids,
          "expected_derived_facts": list of conclusion fact names,
          "expected_alerts": list of alert fact names,
          "rule_details": list of dicts with per-rule evaluation detail
        }
    """
    expected_rules: List[str] = []
    expected_facts: List[str] = []
    expected_alerts: List[str] = []
    rule_details: List[Dict[str, Any]] = []

    for rule in rules:
        rid = rule["rule_id"]
        logic = str(rule.get("logic", "AND")).upper()
        conds = rule.get("conditions", [])
        conclusion = rule.get("conclusion", {})

        cond_results = [_ref_eval_condition(c, facts) for c in conds]

        fired = all(cond_results) if logic == "AND" else any(cond_results)

        detail = {
            "rule_id": rid,
            "rule_name": rule.get("rule_name", ""),
            "fired": fired,
            "logic": logic,
            "condition_results": cond_results,
        }

        if fired:
            expected_rules.append(rid)
            conc_fact = conclusion.get("fact", f"{rid}_fired")
            conc_type = conclusion.get("type", "inference")
            expected_facts.append(conc_fact)
            if conc_type in ALERT_CONCLUSION_TYPES:
                expected_alerts.append(conc_fact)
            detail["conclusion_fact"] = conc_fact
            detail["conclusion_type"] = conc_type

        rule_details.append(detail)

    return {
        "expected_rules": expected_rules,
        "expected_derived_facts": expected_facts,
        "expected_alerts": expected_alerts,
        "rule_details": rule_details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  RULES.CSV LOADER (standalone — does NOT use app.reasoning.loader)
# ═══════════════════════════════════════════════════════════════════════════════

def load_rules_raw(csv_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load rules.csv into raw dicts without Pydantic models."""
    if csv_path is None:
        candidates = [
            _ROOT / "neurosymbolic_clinical" / "rules.csv",
            _BACKEND.parent / "neurosymbolic_clinical" / "rules.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        else:
            raise FileNotFoundError("rules.csv not found in expected locations.")

    rules: List[Dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["rule_id"].strip()
            if not rid:
                continue
            rule = {
                "rule_id": rid,
                "rule_name": row.get("rule_name", "").strip(),
                "domain": row.get("domain", "").strip(),
                "priority": row.get("priority", "medium").strip(),
                "logic": row.get("logic", "AND").strip().upper(),
                "conditions": json.loads(row.get("conditions_json", "[]")),
                "conclusion": json.loads(row.get("conclusion_json", "{}")),
                "explanation": row.get("explanation", "").strip(),
                "evidence_source": row.get("evidence_source", "").strip(),
                "evidence_reference": row.get("evidence_reference", "").strip(),
                "validation_status": row.get("validation_status", "validated").strip(),
            }
            rules.append(rule)
    return rules


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATE AGAINST TEST_CASES.CSV (500 synthetic structured cases)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_against_synthetic_benchmark(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare reference evaluator output against the hand-curated expected_rules
    in test_cases.csv (500 synthetic structured benchmark cases).

    This is NOT circular: the expected_rules in test_cases.csv are externally
    defined and not produced by the reference evaluator.
    """
    tc_path = _ROOT / "neurosymbolic_clinical" / "test_cases.csv"
    if not tc_path.exists():
        return {"error": "test_cases.csv not found"}

    tp = fp = fn = tn = 0
    exact_matches = 0
    n = 0

    with open(tc_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_raw = row.get("patient_json", "{}")
            try:
                pf = json.loads(patient_raw)
            except json.JSONDecodeError:
                continue

            exp_raw = row.get("expected_rules", "").strip()
            expected_set: Set[str] = set()
            if exp_raw:
                for r in exp_raw.split(","):
                    r = r.strip()
                    if r:
                        expected_set.add(r)

            ref_out = reference_evaluate(pf, rules)
            predicted_set = set(ref_out["expected_rules"])

            case_tp = len(predicted_set & expected_set)
            case_fp = len(predicted_set - expected_set)
            case_fn = len(expected_set - predicted_set)
            case_tn = 50 - case_tp - case_fp - case_fn
            tp += case_tp
            fp += case_fp
            fn += case_fn
            tn += max(0, case_tn)

            if predicted_set == expected_set:
                exact_matches += 1
            n += 1

    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "dataset": "synthetic_benchmark_500",
        "n_cases": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match_count": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4) if n > 0 else 0.0,
        "interpretation": (
            "Reference evaluator agreement with externally-curated test_cases.csv. "
            "Scores < 100% reveal differences between reference semantics and test-case "
            "ground truth (e.g. drug expansion / forward chaining cases)."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE CORRECTED HUMANIZED GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

def generate_humanized_ground_truth(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each humanized case, run the reference evaluator on structured_patient_facts.
    This produces the correct ground-truth labels without assuming any rule semantics.
    """
    hum_path = _BACKEND / "evaluation" / "humanized_cases.json"
    if not hum_path.exists():
        return []

    with open(hum_path, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        pf = case.get("patient_facts", {})
        ref = reference_evaluate(pf, rules)
        results.append({
            "case_id": case["case_id"],
            "domain": case.get("domain", ""),
            "case_type": case.get("case_type", ""),
            "reference_rules": ref["expected_rules"],
            "reference_facts": ref["expected_derived_facts"],
            "reference_alerts": ref["expected_alerts"],
            "original_expected_rules": case.get("expected_rules", []),
            "label_changed": sorted(ref["expected_rules"]) != sorted(case.get("expected_rules", [])),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARE REFERENCE VS FULL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compare_reference_vs_engine(
    rules: List[Dict[str, Any]],
    hum_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run both reference evaluator and full engine on structured facts, compare outputs."""
    from app.reasoning.engine import SymbolicRuleEngine
    engine = SymbolicRuleEngine()

    comparison = []
    engine_only_total = 0
    ref_only_total = 0
    both_total = 0
    neither_total = 0

    for case in hum_cases:
        pf = case.get("patient_facts", {})
        cid = case["case_id"]

        ref = reference_evaluate(pf, rules)
        eng = engine.evaluate(pf)

        ref_set = set(ref["expected_rules"])
        eng_set = set(r.rule_id for r in eng.triggered_rules)

        in_both = ref_set & eng_set
        in_engine_only = eng_set - ref_set
        in_ref_only = ref_set - eng_set

        n_total = len(rules)
        in_neither = n_total - len(in_both) - len(in_engine_only) - len(in_ref_only)

        engine_only_total += len(in_engine_only)
        ref_only_total += len(in_ref_only)
        both_total += len(in_both)
        neither_total += max(0, in_neither)

        comparison.append({
            "case_id": cid,
            "in_both": sorted(in_both),
            "engine_only": sorted(in_engine_only),
            "reference_only": sorted(in_ref_only),
            "interpretation": {
                "engine_only": "Engine fires via drug expansion / forward chaining",
                "reference_only": "Potential engine suppression or resolution gap",
                "in_both": "Unambiguously correctly fired",
            }
        })

    return {
        "total_rule_slots": len(rules) * len(hum_cases),
        "in_both": both_total,
        "engine_only_additions": engine_only_total,
        "reference_only_misses": ref_only_total,
        "in_neither": neither_total,
        "engine_additions_meaning": (
            "These rules fire in the full engine but NOT in the reference evaluator. "
            "This is expected: the engine performs drug-class expansion (e.g. "
            "lisinopril -> ace_inhibitor) and forward chaining that the reference "
            "evaluator intentionally omits."
        ),
        "reference_misses_meaning": (
            "These rules fire in the reference but NOT the full engine. "
            "If > 0, investigate for engine suppression bugs."
        ),
        "per_case": comparison,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE VALIDATION ISSUES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rule_validation_issues(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flag structural or coverage issues in rules.csv without modifying the rules.
    """
    issues = []

    # Rules with no conditions
    for r in rules:
        if not r["conditions"]:
            issues.append({
                "rule_id": r["rule_id"],
                "issue": "Rule has zero conditions — fires on every patient",
                "source": "rules.csv schema audit",
                "recommendation": "Add at least one condition or remove rule",
            })

    # Rules that fire on broad single conditions (coverage / false-positive risk)
    for r in rules:
        if len(r["conditions"]) == 1:
            cond = r["conditions"][0]
            fact = cond.get("fact", "")
            op = cond.get("operator", cond.get("op", ""))
            if fact in ("type2_diabetes", "ckd", "heart_failure") and op == "==":
                issues.append({
                    "rule_id": r["rule_id"],
                    "issue": f"Single-condition rule on broad flag '{fact}' — high population coverage",
                    "source": "rules.csv schema audit",
                    "recommendation": "Consider adding specificity conditions",
                })
            if fact == "egfr" and op in ("<", "<="):
                val = cond.get("value", 0)
                if val >= 60:
                    issues.append({
                        "rule_id": r["rule_id"],
                        "issue": f"Single eGFR < {val} rule fires for majority of CKD population",
                        "source": "rules.csv schema audit",
                        "recommendation": "Review whether scope is intentional",
                    })

    # Rules whose conclusion fact matches another rule's condition (forward-chain dependency)
    conclusion_facts = {r["conclusion"].get("fact", ""): r["rule_id"] for r in rules}
    for r in rules:
        for cond in r["conditions"]:
            f = cond.get("fact", "")
            if f in conclusion_facts and conclusion_facts[f] != r["rule_id"]:
                issues.append({
                    "rule_id": r["rule_id"],
                    "issue": f"Condition fact '{f}' is derived by rule {conclusion_facts[f]} — forward chaining dependency",
                    "source": "dependency analysis",
                    "recommendation": "Document dependency chain; reference evaluator will not resolve this",
                })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Canonical Ground-Truth Generator & Engine Validator")
    print("=" * 70)

    # 1. Load rules (independent — no app imports needed)
    print("\n[1/5] Loading rules.csv...")
    rules = load_rules_raw()
    print(f"  Loaded {len(rules)} rules from rules.csv")

    # 2. Validate reference evaluator against 500-case synthetic benchmark
    print("\n[2/5] Validating reference evaluator against synthetic benchmark...")
    synth = validate_against_synthetic_benchmark(rules)
    print(f"  Cases: {synth['n_cases']}")
    print(f"  Micro Precision : {synth['micro_precision']:.4f}")
    print(f"  Micro Recall    : {synth['micro_recall']:.4f}")
    print(f"  Micro F1        : {synth['micro_f1']:.4f}")
    print(f"  Exact Match     : {synth['exact_match_count']}/{synth['n_cases']}")
    print(f"  NOTE: {synth['interpretation']}")

    # 3. Generate humanized ground truth
    print("\n[3/5] Generating corrected humanized ground truth...")
    hum_cases_path = _BACKEND / "evaluation" / "humanized_cases.json"
    if hum_cases_path.exists():
        with open(hum_cases_path, encoding="utf-8") as f:
            hum_cases = json.load(f)

        gt_results = generate_humanized_ground_truth(rules)
        changed = sum(1 for r in gt_results if r["label_changed"])
        print(f"  Humanized cases: {len(gt_results)}")
        print(f"  Cases where original expected_rules were INCORRECT: {changed}")
        for r in gt_results:
            if r["label_changed"]:
                print(f"    {r['case_id']}: was {r['original_expected_rules']} "
                      f"-> now {r['reference_rules']}")

        # Save corrected ground truth
        gt_out = OUT_DIR / "reference_ground_truth.json"
        with open(gt_out, "w", encoding="utf-8") as f:
            json.dump(gt_results, f, indent=2)
        print(f"  [SAVED] {gt_out.name}")
    else:
        hum_cases = []
        gt_results = []
        print("  humanized_cases.json not found — skipping")

    # 4. Compare reference vs full engine
    if hum_cases:
        print("\n[4/5] Comparing reference evaluator vs full SymbolicRuleEngine...")
        comp = compare_reference_vs_engine(rules, hum_cases)
        print(f"  Total rule slots     : {comp['total_rule_slots']}")
        print(f"  Fired by BOTH        : {comp['in_both']}")
        print(f"  Engine-only (expansion/FC) : {comp['engine_only_additions']}")
        print(f"  Reference-only (gaps): {comp['reference_only_misses']}")
        print(f"  Engine additions     : {comp['engine_additions_meaning']}")
        if comp["reference_only_misses"] > 0:
            print(f"  [WARNING] {comp['reference_only_misses']} reference-only cases — investigate engine bugs")

        comp_out = OUT_DIR / "reference_vs_engine_comparison.json"
        with open(comp_out, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        print(f"  [SAVED] {comp_out.name}")

    # 5. Generate rule validation issues
    print("\n[5/5] Generating rule validation issues...")
    issues = generate_rule_validation_issues(rules)
    issues_out = _ROOT / "evaluation_results" / "rule_validation_issues.csv"
    if issues:
        with open(issues_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["rule_id","issue","source","recommendation"])
            writer.writeheader()
            writer.writerows(issues)
        print(f"  Found {len(issues)} structural issues -> {issues_out.name}")
    else:
        print("  No structural issues found.")

    # Save synthetic benchmark results
    synth_out = OUT_DIR / "reference_evaluator_synthetic_validation.json"
    with open(synth_out, "w", encoding="utf-8") as f:
        json.dump(synth, f, indent=2)

    print(f"\n[DONE] Ground truth generator complete.")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
