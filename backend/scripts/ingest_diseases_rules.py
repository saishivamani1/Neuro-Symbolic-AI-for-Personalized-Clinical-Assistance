"""
scripts/ingest_diseases_rules.py

Compiles evidence-grounded symbolic clinical decision rules for diseases in
backend/data/documents/cleaned_diseases.json and populates backend/data/rules.json.

Generates:
1. Diagnostic / Symptom Screening Rules (RuleCategory.DIAGNOSIS)
2. Guideline Treatment Protocol Rules (RuleCategory.TREATMENT_RECOMMENDATION)
3. Urgent Escalation & Complications Rules (RuleCategory.MONITORING / RuleCategory.CONTRAINDICATION)

Preserves core clinical guideline rules (DEFAULT_CLINICAL_RULES).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.logging import configure_logging, get_logger
from app.reasoning.models import (
    ClinicalRule,
    DerivedFact,
    RuleCategory,
    RuleCondition,
    RuleOperator,
    RuleSeverity,
)
from app.reasoning.rules import DEFAULT_CLINICAL_RULES

logger = get_logger(__name__)


def slugify(text: str) -> str:
    """Normalize text into an uppercase alphanumeric underscore code."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").upper()
    return slug[:40] if len(slug) > 40 else slug


def clean_summary_text(text: str, max_length: int = 280) -> str:
    """Clean and truncate medical text to a readable sentence-bounded summary."""
    if not text:
        return ""
    # Normalize whitespace and newlines
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if len(cleaned) <= max_length:
        return cleaned
    truncated = cleaned[:max_length]
    # Break at last sentence or punctuation if possible
    last_dot = max(truncated.rfind(". "), truncated.rfind("; "))
    if last_dot > 100:
        return truncated[: last_dot + 1]
    return truncated.rstrip() + "..."


def extract_symptoms_for_disease(
    record: Dict[str, Any], disease_name: str
) -> List[str]:
    """Extract 1 to 3 distinctive clinical symptoms for a disease."""
    symptoms: List[str] = []
    dis_lower = disease_name.lower()

    # 1. From keywords
    keywords = record.get("keywords") or []
    for kw in keywords:
        kw_clean = kw.strip()
        kw_lower = kw_clean.lower()
        if (
            len(kw_clean) >= 3
            and kw_lower not in dis_lower
            and dis_lower not in kw_lower
            and not any(x in kw_lower for x in ["disease", "syndrome", "disorder", "mayo", "clinic"])
        ):
            capitalized = kw_clean.capitalize()
            if capitalized not in symptoms:
                symptoms.append(capitalized)

    # 2. From Symptoms section if keywords gave few
    if len(symptoms) < 2:
        sym_text = record.get("Symptoms") or ""
        for line in sym_text.split("\n"):
            line_clean = line.strip().rstrip(".")
            if (
                4 <= len(line_clean) <= 40
                and not line_clean.lower().startswith(("symptoms", "when to", "the following", "some people", "types of"))
                and line_clean.lower() not in dis_lower
            ):
                capitalized = line_clean.capitalize()
                if capitalized not in symptoms:
                    symptoms.append(capitalized)
            if len(symptoms) >= 4:
                break

    return symptoms[:3]


def extract_red_flag_warnings(
    record: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """Check for emergency/urgent escalation red-flags in When to see a doctor or Complications.

    Returns (warning_explanation, matched_symptom_or_flag)
    """
    see_doc = record.get("When to see a doctor") or ""
    complications = record.get("Complications") or ""
    combined = f"{see_doc}\n{complications}"
    combined_lower = combined.lower()

    # High-priority red-flag terms
    emergency_patterns = [
        ("chest pain", "Chest pain", "Seek immediate medical attention for chest pain to rule out acute myocardial infarction."),
        ("shortness of breath", "Shortness of breath", "Acute difficulty breathing requires immediate emergency clinical evaluation."),
        ("stroke", "Stroke symptoms", "High risk of thromboembolic stroke. Immediate emergency intervention required."),
        ("blood clot", "Blood clots", "Risk of thromboembolic complications requiring urgent anticoagulant evaluation."),
        ("loss of vision", "Vision loss", "Sudden vision changes warrant emergency ophthalmologic assessment."),
        ("fainting", "Fainting / Syncope", "Unexplained syncope warrants prompt cardiovascular evaluation."),
        ("high fever", "High fever", "Severe fever spikes may signal systemic or bacteremic infection."),
        ("severe pain", "Severe acute pain", "Acute severe pain requires urgent diagnostic workup to exclude acute surgical abdomen or crisis."),
        ("seek immediate", "Emergency symptoms", "Clinical protocol indicates immediate emergency medical attention."),
    ]

    sentences = [s.strip() for s in re.split(r"[.\n]+", combined) if s.strip()]

    for term, flag_name, default_expl in emergency_patterns:
        if term in combined_lower:
            expl = default_expl
            for s in sentences:
                if term in s.lower() and 20 <= len(s) <= 300:
                    expl = s
                    break
            return clean_summary_text(expl, 250), flag_name

    return None, None


class DiseaseRulesCompiler:
    """Compiles evidence-grounded ClinicalRule objects from cleaned_diseases.json."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.raw_data: List[Dict[str, Any]] = []

    def load_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Disease data file not found at: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        logger.info("Loaded %d disease records from %s", len(self.raw_data), self.data_path.name)

    def compile_all_rules(self) -> List[ClinicalRule]:
        """Compile complete clinical ruleset incorporating default and disease rules."""
        if not self.raw_data:
            self.load_data()

        compiled_rules: List[ClinicalRule] = []
        rule_ids_seen: Set[str] = set()

        # 1. First add all DEFAULT_CLINICAL_RULES (preserves core rules & top priority)
        for rule in DEFAULT_CLINICAL_RULES:
            if rule.rule_id not in rule_ids_seen:
                rule_ids_seen.add(rule.rule_id)
                compiled_rules.append(rule)

        # 2. Iterate through each disease and compile domain rules
        for idx, record in enumerate(self.raw_data):
            disease_name = (record.get("disease") or "").strip()
            if not disease_name:
                continue

            slug = slugify(disease_name)
            source_urls = record.get("source_urls") or {}
            main_url = source_urls.get("main") or "https://www.mayoclinic.org"
            tx_url = source_urls.get("diagnosis_treatment") or main_url

            # -------------------------------------------------------------
            # A. Guideline Treatment Recommendation Rule
            # -------------------------------------------------------------
            treatment_text = (record.get("Treatment") or "").strip()
            if len(treatment_text) > 30:
                rule_id = f"RULE_RX_{slug}_{idx:03d}"
                fact_id = f"FACT_RX_{slug}_{idx:03d}"

                if rule_id not in rule_ids_seen:
                    tx_summary = clean_summary_text(treatment_text, 260)
                    tx_rule = ClinicalRule(
                        rule_id=rule_id,
                        name=f"{disease_name} Clinical Management Protocol",
                        category=RuleCategory.TREATMENT_RECOMMENDATION,
                        severity=RuleSeverity.MODERATE,
                        description=f"Evidence-based therapeutic protocol for confirmed {disease_name}.",
                        guideline_source=f"Mayo Clinic Reference: {disease_name}",
                        page_number=1,
                        section="Diagnosis and Treatment",
                        condition_logic="ALL",
                        priority=90,
                        conditions=[
                            RuleCondition(
                                fact_path="conditions",
                                operator=RuleOperator.CONTAINS,
                                value=disease_name,
                            )
                        ],
                        conclusions=[
                            DerivedFact(
                                fact_id=fact_id,
                                category="treatment_recommendation",
                                name=f"{disease_name} Guideline Management Protocol",
                                value="GUIDELINE_THERAPY_RECOMMENDED",
                                severity=RuleSeverity.MODERATE,
                                explanation=(
                                    f"Patient has diagnosed {disease_name}. Guideline protocol: {tx_summary}"
                                ),
                                confidence=0.95,
                                supporting_rules=[rule_id],
                                provenance={
                                    "guideline": f"Mayo Clinic Clinical Reference: {disease_name}",
                                    "section": "Treatment",
                                    "url": tx_url,
                                },
                            )
                        ],
                    )
                    rule_ids_seen.add(rule_id)
                    compiled_rules.append(tx_rule)

            # -------------------------------------------------------------
            # B. Diagnostic Screening Rule
            # -------------------------------------------------------------
            symptoms = extract_symptoms_for_disease(record, disease_name)
            if symptoms:
                rule_id = f"RULE_DX_{slug}_{idx:03d}"
                fact_id = f"FACT_DX_{slug}_{idx:03d}"

                if rule_id not in rule_ids_seen:
                    # Require 1 or 2 characteristic symptoms and absence of diagnosis
                    conditions: List[RuleCondition] = []
                    for sym in symptoms[:2]:
                        conditions.append(
                            RuleCondition(
                                fact_path="symptoms",
                                operator=RuleOperator.CONTAINS,
                                value=sym,
                            )
                        )
                    conditions.append(
                        RuleCondition(
                            fact_path="conditions",
                            operator=RuleOperator.NOT_CONTAINS,
                            value=disease_name,
                        )
                    )

                    dx_summary = clean_summary_text(
                        record.get("Overview") or record.get("Symptoms") or "", 220
                    )

                    dx_rule = ClinicalRule(
                        rule_id=rule_id,
                        name=f"{disease_name} Diagnostic Workup Indication",
                        category=RuleCategory.DIAGNOSIS,
                        severity=RuleSeverity.MODERATE,
                        description=(
                            f"Identification of classic symptom cluster indicating clinical evaluation for {disease_name}."
                        ),
                        guideline_source=f"Mayo Clinic Reference: {disease_name}",
                        page_number=1,
                        section="Symptoms & Clinical Diagnosis",
                        condition_logic="ALL",
                        priority=100,
                        conditions=conditions,
                        conclusions=[
                            DerivedFact(
                                fact_id=fact_id,
                                category="diagnosis",
                                name=f"{disease_name} Diagnostic Workup Indicated",
                                value="ORDER_DIAGNOSTIC_EVALUATION",
                                severity=RuleSeverity.MODERATE,
                                explanation=(
                                    f"Patient presents with characteristic symptoms ({', '.join(symptoms[:2])}) "
                                    f"without a confirmed diagnosis. Clinical overview: {dx_summary} "
                                    f"Formal clinical assessment and diagnostic workup for {disease_name} is indicated."
                                ),
                                confidence=0.88,
                                supporting_rules=[rule_id],
                                provenance={
                                    "guideline": f"Mayo Clinic Clinical Reference: {disease_name}",
                                    "section": "Diagnosis",
                                    "url": main_url,
                                },
                            )
                        ],
                    )
                    rule_ids_seen.add(rule_id)
                    compiled_rules.append(dx_rule)

            # -------------------------------------------------------------
            # C. Urgent Complication / Escalation Alert Rule
            # -------------------------------------------------------------
            warning_expl, flag_symptom = extract_red_flag_warnings(record)
            if warning_expl:
                rule_id = f"RULE_ALERT_{slug}_{idx:03d}"
                fact_id = f"FACT_ALERT_{slug}_{idx:03d}"

                if rule_id not in rule_ids_seen:
                    alert_conditions: List[RuleCondition] = [
                        RuleCondition(
                            fact_path="conditions",
                            operator=RuleOperator.CONTAINS,
                            value=disease_name,
                        )
                    ]
                    if flag_symptom:
                        alert_conditions.append(
                            RuleCondition(
                                fact_path="symptoms",
                                operator=RuleOperator.CONTAINS,
                                value=flag_symptom,
                            )
                        )

                    alert_rule = ClinicalRule(
                        rule_id=rule_id,
                        name=f"{disease_name} Urgent Clinical Escalation Alert",
                        category=RuleCategory.MONITORING,
                        severity=RuleSeverity.CRITICAL,
                        description=(
                            f"Critical escalation alert for high-risk complications in patients with {disease_name}."
                        ),
                        guideline_source=f"Mayo Clinic Reference: {disease_name}",
                        page_number=1,
                        section="When to see a doctor / Complications",
                        condition_logic="ALL",
                        priority=180,
                        conditions=alert_conditions,
                        conclusions=[
                            DerivedFact(
                                fact_id=fact_id,
                                category="monitoring",
                                name=f"{disease_name} Urgent Clinical Escalation Alert",
                                value="IMMEDIATE_CLINICAL_REVIEW",
                                severity=RuleSeverity.CRITICAL,
                                explanation=(
                                    f"CRITICAL CLINICAL ALERT for patient with {disease_name}: {warning_expl} "
                                    f"Immediate healthcare escalation or emergency evaluation required."
                                ),
                                confidence=0.98,
                                supporting_rules=[rule_id],
                                provenance={
                                    "guideline": f"Mayo Clinic Clinical Reference: {disease_name}",
                                    "section": "Emergency Warning Signs",
                                    "url": main_url,
                                },
                            )
                        ],
                    )
                    rule_ids_seen.add(rule_id)
                    compiled_rules.append(alert_rule)

        return compiled_rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and ingest disease clinical rules into data/rules.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/documents/cleaned_diseases.json"),
        help="Input path to cleaned_diseases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/rules.json"),
        help="Output path for compiled rules JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and validate rules without writing to disk",
    )
    args = parser.parse_args()

    configure_logging()

    print("=" * 75)
    print("  Symbolic Clinical Reasoning Engine: Disease Rules Ingestion")
    print("=" * 75)
    print()

    input_path = Path(args.input)
    output_path = Path(args.output)

    compiler = DiseaseRulesCompiler(input_path)
    print(f"[*] Compiling symbolic rules from: {input_path}")
    rules = compiler.compile_all_rules()

    print(f"[+] Successfully compiled and validated {len(rules)} clinical rules!")

    # Categorize counts
    category_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    for r in rules:
        cat = r.category.value if hasattr(r.category, "value") else str(r.category)
        sev = r.severity.value if hasattr(r.severity, "value") else str(r.severity)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print("\nRules by Category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cat:<26}: {count}")

    print("\nRules by Severity:")
    for sev, count in sorted(severity_counts.items(), key=lambda x: -x[1]):
        print(f"  - {sev:<26}: {count}")

    if args.dry_run:
        print("\n[*] Dry-run mode active. No changes written to disk.")
        return

    # Serialize & write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [r.model_dump() for r in rules]
    output_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[+] Saved {len(rules)} rules to {output_path} ({file_size_mb:.2f} MB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
