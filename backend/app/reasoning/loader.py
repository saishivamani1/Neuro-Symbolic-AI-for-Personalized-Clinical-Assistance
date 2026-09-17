"""
app/reasoning/loader.py

External Clinical Rule Knowledge Base Loader.

Loads, validates, and caches the 50 evidence-grounded rules from
neurosymbolic_clinical/rules.csv into immutable ClinicalRule instances.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.exceptions import RuleEngineError
from app.core.logging import get_logger
from app.reasoning.models import ClinicalRule, RuleCondition, RuleConclusion

logger = get_logger(__name__)

_CACHED_RULES: Optional[List[ClinicalRule]] = None
_CACHED_RULE_MAP: Optional[Dict[str, ClinicalRule]] = None


def find_rules_csv_path() -> Path:
    """Locate neurosymbolic_clinical/rules.csv searching workspace directories."""
    search_candidates = [
        Path("neurosymbolic_clinical/rules.csv"),
        Path("../neurosymbolic_clinical/rules.csv"),
        Path(__file__).resolve().parent.parent.parent.parent / "neurosymbolic_clinical" / "rules.csv",
        Path(os.getcwd()) / "neurosymbolic_clinical" / "rules.csv",
        Path(os.getcwd()).parent / "neurosymbolic_clinical" / "rules.csv",
    ]
    for candidate in search_candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find neurosymbolic_clinical/rules.csv. Searched locations: "
        + ", ".join(str(c) for c in search_candidates)
    )


def load_rules_from_csv(csv_path: Optional[Path | str] = None) -> List[ClinicalRule]:
    """Parse neurosymbolic_clinical/rules.csv into validated ClinicalRule models."""
    path = Path(csv_path) if csv_path else find_rules_csv_path()
    if not path.is_file():
        raise RuleEngineError(
            detail=f"Rule knowledge base CSV not found at '{path}'.",
            context={"path": str(path)},
        )

    logger.info("Loading deterministic clinical rules from external CSV: %s", path)
    rules: List[ClinicalRule] = []
    seen_rule_ids = set()

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            rule_id = row.get("rule_id", "").strip()
            if not rule_id:
                continue

            if rule_id in seen_rule_ids:
                raise RuleEngineError(
                    detail=f"Duplicate rule_id '{rule_id}' discovered on CSV line {row_idx}.",
                    context={"rule_id": rule_id, "line": row_idx},
                )
            seen_rule_ids.add(rule_id)

            # Parse conditions_json
            conds_raw = row.get("conditions_json", "").strip()
            try:
                conds_data = json.loads(conds_raw)
                conditions = [RuleCondition(**c) for c in conds_data]
            except Exception as exc:
                raise RuleEngineError(
                    detail=f"Malformed conditions_json in rule '{rule_id}' line {row_idx}: {exc}",
                    context={"rule_id": rule_id, "raw": conds_raw},
                ) from exc

            # Parse conclusion_json
            conc_raw = row.get("conclusion_json", "").strip()
            try:
                conc_data = json.loads(conc_raw)
                conclusion = RuleConclusion(**conc_data)
            except Exception as exc:
                raise RuleEngineError(
                    detail=f"Malformed conclusion_json in rule '{rule_id}' line {row_idx}: {exc}",
                    context={"rule_id": rule_id, "raw": conc_raw},
                ) from exc

            rule = ClinicalRule(
                rule_id=rule_id,
                rule_name=row.get("rule_name", "").strip(),
                domain=row.get("domain", "clinical").strip(),
                priority=row.get("priority", "medium").strip(),
                conditions=conditions,
                logic=row.get("logic", "AND").strip().upper(),
                conclusion=conclusion,
                explanation=row.get("explanation", "").strip(),
                evidence_source=row.get("evidence_source", "").strip(),
                evidence_reference=row.get("evidence_reference", "").strip(),
                validation_status=row.get("validation_status", "validated").strip(),
            )
            rules.append(rule)

    logger.info("Successfully loaded %d deterministic clinical rules from %s", len(rules), path)
    return rules


def get_clinical_rules(force_reload: bool = False) -> List[ClinicalRule]:
    """Retrieve cached clinical rules or load them once from CSV."""
    global _CACHED_RULES, _CACHED_RULE_MAP
    if _CACHED_RULES is None or force_reload:
        _CACHED_RULES = load_rules_from_csv()
        _CACHED_RULE_MAP = {r.rule_id: r for r in _CACHED_RULES}
    return _CACHED_RULES


def get_rule_by_id(rule_id: str) -> Optional[ClinicalRule]:
    """Lookup a cached clinical rule by its ID."""
    global _CACHED_RULE_MAP
    if _CACHED_RULE_MAP is None:
        get_clinical_rules()
    return _CACHED_RULE_MAP.get(rule_id) if _CACHED_RULE_MAP else None
