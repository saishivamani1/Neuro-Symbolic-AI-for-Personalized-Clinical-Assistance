"""
app/reasoning/rules.py

Clinical Rule Knowledge Base interface.

Loads the external deterministic clinical rule set from neurosymbolic_clinical/rules.csv
via app.reasoning.loader instead of hard-coding rule definitions in Python.

Provides backward-compatible exports and legacy rule ID mappings.
"""

from __future__ import annotations

from typing import Dict, List

from app.reasoning.loader import get_clinical_rules, get_rule_by_id
from app.reasoning.models import ClinicalRule

# --------------------------------------------------------------------------- #
# Legacy Rule ID to Benchmark Rule ID Mapping
# --------------------------------------------------------------------------- #

OLD_TO_NEW_RULE_MAP: Dict[str, str] = {
    "RULE_CKD_METFORMIN_CONTRAINDICATION_001": "R001",  # Metformin severe CKD (eGFR < 30)
    "RULE_CKD_METFORMIN_DOSE_LIMIT_002": "R003",        # Metformin reassess (eGFR 30-44)
    "RULE_DDI_ACEI_SPIRONOLACTONE_HYPERKALEMIA_003": "R015", # ACEI / Spironolactone hyperkalemia
    "RULE_LAB_HYPERKALEMIA_CRITICAL_004": "R015",       # Hyperkalemia safety alert
    "RULE_DX_STAGE_2_HYPERTENSION_005": "R022",         # Stage 2 hypertension SBP >= 140
    "RULE_TX_DIABETIC_CKD_ACEI_INDICATION_006": "R011", # ACEI monitoring in low eGFR
    "RULE_TX_SGLT2_CARDIORENAL_INDICATION_007": "R004", # SGLT2 CKD eligible
    "RULE_RISK_UNCONTROLLED_GLYCEMIA_008": "R006",      # Annual screening
    "RULE_MONITOR_ACEI_POTASSIUM_LAB_009": "R011",      # Monitoring low eGFR with ACEi
    "RULE_MONITOR_METFORMIN_ANNUAL_RENAL_010": "R006",  # Annual kidney screening
}

# --------------------------------------------------------------------------- #
# Evidence-Based Clinical Rule Set (Loaded externally from rules.csv)
# --------------------------------------------------------------------------- #

def _load_default_rules() -> List[ClinicalRule]:
    """Load the 50 deterministic clinical rules from rules.csv."""
    try:
        return get_clinical_rules()
    except Exception as exc:
        # Fallback to empty list during isolated import checks if file path unresolved
        return []


DEFAULT_CLINICAL_RULES: List[ClinicalRule] = _load_default_rules()


def get_default_rules() -> List[ClinicalRule]:
    """Always return fresh or cached rules from external rules.csv."""
    return get_clinical_rules()
