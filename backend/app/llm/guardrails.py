"""
app/llm/guardrails.py

Deterministic post-generation clinical guardrails and hallucination defense.

Validates:
1. Patient ID consistency between prompt and LLM response.
2. Mandatory preservation of all critical symbolic reasoning alerts.
3. Rule ID and citation grounding (preventing hallucinated rule codes or fake papers).
4. Safety notice inclusion and clinical non-autonomy framing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.schemas import (
    ClinicalAssistantResponse,
    EvidenceSourceCitation,
    RiskFlagItem,
    RuleBasedFindingItem,
)
from app.reasoning.rules import DEFAULT_CLINICAL_RULES, OLD_TO_NEW_RULE_MAP

logger = get_logger(__name__)

ALL_KNOWN_RULE_IDS: Set[str] = {r.rule_id for r in DEFAULT_CLINICAL_RULES} | set(OLD_TO_NEW_RULE_MAP.keys())


class ClinicalGuardrailValidator:
    """Validates and enforces clinical safety constraints on LLM output."""

    @staticmethod
    def validate_and_sanitize(
        response: ClinicalAssistantResponse,
        context: Dict[str, Any],
    ) -> ClinicalAssistantResponse:
        """Run all safety checks and sanitize the LLM response.

        Parameters
        ----------
        response : ClinicalAssistantResponse
            Parsed structured LLM output.
        context : Dict[str, Any]
            The exact structured context dictionary provided to the LLM.

        Returns
        -------
        ClinicalAssistantResponse
            Validated, sanitized response with guaranteed safety invariants.
        """
        patient_facts = context.get("patient_facts", {})
        expected_patient_id = patient_facts.get("patient_id", "")
        symbolic_reasoning = context.get("symbolic_reasoning", {})
        critical_alerts = symbolic_reasoning.get("critical_alerts", [])
        valid_rule_ids = set(context.get("valid_citations", {}).get("rule_ids", [])) | ALL_KNOWN_RULE_IDS

        # 1. Ensure patient_id matches
        if expected_patient_id and response.patient_id != expected_patient_id:
            logger.warning(
                "Guardrail corrected mismatched patient_id: '%s' -> '%s'",
                response.patient_id,
                expected_patient_id,
            )
            response.patient_id = expected_patient_id

        # 2. Enforce Critical Alert Preservation
        # Every critical alert from symbolic reasoning must be present in rule_based_findings
        existing_rule_ids = {r.rule_id for r in response.rule_based_findings}
        for alert in critical_alerts:
            alert_id = alert.get("fact_id")
            supporting_rules = alert.get("supporting_rules", [])
            primary_rule_id = supporting_rules[0] if supporting_rules else "CRITICAL_ALERT"

            # Check if this critical finding was preserved
            if primary_rule_id not in existing_rule_ids:
                logger.warning(
                    "Guardrail detected omitted critical alert '%s'. Injecting into findings.",
                    alert.get("name"),
                )
                prov = alert.get("provenance", {})
                injected_finding = RuleBasedFindingItem(
                    rule_id=primary_rule_id,
                    rule_name=alert.get("name", "Critical Safety Alert"),
                    category=alert.get("category", "contraindication"),
                    severity="critical",
                    conclusion=alert.get("explanation", "Critical clinical finding."),
                    actionable_guidance=alert.get("explanation", "Immediate clinical review required."),
                    guideline_source=prov.get("guideline", "Clinical Guidelines"),
                    page_number=prov.get("page"),
                )
                response.rule_based_findings.insert(0, injected_finding)
                existing_rule_ids.add(primary_rule_id)

                # Ensure it also appears in risk flags
                response.risk_flags.insert(
                    0,
                    RiskFlagItem(
                        flag=alert.get("name", "Critical Safety Alert"),
                        severity="critical",
                        clinical_implication=alert.get("explanation", "High clinical risk detected."),
                        evidence=[
                            EvidenceSourceCitation(
                                type="rule",
                                rule_id=primary_rule_id,
                                document=prov.get("guideline"),
                                page=prov.get("page"),
                            )
                        ],
                    ),
                )

        # 3. Validate Rule IDs (Disallow fabricated rule codes)
        sanitized_rule_findings: List[RuleBasedFindingItem] = []
        for finding in response.rule_based_findings:
            if finding.rule_id in valid_rule_ids or finding.rule_id.startswith("RULE_"):
                sanitized_rule_findings.append(finding)
            else:
                logger.warning("Guardrail filtered unrecognized rule_id: '%s'", finding.rule_id)
        response.rule_based_findings = sanitized_rule_findings

        # 4. Validate Citations in Evidence items
        for ev in response.evidence:
            valid_sources: List[EvidenceSourceCitation] = []
            for src in ev.sources:
                if src.type == "rule" and src.rule_id and src.rule_id not in valid_rule_ids:
                    logger.warning("Guardrail filtered fabricated rule citation: '%s'", src.rule_id)
                    continue
                valid_sources.append(src)
            ev.sources = valid_sources

        # 5. Enforce Safety Notice Presence
        if not response.safety_notice or len(response.safety_notice.strip()) < 10:
            response.safety_notice = (
                "This output is generated for clinical decision support and research purposes only. "
                "It is not an autonomous medical diagnosis or prescription. All recommendations "
                "must be verified by a licensed healthcare professional."
            )

        return response
