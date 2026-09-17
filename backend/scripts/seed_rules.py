"""
scripts/seed_rules.py

Standalone CLI script: write evidence-grounded clinical decision rules to JSON.

Usage:
    cd backend
    python scripts/seed_rules.py [--output PATH] [--overwrite]

Examples:
    python scripts/seed_rules.py --output ./data/rules.json --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.logging import configure_logging, get_logger
from app.reasoning.models import ClinicalRule
from app.reasoning.rules import DEFAULT_CLINICAL_RULES

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the symbolic rule engine with evidence-grounded clinical rules."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/rules.json"),
        help="Output path for rules JSON file (default: ./data/rules.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rules file if it exists.",
    )
    parser.add_argument(
        "--include-diseases",
        action="store_true",
        default=True,
        help="Include full disease rules from cleaned_diseases.json (default: True).",
    )
    args = parser.parse_args()

    configure_logging()

    print("=" * 70)
    print("  Neuro-Symbolic Healthcare Intelligence Platform")
    print("  Stage 4: Clinical Rules Seeder")
    print("=" * 70)
    print()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"[!] Warning: Rules file already exists at '{output_path}'.")
        print("    Use --overwrite flag to replace it.")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rules_to_seed: list[ClinicalRule] = list(DEFAULT_CLINICAL_RULES)
    if args.include_diseases:
        disease_doc_path = Path("./data/documents/cleaned_diseases.json")
        if disease_doc_path.exists():
            from scripts.ingest_diseases_rules import DiseaseRulesCompiler
            compiler = DiseaseRulesCompiler(disease_doc_path)
            rules_to_seed = compiler.compile_all_rules()

    print(f"[*] Validating {len(rules_to_seed)} clinical decision rules...")
    serialized_rules = []
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}

    for rule in rules_to_seed:
        # Pydantic validation
        validated = ClinicalRule(**rule.model_dump())
        serialized_rules.append(validated.model_dump())

        cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
        sev = rule.severity.value if hasattr(rule.severity, "value") else str(rule.severity)

        category_counts[cat] = category_counts.get(cat, 0) + 1
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print("[*] Writing rules to:", output_path)
    output_path.write_text(json.dumps(serialized_rules, indent=2), encoding="utf-8")

    print()
    print("-" * 70)
    print(f"{'Rule ID':<40} | {'Category':<20} | {'Severity':<10}")
    print("-" * 70)
    for rule in rules_to_seed[:15]:
        cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
        sev = rule.severity.value if hasattr(rule.severity, "value") else str(rule.severity)
        print(f"{rule.rule_id:<40} | {cat:<20} | {sev:<10}")
    if len(rules_to_seed) > 15:
        print(f"... and {len(rules_to_seed) - 15} additional clinical decision rules ...")
    print("-" * 70)
    print(f"Total Rules Seeded: {len(rules_to_seed)}")
    print()
    print("Rules by Category:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    print()
    print("Rules by Severity:")
    for sev, count in severity_counts.items():
        print(f"  {sev}: {count}")
    print()
    print("[+] Clinical rules initialization: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
