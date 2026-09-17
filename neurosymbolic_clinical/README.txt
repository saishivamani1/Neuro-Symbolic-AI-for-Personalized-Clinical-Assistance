Neuro-Symbolic Clinical Reasoning Benchmark

Files:
- rules.csv: 50 deterministic rules.
- test_cases.csv: 500 synthetic patient cases with deterministic ground truth.
- dataset_statistics.json: requested benchmark counts.

Evaluation:
- Pass only patient_json to the reasoning engine.
- Hide expected_rules, expected_derived_facts and expected_alerts from the engine.
- Compare the engine output with those ground-truth columns.
- Boundary cases test exact threshold behavior.
- Multi-rule cases test simultaneous rule triggering.

This is a synthetic software/research benchmark, not a clinical decision tool.
Validate source versions and clinical applicability before publication or clinical use.
