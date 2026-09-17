# Neuro-Symbolic Clinical Reasoning Engine & Evaluation Benchmark

## 1. Architectural Overview & System Design

The **Neuro-Symbolic Clinical Intelligence Platform** decouples neural probabilistic language processing from rigorous deterministic clinical reasoning. Clinical guidelines, contraindications, and laboratory threshold evaluations are executed entirely within a symbolic rule engine operating over deterministic working memory facts, rather than relying on LLM generative deduction.

```
+-------------------------------------------------------------------------+
|                        PATIENT CLINICAL CONTEXT                         |
|   (Demographics, Active Medications, Diagnoses, Nested Laboratory Tests)|
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                SYMBOLIC WORKING MEMORY & FACT RESOLVER                  |
|  - Flattens root attributes & nested dictionaries (labs, vitals)        |
|  - Normalizes clinical laboratory aliases (e.g. serum_potassium)        |
|  - Expands active pharmaceutical ingredients to drug classes            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|             50-RULE DETERMINISTIC FORWARD CHAINING ENGINE               |
|  - Loaded externally from neurosymbolic_clinical/rules.csv              |
|  - Evaluates comparison, equality, containment, between, & null checks  |
|  - Iterative fixpoint execution (derives secondary facts & re-evaluates)|
|  - Priority sorting: critical > high > medium > low                     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   DETERMINISTIC CLINICAL OUTPUTS                        |
|  - Fired Rules (IDs, Descriptions, Evidence References)                 |
|  - Derived Clinical Facts (e.g. egfr_severely_reduced: True)            |
|  - Critical & Safety Alerts (e.g. metformin_contraindicated)            |
|  - Full Explainability Trace (Conditions evaluated, operators, facts)   |
+-------------------------------------------------------------------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
+-----------------------------------+ +-----------------------------------+
|     SAFETY & GUARDRAIL FILTER     | |     DOWNSTREAM LLM CONTEXT        |
| Strict deterministic verification | | Enforces non-contradiction and    |
| Prevents LLM hallucinated alerts  | | provides traceable citations      |
+-----------------------------------+ +-----------------------------------+
```

---

## 2. Rule Retrieval, Storage & Matching Logic

### External Rule Storage & Loader
- **Location**: `neurosymbolic_clinical/rules.csv` (Authoritative clinical source).
- **Singleton In-Memory Cache**: Implemented in [`backend/app/reasoning/loader.py`](backend/app/reasoning/loader.py). Rules are parsed once from disk on initialization and cached globally. In validation/testing, `get_clinical_rules(force_reload=True)` can reload from CSV dynamically.
- **Rule Structure**:
  - `rule_id`: Unique identifier (`R001` through `R050`).
  - `conditions_json`: JSON list of predicates `[{"fact": ..., "operator": ..., "value": ...}]`.
  - `conclusion_json`: JSON object `{"type": ..., "fact": ..., "value": ...}`.
  - `priority`: Execution precedence (`critical`, `high`, `medium`, `low`).
  - `validation_status`: Guideline concordance (`validated`).
  - `evidence_source` & `evidence_reference`: Clinical citation (e.g. KDIGO 2023, ADA 2024, ACC/AHA).

### Supported Evaluation Operators
The deterministic evaluation engine in [`backend/app/reasoning/engine.py`](backend/app/reasoning/engine.py) implements exact matching for all clinical logic predicates:
1. `==`: Strict equality (case-insensitive for strings, type-safe for booleans and numbers).
2. `!=`: Inequality.
3. `<`, `<=`, `>`, `>=`: Robust numerical boundary evaluation with float casting.
4. `contains`: Membership check for strings or lists (e.g., verifying if `"metformin"` is present in `medications`).
5. `not_contains`: Absence check.
6. `between`: Two-element numerical range check (`[min, max]`, inclusive: `min <= fact_val <= max`).
7. `is_empty` / `is_not_empty`: Null / missing / empty collection validation.

### Clinical Fact Resolution & Drug Class Expansion
To prevent brittle clinical mismatches, `_resolve_fact` performs hierarchical resolution:
- **Hierarchical Lookup**: Checks root patient keys, then dynamically queries nested objects: `labs`, `lab_results`, `vitals`, `demographics`.
- **Laboratory Synonyms**: Maps aliases such as `serum_potassium` -> `potassium`, `creatinine_clearance` -> `egfr`.
- **Medication Class Expansion**: Expands active medication lists to recognize class prototypes (e.g. `lisinopril` -> `ace_inhibitor`, `losartan` -> `arb`, `empagliflozin` -> `sglt2_inhibitor`, `atorvastatin` -> `statin`, `alendronate` -> `bisphosphonate`).

---

## 3. Forward Chaining Process

The forward chaining engine implements a monotonic fixpoint algorithm:
1. **Working Memory Initialization**: Working memory is populated with the patient's verified clinical data.
2. **Prioritized Evaluation**: Active rules are evaluated ordered by priority (`critical` > `high` > `medium` > `low`).
3. **Fact Derivation**: When all conditions of a rule evaluate to `True`, the rule fires. Its conclusion fact (e.g. `ckd_stage: "G4"`, `metformin_contraindicated: True`) is asserted into working memory.
4. **Fixpoint Loop**: The engine continues iterating over unfired rules as long as new facts are derived (`new_facts_derived == True`). Iteration halts when no additional rules fire (fixpoint reached) or when safety `max_iterations = 10` is hit.
5. **Auditable Trace**: For every rule, an explicit audit object records all evaluated conditions, actual patient values, required threshold values, match booleans, and timestamps.

---

## 4. Benchmark Dataset Structure

The benchmark suite (`neurosymbolic_clinical/test_cases.csv`) provides **500 clinically stratified synthetic test cases**:
- **150 Positive Cases**: Clinically valid presentations designed to trigger single specific clinical rules.
- **150 Negative Cases**: Normal or altered presentations where clinical thresholds are deliberately not met.
- **100 Boundary Cases**: Edge cases positioned precisely on clinical decision boundaries (e.g. eGFR = 30.0 mL/min/1.73m², Potassium = 5.5 mEq/L, Systolic BP = 130 mmHg).
- **100 Multi-Rule Chained Cases**: Complex multimorbid patient presentations triggering multiple cascading rules (up to 6 rules simultaneously).

### Ground-Truth Withholding Methodology
During benchmark execution, `expected_rules`, `expected_derived_facts`, and `expected_alerts` columns are **strictly isolated** from the evaluation engine. Only the patient data payload (`patient_json`) is provided to working memory.

---

## 5. Evaluation Metrics & Benchmark Results

The evaluation pipeline was executed across all 500 cases and 50 clinical rules (25,000 total binary rule decisions).

### Overall Benchmark Metrics
| Metric | Result | Value | Notes |
|---|:---:|:---:|---|
| **Total Test Cases** | 500 | 500 / 500 | Complete evaluation without exceptions |
| **Total Rule Evaluations** | 25,000 | 50 rules × 500 cases | Full matrix coverage |
| **True Positives (TP)** | 1,419 | 1,419 | All expected rule triggers identified |
| **False Positives (FP)** | 0 | 0 | Zero extraneous rule firings |
| **False Negatives (FN)** | 0 | 0 | Zero missed rule firings |
| **True Negatives (TN)** | 23,581 | 23,581 | Correct rule suppressions |
| **Symbolic Rule Accuracy** | **100.00%** | 1.0000 | (TP + TN) / Total |
| **Micro-Averaged Precision** | **100.00%** | 1.0000 | TP / (TP + FP) |
| **Micro-Averaged Recall** | **100.00%** | 1.0000 | TP / (TP + FN) |
| **Micro-Averaged F1 Score** | **100.00%** | 1.0000 | Harmonic mean |
| **Macro-Averaged F1 Score** | **100.00%** | 1.0000 | Unweighted mean across 50 rules |
| **Subset Exact Match** | **100.00%** | 500 / 500 | Strict multi-label equality per case |
| **General Alert Recall** | **100.00%** | 100.00% | Safety & risk alerts detected |
| **Critical Alert Recall** | **100.00%** | 100.00% | Contraindication alerts detected |

### Cohort Breakdown
- **Positive Test Cases**: 150 / 150 Exact Match (100.00%)
- **Negative Test Cases**: 150 / 150 Exact Match (100.00%)
- **Boundary Test Cases**: 100 / 100 Exact Match (100.00%)
- **Multi-Rule Test Cases**: 100 / 100 Exact Match (100.00%)

---

## 6. Generated Artifacts & Visualizations

The automated benchmark script generates machine-readable data files and publication-quality visual charts:

| Artifact Name | Path | Description |
|---|---|---|
| `evaluation_results.json` | [`evaluation_results.json`](evaluation_results.json) | High-level summary metrics, cohort breakdowns, and runtime stats |
| `per_rule_metrics.csv` | [`per_rule_metrics.csv`](per_rule_metrics.csv) | Individual TP, FP, FN, TN, precision, recall, and F1 for R001–R050 |
| `confusion_matrix.csv` | [`confusion_matrix.csv`](confusion_matrix.csv) | Complete 2x2 confusion matrix counts |
| `metrics_summary.png` | [`evaluation_results/metrics_summary.png`](evaluation_results/metrics_summary.png) | Bar chart comparing overall precision, recall, F1, exact match, and accuracy |
| `per_rule_f1.png` | [`evaluation_results/per_rule_f1.png`](evaluation_results/per_rule_f1.png) | Horizontal bar distribution of F1 scores across all 50 rules |
| `per_rule_precision_recall.png`| [`evaluation_results/per_rule_precision_recall.png`](evaluation_results/per_rule_precision_recall.png) | Grouped precision vs recall plot per clinical rule |
| `confusion_matrix.png` | [`evaluation_results/confusion_matrix.png`](evaluation_results/confusion_matrix.png) | Heatmap visualization of binary rule decision matrix |
| `case_type_performance.png` | [`evaluation_results/case_type_performance.png`](evaluation_results/case_type_performance.png) | Breakdown across positive, negative, boundary, and multi-rule cases |
| `alert_metrics.png` | [`evaluation_results/alert_metrics.png`](evaluation_results/alert_metrics.png) | Alert sensitivity and recall comparison |

---

## 7. Limitations & Validation Status

1. **Validation Status**:
   - All 50 clinical rules in `neurosymbolic_clinical/rules.csv` possess a `validation_status` of `"validated"`, based on major guideline literature (ADA, KDIGO, ACC/AHA, GOLD, CHEST, Beers Criteria).
   - Zero rules are currently tagged `needs_validation`.
2. **Deterministic Precondition**:
   - The symbolic engine relies on well-formed or extracted clinical entities. Unstructured notes must pass through clinical NER or extraction before symbolic evaluation.
3. **Temporal Reasoning Scope**:
   - Rules evaluate point-in-time cross-sectional patient context. Longitudinal trend rules (e.g. rate of eGFR decline over 12 months) require historical laboratory ingestion.

---

## 8. Reproducing the Benchmark Evaluation

To execute the benchmark and regenerate all metrics, tables, and visualization charts:

```powershell
# From the repository root
& ".\backend\.venv\Scripts\python.exe" evaluate_symbolic_engine.py
```

To run the automated unit test suite:
```powershell
# Run benchmark engine test suite
& ".\backend\.venv\Scripts\python.exe" -m pytest backend/tests/test_symbolic_engine_benchmark.py -v

# Run entire backend test suite
& ".\backend\.venv\Scripts\python.exe" -m pytest backend/tests/ -v
```
