"""
scripts/generate_sample_pdf.py

Generate synthetic medical guideline PDF files for testing the RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path
import fitz  # PyMuPDF


def create_sample_pdf(output_path: Path) -> None:
    """Create a multi-page clinical guideline PDF with realistic medical content."""
    doc = fitz.open()

    # Page 1: Title & Hypertension Guidelines
    page1 = doc.new_page()
    text_p1 = (
        "Clinical Practice Guidelines: Management of Essential Hypertension\n"
        "Page 1 of 3\n"
        "--------------------------------------------------\n\n"
        "1. Diagnostic Criteria and Blood Pressure Thresholds\n"
        "Hypertension is diagnosed when systolic blood pressure (SBP) is >= 140 mmHg\n"
        "or diastolic blood pressure (DBP) is >= 90 mmHg on repeated clinic measurements.\n"
        "For patients with documented diabetes or chronic kidney disease (CKD),\n"
        "the recommended target blood pressure is < 130/80 mmHg.\n\n"
        "2. Pharmacological First-Line Therapies\n"
        "First-line pharmacological agents include ACE inhibitors (e.g., Lisinopril 10-40 mg daily),\n"
        "Angiotensin Receptor Blockers (ARBs, e.g., Losartan 50-100 mg daily),\n"
        "Calcium Channel Blockers (CCBs, e.g., Amlodipine 5-10 mg daily), and\n"
        "Thiazide-like diuretics (e.g., Chlorthalidone 12.5-25 mg daily).\n"
        "ACE inhibitors and ARBs are strongly recommended for patients with diabetic nephropathy\n"
        "or proteinuria to delay progression of renal dysfunction."
    )
    rect = fitz.Rect(50, 50, 550, 750)
    page1.insert_textbox(rect, text_p1, fontsize=11, fontname="helv")

    # Page 2: Type 2 Diabetes Guidelines & Contraindications
    page2 = doc.new_page()
    text_p2 = (
        "Clinical Practice Guidelines: Type 2 Diabetes Mellitus Management\n"
        "Page 2 of 3\n"
        "--------------------------------------------------\n\n"
        "3. Glycemic Targets and First-Line Management\n"
        "For non-pregnant adults with Type 2 Diabetes, the general HbA1c target is < 7.0% (53 mmol/mol).\n"
        "Metformin (500 mg to 2000 mg daily with meals) remains the preferred initial pharmacotherapy\n"
        "in the absence of contraindications.\n\n"
        "4. Renal Impairment and Metformin Contraindications\n"
        "Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min/1.73m²)\n"
        "due to the increased risk of fatal lactic acidosis.\n"
        "For patients with eGFR between 30 and 45 mL/min/1.73m², maximum dose should not exceed 1000 mg/day,\n"
        "and renal function must be monitored every 3 to 6 months.\n"
        "In patients with atherosclerotic cardiovascular disease or heart failure,\n"
        "SGLT2 inhibitors (e.g., Empagliflozin, Dapagliflozin) or GLP-1 receptor agonists\n"
        "(e.g., Semaglutide, Liraglutide) with proven cardiovascular benefit are indicated."
    )
    page2.insert_textbox(rect, text_p2, fontsize=11, fontname="helv")

    # Page 3: Drug Interactions & Electrolyte Monitoring
    page3 = doc.new_page()
    text_p3 = (
        "Clinical Practice Guidelines: Drug Interactions & Safety Monitoring\n"
        "Page 3 of 3\n"
        "--------------------------------------------------\n\n"
        "5. Critical Drug Interactions and Hyperkalemia Risk\n"
        "Concurrent administration of ACE inhibitors or ARBs with potassium-sparing diuretics\n"
        "(e.g., Spironolactone, Eplerenone) or potassium supplements significantly elevates\n"
        "the risk of severe hyperkalemia (serum potassium > 5.5 mEq/L).\n"
        "Serum potassium and serum creatinine must be evaluated at baseline, 1-2 weeks after initiation\n"
        "or dosage titration, and at least annually thereafter.\n\n"
        "6. NSAIDs and Antihypertensive Attenuation\n"
        "Nonsteroidal anti-inflammatory drugs (NSAIDs, e.g., Ibuprofen, Naproxen) can blunt the\n"
        "antihypertensive effect of ACE inhibitors, ARBs, and beta-blockers, and can precipitate\n"
        "acute kidney injury in volume-depleted or elderly patients."
    )
    page3.insert_textbox(rect, text_p3, fontsize=11, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Generated sample medical PDF: {output_path}")


if __name__ == "__main__":
    sample_dir = Path("./data/sample")
    docs_dir = Path("./data/documents")
    create_sample_pdf(sample_dir / "clinical_guidelines_sample.pdf")
    create_sample_pdf(docs_dir / "type_2_diabetes_guidelines.pdf")
