"""
scripts/ingest_diseases_kg.py

Medical Knowledge Graph Ingestion Script for Cleaned Diseases Dataset.

Extracts rich clinical entities and directed relationships from:
`backend/data/documents/cleaned_diseases.json` (687 Mayo Clinic disease records):
- Disease nodes (canonical name, overview, Mayo clinic references, prevention notes)
- Symptom nodes (from keywords & symptom text) + [:HAS_SYMPTOM] relations
- Drug & Medication nodes (pharmacological classes & generic agents) + [:TREATED_BY] relations
- RiskFactor nodes (from Risk factors sections) + [:RISK_FACTOR_FOR] relations
- Complication / Condition nodes (from Complications) + [:ASSOCIATED_WITH] relations

Usage:
    cd backend
    .venv/Scripts/python scripts/ingest_diseases_kg.py [--file PATH] [--dry-run] [--batch-size 500]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.knowledge_graph.connection import get_neo4j_manager
from app.knowledge_graph.models import MedicalEntity, MedicalRelationship
from app.knowledge_graph.service import get_kg_service

logger = get_logger(__name__)


def slugify(text: str) -> str:
    """Generate a clean alphanumeric snake_case slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:64] if len(s) > 64 else s


# --------------------------------------------------------------------------- #
# Curated Medication & Pharmacology Catalog
# --------------------------------------------------------------------------- #
DRUG_CATALOG: Dict[str, Tuple[str, str]] = {
    # Anticoagulants & Cardiovascular
    "anticoagulant": ("Anticoagulants (Blood Thinners)", "Antithrombotic agents that prevent blood clot formation and embolisms."),
    "blood thinner": ("Anticoagulants (Blood Thinners)", "Antithrombotic agents that reduce systemic thromboembolic risk."),
    "warfarin": ("Warfarin", "Vitamin K antagonist oral anticoagulant."),
    "apixaban": ("Apixaban (Eliquis)", "Direct oral factor Xa inhibitor anticoagulant."),
    "rivaroxaban": ("Rivaroxaban (Xarelto)", "Oral direct factor Xa inhibitor anticoagulant."),
    "dabigatran": ("Dabigatran (Pradaxa)", "Direct oral thrombin inhibitor anticoagulant."),
    "heparin": ("Heparin", "Parenteral fast-acting anticoagulant."),
    "aspirin": ("Aspirin", "Antiplatelet and cyclooxygenase inhibitor analgesic."),
    "clopidogrel": ("Clopidogrel (Plavix)", "P2Y12 platelet aggregation inhibitor."),
    "beta blocker": ("Beta Blockers", "Adrenergic receptor antagonists reducing myocardial oxygen demand and blood pressure."),
    "metoprolol": ("Metoprolol", "Cardioselective beta-1 adrenergic blocker for hypertension and heart failure."),
    "atenolol": ("Atenolol", "Selective beta-1 antagonist for hypertension and angina."),
    "carvedilol": ("Carvedilol", "Non-selective beta and alpha-1 blocker for heart failure with reduced ejection fraction."),
    "ace inhibitor": ("ACE Inhibitors", "Angiotensin-converting enzyme inhibitors reducing blood pressure and renal decline."),
    "lisinopril": ("Lisinopril", "First-line ACE inhibitor for hypertension and heart failure."),
    "enalapril": ("Enalapril", "ACE inhibitor for arterial hypertension and chronic kidney disease."),
    "ramipril": ("Ramipril", "ACE inhibitor reducing post-infarction cardiovascular events."),
    "arb": ("Angiotensin Receptor Blockers (ARBs)", "Blocks angiotensin II type 1 receptors to lower systemic vascular resistance."),
    "losartan": ("Losartan", "Angiotensin II receptor blocker for hypertension and nephropathy."),
    "valsartan": ("Valsartan", "ARB indicated for heart failure and hypertension."),
    "calcium channel blocker": ("Calcium Channel Blockers", "Inhibits calcium influx into cardiac and vascular smooth muscle."),
    "amlodipine": ("Amlodipine", "Dihydropyridine calcium channel blocker for systemic hypertension and angina."),
    "diltiazem": ("Diltiazem", "Non-dihydropyridine calcium channel blocker for rate control in atrial arrhythmias."),
    "statin": ("Statins (HMG-CoA Reductase Inhibitors)", "Inhibits cholesterol synthesis to lower LDL and stabilize atherosclerotic plaques."),
    "atorvastatin": ("Atorvastatin (Lipitor)", "High-intensity HMG-CoA reductase inhibitor for dyslipidemia and primary prevention."),
    "rosuvastatin": ("Rosuvastatin (Crestor)", "High-intensity statin for severe hypercholesterolemia."),
    "simvastatin": ("Simvastatin (Zocor)", "Moderate-intensity statin for hyperlipidemia."),
    "diuretic": ("Diuretics (Water Pills)", "Promotes natriuresis and water clearance to decrease intravascular volume."),
    "furosemide": ("Furosemide (Lasix)", "Potent loop diuretic for acute congestive edema and fluid retention."),
    "hydrochlorothiazide": ("Hydrochlorothiazide (HCTZ)", "Thiazide diuretic for essential arterial hypertension."),
    "spironolactone": ("Spironolactone (Aldactone)", "Potassium-sparing mineralocorticoid receptor aldosterone antagonist."),
    "nitroglycerin": ("Nitroglycerin", "Vasodilator generating nitric oxide to alleviate ischemic angina pectoris."),
    "digoxin": ("Digoxin", "Cardiac glycoside positive inotrope and vagal rate-control agent."),

    # Endocrine & Metabolic
    "metformin": ("Metformin", "First-line biguanide antihyperglycemic reducing hepatic gluconeogenesis."),
    "insulin": ("Insulin Therapy", "Exogenous hormone replacement essential for glucose homeostasis."),
    "sglt2": ("SGLT2 Inhibitors", "Sodium-glucose cotransporter-2 inhibitors improving glycemic control and cardio-renal outcomes."),
    "empagliflozin": ("Empagliflozin (Jardiance)", "SGLT2 inhibitor for diabetes, heart failure, and chronic kidney disease."),
    "dapagliflozin": ("Dapagliflozin (Farxiga)", "SGLT2 inhibitor reducing mortality in heart failure and renal disease."),
    "glp-1": ("GLP-1 Receptor Agonists", "Incretin mimetics enhancing glucose-dependent insulin secretion and weight loss."),
    "semaglutide": ("Semaglutide (Ozempic/Wegovy)", "Long-acting GLP-1 receptor agonist for diabetes and obesity."),
    "sulfonylurea": ("Sulfonylureas", "Insulin secretagogues stimulating pancreatic beta-cell insulin release."),
    "glipizide": ("Glipizide", "Second-generation sulfonylurea for glycemic control."),
    "levothyroxine": ("Levothyroxine (Synthroid)", "Synthetic levo-isomer of thyroid hormone (T4) for hypothyroidism."),
    "antithyroid": ("Antithyroid Medications", "Thionamide agents (e.g. methimazole) inhibiting thyroid hormone biosynthesis."),
    "methimazole": ("Methimazole (Tapazole)", "Antithyroid drug inhibiting thyroid peroxidase in hyperthyroidism."),

    # Anti-inflammatory, Analgesics & Steroids
    "corticosteroid": ("Corticosteroids", "Steroid hormones with profound anti-inflammatory and immunosuppressive activity."),
    "prednisone": ("Prednisone", "Systemic oral glucocorticoid for acute and chronic inflammatory flare-ups."),
    "dexamethasone": ("Dexamethasone", "Potent, long-acting glucocorticoid with minimal mineralocorticoid activity."),
    "hydrocortisone": ("Hydrocortisone", "Short-acting topical or systemic glucocorticoid for inflammation and adrenal insufficiency."),
    "nsaid": ("Nonsteroidal Anti-inflammatory Drugs (NSAIDs)", "Inhibits cyclooxygenase (COX-1/COX-2) to reduce pain, fever, and inflammation."),
    "ibuprofen": ("Ibuprofen (Advil/Motrin)", "Short-acting non-selective nonsteroidal anti-inflammatory analgesic."),
    "naproxen": ("Naproxen (Aleve)", "Long-acting nonsteroidal anti-inflammatory drug for musculoskeletal pain."),
    "acetaminophen": ("Acetaminophen (Tylenol/Paracetamol)", "Centrally-acting analgesic and antipyretic agent without anti-inflammatory action."),
    "colchicine": ("Colchicine", "Microtubule disruption agent for acute gouty arthritis flares."),
    "allopurinol": ("Allopurinol", "Xanthine oxidase inhibitor reducing serum uric acid levels in chronic gout."),

    # Antimicrobials (Antibiotics, Antivirals, Antifungals)
    "antibiotic": ("Antibiotics", "Antimicrobial pharmaceuticals targeting bacterial infections."),
    "amoxicillin": ("Amoxicillin", "Broad-spectrum aminopenicillin antibiotic for respiratory and ENT infections."),
    "amoxicillin-clavulanate": ("Amoxicillin-Clavulanate (Augmentin)", "Beta-lactamase protected broad-spectrum penicillin combination."),
    "azithromycin": ("Azithromycin (Z-Pak)", "Macrolide antibiotic with immunomodulatory and antibacterial activity."),
    "ciprofloxacin": ("Ciprofloxacin (Cipro)", "Fluoroquinolone antibacterial for complicated urinary and enteric infections."),
    "doxycycline": ("Doxycycline", "Tetracycline class bacteriostatic antibiotic for atypical and tick-borne infections."),
    "cephalexin": ("Cephalexin (Keflex)", "First-generation cephalosporin for skin, soft tissue, and urinary tract infections."),
    "metronidazole": ("Metronidazole (Flagyl)", "Nitroimidazole synthetic antimicrobial active against anaerobic bacteria and protozoa."),
    "antiviral": ("Antivirals", "Targeted pharmaceuticals inhibiting viral attachment, replication, or assembly."),
    "acyclovir": ("Acyclovir (Zovirax)", "Synthetic purine nucleoside analogue active against Herpes simplex and Varicella zoster."),
    "valacyclovir": ("Valacyclovir (Valtrex)", "L-valyl ester prodrug of acyclovir with enhanced oral bioavailability."),
    "oseltamivir": ("Oseltamivir (Tamiflu)", "Neuraminidase inhibitor for influenza A and B prevention and early management."),
    "antifungal": ("Antifungals", "Ergosterol synthesis inhibitors and fungicidal agents."),
    "fluconazole": ("Fluconazole (Diflucan)", "Triazole antifungal for mucosal and systemic Candida infections."),
    "terbinafine": ("Terbinafine (Lamisil)", "Allylamine antifungal for dermatophytosis and onychomycosis."),

    # Respiratory
    "bronchodilator": ("Bronchodilators", "Pharmacological agents inducing bronchial smooth muscle relaxation."),
    "albuterol": ("Albuterol / Salbutamol (Ventolin)", "Short-acting selective beta-2 adrenergic rescue bronchodilator."),
    "salmeterol": ("Salmeterol", "Long-acting beta-2 agonist (LABA) for maintenance airway dilation."),
    "fluticasone": ("Fluticasone", "High-affinity topical/inhaled corticosteroid suppressing bronchial inflammation."),
    "budesonide": ("Budesonide", "Inhaled corticosteroid for maintenance asthma and COPD therapy."),
    "montelukast": ("Montelukast (Singulair)", "Oral cysteinyl leukotriene receptor antagonist for asthma and allergic rhinitis."),
    "ipratropium": ("Ipratropium Bromide (Atrovent)", "Short-acting muscarinic anticholinergic bronchodilator."),
    "tiotropium": ("Tiotropium (Spiriva)", "Long-acting anticholinergic maintenance bronchodilator."),

    # Gastrointestinal
    "proton pump inhibitor": ("Proton Pump Inhibitors (PPIs)", "Inhibits gastric parietal H+/K+ ATPase to suppress acid secretion."),
    "omeprazole": ("Omeprazole (Prilosec)", "First-line proton pump inhibitor for GERD, peptic ulcers, and erosive esophagitis."),
    "pantoprazole": ("Pantoprazole (Protonix)", "Proton pump inhibitor for hypersecretory acid conditions and prophylaxis."),
    "h2 blocker": ("H2 Receptor Antagonists", "Reversible competitive antagonist of gastric histamine H2 receptors."),
    "famotidine": ("Famotidine (Pepcid)", "Potent H2 blocker reducing nocturnal and basal gastric acid volume."),
    "antacid": ("Antacids", "Bases neutralizing gastric hydrochloric acid for fast symptomatic relief."),
    "loperamide": ("Loperamide (Imodium)", "Peripheral opioid-receptor agonist slowing intestinal transit."),
    "laxative": ("Laxatives / Stool Softeners", "Osmotic and stimulant agents relieving chronic and acute constipation."),

    # Neurology & Psychiatry
    "gabapentin": ("Gabapentin (Neurontin)", "Voltage-gated calcium channel alpha-2-delta ligand for neuropathic pain."),
    "pregabalin": ("Pregabalin (Lyrica)", "GABA analogue for neuropathic pain, postherpetic neuralgia, and fibromyalgia."),
    "anticonvulsant": ("Anticonvulsants / Antiepileptic Drugs", "Membrane-stabilizing agents suppressing paroxysmal neuronal firing."),
    "levetiracetam": ("Levetiracetam (Keppra)", "Synaptic vesicle protein SV2A ligand antiepileptic."),
    "antidepressant": ("Antidepressants", "Modulates synaptic monoamines (serotonin, norepinephrine, dopamine)."),
    "ssri": ("Selective Serotonin Reuptake Inhibitors (SSRIs)", "First-line antidepressant and anxiolytic class."),
    "sertraline": ("Sertraline (Zoloft)", "SSRI for major depressive disorder, PTSD, and panic disorder."),
    "fluoxetine": ("Fluoxetine (Prozac)", "SSRI for major depressive disorder and obsessive-compulsive disorder."),
    "duloxetine": ("Duloxetine (Cymbalta)", "Serotonin-norepinephrine reuptake inhibitor (SNRI) for pain and mood."),

    # Rheumatology & Immunology
    "methotrexate": ("Methotrexate (Trexall)", "Folate antimetabolite and disease-modifying antirheumatic drug (DMARD)."),
    "hydroxychloroquine": ("Hydroxychloroquine (Plaquenil)", "Immunomodulatory DMARD for systemic lupus erythematosus and rheumatoid arthritis."),
    "adalimumab": ("Adalimumab (Humira)", "Recombinant human IgG1 anti-TNF monoclonal antibody biologic."),
    "infliximab": ("Infliximab (Remicade)", "Chimeric monoclonal antibody neutralizing tumor necrosis factor alpha."),
    "immunosuppressant": ("Immunosuppressive Agents", "Suppresses pathological immune-mediated inflammatory cascades."),

    # Dermatology & Topical
    "antiperspirant": ("Clinical Antiperspirants", "Topical astringent formulation for localized hyperhidrosis management."),
    "aluminum chloride": ("Aluminum Chloride Hexahydrate (Drysol)", "Prescription topical antiperspirant blocking eccrine sweat pores."),
    "retinoid": ("Topical Retinoids", "Vitamin A derivatives promoting epidermal turnover and comedolysis."),
    "benzoyl peroxide": ("Benzoyl Peroxide", "Topical bactericidal and keratolytic agent."),
    "artificial tear": ("Artificial Tears / Lubricant Eye Drops", "Hypotonic or isotonic demulcent ocular lubricant."),

    # Antihistamines & Allergy
    "antihistamine": ("Antihistamines", "Selective H1-receptor antagonists preventing allergic histamine reactions."),
    "cetirizine": ("Cetirizine (Zyrtec)", "Second-generation peripheral H1 antihistamine with low sedation."),
    "loratadine": ("Loratadine (Claritin)", "Non-sedating second-generation peripheral H1 blocker."),
    "diphenhydramine": ("Diphenhydramine (Benadryl)", "First-generation sedating ethanolamine H1 antihistamine."),
}

# Standard First-Line Validated Disease-to-Medication Mappings
CANONICAL_DISEASE_DRUGS: Dict[str, List[str]] = {
    "atrial fibrillation": ["anticoagulant", "warfarin", "apixaban", "beta blocker", "metoprolol", "diltiazem"],
    "hyperhidrosis": ["antiperspirant", "aluminum chloride"],
    "type 2 diabetes": ["metformin", "sglt2", "glp-1", "insulin"],
    "essential hypertension": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide"],
    "chronic kidney disease": ["lisinopril", "losartan", "sglt2"],
    "heart failure": ["lisinopril", "metoprolol", "furosemide", "spironolactone", "sglt2"],
    "asthma": ["albuterol", "fluticasone", "budesonide", "montelukast"],
    "chronic obstructive pulmonary disease": ["albuterol", "tiotropium", "fluticasone"],
    "gastroesophageal reflux disease": ["omeprazole", "pantoprazole", "famotidine", "antacid"],
    "rheumatoid arthritis": ["methotrexate", "hydroxychloroquine", "prednisone", "nsaid"],
    "osteoarthritis": ["acetaminophen", "ibuprofen", "naproxen"],
    "gout": ["colchicine", "allopurinol", "nsaid"],
    "hypothyroidism": ["levothyroxine"],
    "hyperthyroidism": ["methimazole", "beta blocker"],
    "depression": ["sertraline", "fluoxetine", "ssri"],
    "generalized anxiety disorder": ["sertraline", "duloxetine", "ssri"],
    "migraine": ["ibuprofen", "acetaminophen", "naproxen"],
    "pneumonia": ["amoxicillin", "azithromycin", "doxycycline"],
    "urinary tract infection": ["ciprofloxacin", "cephalexin"],
    "cellulitis": ["cephalexin", "amoxicillin"],
    "shingles": ["valacyclovir", "acyclovir", "gabapentin"],
    "herpes simplex": ["valacyclovir", "acyclovir"],
    "dry eyes": ["artificial tear"],
    "allergic rhinitis": ["cetirizine", "loratadine", "fluticasone"],
}


class DiseaseOntologyExtractor:
    """Extracts standardized MedicalEntities and MedicalRelationships from cleaned_diseases.json."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.raw_data: List[Dict[str, Any]] = []

    def load_data(self) -> None:
        """Load JSON data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Diseases file not found at: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        logger.info("Loaded %d disease records from %s", len(self.raw_data), self.data_path.name)

    def extract_all(self) -> Tuple[List[MedicalEntity], List[MedicalRelationship]]:
        """Extract all clinical entities and relationships."""
        if not self.raw_data:
            self.load_data()

        entities_dict: Dict[str, MedicalEntity] = {}
        relationships: List[MedicalRelationship] = []
        rel_signatures: Set[str] = set()

        def add_entity(e: MedicalEntity) -> None:
            if e.id not in entities_dict:
                entities_dict[e.id] = e
            else:
                # Merge descriptions / properties if new entity has richer info
                existing = entities_dict[e.id]
                if not existing.description and e.description:
                    existing.description = e.description
                if e.properties:
                    existing.properties.update(e.properties)

        def add_relationship(r: MedicalRelationship) -> None:
            sig = f"{r.source_id}|{r.type}|{r.target_id}"
            if sig not in rel_signatures:
                rel_signatures.add(sig)
                relationships.append(r)

        for record in self.raw_data:
            disease_name = record.get("disease", "").strip()
            if not disease_name:
                continue

            disease_id = record.get("id") or f"disease_{slugify(disease_name)}"
            overview = (record.get("Overview") or "").strip()

            # Clean overview snippet (up to 300 chars) for description
            desc_snippet = overview[:300].rsplit(".", 1)[0] + "." if len(overview) > 300 else overview
            if not desc_snippet.strip():
                desc_snippet = f"Clinical disorder: {disease_name}"

            source_urls = record.get("source_urls") or {}

            # 1. Disease Entity
            disease_entity = MedicalEntity(
                id=disease_id,
                name=disease_name,
                type="Disease",
                description=desc_snippet,
                source_document="cleaned_diseases.json",
                confidence=1.0,
                properties={
                    "full_overview": overview,
                    "main_url": source_urls.get("main"),
                    "diagnosis_url": source_urls.get("diagnosis_treatment"),
                    "doctors_url": source_urls.get("doctors_departments"),
                    "prevention": record.get("Prevention"),
                    "diagnosis": record.get("Diagnosis"),
                    "lifestyle": record.get("Lifestyle and home remedies"),
                },
            )
            add_entity(disease_entity)

            # 2. Symptoms Extraction
            symptom_terms: Set[str] = set()
            for kw in record.get("keywords") or []:
                kw_str = str(kw).strip().lower()
                if 2 < len(kw_str) < 50:
                    # Filter out if keyword is identical to disease name
                    if kw_str not in disease_name.lower() and disease_name.lower() not in kw_str:
                        symptom_terms.add(kw_str)

            # Also inspect Symptoms field for line-based symptoms
            symptoms_text = record.get("Symptoms") or ""
            for line in symptoms_text.split("\n"):
                line_str = line.strip()
                if 3 < len(line_str) < 45 and not line_str.lower().startswith(("symptoms", "when to", "atrial")):
                    cleaned_line = line_str.rstrip(".").lower()
                    if cleaned_line not in disease_name.lower():
                        symptom_terms.add(cleaned_line)

            for sym in symptom_terms:
                sym_title = sym.capitalize()
                sym_id = f"symptom_{slugify(sym)}"
                symptom_entity = MedicalEntity(
                    id=sym_id,
                    name=sym_title,
                    type="Symptom",
                    description=f"Clinical sign or reported symptom: {sym_title}.",
                    source_document="cleaned_diseases.json",
                    confidence=0.95,
                )
                add_entity(symptom_entity)

                add_relationship(
                    MedicalRelationship(
                        source_id=disease_id,
                        source_name=disease_name,
                        type="HAS_SYMPTOM",
                        target_id=sym_id,
                        target_name=sym_title,
                        description=f"{disease_name} frequently presents with {sym}.",
                        source_document="cleaned_diseases.json",
                        confidence=0.95,
                    )
                )

            # 3. Drugs & Medicines Extraction (Both Pattern-matched & Canonical)
            treatment_text = (record.get("Treatment") or "") + " " + overview
            treatment_lower = treatment_text.lower()
            disease_lower = disease_name.lower()

            matched_drug_keys: Set[str] = set()

            # Direct regex match in Treatment text
            for pattern_key in DRUG_CATALOG:
                if re.search(r"\b" + re.escape(pattern_key) + r"s?\b", treatment_lower):
                    matched_drug_keys.add(pattern_key)

            # Canonical guideline additions for known diseases
            for canon_dis, canon_drugs in CANONICAL_DISEASE_DRUGS.items():
                if canon_dis in disease_lower or disease_lower in canon_dis:
                    for d_key in canon_drugs:
                        matched_drug_keys.add(d_key)

            for drug_key in matched_drug_keys:
                if drug_key not in DRUG_CATALOG:
                    continue
                drug_name, drug_desc = DRUG_CATALOG[drug_key]
                drug_id = f"drug_{slugify(drug_key)}"

                drug_entity = MedicalEntity(
                    id=drug_id,
                    name=drug_name,
                    type="Drug",
                    description=drug_desc,
                    source_document="cleaned_diseases.json",
                    confidence=0.95,
                    properties={"drug_category": drug_key},
                )
                add_entity(drug_entity)

                add_relationship(
                    MedicalRelationship(
                        source_id=disease_id,
                        source_name=disease_name,
                        type="TREATED_BY",
                        target_id=drug_id,
                        target_name=drug_name,
                        description=f"{drug_name} is an indicated therapeutic medication for {disease_name}.",
                        source_document="cleaned_diseases.json",
                        confidence=0.95,
                    )
                )

            # 4. Risk Factors Extraction
            rf_text = record.get("Risk factors") or ""
            for line in rf_text.split("\n"):
                line_str = line.strip()
                if not line_str or len(line_str) < 5:
                    continue
                match = re.match(r"^([A-Z][A-Za-z0-9\s,\/\-\(\)]{2,40})\.(.*)", line_str)
                if match:
                    rf_title = match.group(1).strip()
                    rf_desc = match.group(2).strip()
                    if not rf_title.lower().startswith(("things that", "risk factor", "factors")):
                        rf_id = f"risk_{slugify(rf_title)}"
                        rf_entity = MedicalEntity(
                            id=rf_id,
                            name=rf_title,
                            type="RiskFactor",
                            description=rf_desc if rf_desc else f"Clinical risk factor: {rf_title}.",
                            source_document="cleaned_diseases.json",
                            confidence=0.90,
                        )
                        add_entity(rf_entity)

                        add_relationship(
                            MedicalRelationship(
                                source_id=rf_id,
                                source_name=rf_title,
                                type="RISK_FACTOR_FOR",
                                target_id=disease_id,
                                target_name=disease_name,
                                description=rf_desc if rf_desc else f"{rf_title} is a clinical risk factor for {disease_name}.",
                                source_document="cleaned_diseases.json",
                                confidence=0.90,
                            )
                        )

            # 5. Complications / Conditions Extraction
            comp_text = record.get("Complications") or ""
            for line in comp_text.split("\n"):
                line_str = line.strip()
                if 4 < len(line_str) < 40 and not line_str.lower().startswith(("complications", "blood", "other")):
                    comp_name = line_str.rstrip(".")
                    comp_id = f"condition_{slugify(comp_name)}"
                    comp_entity = MedicalEntity(
                        id=comp_id,
                        name=comp_name,
                        type="Condition",
                        description=f"Clinical complication or associated condition: {comp_name}.",
                        source_document="cleaned_diseases.json",
                        confidence=0.85,
                    )
                    add_entity(comp_entity)

                    add_relationship(
                        MedicalRelationship(
                            source_id=disease_id,
                            source_name=disease_name,
                            type="ASSOCIATED_WITH",
                            target_id=comp_id,
                            target_name=comp_name,
                            description=f"{disease_name} is associated with risk of {comp_name}.",
                            source_document="cleaned_diseases.json",
                            confidence=0.85,
                        )
                    )

        entities_list = list(entities_dict.values())
        logger.info(
            "Extracted %d unique entities and %d relationships across %d diseases.",
            len(entities_list),
            len(relationships),
            len(self.raw_data),
        )
        return entities_list, relationships


def export_snapshot(
    entities: List[MedicalEntity],
    relationships: List[MedicalRelationship],
    output_path: Path,
) -> None:
    """Save graph snapshot to disk for fast caching and offline resilience."""
    from collections import Counter

    entity_types = dict(Counter(e.type for e in entities))
    rel_types = dict(Counter(r.type for r in relationships))

    # Index top 100 diseases with their subgraphs for instant preview
    sample_subgraphs: Dict[str, Dict[str, Any]] = {}
    rel_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in relationships:
        if r.source_id not in rel_by_source:
            rel_by_source[r.source_id] = []
        rel_by_source[r.source_id].append({
            "rel": r.type,
            "target_id": r.target_id,
            "target_name": r.target_name,
            "desc": r.description,
        })

    for e in entities:
        if e.type == "Disease" and len(sample_subgraphs) < 150:
            sample_subgraphs[e.id] = {
                "name": e.name,
                "description": e.description,
                "connections": rel_by_source.get(e.id, []),
            }

    snapshot = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "entity_counts_by_type": entity_types,
        "relationship_counts_by_type": rel_types,
        "diseases_indexed": len([e for e in entities if e.type == "Disease"]),
        "sample_subgraphs": sample_subgraphs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    logger.info("Saved knowledge graph snapshot to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest cleaned_diseases.json into Neo4j Medical Knowledge Graph.")
    parser.add_argument(
        "--file",
        type=str,
        default=str(backend_dir / "data" / "documents" / "cleaned_diseases.json"),
        help="Path to cleaned_diseases.json file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Neo4j UNWIND upsert queries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and generate snapshot without writing to Neo4j.",
    )
    parser.add_argument(
        "--snapshot-out",
        type=str,
        default=str(backend_dir / "data" / "knowledge_graph_snapshot.json"),
        help="Output path for knowledge graph snapshot.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()

    print("=" * 75)
    print("  Neuro-Symbolic Healthcare Intelligence Platform")
    print("  Medical Knowledge Graph Ingestion Pipeline: Cleaned Diseases Dataset")
    print("=" * 75)
    print()
    print(f"Data file:     {args.file}")
    print(f"Neo4j URI:     {settings.neo4j_uri}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Dry run mode:  {args.dry_run}")
    print()

    # 1. Extraction Phase
    t0 = time.monotonic()
    print("[*] Extracting clinical ontology entities and relationships...")
    extractor = DiseaseOntologyExtractor(Path(args.file))
    entities, relationships = extractor.extract_all()
    extraction_time = round(time.monotonic() - t0, 2)
    print(f"[+] Extraction complete in {extraction_time}s:")
    print(f"    - Total Entities:      {len(entities):,}")
    print(f"    - Total Relationships: {len(relationships):,}")

    from collections import Counter
    ent_counts = Counter(e.type for e in entities)
    print("    - Entities by Type:")
    for k, v in ent_counts.items():
        print(f"        * {k:15s}: {v:,}")

    rel_counts = Counter(r.type for r in relationships)
    print("    - Relationships by Type:")
    for k, v in rel_counts.items():
        print(f"        * {k:20s}: {v:,}")

    # 2. Export Snapshot
    export_snapshot(entities, relationships, Path(args.snapshot_out))
    print(f"[+] Snapshot exported to: {args.snapshot_out}")
    print()

    if args.dry_run:
        print("[!] Dry-run complete. Exiting without database modification.")
        sys.exit(0)

    # 3. Neo4j Verification & Ingestion Phase
    print("[*] Verifying Neo4j connectivity...")
    manager = get_neo4j_manager()
    if not manager.verify_connectivity():
        print("[-] Error: Neo4j database is unreachable at", settings.neo4j_uri)
        print("    Ensure Neo4j is running (Neo4j Desktop or Docker) and password matches .env.")
        sys.exit(1)
    print("[+] Neo4j connection verified successfully.")

    kg_service = get_kg_service()

    print("[*] Initializing schema constraints & indexes...")
    kg_service.initialize_constraints()

    print(f"[*] Ingesting {len(entities):,} entities in batches of {args.batch_size}...")
    t_ingest = time.monotonic()
    ent_upserted = kg_service.batch_create_entities(entities, batch_size=args.batch_size)
    print(f"[+] Upserted {ent_upserted:,} entity nodes.")

    print(f"[*] Ingesting {len(relationships):,} relationships in batches of {args.batch_size}...")
    rel_upserted = kg_service.batch_create_relationships(relationships, batch_size=args.batch_size)
    print(f"[+] Upserted {rel_upserted:,} relationships.")
    ingest_time = round(time.monotonic() - t_ingest, 2)

    # 4. Final Graph Statistics
    stats = kg_service.get_stats()
    print()
    print("=" * 75)
    print("  Ingestion Complete — Live Neo4j Graph Statistics")
    print("=" * 75)
    print(f"  Total Graph Entities:      {stats.total_entities:,}")
    print(f"  Total Graph Relationships: {stats.total_relationships:,}")
    print("  Entity breakdown:")
    for ent_type, count in sorted(stats.entity_counts_by_type.items()):
        print(f"    - {ent_type:15s}: {count:,}")
    print("  Relationship breakdown:")
    for rel_type, count in sorted(stats.relationship_counts_by_type.items()):
        print(f"    - {rel_type:20s}: {count:,}")
    print(f"  Ingestion time:            {ingest_time}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
