# dental-notes-benchmark

[![Dataset](https://img.shields.io/badge/dataset-20,000_notes-blue)](#current-release)
[![Format](https://img.shields.io/badge/format-JSONL-green)](#data-format)
[![Diagnosis](https://img.shields.io/badge/diagnosis-ICD--10--CM-orange)](#coding-strategy)
[![Procedures](https://img.shields.io/badge/procedures-open_internal_codes-purple)](#coding-strategy)


**dental-notes-benchmark** is an open synthetic benchmark dataset for **dental clinical note extraction and summarization**. It provides multi-specialty synthetic dental encounter notes with **ICD-10-CM diagnosis mappings**, **open internal procedure codes**, **tooth-level structured annotations**, and **summary targets** for reproducible dental NLP research.

This repository contains the public release of a synthetic dataset designed to support research in:
- dental clinical NLP
- note summarization
- diagnosis extraction
- procedure prediction
- tooth-level entity recognition
- structured documentation analysis
- dental informatics benchmarking

The dataset is intended for **open research, benchmarking, and educational use** and does **not** contain real patient records.

---

## Overview

Clinical documentation in dentistry is often unstructured, which makes it difficult to build and evaluate systems for summarization, diagnosis extraction, structured charting, and quality review. Public dental note benchmarks remain limited because real encounter notes are difficult to share due to privacy, governance, and institutional barriers.

**dental-notes-benchmark** addresses this gap with a fully synthetic, openly shareable dataset that reflects realistic dental documentation patterns across multiple specialties while preserving structured labels for machine learning and evaluation.

The design emphasizes:
- **open usability** for public research use
- **clinical plausibility** in note structure and terminology
- **multi-task support** for summarization and extraction
- **tooth-level granularity** beyond billing-oriented code systems
- **transparent provenance** through metadata fields
- **reproducibility** through codebooks, scripts, and validation artifacts

---

## Current Release

**Version:** `v2.0`  
**Format:** `JSONL` (one JSON object per line)

### Dataset summary
- **Total notes:** 20,000
- **Train split:** 14,000
- **Validation split:** 3,000
- **Test split:** 3,000
  - 1,000 train
  - 500 validation
  - 500 test
- **Predefined benchmark subset:** 2,000 records (1,000 train / 500 validation / 500 test)
- **Diagnosis coding:** ICD-10-CM
- **Procedure coding:** `DPO_INTERNAL_V1` (open internal procedure ontology)
- **Structured detail:** tooth-level findings, surfaces, risk factors, medications, follow-up, and periodontal metrics

### Supported specialties
- Restorative dentistry
- Periodontics
- Endodontics
- Pediatric dentistry
- Implant dentistry
- Oral surgery
- General dentistry / recall
- Emergency dental care

---

## Benchmark Tasks

The dataset supports the following benchmark tasks:

1. **SOAP note summarization**  
   Input: full clinical note  
   Output: structured or concise clinical summary

2. **Multi-label diagnosis prediction**  
   Input: full clinical note  
   Output: one or more ICD-10-CM diagnosis codes

3. **Procedure prediction / extraction**  
   Input: full clinical note  
   Output: one or more internal procedure codes

4. **Tooth-level entity extraction**  
   Input: full clinical note  
   Output: tooth number, surfaces, findings, and linked procedures

5. **Visit-type classification**  
   Input: full clinical note  
   Output: encounter category such as recall, restorative, endodontic, emergency, or periodontal maintenance

6. **Periodontal metric extraction**  
   Input: periodontal note text  
   Output: structured perio metrics where applicable

---

## Coding Strategy

### Diagnosis labels
Diagnosis labels are normalized to **ICD-10-CM**.

Example:

```json
{
  "diagnosis_code_system": "ICD-10-CM",
  "diagnosis_code": "K04.02",
  "diagnosis_text": "Irreversible pulpitis"
}
```

### Procedure labels
Procedure labels use an **open internal procedure code system** rather than CDT.

Example:

```json
{
  "procedure_code_system": "DPO_INTERNAL_V1",
  "procedure_code": "PROC_PULPOTOMY",
  "procedure_text": "Emergency pulpotomy"
}
```

This approach keeps the public release openly reusable without dependence on proprietary dental procedure terminologies.

---

## Data Format

The dataset is stored in **JSONL** format, with **one JSON object per line**.

Each record may include:
- note metadata
- specialty and visit type
- full synthetic note text
- chief complaint
- findings text
- summary text
- diagnoses
- procedures
- tooth-level annotations
- periodontal metrics
- risk factors and SDOH
- medications and allergies
- follow-up recommendations
- task tags and recommended metrics
- provenance and annotation metadata

---

## Example Record

```json
{
  "note_id": "DN_000010",
  "version": "v2.0",
  "split": "test",
  "generation_method": "template_llm",
  "source_prompt_id": "P_PEDIATRIC_EMERG_01",
  "visit_type": "emergency",
  "specialty": "pediatric_dentistry",
  "patient_age_group": "child",
  "patient_sex": "female",
  "emergency_flag": true,
  "input_note": "SUBJECTIVE: 9-year-old female presents with her parent after falling off a bicycle earlier today ...",
  "chief_complaint": "Front tooth pushed back and painful after bicycle fall.",
  "findings_text": "Lateral luxation of #8 with palatal displacement and mobility; widened PDL space, no fracture.",
  "summary_SOAP": "9-year-old healthy female presented on an emergency basis after a bicycle fall with palatally displaced #8 ...",
  "diagnoses": [
    {
      "diagnosis_code_system": "ICD-10-CM",
      "diagnosis_code": "S03.2XXA",
      "diagnosis_text": "Dislocation of tooth: initial encounter"
    }
  ],
  "procedures": [
    {
      "procedure_code_system": "DPO_INTERNAL_V1",
      "procedure_code": "PROC_REPOSITION",
      "procedure_text": "Repositioning of injured tooth"
    }
  ]
}
```

---

## Files Included

### Main dataset
- Download the full dataset from IEEE Data Port

DOI:10.21227/tt32-vp70

https://ieee-dataport.org/documents/synthetic-dental-clinical-notes-dataset-dental-nlp-benchmarking

### Codebooks
- `codebooks/icd10_cm_diagnosis_mappings_v20.csv`
- `codebooks/internal_procedure_code_definitions_v20.csv`

### Generator / scripts
- `generators/generate_dentalnote_bench_20000_icd10_internalproc_v20.py`

### Recommended companion docs
- `dataset_card.md`
- `docs/schema.md`
- `docs/validation.md`
- `docs/annotation_guidelines.md`

---

## Validation and Quality

The dataset is designed with validation across four dimensions:

1. **Realism and completeness**
2. **Annotation and structural consistency**
3. **Benchmark utility**
4. **Privacy and memorization risk**

Examples of quality controls include:
- summary coverage checks
- structured field consistency checks
- split stratification
- note-style variability
- predefined benchmark subset construction (2,000 records; 1,000 training, 500 validation, 500 test)
The predefined 2,000-record subset is a research benchmark subset and should not be interpreted as an independently clinician-reviewed cohort. The `annotator_type` metadata was assigned programmatically and does not establish human clinical adjudication. Independent dentist assessments are documented separately.
- diagnosis normalization through ICD-10-CM
- open procedure code definitions for reproducible modeling

---

## Intended Use

This dataset is suitable for:
- NLP benchmarking
- structured extraction research
- summarization research
- synthetic data research
- educational demonstrations
- academic publication and reproducible experimentation

---

## Out-of-Scope / Not Intended For

This dataset should **not** be used as:
- direct clinical decision support
- billing advice
- payer submission logic
- autonomous diagnosis
- autonomous treatment recommendation
- legal or compliance documentation
- substitute for real patient records

---

## Installation / Quick Start

### Load with Python

```python
import json

records = []
with open("data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(len(records))
print(records[0]["note_id"])
```

### Load with pandas

```python
import pandas as pd

df = pd.read_json(
    "data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl",
    lines=True
)
print(df.shape)
```

---

## Repository Structure

```text
.
├── README.md
├── dataset_card.md
├── CITATION.cff
├── LICENSE
├── data/
├── codebooks/
├── generators/
├── docs/
└── examples/
```

---

## License

This repository uses a **license model**:
- **This repository, including its dataset, source code, scripts, and documentation, is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.** CC BY 4.0

See the [`LICENSE`](LICENSE) file for details.

> Important: ICD-10-CM diagnosis codes are included to improve diagnosis normalization for research. Internal procedure codes are benchmark-oriented open labels and do not replace licensed billing code systems.

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{selvam_dental_notes_benchmark_2026,
  author       = {Thanigachalam Tamizh Selvam},
  title        = {dental-notes-benchmark: A Synthetic Benchmark Dataset for Dental Clinical Note Extraction and Summarization},
  year         = {2026},
  version      = {v2.0},
  publisher    = {GitHub / IEEE DataPort},
  url          = {https://github.com/thanigachalam/dental-notes-benchmark}
  DOI          = 10.21227/tt32-vp70
}
```

---

## Companion Paper

**Title:**  
*Synthetic Dental Clinical Notes Dataset for Dental NLP Benchmarking (SDCN-DNLP)*

This repository accompanies the dataset generation script and evaluation release and supporting manuscript.

---

## Disclaimer

**dental-notes-benchmark** is a synthetic research dataset. It does not contain real patient records and should not be interpreted as clinical guidance, billing guidance, or a substitute for professional dental judgment.

---

## Contact

**Thanigachalam Tamizh Selvam**  
Prosper, Texas, USA  
Email: `thanigachalamtamizhselvam@gmail.com`

---

## Acknowledgments

We thank the dental clinicians and reviewers who contributed domain feedback on note realism, clinical structure, and benchmark design.
