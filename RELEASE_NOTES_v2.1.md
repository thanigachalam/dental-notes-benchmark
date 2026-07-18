# dental-notes-benchmark v2.1

## Release summary

This release contains the public **v2.0** version of **dental-notes-benchmark**, an open synthetic benchmark dataset for dental clinical note extraction and summarization.

The release is designed for open research and reproducible experimentation in:

- dental NLP
- clinical summarization
- diagnosis extraction
- procedure prediction
- tooth-level entity extraction
- structured documentation analysis
- DentalNoteBench v2.1 includes a base synthetic dataset and an LLM-expanded clinical handoff summary layer generated with a locally hosted Ollama model. The final release file is the repaired and validated JSONL file:
data/dentalnote_bench_20000_icd10_internalproc_v21.jsonl


## What is included

- **20,000 synthetic dental clinical notes** in JSONL format
- **ICD-10-CM diagnosis mappings**
- **open internal procedure ontology** (`DPO_INTERNAL_V1`)
- structured tooth-level annotations
- summary targets for benchmarking
- split assignments for train / validation / test
- dataset statistics file
- diagnosis and procedure codebooks
- dataset generator script
- repository documentation files

## Release highlights

### Open public release design
This release uses:

- **ICD-10-CM** for diagnoses
- **open internal procedure codes** for procedures

This approach preserves public usability while avoiding reliance on proprietary public procedure code redistribution.

### Benchmark-oriented structure
The dataset supports multiple tasks:

- SOAP or concise summary generation
- multi-label diagnosis prediction
- procedure extraction / prediction
- tooth-level entity recognition
- visit-type classification
- periodontal metric extraction where applicable

### Multi-specialty coverage
The dataset includes synthetic encounters spanning:

- restorative dentistry
- periodontics
- endodontics
- pediatric dentistry
- implant dentistry
- oral surgery
- general recall and preventive care
- emergency dental care

## Intended use

This release is intended for:

- academic research
- benchmarking
- model evaluation
- educational demonstrations
- reproducible dental NLP experiments

## Not intended for

This dataset is **not** intended for:

- direct patient care
- autonomous diagnosis
- billing submission logic
- legal documentation
- real-world claims adjudication

## Citation title

**dental-notes-benchmark: A Synthetic Benchmark Dataset for Dental Clinical Note Extraction and Summarization**

## Suggested repository description

Open synthetic benchmark dataset for dental clinical note extraction and summarization with ICD-10-CM diagnoses and structured tooth-level annotations.
