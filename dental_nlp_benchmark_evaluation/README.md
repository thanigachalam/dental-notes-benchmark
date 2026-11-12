# SDCN-DNLP v21

This folder supplements **IEEE DataPort DOI 10.21227/tt32-vp70**. It does not mutate or overwrite any dataset records, scripts, or the DataPort DOI. Only scripts, small ID-only split manifests, reports, and reproducibility materials are shared in the GitHub.

## Quick start (macOS / Linux)

From the existing local `dental-notes-benchmark` repository root:

```bash
mkdir -p dental_nlp_benchmark_evaluation/data
# Copy these packaged files into dental_nlp_benchmark_evaluation/ (or unzip Dental_NLP_Benchmark_Evaluation_v21.zip here)
python3 -m venv dental_nlp_benchmark_evaluation/.venv
source dental_nlp_benchmark_evaluation/.venv/bin/activate
pip install -r dental_nlp_benchmark_evaluation/requirements.txt
```

Download your **unchanged** DataPort ZIP and keep it locally, e.g. `data/dentalnote_bench_20000_icd10_internalproc_v21.jsonl.zip`. The scripts accept both `.jsonl` and a ZIP containing a single JSONL. ZIP must contain exactly one JSONL; macOS `__MACOSX` entries are ignored.

```bash
DATA=data/dentalnote_bench_20000_icd10_internalproc_v21.jsonl.zip
ROOT=dental_nlp_benchmark_evaluation
python "$ROOT/scripts/01_dataset_audit.py" --input "$DATA" --out "$ROOT/outputs/audit"
python "$ROOT/scripts/02_build_splits.py" --input "$DATA" --out "$ROOT/outputs/splits"
python "$ROOT/scripts/05_check_split_integrity.py" --input "$DATA" --splits "$ROOT/outputs/splits/"*.csv --out "$ROOT/outputs/split_integrity.json"
python "$ROOT/scripts/03_duplicate_analysis.py" --input "$DATA" --out "$ROOT/outputs/duplicates" --threshold 0.85
python "$ROOT/scripts/04_baseline_benchmark.py" --input "$DATA" --splits "$ROOT/outputs/splits/published_70_15_15.csv" "$ROOT/outputs/splits/prompt_disjoint.csv" --out "$ROOT/outputs/benchmarks"
```

### Publishing split manifests to GitHub

Copied only the validated ID-only split CSVs into `dental_nlp_benchmark_evaluation/benchmarks/splits/`. This is not having any original note text or extracted JSONL. Pair reports contain IDs but **no note text**.

### Template-disjoint split is intentionally gated

The dataset has `source_prompt_id`, but that is *not automatically* proof of a distinct higher-level template family. If the generator or source logs provide a defensible source-prompt-to-template-family mapping, create `dental_nlp_benchmark_evaluation/config/verified_template_map.csv`:

```csv
source_prompt_id,template_family_id
P_EXAMPLE_01,FAMILY_A
```

Populate **every** real prompt ID and independently verify the grouping. Then run:

```bash
python dental_nlp_benchmark_evaluation/scripts/02_build_splits.py --input "$DATA" --out dental_nlp_benchmark_evaluation/outputs/splits --template-map dental_nlp_benchmark_evaluation/config/verified_template_map.csv
```

The script refuses incomplete mappings and otherwise **does not generate** `template_disjoint.csv`. Never invent grouping provenance to satisfy a reviewer.

### Important methodological restrictions

- Original data and original `split` assignments remain unchanged. Alternative split CSVs are external benchmark protocols.
- Near-duplicate search is **approximate** MinHash-LSH candidate retrieval with exact **word four-gram Jaccard** confirmation. A zero count is **not** proof no near-duplicates exist. Record chosen threshold and sensitivity analyses (e.g. `0.80`, `0.85`, `0.90`, `0.95`).
- Rule-based and TF-IDF baselines use **`input_note` only** at prediction time, not generated summaries or gold targets. Rule lexicon is learned only from training annotations. This is exploratory and should be compared carefully to whatever rule-based baseline the submitted paper originally described.
- `04_baseline_benchmark.py` currently implements **rule-based and TF-IDF OvR**, not ClinicalBERT. ClinicalBERT requires a separately reproducible Hugging Face training/evaluation implementation and compute budget; do not claim the three-baseline request is complete from this toolkit alone.
- Model quality is measured against **synthetically generated labels**, not independently adjudicated clinical ground truth.
- Source data include `annotator_type=expert_dentist` on some records. This field is *not independent proof* that a dentist scored those records.
- For publication, report the actual class distributions, unseen test-label counts and potential group/label confounding. Label-disjoint or group-disjoint evaluation may be harder than random evaluation.

### Output inventory

| Script | Main outputs | Reviewer item |
|---|---|---|
| 01 | audit JSON, field_states.csv, specialty/diagnosis/procedure counts | R2-11, R2-12 |
| 02 | published and prompt-disjoint ID-only manifests, conditional template manifest | R2-08, R2-09 |
| 05 | manifest coverage, duplicate/missing IDs, prompt group overlap | R2-08 |
| 03 | near-duplicate pairs, cluster membership, exact text duplicates, configuration | R2-06 |
| 04 | task/model/split metrics and config | R2-07, R2-08 |

### Dependencies / provenance

Python 3.10+ recommended. `requirements.txt` contains analysis dependencies and does **not** alter the base generation environment. Freeze installed versions with `pip freeze > dental_nlp_benchmark_evaluation/outputs/requirements_lock.txt` after successful runs. Preserve original scripts and git revision ID in the experiment notes.
