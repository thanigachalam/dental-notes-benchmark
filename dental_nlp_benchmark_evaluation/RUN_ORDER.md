# Recommended workflow

1. Run `01_dataset_audit.py` to reproduce counts and inspect `field_states.csv`.
2. Run `02_build_splits.py` and `05_check_split_integrity.py`. Confirm prompt overlap is zero in the prompt-disjoint manifest.
3. Run `03_duplicate_analysis.py` at 0.85 and repeat at 0.80, 0.90, 0.95 into separate output folders.
4. Run `04_baseline_benchmark.py` on both published and prompt-disjoint manifests, recording results and label coverage. Do not claim ClinicalBERT or template-disjoint results until implemented and tested.
5. Publish *only* ID-only manifests, verified summary tables, code and methodology. Keep released clinical records on IEEE DataPort.
6. Added completed numbers and caveats into the revised paper and point-by-point response.
