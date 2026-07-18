#!/usr/bin/env python3
"""Validate nested annotation structures in the dental dataset.

Checks parseability and expected keys for diagnoses, procedures, tooth-level
annotations, periodontal metrics, and coverage-check objects. Also checks that
`tooth_level[].procedures_here` codes are present in the row-level `procedures`.

Example:
  python scripts/verify_nested_annotations.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --out-dir validation_outputs/nested_annotations
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from typing import Any

from _dataset_utils import ensure_dir, load_dataset, normalize_nested_columns, parse_jsonish, write_json

EXPECTED_KEYS = {
    "diagnoses": {"diagnosis_code_system", "diagnosis_code", "diagnosis_text", "summary_text", "diagnosis_source_label"},
    "procedures": {"procedure_code_system", "procedure_code", "procedure_text", "procedure_category"},
    "tooth_level": {"tooth_number", "surfaces", "findings", "prior_restorations", "procedures_here"},
    "perio_metrics": {"generalized_or_localized", "worst_pd_mm", "worst_cal_mm", "bop_percent", "bone_loss_percent", "perio_stage", "perio_grade"},
    "coverage_checks": {"note_has_cc", "note_has_findings", "note_has_followup", "summary_has_dx", "summary_has_plan", "summary_has_tooth"},
}

LIST_COLUMNS = {"diagnoses", "procedures", "tooth_level"}
DICT_COLUMNS = {"perio_metrics", "coverage_checks"}


def row_note_id(row: Any, idx: int) -> str:
    value = row.get("note_id", None)
    return str(value) if value is not None else str(idx)


def validate_row(row: Any, idx: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    note_id = row_note_id(row, idx)

    procedure_codes = set()
    procedures = parse_jsonish(row.get("procedures"))
    if isinstance(procedures, list):
        for proc in procedures:
            if isinstance(proc, dict) and proc.get("procedure_code"):
                procedure_codes.add(str(proc["procedure_code"]))

    for column in LIST_COLUMNS:
        value = parse_jsonish(row.get(column))
        if not isinstance(value, list):
            errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "expected_list", "detail": type(value).__name__})
            continue
        if len(value) == 0:
            errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "empty_list", "detail": "no nested objects present"})
        for obj_idx, obj in enumerate(value):
            if not isinstance(obj, dict):
                errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "nested_item_not_object", "detail": f"item_index={obj_idx}"})
                continue
            missing = sorted(EXPECTED_KEYS[column] - set(obj.keys()))
            if missing:
                errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "missing_keys", "detail": ",".join(missing)})
            if column == "tooth_level":
                procedures_here = obj.get("procedures_here", [])
                if isinstance(procedures_here, list):
                    unknown = sorted([str(code) for code in procedures_here if str(code) not in procedure_codes])
                    if unknown:
                        errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "procedures_here_not_in_row_procedures", "detail": ",".join(unknown)})
                else:
                    errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "procedures_here_not_list", "detail": type(procedures_here).__name__})

    for column in DICT_COLUMNS:
        value = parse_jsonish(row.get(column))
        if not isinstance(value, dict):
            errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "expected_object", "detail": type(value).__name__})
            continue
        missing = sorted(EXPECTED_KEYS[column] - set(value.keys()))
        if missing:
            errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "missing_keys", "detail": ",".join(missing)})
        if column == "coverage_checks":
            for key in EXPECTED_KEYS[column]:
                if key in value and not isinstance(value[key], bool):
                    errors.append({"note_id": note_id, "row_index": idx, "column": column, "issue": "coverage_value_not_boolean", "detail": f"{key}={value[key]}"})

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify nested JSON annotations in the dental dataset.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--out-dir", default="validation_outputs/nested_annotations")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = normalize_nested_columns(load_dataset(args.input, sample_limit=args.sample_limit))

    errors: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        errors.extend(validate_row(row, int(idx)))

    with (out_dir / "nested_annotation_errors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["note_id", "row_index", "column", "issue", "detail"])
        writer.writeheader()
        writer.writerows(errors)

    issue_counts = Counter(error["issue"] for error in errors)
    column_counts = Counter(error["column"] for error in errors)
    report = {
        "input": str(args.input),
        "row_count": int(len(df)),
        "error_count": int(len(errors)),
        "issue_counts": dict(issue_counts),
        "column_counts": dict(column_counts),
        "error_csv": str(out_dir / "nested_annotation_errors.csv"),
    }
    write_json(report, out_dir / "nested_annotation_report.json")

    print(f"Nested annotation errors: {len(errors):,}")
    print("Report:", out_dir / "nested_annotation_report.json")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
