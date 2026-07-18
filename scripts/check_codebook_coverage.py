#!/usr/bin/env python3
"""Check dataset diagnosis/procedure code coverage against codebooks.

This is useful for the IEEE Data Descriptor repository because it shows that the
released records use documented ICD-10-CM mappings and internal procedure codes.

Example:
  python scripts/check_codebook_coverage.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --diagnosis-codebook codebooks/icd10_cm_diagnosis_mappings_v20.csv \
    --procedure-codebook codebooks/internal_procedure_code_definitions_v20.csv \
    --out-dir validation_outputs/codebook_coverage
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _dataset_utils import (
    ensure_dir,
    load_codebook_codes,
    load_dataset,
    nested_code_counts,
    normalize_nested_columns,
    write_json,
)


def write_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("code,count,in_codebook\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check diagnosis/procedure code coverage against codebooks.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--diagnosis-codebook", default="codebooks/icd10_cm_diagnosis_mappings_v20.csv")
    parser.add_argument("--procedure-codebook", default="codebooks/internal_procedure_code_definitions_v20.csv")
    parser.add_argument("--out-dir", default="validation_outputs/codebook_coverage")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = normalize_nested_columns(load_dataset(args.input, sample_limit=args.sample_limit))

    diagnosis_counts = nested_code_counts(df, "diagnoses", "diagnosis_code")
    procedure_counts = nested_code_counts(df, "procedures", "procedure_code")

    diagnosis_codebook = load_codebook_codes(
        args.diagnosis_codebook,
        ["diagnosis_code", "icd10_code", "icd10_cm_code", "code"],
    )
    procedure_codebook = load_codebook_codes(
        args.procedure_codebook,
        ["procedure_code", "internal_procedure_code", "code"],
    )

    diagnosis_rows = [
        {"code": code, "count": int(count), "in_codebook": bool(code in diagnosis_codebook)}
        for code, count in diagnosis_counts.most_common()
    ]
    procedure_rows = [
        {"code": code, "count": int(count), "in_codebook": bool(code in procedure_codebook)}
        for code, count in procedure_counts.most_common()
    ]

    write_counts(out_dir / "diagnosis_code_coverage.csv", diagnosis_rows)
    write_counts(out_dir / "procedure_code_coverage.csv", procedure_rows)

    missing_dx = sorted([r["code"] for r in diagnosis_rows if not r["in_codebook"]])
    missing_proc = sorted([r["code"] for r in procedure_rows if not r["in_codebook"]])

    report = {
        "input": str(args.input),
        "row_count": int(len(df)),
        "diagnosis_codebook": str(args.diagnosis_codebook),
        "procedure_codebook": str(args.procedure_codebook),
        "unique_diagnosis_codes_in_dataset": int(len(diagnosis_counts)),
        "unique_procedure_codes_in_dataset": int(len(procedure_counts)),
        "unique_diagnosis_codes_in_codebook": int(len(diagnosis_codebook)),
        "unique_procedure_codes_in_codebook": int(len(procedure_codebook)),
        "diagnosis_codes_not_in_codebook": missing_dx,
        "procedure_codes_not_in_codebook": missing_proc,
        "diagnosis_coverage_percent": round(100.0 * (len(diagnosis_counts) - len(missing_dx)) / len(diagnosis_counts), 2) if diagnosis_counts else 0.0,
        "procedure_coverage_percent": round(100.0 * (len(procedure_counts) - len(missing_proc)) / len(procedure_counts), 2) if procedure_counts else 0.0,
    }
    write_json(report, out_dir / "codebook_coverage_report.json")

    print("Diagnosis coverage:", report["diagnosis_coverage_percent"], "%")
    print("Procedure coverage:", report["procedure_coverage_percent"], "%")
    print("Report:", out_dir / "codebook_coverage_report.json")
    return 1 if missing_dx or missing_proc else 0


if __name__ == "__main__":
    raise SystemExit(main())
