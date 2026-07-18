#!/usr/bin/env python3
"""Export tables that directly support the IEEE Data Descriptor manuscript.

Outputs CSV and Markdown tables for:
  - field-level schema inferred from the released dataset
  - missingness/completeness
  - split, visit type, specialty, age group, primary task distributions
  - note/summary length summary
  - task tags and recommended metrics

Example:
  python scripts/export_ieee_descriptor_tables.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --out-dir paper_tables
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from typing import Any

import pandas as pd

from _dataset_utils import (
    EXPECTED_COLUMNS,
    FIELD_DESCRIPTIONS,
    ensure_dir,
    flatten_list_counts,
    infer_ieee_dtype,
    load_dataset,
    normalize_nested_columns,
    value_counts_rows,
    write_json,
    write_markdown_table,
)


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def schema_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    ordered_columns = [c for c in EXPECTED_COLUMNS if c in df.columns] + [c for c in df.columns if c not in EXPECTED_COLUMNS]
    for col in ordered_columns:
        series = df[col]
        rows.append({
            "field_name": col,
            "data_type": infer_ieee_dtype(series),
            "required": "Yes" if int(series.isna().sum()) == 0 else "Review",
            "non_null": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "description": FIELD_DESCRIPTIONS.get(col, "Dataset field included in the released file."),
            "example": str(series.dropna().iloc[0])[:160] if series.notna().any() else "",
        })
    return rows


def length_summary_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for col in ["note_length", "summary_length"]:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        rows.append({
            "field": col,
            "count": int(numeric.count()),
            "mean": round(float(numeric.mean()), 2),
            "std": round(float(numeric.std()), 2),
            "min": int(numeric.min()),
            "p25": round(float(numeric.quantile(0.25)), 2),
            "median": round(float(numeric.median()), 2),
            "p75": round(float(numeric.quantile(0.75)), 2),
            "max": int(numeric.max()),
        })
    return rows


def counter_rows(counter: Counter, name: str, top_n: int | None = None) -> list[dict[str, Any]]:
    items = counter.most_common(top_n)
    total = sum(counter.values())
    return [
        {name: key, "count": int(count), "percent": round(count * 100.0 / total, 2) if total else 0.0}
        for key, count in items
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export IEEE Data Descriptor support tables.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--out-dir", default="paper_tables")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=50, help="Top N rows for list/coded tables")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = normalize_nested_columns(load_dataset(args.input, sample_limit=args.sample_limit))

    outputs: dict[str, str] = {}

    tables: dict[str, list[dict[str, Any]]] = {
        "field_level_schema": schema_rows(df),
        "length_summary": length_summary_rows(df),
    }

    for col in ["split", "visit_type", "specialty", "patient_age_group", "patient_sex", "generation_method", "primary_task", "summary_type", "note_style", "summary_style"]:
        if col in df.columns:
            tables[f"{col}_distribution"] = value_counts_rows(df[col], col)

    for col, label in [("task_tags", "task_tag"), ("metrics_recommended", "metric"), ("materials_used", "material"), ("risk_factors", "risk_factor"), ("sdoh_factors", "sdoh_factor"), ("medications", "medication"), ("allergies", "allergy")]:
        if col in df.columns:
            tables[f"{col}_distribution"] = counter_rows(flatten_list_counts(df, col), label, args.top_n)

    for name, rows in tables.items():
        csv_path = out_dir / f"{name}.csv"
        md_path = out_dir / f"{name}.md"
        write_csv(csv_path, rows)
        write_markdown_table(rows, md_path)
        outputs[name] = str(csv_path)

    manifest = {
        "input": str(args.input),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "outputs": outputs,
    }
    write_json(manifest, out_dir / "paper_tables_manifest.json")
    print(f"Exported {len(tables)} table groups to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
