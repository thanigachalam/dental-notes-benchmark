#!/usr/bin/env python3
"""Export a reviewer-friendly CSV from the JSONL dataset.

Nested fields are serialized as compact JSON strings so reviewers can open the
file in Excel/Sheets while preserving the original structured annotations.

Example:
  python scripts/export_review_csv.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --output review_exports/dentalnote_bench_v20_review.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from _dataset_utils import NESTED_COLUMNS, ensure_dir, load_dataset, normalize_nested_columns


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Export JSONL dataset to reviewer-friendly CSV.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--output", default="review_exports/dentalnote_bench_v20_review.csv")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optionally export only first N rows")
    args = parser.parse_args()

    df = normalize_nested_columns(load_dataset(args.input, sample_limit=args.sample_limit))
    for col in NESTED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(serialize_cell)
    output = Path(args.output)
    ensure_dir(output.parent)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows and {len(df.columns):,} columns to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
