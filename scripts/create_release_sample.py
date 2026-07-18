#!/usr/bin/env python3
"""Create a small reviewer-friendly sample JSONL file.

The sample can be committed under `examples/` while keeping the full dataset in
`data/` or a DOI-backed repository. Sampling is deterministic and can be
stratified by a column such as `visit_type`, `primary_task`, or `split`.

Example:
  python scripts/create_release_sample.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --output examples/sample_records_v20.jsonl \
    --n 100 \
    --stratify-by visit_type
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from _dataset_utils import ensure_dir, load_dataset


def stratified_sample(df: pd.DataFrame, n: int, stratify_by: str, random_state: int) -> pd.DataFrame:
    if stratify_by not in df.columns or n >= len(df):
        return df.sample(n=min(n, len(df)), random_state=random_state).sort_index()
    groups = df.groupby(stratify_by, dropna=False, group_keys=False)
    base = max(1, n // max(1, groups.ngroups))
    pieces = []
    for _, group in groups:
        pieces.append(group.sample(n=min(base, len(group)), random_state=random_state))
    sample = pd.concat(pieces).drop_duplicates()
    if len(sample) < n:
        remainder = df.drop(index=sample.index)
        if not remainder.empty:
            extra = remainder.sample(n=min(n - len(sample), len(remainder)), random_state=random_state)
            sample = pd.concat([sample, extra])
    elif len(sample) > n:
        sample = sample.sample(n=n, random_state=random_state)
    return sample.sort_index()


def write_jsonl(df: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            cleaned: dict[str, Any] = {}
            for k, v in record.items():
                if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                    cleaned[k] = None
                else:
                    cleaned[k] = v
            f.write(json.dumps(cleaned, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic sample JSONL file for GitHub examples.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--output", default="examples/sample_records_v20.jsonl")
    parser.add_argument("--n", type=int, default=100, help="Number of sample records")
    parser.add_argument("--stratify-by", default="visit_type", help="Column to preserve distribution approximately")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.input)
    sample = stratified_sample(df, args.n, args.stratify_by, args.random_state)
    write_jsonl(sample, args.output)
    print(f"Wrote {len(sample):,} sample records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
