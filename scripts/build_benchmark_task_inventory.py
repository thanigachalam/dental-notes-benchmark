#!/usr/bin/env python3
"""Build task/metric inventory tables for benchmark-use documentation.

This supports the manuscript's reuse narrative by showing which NLP tasks and
recommended metrics are represented in the released records.

Example:
  python scripts/build_benchmark_task_inventory.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --out-dir benchmark_inventory
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from typing import Any

from _dataset_utils import ensure_dir, iter_list_items, load_dataset, normalize_nested_columns, parse_jsonish, write_json


def write_rows(path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmark task and metric inventory tables.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--out-dir", default="benchmark_inventory")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = normalize_nested_columns(load_dataset(args.input, sample_limit=args.sample_limit))

    task_tag_counts = Counter(str(x) for x in iter_list_items(df["task_tags"])) if "task_tags" in df.columns else Counter()
    metric_counts = Counter(str(x) for x in iter_list_items(df["metrics_recommended"])) if "metrics_recommended" in df.columns else Counter()

    task_by_primary: dict[str, Counter] = defaultdict(Counter)
    if "primary_task" in df.columns and "task_tags" in df.columns:
        for _, row in df.iterrows():
            primary = str(row.get("primary_task", "unknown"))
            tags = parse_jsonish(row.get("task_tags"))
            if isinstance(tags, list):
                for tag in tags:
                    task_by_primary[primary][str(tag)] += 1

    metrics_by_task: dict[str, Counter] = defaultdict(Counter)
    if "primary_task" in df.columns and "metrics_recommended" in df.columns:
        for _, row in df.iterrows():
            primary = str(row.get("primary_task", "unknown"))
            metrics = parse_jsonish(row.get("metrics_recommended"))
            if isinstance(metrics, list):
                for metric in metrics:
                    metrics_by_task[primary][str(metric)] += 1

    task_rows = [{"task_tag": k, "count": int(v)} for k, v in task_tag_counts.most_common()]
    metric_rows = [{"metric": k, "count": int(v)} for k, v in metric_counts.most_common()]
    primary_task_rows = []
    for primary, counter in sorted(task_by_primary.items()):
        for tag, count in counter.most_common():
            primary_task_rows.append({"primary_task": primary, "task_tag": tag, "count": int(count)})
    metric_by_task_rows = []
    for primary, counter in sorted(metrics_by_task.items()):
        for metric, count in counter.most_common():
            metric_by_task_rows.append({"primary_task": primary, "metric": metric, "count": int(count)})

    write_rows(out_dir / "task_tag_inventory.csv", task_rows)
    write_rows(out_dir / "metric_inventory.csv", metric_rows)
    write_rows(out_dir / "task_tags_by_primary_task.csv", primary_task_rows)
    write_rows(out_dir / "metrics_by_primary_task.csv", metric_by_task_rows)

    report = {
        "input": str(args.input),
        "row_count": int(len(df)),
        "unique_task_tags": int(len(task_tag_counts)),
        "unique_metrics": int(len(metric_counts)),
        "primary_task_count": int(df["primary_task"].nunique()) if "primary_task" in df.columns else 0,
    }
    write_json(report, out_dir / "benchmark_task_inventory_report.json")
    print(f"Task tags: {len(task_tag_counts):,}; Metrics: {len(metric_counts):,}. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
