#!/usr/bin/env python3
"""Export paper-ready validation tables for the LLM-expanded summary layer.

This script summarizes the per-record LLM summary validation fields added by
`scripts/add_llm_summary_note_ollama_expanded.py`.

It writes CSV, Markdown, and JSON outputs that can be used directly in the
IEEE Data Descriptor paper's Validation and Quality section.

Example:
  python scripts/export_llm_summary_validation_table.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v21_llm_summary.jsonl \
    --out-dir validation_outputs/llm_summary_validation

Optional:
  python scripts/export_llm_summary_validation_table.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v21_llm_summary.jsonl \
    --out-dir validation_outputs/llm_summary_validation \
    --include-all-error-types
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TABLE_ROWS = [
    ("total", "Total records evaluated"),
    ("passed", "Passed all validation checks"),
    ("failed", "Failed one or more validation checks"),
    ("expected_tooth_not_mentioned", "Missing expected tooth/site mention"),
    ("new_tooth_number_mentioned", "Introduced unsupported tooth number"),
    ("sentence_count_below_target", "Below target sentence count"),
    ("summary_too_short", "Summary too short"),
    ("summary_too_short_for_expanded_handoff", "Summary too short for expanded handoff"),
    ("privacy_language_flag", "Privacy-language flag"),
    ("generation_runtime_failure", "Generation/runtime failure"),
]

GENERATION_FAILURE_SOURCES = {
    "llm_generation_failed",
    "ollama_generation_failed",
    "openai_generation_failed",
    "local_rag_generation_failed",
}

GENERATION_FAILURE_ERROR_HINTS = (
    "readtimeout",
    "connecttimeout",
    "connectionerror",
    "httperror",
    "timeout",
    "ollama api error",
    "openai api error",
    "local rag api error",
    "generation failed",
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path, sample_limit: int | None = None) -> Iterable[dict[str, Any]]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
            count += 1
            if sample_limit and count >= sample_limit:
                break


def normalize_error_type(error: Any) -> str:
    text = str(error or "").strip()
    if not text:
        return "blank_error"
    return text.split(":", 1)[0].strip()


def is_generation_failure(record: dict[str, Any]) -> bool:
    source = str(record.get("llm_summary_source", "")).strip().lower()
    if source in GENERATION_FAILURE_SOURCES or "generation_failed" in source:
        return True

    summary = str(record.get("llm_summary_note", "") or "").strip()
    errors = [str(e).lower() for e in (record.get("llm_summary_validation_errors") or [])]
    if not summary and errors:
        joined = " | ".join(errors)
        return any(hint in joined for hint in GENERATION_FAILURE_ERROR_HINTS)
    return False


def percent(count: int, total: int) -> float:
    return round((count * 100.0 / total), 2) if total else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [str(row.get(h, "")).replace("|", "\\|").replace("\n", " ") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LLM summary validation table for paper/report use.")
    parser.add_argument("--input", required=True, help="Final v2.1 JSONL file with LLM summary fields")
    parser.add_argument("--out-dir", default="validation_outputs/llm_summary_validation")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--include-all-error-types",
        action="store_true",
        help="Append every observed validation error type to the main paper table.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    total = 0
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    generation_runtime_failure_count = 0

    length_values: list[int] = []
    sentence_values: list[int] = []

    for record in read_jsonl(args.input, args.sample_limit):
        total += 1
        status = str(record.get("llm_summary_validation_status", "missing")).strip().lower() or "missing"
        status_counts[status] += 1

        source_counts[str(record.get("llm_summary_source", "missing"))] += 1
        model_counts[str(record.get("llm_summary_model", "missing"))] += 1
        prompt_counts[str(record.get("llm_summary_prompt_version", "missing"))] += 1

        try:
            length_values.append(int(record.get("llm_summary_length", 0) or 0))
        except Exception:
            pass
        try:
            sentence_values.append(int(record.get("llm_summary_sentence_count", 0) or 0))
        except Exception:
            pass

        for error in record.get("llm_summary_validation_errors") or []:
            error_counts[normalize_error_type(error)] += 1

        if is_generation_failure(record):
            generation_runtime_failure_count += 1

    table_metric_counts: dict[str, int] = {
        "total": total,
        "passed": status_counts.get("passed", 0),
        "failed": status_counts.get("failed", 0),
        "generation_runtime_failure": generation_runtime_failure_count,
    }
    for key, count in error_counts.items():
        table_metric_counts[key] = count

    table_rows: list[dict[str, Any]] = []
    included_keys = set()
    for key, label in DEFAULT_TABLE_ROWS:
        # Keep the full table stable, but avoid duplicate short/expanded summary rows when count is zero.
        if key == "summary_too_short" and table_metric_counts.get(key, 0) == 0 and table_metric_counts.get("summary_too_short_for_expanded_handoff", 0) > 0:
            continue
        if key == "summary_too_short_for_expanded_handoff" and table_metric_counts.get(key, 0) == 0:
            continue
        count = int(table_metric_counts.get(key, 0))
        table_rows.append({
            "Validation measure": label,
            "Count": count,
            "Percent": f"{percent(count, total):.2f}%",
        })
        included_keys.add(key)

    if args.include_all_error_types:
        for key, count in error_counts.most_common():
            if key in included_keys:
                continue
            label = key.replace("_", " ").capitalize()
            table_rows.append({
                "Validation measure": label,
                "Count": int(count),
                "Percent": f"{percent(int(count), total):.2f}%",
            })

    error_rows = [
        {"validation_error_type": key, "count": int(count), "percent": f"{percent(int(count), total):.2f}%"}
        for key, count in error_counts.most_common()
    ]

    source_rows = [
        {"llm_summary_source": key, "count": int(count), "percent": f"{percent(int(count), total):.2f}%"}
        for key, count in source_counts.most_common()
    ]

    model_rows = [
        {"llm_summary_model": key, "count": int(count), "percent": f"{percent(int(count), total):.2f}%"}
        for key, count in model_counts.most_common()
    ]

    def numeric_summary(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": len(values),
            "min": min(values),
            "mean": round(sum(values) / len(values), 2),
            "max": max(values),
        }

    report = {
        "input": str(args.input),
        "row_count": total,
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "source_counts": dict(source_counts),
        "model_counts": dict(model_counts),
        "prompt_version_counts": dict(prompt_counts),
        "generation_runtime_failure_count": generation_runtime_failure_count,
        "llm_summary_length": numeric_summary(length_values),
        "llm_summary_sentence_count": numeric_summary(sentence_values),
        "outputs": {
            "paper_table_csv": str(out_dir / "llm_summary_validation_table.csv"),
            "paper_table_md": str(out_dir / "llm_summary_validation_table.md"),
            "error_types_csv": str(out_dir / "llm_summary_validation_error_types.csv"),
            "source_distribution_csv": str(out_dir / "llm_summary_source_distribution.csv"),
            "model_distribution_csv": str(out_dir / "llm_summary_model_distribution.csv"),
        },
    }

    write_csv(out_dir / "llm_summary_validation_table.csv", table_rows)
    write_markdown(out_dir / "llm_summary_validation_table.md", table_rows)
    write_csv(out_dir / "llm_summary_validation_error_types.csv", error_rows)
    write_csv(out_dir / "llm_summary_source_distribution.csv", source_rows)
    write_csv(out_dir / "llm_summary_model_distribution.csv", model_rows)
    with (out_dir / "llm_summary_validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({
        "input": str(args.input),
        "out_dir": str(out_dir),
        "row_count": total,
        "passed": status_counts.get("passed", 0),
        "failed": status_counts.get("failed", 0),
        "paper_table": str(out_dir / "llm_summary_validation_table.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
