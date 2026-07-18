#!/usr/bin/env python3
"""Record-level consistency checks for the dental dataset.

This script complements schema validation. It checks logical consistency within
each record: unique IDs, split values, summary flags, note/summary lengths,
emergency flag vs visit type, and row-level coding system consistency.

Example:
  python scripts/validate_record_consistency.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --out-dir validation_outputs/record_consistency
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from typing import Any

from _dataset_utils import ensure_dir, load_dataset, normalize_nested_columns, parse_jsonish, write_json

VALID_SPLITS = {"train", "validation", "test"}
NOTE_ID_PATTERN = re.compile(r"^DN_\d{6}$")


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "y"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "n"}:
            return False
    return None


def add_issue(issues: list[dict[str, Any]], note_id: Any, row_index: int, issue: str, detail: str) -> None:
    issues.append({"note_id": note_id, "row_index": row_index, "issue": issue, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run record-level logical consistency checks.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--out-dir", default="validation_outputs/record_consistency")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--length-tolerance", type=int, default=0, help="Allowed length mismatch when checking character lengths")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = normalize_nested_columns(load_dataset(args.input, sample_limit=args.sample_limit))
    issues: list[dict[str, Any]] = []

    if "note_id" in df.columns:
        dup_ids = set(df.loc[df["note_id"].duplicated(keep=False), "note_id"].astype(str))
    else:
        dup_ids = set()

    for idx, row in df.iterrows():
        note_id = row.get("note_id", f"row_{idx}")
        row_index = int(idx)

        if "note_id" not in df.columns or not str(note_id).strip():
            add_issue(issues, note_id, row_index, "missing_note_id", "note_id is missing or blank")
        elif not NOTE_ID_PATTERN.match(str(note_id)):
            add_issue(issues, note_id, row_index, "note_id_pattern", "expected DN_000001 style identifier")
        if str(note_id) in dup_ids:
            add_issue(issues, note_id, row_index, "duplicate_note_id", "note_id is not unique")

        split = str(row.get("split", "")).strip()
        if split not in VALID_SPLITS:
            add_issue(issues, note_id, row_index, "invalid_split", f"split={split}")

        input_note = "" if row.get("input_note") is None else str(row.get("input_note"))
        summary = "" if row.get("summary_SOAP") is None else str(row.get("summary_SOAP"))
        has_summary = as_bool(row.get("has_summary"))
        if has_summary is True and not summary.strip():
            add_issue(issues, note_id, row_index, "has_summary_but_blank_summary", "has_summary=True but summary_SOAP is blank")
        if has_summary is False and summary.strip():
            add_issue(issues, note_id, row_index, "summary_present_but_has_summary_false", "summary_SOAP populated but has_summary=False")

        if "note_length" in df.columns:
            try:
                reported = int(row.get("note_length"))
                actual = len(input_note)
                if abs(reported - actual) > args.length_tolerance:
                    add_issue(issues, note_id, row_index, "note_length_mismatch", f"reported={reported}; actual_chars={actual}")
            except Exception:
                add_issue(issues, note_id, row_index, "note_length_not_integer", str(row.get("note_length")))
        if "summary_length" in df.columns:
            try:
                reported = int(row.get("summary_length"))
                actual = len(summary)
                if abs(reported - actual) > args.length_tolerance:
                    add_issue(issues, note_id, row_index, "summary_length_mismatch", f"reported={reported}; actual_chars={actual}")
            except Exception:
                add_issue(issues, note_id, row_index, "summary_length_not_integer", str(row.get("summary_length")))

        emergency_flag = as_bool(row.get("emergency_flag"))
        visit_type = str(row.get("visit_type", "")).lower()
        if emergency_flag is True and "emergency" not in visit_type:
            add_issue(issues, note_id, row_index, "emergency_flag_visit_type_mismatch", f"emergency_flag=True; visit_type={visit_type}")
        if emergency_flag is False and visit_type == "emergency":
            add_issue(issues, note_id, row_index, "visit_type_emergency_flag_false", "visit_type=emergency but emergency_flag=False")

        diagnoses = parse_jsonish(row.get("diagnoses"))
        if isinstance(diagnoses, list):
            for obj in diagnoses:
                if isinstance(obj, dict) and obj.get("diagnosis_code_system") != row.get("diagnosis_code_system"):
                    add_issue(issues, note_id, row_index, "diagnosis_code_system_mismatch", str(obj.get("diagnosis_code_system")))

        procedures = parse_jsonish(row.get("procedures"))
        if isinstance(procedures, list):
            for obj in procedures:
                if isinstance(obj, dict) and obj.get("procedure_code_system") != row.get("procedure_code_system"):
                    add_issue(issues, note_id, row_index, "procedure_code_system_mismatch", str(obj.get("procedure_code_system")))

    with (out_dir / "record_consistency_issues.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["note_id", "row_index", "issue", "detail"])
        writer.writeheader()
        writer.writerows(issues)

    issue_counts = Counter(issue["issue"] for issue in issues)
    report = {
        "input": str(args.input),
        "row_count": int(len(df)),
        "issue_count": int(len(issues)),
        "issue_counts": dict(issue_counts),
        "issue_csv": str(out_dir / "record_consistency_issues.csv"),
    }
    write_json(report, out_dir / "record_consistency_report.json")
    print(f"Record consistency issues: {len(issues):,}")
    print("Report:", out_dir / "record_consistency_report.json")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
