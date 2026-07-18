#!/usr/bin/env python3
"""Pattern-based PHI-risk screen for synthetic dental note datasets.

This script is designed for synthetic datasets. It does not prove that data are
PHI-free, but it provides a reproducible screen for common real-world identifier
patterns that reviewers may ask about: email, phone, SSN, MRN-like strings,
full dates, URLs, and street-address-like strings.

Example:
  python scripts/check_synthetic_phi_risk.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --out-dir validation_outputs/phi_risk
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from _dataset_utils import TEXT_COLUMNS, ensure_dir, load_dataset, write_json

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone_us": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn_like": re.compile(r"\b(?:MRN|Medical Record|Record No\.?|Patient ID)\s*[:#-]?\s*[A-Z0-9-]{5,}\b", re.I),
    "dob_like": re.compile(r"\b(?:DOB|Date of Birth)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I),
    "full_date_numeric": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b"),
    "street_address_like": re.compile(r"\b\d{2,6}\s+[A-Za-z0-9.'-]+\s+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?)\b", re.I),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
}


def redact_match(text: str, start: int, end: int, window: int = 35) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:start] + "[MATCH]" + text[end:right]
    return snippet.replace("\n", " ")


def scan_text(value: Any) -> list[dict[str, Any]]:
    text = "" if value is None else str(value)
    findings: list[dict[str, Any]] = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({
                "pattern": label,
                "start": match.start(),
                "end": match.end(),
                "snippet": redact_match(text, match.start(), match.end()),
            })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pattern-based PHI-risk screen on synthetic dental notes.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--out-dir", default="validation_outputs/phi_risk", help="Output directory")
    parser.add_argument("--text-columns", nargs="*", default=None, help="Columns to scan. Defaults to known text columns present in the dataset.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optionally scan only the first N records")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = load_dataset(args.input, sample_limit=args.sample_limit)
    text_columns = args.text_columns or [c for c in TEXT_COLUMNS if c in df.columns]
    if not text_columns:
        raise ValueError("No text columns found to scan. Pass --text-columns explicitly.")

    rows: list[dict[str, Any]] = []
    by_pattern = {name: 0 for name in PATTERNS}
    records_with_findings = set()

    for idx, row in df.iterrows():
        note_id = row.get("note_id", idx)
        for column in text_columns:
            for finding in scan_text(row.get(column, "")):
                by_pattern[finding["pattern"]] += 1
                records_with_findings.add(note_id)
                rows.append({
                    "note_id": note_id,
                    "row_index": int(idx),
                    "column": column,
                    **finding,
                })

    csv_path = out_dir / "phi_risk_findings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["note_id", "row_index", "column", "pattern", "start", "end", "snippet"])
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "input": str(args.input),
        "row_count_scanned": int(len(df)),
        "text_columns_scanned": text_columns,
        "finding_count": int(len(rows)),
        "record_count_with_findings": int(len(records_with_findings)),
        "findings_by_pattern": by_pattern,
        "output_csv": str(csv_path),
        "interpretation": "Pattern-based screen only; zero findings supports synthetic-release quality control but does not independently certify PHI absence.",
    }
    write_json(report, out_dir / "phi_risk_report.json")

    print(f"Scanned {len(df):,} records across {len(text_columns)} text columns.")
    print(f"Findings: {len(rows):,}. Report: {out_dir / 'phi_risk_report.json'}")
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
