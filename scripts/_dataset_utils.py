#!/usr/bin/env python3
"""Shared utilities for dental dataset release scripts.

These helpers are intentionally lightweight and use only the Python standard
library plus pandas. Keep this file in the same `scripts/` folder as the other
scripts.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

JSON_LIKE_NULLS = {"", "null", "none", "nan", "NaN", "NULL", None}

NESTED_COLUMNS = [
    "diagnoses",
    "procedures",
    "tooth_level",
    "materials_used",
    "perio_metrics",
    "risk_factors",
    "sdoh_factors",
    "medications",
    "allergies",
    "task_tags",
    "metrics_recommended",
    "coverage_checks",
]

TEXT_COLUMNS = [
    "input_note",
    "chief_complaint",
    "findings_text",
    "summary_SOAP",
]

EXPECTED_COLUMNS = [
    "note_id",
    "version",
    "split",
    "generation_method",
    "source_prompt_id",
    "visit_type",
    "specialty",
    "patient_age_group",
    "patient_sex",
    "emergency_flag",
    "diagnosis_code_system",
    "procedure_code_system",
    "input_note",
    "chief_complaint",
    "findings_text",
    "summary_SOAP",
    "summary_type",
    "summary_source",
    "has_summary",
    "diagnoses",
    "procedures",
    "tooth_level",
    "materials_used",
    "perio_metrics",
    "risk_factors",
    "sdoh_factors",
    "medications",
    "allergies",
    "follow_up_recommendation",
    "task_tags",
    "primary_task",
    "metrics_recommended",
    "annotator_type",
    "annotation_round",
    "note_style",
    "summary_style",
    "coverage_checks",
    "note_length",
    "summary_length",
]

FIELD_DESCRIPTIONS = {
    "note_id": "Unique identifier for each synthetic dental note record.",
    "version": "Dataset release or generation version.",
    "split": "Recommended machine learning split for benchmark use.",
    "generation_method": "Method used to generate the synthetic note.",
    "source_prompt_id": "Identifier of the source prompt or scenario template used to generate the note.",
    "visit_type": "Dental visit or encounter type represented in the note.",
    "specialty": "Dental specialty associated with the encounter scenario.",
    "patient_age_group": "Synthetic patient age category used for scenario diversity.",
    "patient_sex": "Synthetic patient sex field used for scenario diversity.",
    "emergency_flag": "Indicates whether the note represents an emergency dental visit.",
    "diagnosis_code_system": "Diagnosis coding system used for diagnosis labels.",
    "procedure_code_system": "Procedure coding system used for procedure labels.",
    "input_note": "Full synthetic dental clinical note text.",
    "chief_complaint": "Patient's main dental complaint or reason for visit.",
    "findings_text": "Narrative description of clinical findings and tooth or site involvement.",
    "summary_SOAP": "Target or reference summary of the note in SOAP, handoff, or clinical-summary style.",
    "summary_type": "Type of summary associated with the note.",
    "summary_source": "Source or reviewer type for the summary.",
    "has_summary": "Indicates whether the record includes a reference summary.",
    "diagnoses": "Structured diagnosis objects with diagnosis code system, code, text, summary label, and source label.",
    "procedures": "Structured procedure objects with procedure code system, code, text, and procedure category.",
    "tooth_level": "Tooth-level structured annotation including tooth number, surfaces, findings, prior restorations, and site procedures.",
    "materials_used": "Dental materials, medications, or clinical materials used during the encounter.",
    "perio_metrics": "Periodontal metrics such as probing depth, attachment loss, bleeding on probing, bone loss, stage, and grade.",
    "risk_factors": "Clinical or contextual risk factors relevant to the synthetic scenario.",
    "sdoh_factors": "Social determinants of health factors included in the scenario, if any.",
    "medications": "Medications or anesthetic agents referenced in the note.",
    "allergies": "Synthetic allergy information included in the note.",
    "follow_up_recommendation": "Recommended follow-up action or care pathway.",
    "task_tags": "NLP or AI benchmark task labels supported by the record.",
    "primary_task": "Primary benchmark task assigned to the record.",
    "metrics_recommended": "Recommended evaluation metrics for model benchmarking.",
    "annotator_type": "Type of annotator or validation source associated with the record.",
    "annotation_round": "Annotation or review round number.",
    "note_style": "Writing style of the generated clinical note.",
    "summary_style": "Style of the reference summary.",
    "coverage_checks": "Rule-based validation indicators confirming required note and summary elements.",
    "note_length": "Character length of the synthetic input note.",
    "summary_length": "Character length of the reference summary.",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_jsonish(value: Any) -> Any:
    """Parse JSON/Python-literal strings into native objects.

    JSONL input normally already contains lists/dicts. CSV export may store these
    fields as strings. This function accepts either form.
    """
    if isinstance(value, (dict, list, tuple)):
        return value
    if pd.isna(value):
        return None
    if value in JSON_LIKE_NULLS:
        return None
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s in JSON_LIKE_NULLS:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return value


def load_dataset(path: str | Path, sample_limit: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if sample_limit and len(records) >= sample_limit:
                    break
        return pd.DataFrame(records)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload[:sample_limit] if sample_limit else payload)
        if isinstance(payload, dict) and "records" in payload:
            data = payload["records"]
            return pd.DataFrame(data[:sample_limit] if sample_limit else data)
        raise ValueError("JSON input must be a list of records or contain a 'records' list.")
    if suffix == ".csv":
        return pd.read_csv(path, nrows=sample_limit)
    raise ValueError(f"Unsupported dataset format: {suffix}. Use .jsonl, .ndjson, .json, or .csv")


def normalize_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NESTED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_jsonish)
    return df


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def write_markdown_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = [str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_jsonl_records(path: str | Path) -> int:
    path = Path(path)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_list_items(series: Iterable[Any]) -> Iterable[Any]:
    for value in series:
        parsed = parse_jsonish(value)
        if isinstance(parsed, list):
            for item in parsed:
                yield item
        elif parsed not in JSON_LIKE_NULLS:
            yield parsed


def value_counts_rows(series: pd.Series, field_name: str, top_n: int | None = None) -> list[dict[str, Any]]:
    counts = series.astype(str).value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    total = len(series)
    return [
        {field_name: idx, "count": int(count), "percent": round(float(count) * 100 / total, 2) if total else 0.0}
        for idx, count in counts.items()
    ]


def flatten_list_counts(df: pd.DataFrame, column: str) -> Counter:
    counter: Counter = Counter()
    if column not in df.columns:
        return counter
    for item in iter_list_items(df[column]):
        if isinstance(item, str):
            counter[item] += 1
        elif isinstance(item, dict):
            counter[json.dumps(item, sort_keys=True)] += 1
        else:
            counter[str(item)] += 1
    return counter


def nested_code_counts(df: pd.DataFrame, column: str, code_key: str) -> Counter:
    counter: Counter = Counter()
    if column not in df.columns:
        return counter
    for value in df[column]:
        parsed = parse_jsonish(value)
        if isinstance(parsed, list):
            for obj in parsed:
                if isinstance(obj, dict) and obj.get(code_key):
                    counter[str(obj[code_key])] += 1
    return counter


def load_codebook_codes(path: str | Path, candidate_columns: list[str]) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set()
        lower_to_original = {name.lower().strip(): name for name in reader.fieldnames}
        selected_col = None
        for candidate in candidate_columns:
            if candidate.lower() in lower_to_original:
                selected_col = lower_to_original[candidate.lower()]
                break
        if selected_col is None:
            # Fall back to the first column containing 'code'.
            for name in reader.fieldnames:
                if "code" in name.lower():
                    selected_col = name
                    break
        if selected_col is None:
            return set()
        return {str(row[selected_col]).strip() for row in reader if str(row.get(selected_col, "")).strip()}


def infer_ieee_dtype(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "unknown"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    # Detect nested string values.
    sample_values = s.head(20).tolist()
    parsed_values = [parse_jsonish(v) for v in sample_values]
    if any(isinstance(v, list) for v in parsed_values):
        return "JSON array"
    if any(isinstance(v, dict) for v in parsed_values):
        return "JSON object"
    avg_len = s.astype(str).str.len().mean()
    if avg_len and avg_len > 120:
        return "text"
    nunique = s.astype(str).nunique(dropna=True)
    if len(s) > 0 and nunique / len(s) < 0.25:
        return "categorical string"
    return "string"


def clean_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
