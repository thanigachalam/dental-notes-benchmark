#!/usr/bin/env python3
"""Repair hard validation failures in llm_summary_note by appending missing required facts.

Run AFTER Ollama generation completes. It does not regenerate summaries.
It appends a short deterministic sentence only when the LLM summary failed
hard checks such as expected_tooth_not_mentioned or follow_up_not_mentioned.

Original validation errors are preserved in:
  llm_summary_original_validation_errors

Example:
  python3 scripts/repair_llm_summary_required_facts.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v21_llm_summary.jsonl \
    --output data/dentalnote_bench_20000_icd10_internalproc_v21_llm_summary_repaired.jsonl \
    --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any

TOOTH_RE = re.compile(r"#([A-T]|[1-9]|[12][0-9]|3[0-2])\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def write_jsonl_record(f, record: dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sentence_count(text: str) -> int:
    return len(SENTENCE_RE.findall(clean_text(text)))


def extract_tooth_numbers(record: dict[str, Any]) -> list[str]:
    teeth = set()
    for m in TOOTH_RE.finditer(record.get("input_note") or ""):
        teeth.add(m.group(1).upper())
    for obj in record.get("tooth_level") or []:
        if isinstance(obj, dict) and obj.get("tooth_number"):
            teeth.add(str(obj["tooth_number"]).replace("#", "").upper())
    return sorted(teeth, key=lambda x: (len(x), x))


def extract_follow_up(record: dict[str, Any]) -> str:
    value = clean_text(record.get("follow_up_recommendation"))
    if value:
        return value.replace("_", " ")
    note = record.get("input_note") or ""
    m = re.search(r"(?:FU|Follow-up)\s*:\s*([^\.\n]+)", note, flags=re.IGNORECASE)
    if m:
        return clean_text(m.group(1)).rstrip(".")
    return ""


def validate_repaired(record: dict[str, Any], summary: str, min_sentences: int) -> list[str]:
    errors: list[str] = []
    text = clean_text(summary)
    lower = text.lower()

    if not text:
        return ["summary_blank"]
    if len(text) < 400:
        errors.append("summary_too_short_for_expanded_handoff")
    sc = sentence_count(text)
    if sc < min_sentences:
        errors.append(f"sentence_count_below_target:{sc}<{min_sentences}")

    allowed_teeth = set(extract_tooth_numbers(record))
    mentioned_teeth = set(m.group(1).upper() for m in TOOTH_RE.finditer(text))
    new_teeth = mentioned_teeth - allowed_teeth
    if new_teeth:
        errors.append("new_tooth_number_mentioned:" + ",".join(sorted(new_teeth)))
    if 0 < len(allowed_teeth) <= 3:
        missing = allowed_teeth - mentioned_teeth
        if missing:
            errors.append("expected_tooth_not_mentioned:" + ",".join(sorted(missing)))

    follow_up = extract_follow_up(record)
    input_lower = (record.get("input_note") or "").lower()
    if "fu:" in input_lower or "follow-up" in input_lower or follow_up:
        if not any(w in lower for w in ["follow", "recheck", "recall", "monitor", "workup", "prn"]):
            errors.append("follow_up_not_mentioned")

    if "patient name" in lower or "real patient" in lower:
        errors.append("privacy_language_flag")

    return errors


def build_repair_sentence(record: dict[str, Any], errors: list[str]) -> str:
    parts = []
    teeth = extract_tooth_numbers(record)
    follow = extract_follow_up(record)
    error_keys = {e.split(":")[0] for e in errors}

    if "expected_tooth_not_mentioned" in error_keys and teeth:
        if len(teeth) == 1:
            parts.append(f"the documented tooth/site involvement included #{teeth[0]}")
        else:
            parts.append("the documented tooth/site involvement included " + ", ".join(f"#{t}" for t in teeth))

    if "follow_up_not_mentioned" in error_keys and follow:
        parts.append(f"the follow-up recommendation was {follow}")

    if not parts:
        return ""

    sentence = "; ".join(parts).strip()
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair missing required facts in LLM summary JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-sentences", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}. Use --overwrite to replace.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    repaired = 0
    before_status = Counter()
    after_status = Counter()
    before_errors = Counter()
    after_errors = Counter()

    with open(args.output, "w", encoding="utf-8") as out:
        for _, record in read_jsonl(args.input):
            total += 1
            old_errors = list(record.get("llm_summary_validation_errors") or [])
            before_status[record.get("llm_summary_validation_status", "missing")] += 1
            for e in old_errors:
                before_errors[e.split(":")[0]] += 1

            summary = clean_text(record.get("llm_summary_note"))
            repair_sentence = build_repair_sentence(record, old_errors)

            if repair_sentence and repair_sentence not in summary:
                if summary and not summary.endswith((".", "!", "?")):
                    summary += "."
                summary = clean_text(summary + " " + repair_sentence)
                repaired += 1
                record["llm_summary_note"] = summary
                record["llm_summary_length"] = len(summary)
                record["llm_summary_sentence_count"] = sentence_count(summary)
                record.setdefault("llm_summary_risk_flags", [])
                record["llm_summary_risk_flags"].append("deterministic_required_fact_repair")
                record["llm_summary_original_validation_status"] = record.get("llm_summary_validation_status")
                record["llm_summary_original_validation_errors"] = old_errors

            new_errors = validate_repaired(record, summary, args.min_sentences)
            record["llm_summary_validation_status"] = "passed" if not new_errors else "failed"
            record["llm_summary_validation_errors"] = new_errors

            after_status[record["llm_summary_validation_status"]] += 1
            for e in new_errors:
                after_errors[e.split(":")[0]] += 1

            write_jsonl_record(out, record)

    report = {
        "input": args.input,
        "output": args.output,
        "total": total,
        "repaired": repaired,
        "before_status": dict(before_status),
        "after_status": dict(after_status),
        "before_errors": dict(before_errors),
        "after_errors": dict(after_errors),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
