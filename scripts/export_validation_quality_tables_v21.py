#!/usr/bin/env python3
"""Export reviewer-safe validation and quality tables for dental-notes benchmark v2.1.

This script replaces the earlier ad-hoc validation table logic with two fixes:
  1) Procedure labels are evaluated using direct text evidence plus documented
     contextual evidence cues (for example, radiograph/imaging for diagnostic
     radiographs and local anesthetic terms for anesthesia).
  2) Tooth/site validation accepts dental-specific site descriptors such as
     implant_site, full_coverage, root, cervical/C, facial/F, and standard
     restorative surface combinations.

Outputs are designed for the manuscript and GitHub validation folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOTH_RE = re.compile(r"#([A-T]|[1-9]|[12][0-9]|3[0-2])\b", re.IGNORECASE)
PHI_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone_us": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn_like": re.compile(r"\b(?:MRN|Medical Record|Record No\.?|Patient ID)\s*[:#-]?\s*[A-Z0-9-]{5,}\b", re.I),
    "dob_like": re.compile(r"\b(?:DOB|Date of Birth)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I),
    "full_date_numeric": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b"),
    "street_address_like": re.compile(r"\b\d{2,6}\s+[A-Za-z0-9.'-]+\s+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?)\b", re.I),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
}

REQUIRED_FIELDS = [
    "note_id", "version", "split", "visit_type", "specialty", "patient_age_group", "patient_sex",
    "input_note", "chief_complaint", "findings_text", "summary_SOAP", "diagnoses", "procedures",
    "tooth_level", "follow_up_recommendation", "task_tags", "primary_task", "metrics_recommended",
]

LLM_FIELDS = [
    "llm_summary_note", "llm_summary_source", "llm_summary_model", "llm_summary_provider",
    "llm_summary_prompt_version", "llm_summary_style", "llm_summary_generated_at",
    "llm_summary_length", "llm_summary_sentence_count", "llm_summary_validation_status",
    "llm_summary_validation_errors",
]

ALLOWED_SURFACES = {
    "M", "D", "O", "B", "L", "F", "I", "C",
    "MO", "DO", "MOD", "BO", "LO", "BL", "MB", "DB", "ML", "DL",
    "mesial", "distal", "occlusal", "buccal", "lingual", "facial", "incisal",
    "cervical", "root", "implant_site", "full_coverage", "site", "edentulous_site",
}

CONTEXTUAL_CUES_BY_BASE = {
    "local anesthesia": ["local anesthesia", "local anesthetic", "anesthesia", "anesthetic", "lidocaine", "articaine", "mepivacaine"],
    "periapical radiograph": ["periapical radiograph", "radiograph", "radiographic", "imaging", "x-ray", "xray"],
    "bitewing radiograph": ["bitewing", "radiograph", "radiographic", "imaging", "x-ray", "xray"],
    "radiographic review": ["radiographic review", "radiograph", "radiographic", "imaging", "x-ray", "xray"],
    "dental prophylaxis": ["prophylaxis", "prophy", "cleaning", "preventive services"],
    "adult prophylaxis": ["prophylaxis", "prophy", "cleaning", "preventive services"],
    "child prophylaxis": ["prophylaxis", "prophy", "cleaning", "preventive services"],
    "periodic oral evaluation": ["periodic exam", "periodic oral evaluation", "exam", "recall"],
    "oral hygiene instruction": ["oral hygiene", "brushing", "flossing", "interdental cleaning", "diet/oral hygiene"],
    "topical fluoride varnish": ["fluoride", "fluoride varnish", "topical fluoride"],
    "sealant placement": ["sealant", "pit and fissure sealant"],
    "post-operative instructions": ["post operative", "post-operative", "post-op", "precautions", "instructions"],
    "emergency evaluation": ["emergency evaluation", "emergency management", "urgent"],
    "dental trauma evaluation": ["trauma evaluation", "traumatic", "trauma", "emergency management"],
    "prescription management": ["medication", "meds", "rx", "prescribed", "prescription"],
}

CONTEXTUAL_CUES_BY_CATEGORY = {
    "diagnostic": ["radiograph", "radiographic", "imaging", "x-ray", "xray", "exam", "review", "testing"],
    "anesthesia": ["anesthesia", "anesthetic", "lidocaine", "articaine", "mepivacaine"],
    "preventive": ["preventive", "prophy", "prophylaxis", "cleaning", "fluoride", "sealant", "oral hygiene"],
    "restorative": ["composite", "restoration", "occlusion", "adjustment", "filling"],
    "periodontics": ["periodontal", "debridement", "scaling", "maintenance", "oral hygiene"],
    "endodontics": ["pulp", "pulpotomy", "vitality", "temporary restoration", "endodontic"],
    "emergency": ["emergency", "urgent", "precautions", "management"],
    "trauma": ["trauma", "reposition", "splint", "fracture", "displacement"],
    "implant": ["implant", "osseointegration", "abutment"],
    "oral_surgery": ["extraction", "surgical", "post-operative", "post operative", "sutures", "gelfoam"],
    "prosthodontics": ["crown", "impression", "scan", "provisional", "cement", "occlusion", "prosthodontic"],
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9#]+", " ", value.lower()).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = [str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(n: int | float, d: int | float) -> float:
    return round(float(n) * 100.0 / float(d), 2) if d else 0.0


def procedure_base_name(proc: dict[str, Any]) -> str:
    text = clean_text(proc.get("procedure_text"))
    text = re.split(r",\s*(?:tooth|site)\s*#?\d+", text, flags=re.IGNORECASE)[0]
    return clean_text(text)


def text_blob(record: dict[str, Any]) -> str:
    parts = [
        record.get("input_note"), record.get("findings_text"), record.get("summary_SOAP"), record.get("llm_summary_note"),
        " ".join(str(x) for x in record.get("materials_used") or []),
        " ".join(str(x) for x in record.get("medications") or []),
    ]
    return normalize(" ".join(str(x or "") for x in parts))


def diagnosis_supported(record: dict[str, Any], dx: dict[str, Any]) -> bool:
    blob = text_blob(record)
    candidates = [
        str(dx.get("diagnosis_text", "")).split(",")[0],
        dx.get("summary_text"),
        dx.get("diagnosis_source_label"),
    ]
    return any(normalize(str(c)) and normalize(str(c)) in blob for c in candidates)


def procedure_support(record: dict[str, Any], proc: dict[str, Any]) -> tuple[str, str]:
    blob = text_blob(record)
    base = procedure_base_name(proc)
    base_norm = normalize(base)
    if base_norm and base_norm in blob:
        return "direct_text", base

    cues = []
    cues.extend(CONTEXTUAL_CUES_BY_BASE.get(base_norm, []))
    category = str(proc.get("procedure_category") or "").strip()
    cues.extend(CONTEXTUAL_CUES_BY_CATEGORY.get(category, []))
    for cue in cues:
        if normalize(cue) in blob:
            return "contextual_text", cue
    return "unsupported", base or str(proc.get("procedure_category") or "unknown")


def valid_tooth_entry(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    issues = []
    tooth = clean_text(entry.get("tooth_number"))
    if not tooth:
        issues.append("missing_tooth_number")
    elif not re.fullmatch(r"(?:[A-T]|[1-9]|[12][0-9]|3[0-2])", tooth, flags=re.IGNORECASE):
        issues.append(f"unexpected_tooth_number:{tooth}")

    surfaces = entry.get("surfaces")
    if not isinstance(surfaces, list):
        issues.append("surfaces_not_list")
    elif not surfaces:
        issues.append("surfaces_empty")
    else:
        for s in surfaces:
            s_text = clean_text(s)
            if s_text not in ALLOWED_SURFACES:
                issues.append(f"unexpected_surface:{s_text}")

    for key in ["findings", "prior_restorations", "procedures_here"]:
        if key not in entry:
            issues.append(f"missing_{key}")
        elif not isinstance(entry.get(key), list):
            issues.append(f"{key}_not_list")

    return not issues, issues


def scan_phi(record: dict[str, Any]) -> list[dict[str, str]]:
    cols = ["input_note", "chief_complaint", "findings_text", "summary_SOAP", "llm_summary_note"]
    findings = []
    for col in cols:
        text = str(record.get(col) or "")
        for label, pattern in PHI_PATTERNS.items():
            if pattern.search(text):
                findings.append({"column": col, "pattern": label})
    return findings


def summarize_validation(records: list[dict[str, Any]], out_dir: Path) -> None:
    total = len(records)

    # Required fields.
    available_fields = set().union(*(r.keys() for r in records)) if records else set()
    required = REQUIRED_FIELDS + [f for f in LLM_FIELDS if f in available_fields]
    required_rows = []
    for field in required:
        present = sum(1 for r in records if r.get(field) not in [None, "", []])
        required_rows.append({
            "field": field,
            "records_present": present,
            "records_total": total,
            "presence_percent": pct(present, total),
            "status": "ok" if present == total else "review",
        })

    # Coverage checks.
    coverage_counts = Counter()
    coverage_total = Counter()
    for r in records:
        cc = r.get("coverage_checks") if isinstance(r.get("coverage_checks"), dict) else {}
        for key, value in cc.items():
            coverage_total[key] += 1
            if value is True:
                coverage_counts[key] += 1
    coverage_rows = [{
        "coverage_check": key,
        "records_passing": coverage_counts[key],
        "records_total": coverage_total[key],
        "pass_percent": pct(coverage_counts[key], coverage_total[key]),
        "status": "ok" if coverage_counts[key] == coverage_total[key] else "review",
    } for key in sorted(coverage_total)]

    # Label support.
    label_detail = []
    summary_counts = {
        "diagnosis": Counter(),
        "procedure": Counter(),
    }
    for r in records:
        note_id = r.get("note_id")
        diagnoses = r.get("diagnoses") if isinstance(r.get("diagnoses"), list) else []
        procs = r.get("procedures") if isinstance(r.get("procedures"), list) else []

        dx_checked = len([d for d in diagnoses if isinstance(d, dict)])
        dx_supported = sum(1 for d in diagnoses if isinstance(d, dict) and diagnosis_supported(r, d))
        summary_counts["diagnosis"]["checked"] += dx_checked
        summary_counts["diagnosis"]["direct_text"] += dx_supported
        summary_counts["diagnosis"]["unsupported"] += dx_checked - dx_supported
        if dx_checked:
            label_detail.append({
                "note_id": note_id, "label_type": "diagnosis", "labels_checked": dx_checked,
                "direct_text_supported": dx_supported, "contextual_text_supported": 0,
                "unsupported": dx_checked - dx_supported,
                "unsupported_labels": json.dumps([d.get("summary_text") for d in diagnoses if isinstance(d, dict) and not diagnosis_supported(r, d)], ensure_ascii=False),
            })

        proc_checked = 0
        proc_direct = 0
        proc_context = 0
        proc_unsupported = []
        for p in procs:
            if not isinstance(p, dict):
                continue
            proc_checked += 1
            mode, evidence = procedure_support(r, p)
            if mode == "direct_text":
                proc_direct += 1
            elif mode == "contextual_text":
                proc_context += 1
            else:
                proc_unsupported.append(evidence)
        summary_counts["procedure"]["checked"] += proc_checked
        summary_counts["procedure"]["direct_text"] += proc_direct
        summary_counts["procedure"]["contextual_text"] += proc_context
        summary_counts["procedure"]["unsupported"] += len(proc_unsupported)
        if proc_checked:
            label_detail.append({
                "note_id": note_id, "label_type": "procedure", "labels_checked": proc_checked,
                "direct_text_supported": proc_direct, "contextual_text_supported": proc_context,
                "unsupported": len(proc_unsupported),
                "unsupported_labels": json.dumps(proc_unsupported, ensure_ascii=False),
            })

    label_summary = []
    for label_type, counts in summary_counts.items():
        checked = counts["checked"]
        supported = counts["direct_text"] + counts["contextual_text"]
        label_summary.append({
            "quality_check": f"{label_type}_label_evidence_support",
            "labels_checked": checked,
            "direct_text_supported": counts["direct_text"],
            "contextual_text_supported": counts["contextual_text"],
            "labels_supported": supported,
            "unsupported_labels": counts["unsupported"],
            "support_rate": round(supported / checked, 4) if checked else 0.0,
            "status": "ok" if checked and supported / checked >= 0.95 else "review",
        })

    # Tooth-level summary.
    tooth_detail = []
    tooth_records = 0
    tooth_pass = 0
    for r in records:
        entries = r.get("tooth_level") if isinstance(r.get("tooth_level"), list) else []
        if entries:
            tooth_records += 1
        issues = []
        for entry in entries:
            if not isinstance(entry, dict):
                issues.append("tooth_entry_not_object")
                continue
            _, entry_issues = valid_tooth_entry(entry)
            issues.extend(entry_issues)
        ok = bool(entries) and not issues
        if ok:
            tooth_pass += 1
        tooth_detail.append({
            "note_id": r.get("note_id"),
            "tooth_entries": len(entries),
            "valid_tooth_structure": ok,
            "issues": ";".join(sorted(set(issues))),
        })
    tooth_summary = [{
        "quality_check": "tooth_level_structure",
        "records_with_tooth_annotations": tooth_records,
        "records_passing_structure_check": tooth_pass,
        "pass_rate": round(tooth_pass / tooth_records, 4) if tooth_records else 0.0,
        "status": "ok" if tooth_records and tooth_pass == tooth_records else "review",
    }]

    # PHI / privacy.
    phi_findings = []
    for r in records:
        for finding in scan_phi(r):
            phi_findings.append({"note_id": r.get("note_id"), **finding})
    phi_rows = [{
        "quality_check": "identifier_pattern_screen",
        "records_scanned": total,
        "finding_count": len(phi_findings),
        "records_with_findings": len(set(x["note_id"] for x in phi_findings)),
        "status": "ok" if not phi_findings else "review",
    }]

    # LLM summary validation if present.
    llm_rows = []
    if "llm_summary_validation_status" in available_fields:
        status_counts = Counter(str(r.get("llm_summary_validation_status", "missing")) for r in records)
        llm_rows = [
            {"validation_measure": "Total generated summaries", "count": total, "percent": 100.0},
            {"validation_measure": "Passed all validation checks", "count": status_counts.get("passed", 0), "percent": pct(status_counts.get("passed", 0), total)},
            {"validation_measure": "Failed one or more validation checks", "count": status_counts.get("failed", 0), "percent": pct(status_counts.get("failed", 0), total)},
        ]
        error_counts = Counter()
        for r in records:
            for e in r.get("llm_summary_validation_errors") or []:
                error_counts[str(e).split(":")[0]] += 1
        for key, count in error_counts.most_common():
            llm_rows.append({"validation_measure": key, "count": count, "percent": pct(count, total)})

    # One compact paper summary.
    required_ok = sum(1 for row in required_rows if row["status"] == "ok") == len(required_rows)
    coverage_ok = sum(row["records_passing"] for row in coverage_rows)
    coverage_den = sum(row["records_total"] for row in coverage_rows)
    dx = next((row for row in label_summary if row["quality_check"].startswith("diagnosis")), None)
    proc = next((row for row in label_summary if row["quality_check"].startswith("procedure")), None)
    quality_rows = [
        {"quality_check": "required_field_completeness", "result": "100.00%" if required_ok else "review", "interpretation": "Required release fields populated"},
        {"quality_check": "coverage_checks", "result": f"{pct(coverage_ok, coverage_den):.2f}%", "interpretation": "Chief complaint, findings, follow-up, diagnosis, plan, and tooth/site coverage"},
        {"quality_check": "diagnosis_label_evidence_support", "result": f"{dx['support_rate']*100:.2f}%" if dx else "n/a", "interpretation": "Diagnosis labels supported by note, finding, or summary text"},
        {"quality_check": "procedure_label_evidence_support", "result": f"{proc['support_rate']*100:.2f}%" if proc else "n/a", "interpretation": "Procedure labels supported by direct or contextual text evidence"},
        {"quality_check": "tooth_level_structure", "result": f"{tooth_summary[0]['pass_rate']*100:.2f}%", "interpretation": "Tooth/site annotations conform to accepted dental surface/site vocabulary"},
        {"quality_check": "identifier_pattern_screen", "result": str(len(phi_findings)), "interpretation": "Pattern-based screen for common direct identifiers"},
    ]
    if llm_rows:
        passed = next((r["count"] for r in llm_rows if r["validation_measure"] == "Passed all validation checks"), 0)
        quality_rows.append({
            "quality_check": "llm_expanded_summary_validation",
            "result": f"{pct(passed, total):.2f}%",
            "interpretation": "Automated fact-preservation and length checks for LLM handoff summaries",
        })

    outputs = {
        "table3_required_field_presence": required_rows,
        "table3_coverage_checks": coverage_rows,
        "table3_label_support_summary": label_summary,
        "table3_label_support_detail": label_detail,
        "table3_tooth_level_summary": tooth_summary,
        "table3_tooth_level_detail": tooth_detail,
        "table3_privacy_memorization_summary": phi_rows,
        "table3_identifier_pattern_detail": phi_findings,
        "table3_validation_quality_controls": quality_rows,
    }
    if llm_rows:
        outputs["table3_llm_summary_validation"] = llm_rows

    for name, rows in outputs.items():
        write_csv(out_dir / f"{name}.csv", rows)
        write_md(out_dir / f"{name}.md", rows)

    report = {
        "input_record_count": total,
        "outputs": {name: str(out_dir / f"{name}.csv") for name in outputs},
        "procedure_support_policy": "direct text evidence plus contextual dental evidence cues are counted separately and transparently",
        "accepted_surface_vocabulary": sorted(ALLOWED_SURFACES),
    }
    (out_dir / "tables3_validation_report_v21.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export v2.1 validation and quality tables for paper/release.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="validation_outputs/tables3_v21")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    records = []
    for _, record in read_jsonl(args.input):
        records.append(record)
        if args.sample_limit and len(records) >= args.sample_limit:
            break
    out_dir = ensure_dir(args.out_dir)
    summarize_validation(records, out_dir)
    print(f"Exported v2.1 validation tables for {len(records):,} records to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
