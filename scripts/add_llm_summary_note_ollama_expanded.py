#!/usr/bin/env python3
"""Add expanded Ollama-generated llm_summary_note to dentalnote_bench JSONL.

This script is intentionally Ollama-only and expanded-summary-only to avoid
confusion with earlier OpenAI/local-RAG scripts.

Example:
  export OLLAMA_BASE_URL=http://127.0.0.1:11434
  export OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M

  python scripts/add_llm_summary_note_ollama_expanded.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --output data/dentalnote_bench_20000_icd10_internalproc_v21_llm_summary_expanded_test.jsonl \
    --limit 10 \
    --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

PROMPT_VERSION = "llm_summary_note_v2.0_ollama_expanded_6_8_sentence_handoff"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b-instruct-q4_K_M"
TOOTH_RE = re.compile(r"#([A-T]|[1-9]|[12][0-9]|3[0-2])\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e


def write_jsonl_record(f, record: dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_token_text(value: str) -> str:
    return re.sub(r"[^a-z0-9#]+", " ", value.lower()).strip()


def sentence_count(text: str) -> int:
    return len(SENTENCE_RE.findall(text.strip()))


def extract_facts(record: dict[str, Any]) -> dict[str, Any]:
    input_note = record.get("input_note") or ""
    tooth_level = record.get("tooth_level") or []
    diagnoses = record.get("diagnoses") or []
    procedures = record.get("procedures") or []

    teeth = set(m.group(1).upper() for m in TOOTH_RE.finditer(input_note))
    surfaces = set()
    for t in tooth_level:
        if not isinstance(t, dict):
            continue
        n = clean_text(t.get("tooth_number"))
        if n:
            teeth.add(n.upper())
        for s in t.get("surfaces") or []:
            if s:
                surfaces.add(str(s))

    diagnosis_texts = []
    for d in diagnoses:
        if isinstance(d, dict):
            diagnosis_texts.append(clean_text(d.get("diagnosis_text") or d.get("summary_text")))
    diagnosis_texts = [x for x in diagnosis_texts if x]

    procedure_texts = []
    for p in procedures:
        if isinstance(p, dict):
            procedure_texts.append(clean_text(p.get("procedure_text")))
    procedure_texts = [x for x in procedure_texts if x]

    return {
        "note_id": record.get("note_id"),
        "visit_type": record.get("visit_type"),
        "specialty": record.get("specialty"),
        "patient_age_group": record.get("patient_age_group"),
        "patient_sex": record.get("patient_sex"),
        "chief_complaint": record.get("chief_complaint"),
        "tooth_numbers": sorted(teeth, key=lambda x: (len(x), x)),
        "surfaces": sorted(surfaces),
        "diagnoses": diagnosis_texts[:6],
        "procedures": procedure_texts[:8],
        "medications": record.get("medications") or [],
        "follow_up_recommendation": record.get("follow_up_recommendation"),
        "findings_text": record.get("findings_text"),
        "summary_SOAP": record.get("summary_SOAP"),
    }


def build_expanded_prompt(record: dict[str, Any], attempt_feedback: str = "") -> str:
    facts = extract_facts(record)

    prompt = f"""
You are writing an expanded natural clinical handoff note for a synthetic dental NLP benchmark.

TASK:
Write ONE paragraph only. Write 6 to 8 complete clinical sentences. Target 110 to 190 words.
Do NOT write bullets. Do NOT write SOAP labels. Do NOT return JSON. Do NOT stop after 1 or 2 sentences.

MUST INCLUDE WHEN PRESENT IN THE SOURCE:
1. Patient age/sex and visit reason.
2. Tooth number/site.
3. Key objective findings.
4. Imaging, radiograph, percussion, or exam tests if mentioned.
5. Assessment/diagnosis.
6. Treatment/procedures completed or staged.
7. Medications and local anesthesia if mentioned.
8. Follow-up recommendation.

SAFETY/CONSISTENCY RULES:
- Use only the source facts below.
- Do not invent new teeth, diagnoses, procedures, medications, or follow-up.
- Do not mention allergies unless they are explicitly stated in the input_note.
- Keep the language natural and human-like, like a dentist handoff note.
- This should be more detailed than summary_SOAP, not shorter.

{attempt_feedback}

SOURCE INPUT NOTE:
{record.get('input_note', '')}

SOURCE FINDINGS_TEXT:
{record.get('findings_text', '')}

SOURCE SUMMARY_SOAP FOR COVERAGE ONLY:
{record.get('summary_SOAP', '')}

STRUCTURED FACTS FOR FACT PRESERVATION:
{json.dumps(facts, ensure_ascii=False)}

FINAL EXPANDED CLINICAL HANDOFF PARAGRAPH:
""".strip()
    return prompt


def ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
    num_predict: int,
    num_ctx: int,
) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: pip install requests")

    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "repeat_penalty": 1.08,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = clean_model_output(data.get("response", ""))
    return text


def clean_model_output(text: str) -> str:
    text = text.strip()
    # Remove common wrappers if the local model ignores instructions.
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(FINAL EXPANDED CLINICAL HANDOFF PARAGRAPH:|Summary:|Expanded summary:)\s*", "", text, flags=re.IGNORECASE).strip()
    # If model returned JSON anyway, extract likely field.
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ["llm_summary_note", "summary", "expanded_summary", "text"]:
                if obj.get(key):
                    return clean_text(obj[key])
        except Exception:
            pass
    return clean_text(text)


def warmup_ollama(model: str, base_url: str, timeout: int) -> None:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: pip install requests")
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": "Reply with: ready",
        "stream": False,
        "options": {"num_predict": 8, "num_ctx": 512, "temperature": 0.0},
    }
    requests.post(url, json=payload, timeout=timeout).raise_for_status()


def validate_llm_summary(record: dict[str, Any], summary: str, min_sentences: int) -> list[str]:
    errors: list[str] = []
    text = clean_text(summary)
    lower = text.lower()
    input_note = record.get("input_note", "") or ""
    input_lower = input_note.lower()
    facts = extract_facts(record)

    if not text:
        errors.append("summary_blank")
        return errors

    if len(text) < 400:
        errors.append("summary_too_short_for_expanded_handoff")
    if len(text) > 1800:
        errors.append("summary_too_long")

    sc = sentence_count(text)
    if sc < min_sentences:
        errors.append(f"sentence_count_below_target:{sc}<" + str(min_sentences))

    # Tooth consistency: no new teeth and retain source teeth when 1-3 sites are involved.
    allowed_teeth = set(t.upper() for t in facts["tooth_numbers"])
    mentioned_teeth = set(m.group(1).upper() for m in TOOTH_RE.finditer(text))
    new_teeth = mentioned_teeth - allowed_teeth
    if new_teeth:
        errors.append("new_tooth_number_mentioned:" + ",".join(sorted(new_teeth)))
    if 0 < len(allowed_teeth) <= 3:
        missing = allowed_teeth - mentioned_teeth
        if missing:
            errors.append("expected_tooth_not_mentioned:" + ",".join(sorted(missing)))

    # Coverage checks for common clinical facts.
    if "radiograph" in input_lower or "imaging" in input_lower or "xray" in input_lower or "x-ray" in input_lower:
        if not any(w in lower for w in ["radiograph", "imaging", "x-ray", "xray"]):
            errors.append("imaging_or_radiograph_not_mentioned")

    if "percussion" in input_lower:
        if "percussion" not in lower:
            errors.append("percussion_not_mentioned")

    if "fu:" in input_lower or "follow-up" in input_lower or facts.get("follow_up_recommendation"):
        if not any(w in lower for w in ["follow", "recheck", "recall", "monitor", "workup", "prn"]):
            errors.append("follow_up_not_mentioned")

    meds = [clean_text(m) for m in (record.get("medications") or []) if clean_text(m)]
    # Only require meds if the input note explicitly has medication cue.
    if any(cue in input_lower for cue in ["meds:", "rx/meds", "medications reviewed", "rx:"]):
        med_hit = False
        for med in meds:
            med_l = med.lower()
            first = med_l.split()[0].replace(";", "")
            if first and first in lower:
                med_hit = True
                break
            if "local anesthetic" in med_l and ("local anesthetic" in lower or "anesthesia" in lower or "anesthetic" in lower):
                med_hit = True
                break
        if meds and not med_hit:
            errors.append("medication_or_anesthesia_not_mentioned")

    # Privacy / real patient language guard.
    if "real patient" in lower or "patient name" in lower:
        errors.append("privacy_language_flag")

    return errors


def fallback_summary(record: dict[str, Any]) -> str:
    # Rich fallback from existing source fields. This is deterministic and marked as fallback.
    pieces = []
    cc = clean_text(record.get("chief_complaint"))
    facts = extract_facts(record)
    sex = clean_text(record.get("patient_sex"))
    age_group = clean_text(record.get("patient_age_group"))
    visit = clean_text(record.get("visit_type"))
    teeth = ", ".join(f"#{t}" for t in facts["tooth_numbers"])
    if cc:
        pieces.append(f"The patient presented for {visit or 'dental'} care with {cc.lower()}")
    if age_group or sex:
        pieces.append(f"The encounter involved a {age_group} {sex}".strip())
    if teeth:
        pieces.append(f"The documented tooth/site involvement included {teeth}")
    if record.get("findings_text"):
        pieces.append(clean_text(record.get("findings_text")))
    if record.get("summary_SOAP"):
        pieces.append(clean_text(record.get("summary_SOAP")))
    if record.get("medications"):
        pieces.append("Medications/anesthesia documented: " + "; ".join(map(str, record.get("medications") or [])))
    if record.get("follow_up_recommendation"):
        pieces.append("Follow-up recommendation: " + str(record.get("follow_up_recommendation")).replace("_", " "))
    text = ". ".join(p.rstrip(".") for p in pieces if p) + "."
    return clean_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Add expanded Ollama llm_summary_note fields to dental dataset JSONL.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    ap.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--num-predict", type=int, default=320)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-sentences", type=int, default=5, help="Validation minimum. Prompt still asks for 6-8 sentences.")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--fallback-on-error", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict-exit-on-fail", action="store_true", help="Return non-zero if any row fails validation.")
    args = ap.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print(f"Output exists: {args.output}. Use --overwrite to replace.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.warmup:
        print(f"Warming up Ollama model {args.model}...", file=sys.stderr)
        warmup_ollama(args.model, args.ollama_base_url, args.timeout)

    generated_at = datetime.now(timezone.utc).isoformat()
    processed = 0
    failed = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for line_no, record in read_jsonl(args.input):
            if args.limit and processed >= args.limit:
                break

            provider_status = "ollama_generated"
            risk_flags: list[str] = []
            validation_errors: list[str] = []
            summary = ""

            try:
                prompt = build_expanded_prompt(record)
                summary = ollama_generate(
                    prompt=prompt,
                    model=args.model,
                    base_url=args.ollama_base_url,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                )
                validation_errors = validate_llm_summary(record, summary, args.min_sentences)

                # One corrective retry if the model was too brief.
                if validation_errors:
                    feedback = (
                        "IMPORTANT RETRY INSTRUCTION: The previous answer failed validation. "
                        "Write a fuller 6 to 8 sentence clinical handoff. "
                        "You MUST explicitly include every tooth number from the source note, the follow-up recommendation, "
                        "and any radiograph, imaging, percussion, medication, or anesthesia details when present. "
                        "Do not omit these required facts. Do not invent new facts."
                    )
                    prompt = build_expanded_prompt(record, attempt_feedback=feedback)
                    summary_retry = ollama_generate(
                        prompt=prompt,
                        model=args.model,
                        base_url=args.ollama_base_url,
                        timeout=args.timeout,
                        temperature=args.temperature,
                        num_predict=args.num_predict,
                        num_ctx=args.num_ctx,
                    )
                    retry_errors = validate_llm_summary(record, summary_retry, args.min_sentences)
                    # Keep retry if it improves validation count or length.
                    if len(retry_errors) <= len(validation_errors) or len(summary_retry) > len(summary):
                        summary = summary_retry
                        validation_errors = retry_errors
                        risk_flags.append("used_corrective_retry")

            except Exception as e:
                validation_errors = [f"line_{line_no}: {type(e).__name__}: {e}"]
                provider_status = "ollama_generation_failed"
                if args.fallback_on_error:
                    summary = fallback_summary(record)
                    risk_flags.append("fallback_from_existing_summary_due_to_ollama_error")
                    validation_errors = validate_llm_summary(record, summary, args.min_sentences)
                    provider_status = "fallback_existing_summary"
                else:
                    summary = ""

            SOFT_VALIDATION_WARNINGS = {
                "percussion_not_mentioned",
                "imaging_or_radiograph_not_mentioned",
                "medication_or_anesthesia_not_mentioned",
            }

            soft_warnings = []
            hard_errors = []

            for err in validation_errors:
                err_key = err.split(":")[0]
                if err_key in SOFT_VALIDATION_WARNINGS:
                    soft_warnings.append(err)
                else:
                    hard_errors.append(err)

            if soft_warnings:
                risk_flags.extend(["soft_validation_warning:" + e for e in soft_warnings])

            validation_errors = hard_errors

            if validation_errors:
                failed += 1

            record["llm_summary_note"] = summary
            record["llm_summary_source"] = provider_status
            record["llm_summary_model"] = args.model
            record["llm_summary_provider"] = "ollama"
            record["llm_summary_prompt_version"] = PROMPT_VERSION
            record["llm_summary_style"] = "expanded_clinical_handoff"
            record["llm_summary_generated_at"] = generated_at
            record["llm_summary_length"] = len(summary)
            record["llm_summary_sentence_count"] = sentence_count(summary)
            record["llm_summary_risk_flags"] = risk_flags
            record["llm_summary_validation_status"] = "passed" if not validation_errors else "failed"
            record["llm_summary_validation_errors"] = validation_errors

            write_jsonl_record(out, record)
            processed += 1

            if processed % 25 == 0:
                print(f"processed={processed:,} failed={failed:,}", file=sys.stderr)

    result = {
        "input": args.input,
        "output": args.output,
        "processed": processed,
        "failed": failed,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
    }
    print(json.dumps(result, indent=2))
    if args.strict_exit_on_fail and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
