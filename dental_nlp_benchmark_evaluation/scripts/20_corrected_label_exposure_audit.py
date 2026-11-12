import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "ToUpload/dental_nlp_benchmark_evaluation"
DATA = Path.home() / "ToUpload/data/dentalnote_bench_20000_icd10_internalproc_v21.jsonl.zip"
OUT = ROOT / "final_evidence/label_template_audit"
OUT.mkdir(parents=True, exist_ok=True)

stats = {
    "records": 0,
    "diagnosis": Counter(),
    "procedure": Counter(),
}
label_counts = {
    "diagnosis": Counter(),
    "procedure": Counter(),
}

with zipfile.ZipFile(DATA) as archive:
    members = [
        name for name in archive.namelist()
        if name.endswith(".jsonl")
        and not name.startswith("__MACOSX/")
        and not Path(name).name.startswith("._")
    ]

    if len(members) != 1:
        raise ValueError(f"Expected one data JSONL; found {members}")

    with archive.open(members[0]) as stream:
        for raw in stream:
            if not raw.strip():
                continue

            record = json.loads(raw)
            stats["records"] += 1
            text = record.get("input_note", "")

            if not isinstance(text, str):
                raise ValueError("Unexpected input_note type")

            for task, field, code_field in [
                ("diagnosis", "diagnoses", "diagnosis_code"),
                ("procedure", "procedures", "procedure_code_base"),
            ]:
                items = record.get(field, [])

                if not isinstance(items, list):
                    raise ValueError(f"Unexpected {field} type")

                codes = set()
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError(f"Unexpected item in {field}")
                    code = item.get(code_field)
                    if code:
                        codes.add(str(code))

                stats[task]["records_with_labels"] += bool(codes)
                stats[task]["label_occurrences"] += len(codes)

                any_match = False

                for code in codes:
                    label_counts[task][code] += 1

                    pattern = (
                        r"(?<![A-Za-z0-9_])"
                        + re.escape(code)
                        + r"(?![A-Za-z0-9_])"
                    )

                    if re.search(pattern, text, re.IGNORECASE):
                        stats[task]["literal_code_matches"] += 1
                        any_match = True

                stats[task]["records_with_code_match"] += any_match

report = {
    "total_records": stats["records"],
    "diagnosis": dict(stats["diagnosis"]),
    "procedure": dict(stats["procedure"]),
    "unique_diagnosis_codes": len(label_counts["diagnosis"]),
    "unique_procedure_base_codes": len(label_counts["procedure"]),
    "limitations": [
        "Only literal target-code occurrences are measured.",
        "Clinical terminology, synonyms, and scenario-specific wording may reveal labels without literal codes.",
        "This check does not establish absence of target leakage.",
        "Counts are based on unique target codes per record."
    ],
}

path = OUT / "corrected_literal_code_exposure.json"
path.write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
print("\nSaved:", path)
