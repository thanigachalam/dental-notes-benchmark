import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path("dental_nlp_benchmark_evaluation")
DATA = Path("data/dentalnote_bench_20000_icd10_internalproc_v21.jsonl.zip")
OUT = ROOT / "final_evidence/template_audit"
OUT.mkdir(parents=True, exist_ok=True)

def read_records(path):
    with zipfile.ZipFile(path) as archive:
        members = [
            n for n in archive.namelist()
            if n.endswith(".jsonl")
            and not n.startswith("__MACOSX/")
            and not Path(n).name.startswith("._")
        ]
        if len(members) != 1:
            raise ValueError(f"Expected one JSONL file; found: {members}")
        with archive.open(members[0]) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

def get_value(record, field):
    if field in record:
        return record[field]
    for key in ("metadata", "provenance", "generation_metadata"):
        nested = record.get(key)
        if isinstance(nested, dict) and field in nested:
            return nested[field]
    return None

def family_from_id(prompt_id):
    return re.sub(r"_\d{2}$", "", str(prompt_id))

records = list(read_records(DATA))
families = Counter()
prompts = Counter()
methods = Counter()
missing_prompt = 0

for record in records:
    prompt = get_value(record, "source_prompt_id")
    method = get_value(record, "generation_method")

    if prompt is None or str(prompt).strip() == "":
        missing_prompt += 1
        continue

    prompt = str(prompt)
    prompts[prompt] += 1
    families[family_from_id(prompt)] += 1
    methods[str(method)] += 1

with (OUT / "scenario_family_counts.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario_family", "record_count"])
    writer.writerows(sorted(families.items()))

with (OUT / "source_prompt_counts.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_prompt_id", "scenario_family", "record_count"])
    for prompt, count in sorted(prompts.items()):
        writer.writerow([prompt, family_from_id(prompt), count])

split_files = sorted((ROOT / "outputs/splits").glob("*.csv"))
split_schemas = {}

for path in split_files:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        split_schemas[path.name] = {
            "columns": reader.fieldnames,
            "first_two_rows": [
                row for _, row in zip(range(2), reader)
            ],
        }

summary = {
    "total_records": len(records),
    "unique_source_prompt_ids": len(prompts),
    "unique_scenario_families": len(families),
    "missing_source_prompt_id": missing_prompt,
    "generation_method_metadata_counts": dict(methods),
    "scenario_family_counts": dict(sorted(families.items())),
    "split_manifest_schemas": split_schemas,
    "methodological_note": (
        "Scenario family is inferred by removing a trailing "
        "two-digit source-prompt suffix. This does not verify "
        "independently authored prompts or text templates. "
        "Generation-method metadata is descriptive and has "
        "not been verified against execution-level provenance."
    ),
}

(OUT / "generation_family_audit.json").write_text(
    json.dumps(summary, indent=2)
)

print("\n===== GENERATION FAMILY AUDIT =====")
print("Total records:", len(records))
print("Unique source-prompt IDs:", len(prompts))
print("Unique scenario families:", len(families))
print("Missing source-prompt IDs:", missing_prompt)

print("\n===== SCENARIO FAMILY COUNTS =====")
for name, count in sorted(families.items()):
    print(f"{name}: {count}")

print("\n===== SPLIT MANIFEST SCHEMAS =====")
for name, info in split_schemas.items():
    print(name, "=>", info["columns"])
    print("Sample:", info["first_two_rows"][:1])

print("\nSaved audit files:", OUT)
