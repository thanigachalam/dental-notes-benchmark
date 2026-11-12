import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("dental_nlp_benchmark_evaluation")
DATA = Path("data/dentalnote_bench_20000_icd10_internalproc_v21.jsonl.zip")
OUT = ROOT / "final_evidence/template_audit"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "published": ROOT / "outputs/splits/published_70_15_15.csv",
    "prompt_disjoint": ROOT / "outputs/splits/prompt_disjoint.csv",
}

ID_FIELDS = ("record_id", "note_id", "sample_id", "id")
SPLIT_FIELDS = ("split", "partition", "subset", "set")

def field_value(obj, key):
    if key in obj:
        return obj[key]
    for parent in ("metadata", "provenance", "generation_metadata"):
        nested = obj.get(parent)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None

def choose_field(fields, candidates, purpose):
    for name in candidates:
        if name in fields:
            return name
    raise ValueError(
        f"Could not identify {purpose}. Available columns: {fields}"
    )

def normalized_split(value):
    name = str(value).strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
        "test": "test",
        "testing": "test",
    }
    if name not in mapping:
        raise ValueError(f"Unknown split value: {value!r}")
    return mapping[name]

def family_from_prompt(prompt):
    return re.sub(r"_\d{2}$", "", str(prompt))

# Read the unchanged ZIP without extracting or rewriting it.
records = {}

with zipfile.ZipFile(DATA) as z:
    members = [
        n for n in z.namelist()
        if n.endswith(".jsonl")
        and not n.startswith("__MACOSX/")
        and not Path(n).name.startswith("._")
    ]

    if len(members) != 1:
        raise ValueError(f"Expected one data JSONL: {members}")

    with z.open(members[0]) as f:
        for raw in f:
            if not raw.strip():
                continue

            record = json.loads(raw)

            identifier = next(
                (
                    field_value(record, k)
                    for k in ID_FIELDS
                    if field_value(record, k) is not None
                ),
                None,
            )

            if identifier is None:
                raise ValueError(
                    "No usable record ID. Dataset keys: "
                    + str(list(record))
                )

            identifier = str(identifier)
            prompt = field_value(record, "source_prompt_id")

            if prompt is None:
                raise ValueError(
                    f"Missing source_prompt_id: {identifier}"
                )

            if identifier in records:
                raise ValueError(f"Duplicate dataset ID: {identifier}")

            records[identifier] = {
                "prompt": str(prompt),
                "family": family_from_prompt(prompt),
            }

print("Dataset records:", len(records))

report = {"dataset_records": len(records), "splits": {}}
matrix = []

for split_name, path in SPLITS.items():
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        id_field = choose_field(
            reader.fieldnames, ID_FIELDS, "manifest record ID"
        )
        split_field = choose_field(
            reader.fieldnames, SPLIT_FIELDS, "split name"
        )

        assignments = {}

        for row in reader:
            identifier = str(row[id_field]).strip()
            partition = normalized_split(row[split_field])

            if identifier in assignments:
                raise ValueError(
                    f"Duplicate manifest ID: {identifier}"
                )

            assignments[identifier] = partition

    missing = set(records) - set(assignments)
    extra = set(assignments) - set(records)

    counts = Counter()
    family_sets = defaultdict(set)
    prompt_sets = defaultdict(set)
    family_counts = defaultdict(Counter)

    for identifier in set(records) & set(assignments):
        partition = assignments[identifier]
        info = records[identifier]

        counts[partition] += 1
        family_sets[partition].add(info["family"])
        prompt_sets[partition].add(info["prompt"])
        family_counts[info["family"]][partition] += 1

    partitions = ("train", "validation", "test")
    comparisons = {}

    for a, b in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        comparisons[f"{a}_vs_{b}"] = {
            "shared_source_prompt_ids": len(
                prompt_sets[a] & prompt_sets[b]
            ),
            "shared_scenario_families": len(
                family_sets[a] & family_sets[b]
            ),
            "shared_family_names": sorted(
                family_sets[a] & family_sets[b]
            ),
        }

    for family, values in sorted(family_counts.items()):
        matrix.append({
            "split_strategy": split_name,
            "scenario_family": family,
            **{p: values[p] for p in partitions},
        })

    report["splits"][split_name] = {
        "manifest": str(path),
        "id_field": id_field,
        "split_field": split_field,
        "counts": dict(counts),
        "missing_dataset_ids": len(missing),
        "unknown_manifest_ids": len(extra),
        "unique_prompt_ids": {
            p: len(prompt_sets[p]) for p in partitions
        },
        "unique_families": {
            p: len(family_sets[p]) for p in partitions
        },
        "overlap": comparisons,
    }

    print(f"\n===== {split_name.upper()} =====")
    print("Partition counts:", dict(counts))
    print("Missing dataset IDs:", len(missing))
    print("Unknown manifest IDs:", len(extra))

    for name, result in comparisons.items():
        print(
            name,
            "shared prompt IDs:",
            result["shared_source_prompt_ids"],
            "shared families:",
            result["shared_scenario_families"],
        )

    if missing or extra:
        print("WARNING: Manifest ID coverage is incomplete.")

with (OUT / "cross_split_family_matrix.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split_strategy",
            "scenario_family",
            "train",
            "validation",
            "test",
        ],
    )
    writer.writeheader()
    writer.writerows(matrix)

(OUT / "cross_split_family_audit.json").write_text(
    json.dumps(report, indent=2)
)

print("\nSaved:")
print(OUT / "cross_split_family_matrix.csv")
print(OUT / "cross_split_family_audit.json")
