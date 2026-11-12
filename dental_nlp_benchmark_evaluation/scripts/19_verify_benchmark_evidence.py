import csv
import json
from pathlib import Path

ROOT = Path.home() / "ToUpload/dental_nlp_benchmark_evaluation"
OUT = ROOT / "final_evidence/paper_outputs"
OUT.mkdir(parents=True, exist_ok=True)

REPORTED = OUT / "benchmark_scores_REPORTED_UNVERIFIED.csv"
BASELINES = ROOT / "outputs/benchmarks/baseline_results.csv"

RUNS = {
    ("Diagnosis", "Published"): "full_published_diagnosis",
    ("Procedure", "Published"): "full_published_procedure",
    ("Diagnosis", "Prompt-ID-disjoint"): "full_prompt_disjoint_diagnosis",
    ("Procedure", "Prompt-ID-disjoint"): "full_prompt_disjoint_procedure",
}

MANIFESTS = {
    "published_70_15_15.csv": "Published",
    "prompt_disjoint.csv": "Prompt-ID-disjoint",
}

COUNTS = {
    "Published": (14000, 3000, 3000),
    "Prompt-ID-disjoint": (13971, 3012, 3017),
}

def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def normalize_task(s):
    s = s.strip().lower()
    if s not in ("diagnosis", "procedure"):
        raise ValueError(f"Unexpected task: {s}")
    return s.capitalize()

def normalize_model(s):
    s = s.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if s in ("rule", "rulebased"):
        return "Rule-based"
    if s in ("tfidf", "tfidflogistic", "tfidfovr"):
        return "TF-IDF"
    if s == "clinicalbert":
        return "ClinicalBERT"
    raise ValueError(f"Unexpected model: {s}")

def validate(row, task, split, full):
    counts = tuple(int(row[k]) for k in ("n_train", "n_validation", "n_test"))
    if counts != COUNTS[split]:
        raise ValueError(f"{task}/{split}: unexpected counts {counts}")

    labels = 49 if task == "Diagnosis" else 31
    if int(row["labels_train"]) != labels:
        raise ValueError(f"{task}/{split}: unexpected label coverage")

    if int(row["unseen_test_labels"]) != 0:
        raise ValueError(f"{task}/{split}: unseen labels detected")

    if full and row.get("smoke_test", "").lower() != "false":
        raise ValueError(f"{task}/{split}: not a full ClinicalBERT run")

actual = {}

for row in load(BASELINES):
    task = normalize_task(row["task"])
    split = MANIFESTS[row["split_manifest"]]
    model = normalize_model(row["model"])
    key = (task, split, model)

    if key in actual:
        raise ValueError(f"Duplicate baseline result: {key}")

    validate(row, task, split, full=False)
    actual[key] = float(row["macro_f1"])

for (task, split), folder in RUNS.items():
    path = ROOT / "outputs/clinicalbert" / folder / "clinicalbert_results.csv"
    rows = load(path)

    if len(rows) != 1:
        raise ValueError(f"Expected one result row in {path}")

    row = rows[0]
    if normalize_task(row["task"]) != task:
        raise ValueError(f"Task mismatch: {path}")

    if MANIFESTS[row["split_manifest"]] != split:
        raise ValueError(f"Split mismatch: {path}")

    validate(row, task, split, full=True)
    actual[(task, split, "ClinicalBERT")] = float(row["macro_f1"])

reported = {}
for row in load(REPORTED):
    key = (
        normalize_task(row["task"]),
        row["split"].strip(),
        normalize_model(row["model"]),
    )
    if key in reported:
        raise ValueError(f"Duplicate reported score: {key}")
    reported[key] = float(row["test_macro_f1"])

expected = {
    (task, split, model)
    for task in ("Diagnosis", "Procedure")
    for split in ("Published", "Prompt-ID-disjoint")
    for model in ("Rule-based", "TF-IDF", "ClinicalBERT")
}

if set(actual) != expected or set(reported) != expected:
    raise ValueError(
        f"Actual missing: {expected - set(actual)}; "
        f"Reported missing: {expected - set(reported)}; "
        f"Unexpected actual: {set(actual) - expected}; "
        f"Unexpected reported: {set(reported) - expected}"
    )

audit = []
verified = []

for task, split, model in sorted(expected):
    key = (task, split, model)
    difference = abs(actual[key] - reported[key])
    match = difference <= 0.0000051

    audit.append({
        "task": task,
        "split": split,
        "model": model,
        "reported_macro_f1": reported[key],
        "source_macro_f1": actual[key],
        "absolute_difference": difference,
        "match_5_decimals": match,
    })

    verified.append({
        "task": task,
        "split": split,
        "model": model,
        "test_macro_f1": f"{actual[key]:.8f}",
        "evidence_status": "SOURCE_CSV_VERIFIED",
    })

audit_path = OUT / "benchmark_evidence_reconciliation.json"
audit_path.write_text(json.dumps(audit, indent=2))

print("\nBENCHMARK RECONCILIATION")
for row in audit:
    print(row)

if not all(row["match_5_decimals"] for row in audit):
    raise SystemExit(
        "DISCREPANCIES FOUND. Review audit JSON; no verified CSV produced."
    )

verified_path = OUT / "benchmark_scores_SOURCE_CSV_VERIFIED.csv"
with verified_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "task", "split", "model", "test_macro_f1", "evidence_status"
    ])
    writer.writeheader()
    writer.writerows(verified)

print("\nPASS: All 12 benchmark scores match their source CSVs.")
print("Saved:", audit_path)
print("Saved:", verified_path)
