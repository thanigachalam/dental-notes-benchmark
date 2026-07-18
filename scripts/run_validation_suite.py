#!/usr/bin/env python3
"""Run the additional validation/support scripts as one suite.

This orchestrator does not replace your main schema validator. It runs the
additional IEEE-support checks included in this script package.

Example:
  python scripts/run_validation_suite.py \
    --input data/dentalnote_bench_20000_icd10_internalproc_v20.jsonl \
    --repo-root . \
    --out-dir validation_outputs/additional_suite
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _dataset_utils import ensure_dir, write_json


def run_step(name: str, cmd: list[str]) -> dict[str, Any]:
    print("\n===", name, "===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return {
        "name": name,
        "command": cmd,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run additional dental dataset validation suite.")
    parser.add_argument("--input", required=True, help="Dataset path: JSONL, NDJSON, JSON, or CSV")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="validation_outputs/additional_suite")
    parser.add_argument("--diagnosis-codebook", default="codebooks/icd10_cm_diagnosis_mappings_v20.csv")
    parser.add_argument("--procedure-codebook", default="codebooks/internal_procedure_code_definitions_v20.csv")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--allow-phi-findings", action="store_true", help="Do not fail suite if PHI-pattern screen finds matches")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    scripts_dir = Path(__file__).resolve().parent
    py = sys.executable
    sample_args = ["--sample-limit", str(args.sample_limit)] if args.sample_limit else []

    steps = [
        (
            "record_consistency",
            [py, str(scripts_dir / "validate_record_consistency.py"), "--input", args.input, "--out-dir", str(out_dir / "record_consistency"), *sample_args],
        ),
        (
            "nested_annotations",
            [py, str(scripts_dir / "verify_nested_annotations.py"), "--input", args.input, "--out-dir", str(out_dir / "nested_annotations"), *sample_args],
        ),
        (
            "codebook_coverage",
            [
                py,
                str(scripts_dir / "check_codebook_coverage.py"),
                "--input",
                args.input,
                "--diagnosis-codebook",
                args.diagnosis_codebook,
                "--procedure-codebook",
                args.procedure_codebook,
                "--out-dir",
                str(out_dir / "codebook_coverage"),
                *sample_args,
            ],
        ),
        (
            "phi_risk",
            [py, str(scripts_dir / "check_synthetic_phi_risk.py"), "--input", args.input, "--out-dir", str(out_dir / "phi_risk"), *sample_args],
        ),
        (
            "paper_tables",
            [py, str(scripts_dir / "export_ieee_descriptor_tables.py"), "--input", args.input, "--out-dir", str(out_dir / "paper_tables"), *sample_args],
        ),
        (
            "benchmark_inventory",
            [py, str(scripts_dir / "build_benchmark_task_inventory.py"), "--input", args.input, "--out-dir", str(out_dir / "benchmark_inventory"), *sample_args],
        ),
        (
            "release_audit",
            [py, str(scripts_dir / "audit_release_files.py"), "--repo-root", args.repo_root, "--out-dir", str(out_dir / "release_audit")],
        ),
    ]

    results = []
    for name, cmd in steps:
        result = run_step(name, cmd)
        if name == "phi_risk" and args.allow_phi_findings and result["returncode"] != 0:
            result["returncode_original"] = result["returncode"]
            result["returncode"] = 0
            result["note"] = "PHI findings allowed by --allow-phi-findings"
        results.append(result)

    failed = [r for r in results if r["returncode"] != 0]
    suite_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "out_dir": str(out_dir),
        "step_count": len(results),
        "failed_step_count": len(failed),
        "failed_steps": [r["name"] for r in failed],
        "steps": results,
    }
    write_json(suite_report, out_dir / "additional_validation_suite_report.json")
    print("\nSuite complete.")
    print("Failed steps:", ", ".join(suite_report["failed_steps"]) if failed else "none")
    print("Report:", out_dir / "additional_validation_suite_report.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
