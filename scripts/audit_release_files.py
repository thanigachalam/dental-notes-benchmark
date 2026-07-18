#!/usr/bin/env python3
"""Create a release manifest with file sizes, SHA-256 hashes, and row counts.

Use this before GitHub/Zenodo/IEEE DataPort release to document exactly which
files were included in the dataset package.

Example:
  python scripts/audit_release_files.py \
    --repo-root . \
    --out-dir release_audit
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _dataset_utils import count_jsonl_records, ensure_dir, sha256_file, write_json

DEFAULT_INCLUDE = [
    "README.md",
    "dataset_card.md",
    "CITATION.cff",
    "LICENSE",
    "CHANGELOG.md",
    "RELEASE_NOTES_v2.0.md",
    "data/*.jsonl",
    "data/*.json",
    "codebooks/*.csv",
    "docs/*.md",
    "examples/*.jsonl",
    "generators/*.py",
    "scripts/*.py",
]


def count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return None


def collect_files(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                files.add(path)
    return sorted(files)


def row_count_for(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        try:
            return count_jsonl_records(path)
        except Exception:
            return None
    if suffix == ".csv":
        return count_csv_rows(path)
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return len(payload)
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return len(payload["records"])
        except Exception:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit release files and produce checksum manifest.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="release_audit")
    parser.add_argument("--include", nargs="*", default=None, help="Glob patterns relative to repo root")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = ensure_dir(args.out_dir)
    patterns = args.include or DEFAULT_INCLUDE
    files = collect_files(root, patterns)

    rows: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "size_bytes": int(stat.st_size),
            "sha256": sha256_file(path),
            "row_count_if_table": row_count_for(path),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })

    csv_path = out_dir / "release_file_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "sha256", "row_count_if_table", "modified_utc"])
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "repo_root": str(root),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "patterns": patterns,
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "files": rows,
    }
    write_json(manifest, out_dir / "release_file_manifest.json")
    print(f"Audited {len(rows):,} files. Manifest: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
