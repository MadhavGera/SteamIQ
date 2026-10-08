#!/usr/bin/env python3
"""
SteamIQ Integrity Linter: No Hardcoded Metrics Enforcer
Rule: STEAMIQ-FE-0002

Scans all TypeScript/TSX source files in frontend and Python files in backend/recsys & backend/jobs.
Ensures that no hardcoded precision metric literals (e.g., 0.0xxx accuracy values)
are embedded directly in production source files outside tests and fixtures.
All metrics must flow dynamically from computed reports or serving APIs.
"""

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIRS = [
    ROOT_DIR / "frontend" / "app",
    ROOT_DIR / "frontend" / "components",
    ROOT_DIR / "frontend" / "lib",
]
BACKEND_DIRS = [
    ROOT_DIR / "backend" / "recsys",
    ROOT_DIR / "backend" / "jobs",
]

# Pattern matching 4+ decimal places like 0.0123 or 0.1423
METRIC_LITERAL_PATTERN = re.compile(r"\b0\.0\d{3,}\b")

# Whitelisted test/fixture file substrings
EXCLUDE_SUBSTRINGS = ["test", "fixture", "spec", "mock"]


def is_comment_or_doc(line: str, ext: str = ".ts") -> bool:
    stripped = line.strip()
    if ext == ".py":
        return (
            stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
            or stripped.startswith("*")
        )
    return (
        stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("<!--")
    )


def check_file(file_path: Path) -> list[tuple[int, str, str]]:
    violations = []
    # Skip test/fixture files
    if any(ex in file_path.name.lower() for ex in EXCLUDE_SUBSTRINGS):
        return violations

    lines = file_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if is_comment_or_doc(line, file_path.suffix):
            continue

        matches = METRIC_LITERAL_PATTERN.findall(line)
        if matches:
            for m in matches:
                violations.append((i + 1, m, line.strip()))

    return violations


def main() -> int:
    all_violations = {}
    total_files_scanned = 0

    # 1. Frontend source files
    for base_dir in FRONTEND_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.ts", "*.tsx"):
            for file_path in base_dir.rglob(ext):
                total_files_scanned += 1
                v = check_file(file_path)
                if v:
                    all_violations[file_path] = v

    # 2. Backend source files (recsys and jobs)
    for base_dir in BACKEND_DIRS:
        if not base_dir.exists():
            continue
        for file_path in base_dir.rglob("*.py"):
            total_files_scanned += 1
            v = check_file(file_path)
            if v:
                all_violations[file_path] = v

    print(f"[CHECK] Scanned {total_files_scanned} frontend and backend source files for hardcoded metric literals...")

    if all_violations:
        print("\n[ERROR] Found hardcoded metric literals in production code:")
        for path, violations in all_violations.items():
            rel_path = path.relative_to(ROOT_DIR)
            print(f"\n  File: {rel_path}:")
            for line_num, match, code in violations:
                print(f"    Line {line_num}: Found '{match}' in: {code}")
        print("\nAll evaluation metrics must be loaded dynamically from reports or registry.")
        return 1

    print("[SUCCESS] Zero hardcoded metric literals found in production files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

