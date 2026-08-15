#!/usr/bin/env python3
"""
SteamIQ Integrity Linter: Honest Fallbacks Enforcer
Rule: STEAMIQ-FE-0001

Scans all TypeScript/TSX source files in frontend/components and frontend/app.
Ensures that any null-coalescing operator (`??`) used for fallback data is
accompanied by an explicit `// HONEST-FALLBACK:` annotation on the current
or immediately preceding line.
"""

import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIRS = [
    ROOT_DIR / "frontend" / "components",
    ROOT_DIR / "frontend" / "app",
]

# Pattern matching ?? operator
COALESCE_PATTERN = re.compile(r"\?\?")
# Pattern matching required comment
HONEST_COMMENT_PATTERN = re.compile(r"HONEST-FALLBACK:")

def check_file(file_path: Path) -> list[tuple[int, str]]:
    violations = []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    
    for i, line in enumerate(lines):
        # Skip pure comment lines
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
            
        if "??" in line:
            # Check current line, previous line, or 2 lines up for HONEST-FALLBACK:
            has_comment = bool(HONEST_COMMENT_PATTERN.search(line))
            if not has_comment and i > 0:
                has_comment = bool(HONEST_COMMENT_PATTERN.search(lines[i - 1]))
            if not has_comment and i > 1:
                has_comment = bool(HONEST_COMMENT_PATTERN.search(lines[i - 2]))
                
            if not has_comment:
                violations.append((i + 1, line.strip()))
                
    return violations

def main() -> int:
    all_violations = {}
    total_files_scanned = 0
    
    for base_dir in FRONTEND_DIRS:
        if not base_dir.exists():
            continue
        for ext in ("*.ts", "*.tsx"):
            for file_path in base_dir.rglob(ext):
                total_files_scanned += 1
                v = check_file(file_path)
                if v:
                    all_violations[file_path] = v
                    
    print(f"[CHECK] Scanned {total_files_scanned} frontend source files for HONEST-FALLBACK compliance...")
    
    if all_violations:
        print("\n[ERROR] Found unannotated ?? fallback operators in UI code:")
        for path, violations in all_violations.items():
            rel_path = path.relative_to(ROOT_DIR)
            print(f"\n  File: {rel_path}:")
            for line_num, code in violations:
                print(f"    Line {line_num}: {code}")
        print("\nRule STEAMIQ-FE-0001 requires all ?? fallback expressions to be verified and annotated")
        print("with a `// HONEST-FALLBACK: <rationale>` comment confirming they do not fabricate domain metrics.")
        return 1
        
    print("[SUCCESS] All ?? fallback operators are strictly annotated and compliant with STEAMIQ-FE-0001.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
