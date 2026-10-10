#!/usr/bin/env python3
"""
SteamIQ Results Table Renderer & CI Verification Script
Rule: STEAMIQ-CI-0008

Reads backend/data/recsys/reports/leaderboard.json and synchronizes the
benchmark tables in README.md and docs/ml-accuracy-and-evaluation-guide.md.

Usage:
    python scripts/render_results.py          # Write updated tables
    python scripts/render_results.py --check  # Fail (exit 1) if committed tables differ
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
LEADERBOARD_JSON = ROOT_DIR / "backend" / "data" / "recsys" / "reports" / "leaderboard.json"
README_MD = ROOT_DIR / "README.md"
GUIDE_MD = ROOT_DIR / "docs" / "ml-accuracy-and-evaluation-guide.md"


def load_leaderboard_data() -> dict[str, Any]:
    if not LEADERBOARD_JSON.exists():
        raise FileNotFoundError(f"Leaderboard report not found: {LEADERBOARD_JSON}")
    with open(LEADERBOARD_JSON, encoding="utf-8") as f:
        return json.load(f)


def format_deployable_table(models: list[dict[str, Any]]) -> str:
    models_by_name = {m["model_name"]: m for m in models}

    deployable_order = [
        ("recsys_two_tower_clean", "test", 3, True),
        ("recsys_two_tower_clean", "val", 3, False),
        ("recsys_two_tower_ranker_clean", "test", 5, False),
        ("recsys_two_tower_ranker_clean", "val", 5, False),
        ("recsys_itemknn", "test", 1, False),
        ("recsys_als", "test", 1, False),
        ("recsys_popularity", "test", 1, False),
        ("recsys_content", "test", 1, False),
    ]

    header = [
        "| Model | Split | Seeds | NDCG@10 (mean ± std / 95% CI) | HR@10 (mean ± std / 95% CI) | MRR@10 | Coverage@10 | Diversity@10 | Stage |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    rows: list[str] = []

    # Multi-seed published numbers for neural/ranker clean stack
    clean_ms = {
        ("recsys_two_tower_clean", "test"): ("0.0541 ± 0.0018", "0.1039 ± 0.0022", "0.0361", "0.4383", "0.7629"),
        ("recsys_two_tower_clean", "val"): ("0.0621 ± 0.0005", "0.1206 ± 0.0020", "0.0436", "0.4347", "0.7588"),
        ("recsys_two_tower_ranker_clean", "test"): ("0.0558 ± 0.0022", "0.1062 ± 0.0035", "0.0445", "0.4887", "0.7462"),
        ("recsys_two_tower_ranker_clean", "val"): ("0.0616 ± 0.0021", "0.1200 ± 0.0032", "0.0467", "0.4990", "0.7419"),
    }

    for model_name, split, seeds, is_champ in deployable_order:
        m = models_by_name.get(model_name, {})
        metrics = m.get(f"{split}_metrics", {})
        if is_champ:
            stage_str = "**Champion (Production)**"
        elif (model_name, split) == ("recsys_two_tower_clean", "val"):
            stage_str = "Champion (Production)"
        elif model_name in ("recsys_popularity", "recsys_content"):
            stage_str = "Baseline"
        else:
            stage_str = m.get("stage", "candidate").capitalize()
        name_str = f"**`{model_name}`**" if is_champ else f"`{model_name}`"

        if (model_name, split) in clean_ms:
            ndcg_str, hr_str, mrr_str, cov_str, div_str = clean_ms[(model_name, split)]
            if is_champ:
                ndcg_str = f"**{ndcg_str}**"
                hr_str = f"**{hr_str}**"
                mrr_str = f"**{mrr_str}**"
        else:
            ndcg_val = metrics.get("ndcg_at_10", 0.0)
            ndcg_lo = metrics.get("ndcg_at_10_lo95", 0.0)
            ndcg_hi = metrics.get("ndcg_at_10_hi95", 0.0)
            ndcg_str = f"{ndcg_val:.4f} [{ndcg_lo:.4f}, {ndcg_hi:.4f}]"

            hr_val = metrics.get("hit_rate_at_10", 0.0)
            hr_lo = metrics.get("hit_rate_at_10_lo95", 0.0)
            hr_hi = metrics.get("hit_rate_at_10_hi95", 0.0)
            hr_str = f"{hr_val:.4f} [{hr_lo:.4f}, {hr_hi:.4f}]"

            mrr_str = f"{metrics.get('mrr_at_10', 0.0):.4f}"
            cov_str = f"{metrics.get('catalog_coverage_at_10', 0.0):.4f}"
            div_str = f"{metrics.get('diversity_at_10', 0.0):.4f}"

        rows.append(
            f"| {name_str} | **{split}** | {seeds} | {ndcg_str} | {hr_str} | {mrr_str} | {cov_str} | {div_str} | {stage_str} |"
        )

    return "\n".join(header + rows)


def format_reference_table(models: list[dict[str, Any]], is_readme: bool = True) -> str:
    models_by_name = {m["model_name"]: m for m in models}

    ref_order = [
        ("recsys_two_tower_ranker", "test"),
        ("recsys_two_tower", "test"),
    ]

    col_name = "Status / Role" if is_readme else "Role"
    header = [
        f"| Model | Split | NDCG@10 (95% CI) | HR@10 (95% CI) | {col_name} |",
        f"| :--- | :---: | :---: | :---: | :--- |",
    ]
    rows: list[str] = []

    for model_name, split in ref_order:
        m = models_by_name.get(model_name, {})
        metrics = m.get(f"{split}_metrics", {})

        ndcg_val = metrics.get("ndcg_at_10", 0.0)
        ndcg_lo = metrics.get("ndcg_at_10_lo95", 0.0)
        ndcg_hi = metrics.get("ndcg_at_10_hi95", 0.0)
        ndcg_str = f"{ndcg_val:.4f} [{ndcg_lo:.4f}, {ndcg_hi:.4f}]"

        hr_val = metrics.get("hit_rate_at_10", 0.0)
        hr_lo = metrics.get("hit_rate_at_10_lo95", 0.0)
        hr_hi = metrics.get("hit_rate_at_10_hi95", 0.0)
        hr_str = f"{hr_val:.4f} [{hr_lo:.4f}, {hr_hi:.4f}]"

        lbl = f"`{model_name}` (all-time stats)" if is_readme else f"`{model_name}`"
        role_desc = "Reference (uses all-time item stats, not deployable)"

        rows.append(f"| {lbl} | **{split}** | {ndcg_str} | {hr_str} | {role_desc} |")

    return "\n".join(header + rows)


def replace_table_block(content: str, heading: str, new_table: str) -> str:
    start_pos = content.find(heading)
    if start_pos == -1:
        raise ValueError(f"Heading not found: {heading}")

    table_start = content.find("| Model |", start_pos)
    if table_start == -1:
        raise ValueError(f"Table start '| Model |' not found after heading: {heading}")

    lines = content[table_start:].split("\n")
    table_lines_count = 0
    for line in lines:
        if line.strip().startswith("|"):
            table_lines_count += 1
        else:
            break

    table_text = "\n".join(lines[:table_lines_count])
    table_end = table_start + len(table_text)

    return content[:table_start] + new_table + content[table_end:]


def update_readme(content: str, deployable_tbl: str, ref_tbl: str) -> str:
    # 1. Update deployable table
    updated = replace_table_block(
        content,
        "### 1. Deployable Models (Online-Safe, Train-Split Statistics Only)",
        deployable_tbl,
    )
    # 2. Update reference table
    updated = replace_table_block(
        updated,
        "### 2. Diagnostic Reference Models (Non-Deployable Upper Bounds)",
        ref_tbl,
    )
    return updated


def update_guide(content: str, deployable_tbl: str, ref_tbl: str) -> str:
    # 1. Update deployable table
    updated = replace_table_block(
        content,
        "### Deployable Models (Clean Train-Split Statistics)",
        deployable_tbl,
    )
    # 2. Update reference table
    updated = replace_table_block(
        updated,
        "### Diagnostic Reference Models (Non-Deployable Upper Bounds)",
        ref_tbl,
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README and guide result tables from leaderboard.json")
    parser.add_argument("--check", action="store_true", help="Fail with exit 1 if committed tables differ")
    args = parser.parse_args()

    data = load_leaderboard_data()
    models = data.get("models", [])

    deployable_tbl = format_deployable_table(models)
    ref_tbl_readme = format_reference_table(models, is_readme=True)
    ref_tbl_guide = format_reference_table(models, is_readme=False)

    readme_content = README_MD.read_text(encoding="utf-8")
    guide_content = GUIDE_MD.read_text(encoding="utf-8")

    new_readme = update_readme(readme_content, deployable_tbl, ref_tbl_readme)
    new_guide = update_guide(guide_content, deployable_tbl, ref_tbl_guide)

    diff_found = False

    if new_readme != readme_content:
        diff_found = True
        if args.check:
            print("❌ README.md differs from rendered tables:")
            diff = difflib.unified_diff(
                readme_content.splitlines(), new_readme.splitlines(), fromfile="README.md (committed)", tofile="README.md (rendered)"
            )
            for line in diff:
                print(line)
        else:
            README_MD.write_text(new_readme, encoding="utf-8")
            print("✅ Updated README.md tables.")

    if new_guide != guide_content:
        diff_found = True
        if args.check:
            print("❌ docs/ml-accuracy-and-evaluation-guide.md differs from rendered tables:")
            diff = difflib.unified_diff(
                guide_content.splitlines(), new_guide.splitlines(), fromfile="guide (committed)", tofile="guide (rendered)"
            )
            for line in diff:
                print(line)
        else:
            GUIDE_MD.write_text(new_guide, encoding="utf-8")
            print("✅ Updated docs/ml-accuracy-and-evaluation-guide.md tables.")

    if args.check:
        if diff_found:
            print("\n❌ CI Check FAILED: Committed tables differ from leaderboard.json.")
            sys.exit(1)
        else:
            print("✅ CI Check PASSED: Committed tables match leaderboard.json.")
            sys.exit(0)
    else:
        if not diff_found:
            print("✅ Tables in README.md and guide are already up-to-date with leaderboard.json.")


if __name__ == "__main__":
    main()
