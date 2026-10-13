"""
Snapshot Regression Fixture Updater — Phase 7.

Generates recommendations for the 50 fixed test user histories using
the registered production model (recsys_two_tower_clean), compares against
the existing golden snapshot fixture, prints unified diff, and updates fixture.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core.config import settings
from recsys.inference import RecommenderService


def main() -> None:
    data_dir = settings.resolved_recsys_data_dir
    artifacts_dir = data_dir / "artifacts"
    if not artifacts_dir.exists():
        print("ERROR: Artifacts directory does not exist.")
        sys.exit(1)

    service = RecommenderService.get_instance()
    service.initialize()

    catalog_app_ids = list(service.app_id_to_idx.keys())
    if len(catalog_app_ids) < 20:
        print("ERROR: Catalog size too small.")
        sys.exit(1)

    fixed_histories = []
    for i in range(50):
        step = (i % 7) + 1
        length = (i % 8) + 3
        hist = [int(catalog_app_ids[(i * step + j * 13) % len(catalog_app_ids)]) for j in range(length)]
        hist = list(dict.fromkeys(hist))
        fixed_histories.append(hist)

    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "backend"
        / "tests"
        / "golden"
        / "fixtures"
        / "golden_artifact_recommendations.json"
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    current_outputs = []
    for h in fixed_histories:
        rec_res = service.recommend(
            liked_app_ids=h,
            model="recsys_two_tower_clean",
            k=10,
            explain=False,
        )
        rec_ids = [it["app_id"] for it in rec_res["items"]]
        current_outputs.append({
            "history": h,
            "recommended_app_ids": rec_ids,
        })

    new_content = json.dumps(current_outputs, indent=2) + "\n"

    if fixture_path.exists():
        old_content = fixture_path.read_text(encoding="utf-8")
        diff = list(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile="golden_artifact_recommendations.json (existing)",
                tofile="golden_artifact_recommendations.json (updated)",
            )
        )
        if diff:
            print(">>> Differences detected in snapshot regression fixtures:")
            sys.stdout.writelines(diff)
        else:
            print("[OK] Snapshot regression fixtures are identical. Zero drift detected.")
    else:
        print(">>> No existing fixture found. Creating new golden artifact snapshot...")

    fixture_path.write_text(new_content, encoding="utf-8")
    print(f"[OK] Successfully wrote 50 golden histories to {fixture_path}")


if __name__ == "__main__":
    main()
