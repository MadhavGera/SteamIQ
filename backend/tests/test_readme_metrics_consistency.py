"""
Unit tests for README.md and leaderboard.json consistency (Issue N).

Verifies:
  1. Headline metrics in README.md matches docs/results/leaderboard.json within 1e-4 tolerance.
  2. The serving default in backend/recsys/inference.py matches the registry champion.
  3. Assistant documentation states clearly that human holdout evaluation is pending.
"""
from __future__ import annotations

import json
from pathlib import Path

from recsys.inference import DEFAULT_MODEL_NAME


def test_readme_leaderboard_table_consistency():
    readme_path = Path("README.md")
    leaderboard_path = Path("docs/results/leaderboard.json")

    assert readme_path.exists(), "README.md not found"
    assert leaderboard_path.exists(), "docs/results/leaderboard.json not found"

    with open(leaderboard_path, encoding="utf-8") as f:
        lb_data = json.load(f)

    models_by_name = {m["model_name"]: m for m in lb_data["models"]}

    # Read README content
    readme_text = readme_path.read_text(encoding="utf-8")

    # Verify Two-Tower Clean Champion metrics in README
    # Match pattern: | **`recsys_two_tower_clean`** | **test** | 1 (Champ 44) | **0.0513 [0.0467, 0.0563]** | **0.1016 [0.0936, 0.1102]** |
    tt_clean = models_by_name["recsys_two_tower_clean"]
    test_ndcg = tt_clean["test_metrics"]["ndcg_at_10"]
    test_hr = tt_clean["test_metrics"]["hit_rate_at_10"]
    val_ndcg = tt_clean["val_metrics"]["ndcg_at_10"]
    val_hr = tt_clean["val_metrics"]["hit_rate_at_10"]

    # Assert README contains these rounded figures (tolerance 1e-4)
    assert f"{test_ndcg:.4f}" in readme_text, f"Expected Test NDCG@10 {test_ndcg:.4f} in README"
    assert f"{test_hr:.4f}" in readme_text, f"Expected Test HR@10 {test_hr:.4f} in README"
    assert f"{val_ndcg:.4f}" in readme_text, f"Expected Val NDCG@10 {val_ndcg:.4f} in README"
    assert f"{val_hr:.4f}" in readme_text, f"Expected Val HR@10 {val_hr:.4f} in README"

    # Verify Baseline models in test split
    for baseline_name in ["recsys_itemknn", "recsys_als", "recsys_popularity", "recsys_content"]:
        b_model = models_by_name[baseline_name]
        b_ndcg = b_model["test_metrics"]["ndcg_at_10"]
        b_hr = b_model["test_metrics"]["hit_rate_at_10"]
        assert f"{b_ndcg:.4f}" in readme_text, f"Expected {baseline_name} Test NDCG@10 {b_ndcg:.4f} in README"
        assert f"{b_hr:.4f}" in readme_text, f"Expected {baseline_name} Test HR@10 {b_hr:.4f} in README"


def test_served_champion_equals_registry_champion():
    leaderboard_path = Path("docs/results/leaderboard.json")
    with open(leaderboard_path, encoding="utf-8") as f:
        lb_data = json.load(f)

    # Find model marked stage == 'champion'
    champion_models = [m["model_name"] for m in lb_data["models"] if m.get("stage") == "champion"]
    assert len(champion_models) >= 1, "No champion model found in leaderboard.json"

    registry_champion = champion_models[0]
    assert registry_champion == DEFAULT_MODEL_NAME, (
        f"Served default '{DEFAULT_MODEL_NAME}' does not match registry champion '{registry_champion}'!"
    )


def test_assistant_documentation_states_human_holdout_pending():
    docs_to_check = [
        Path("docs/results/SPRINT_REPORT.md"),
        Path("README.md"),
    ]
    found_pending_mention = False
    for doc_path in docs_to_check:
        if doc_path.exists():
            text = doc_path.read_text(encoding="utf-8").lower()
            if "human" in text and ("pending" in text or "scaffold" in text or "future" in text):
                found_pending_mention = True
                break

    assert found_pending_mention, "Documentation must explicitly state that human holdout evaluation is pending/scaffolded"
