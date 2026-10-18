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


def test_claims_and_evidence_consistency():
    claims_path = Path("docs/claims-and-evidence.md")
    assert claims_path.exists(), "docs/claims-and-evidence.md not found"
    claims_text = claims_path.read_text(encoding="utf-8")

    # 1. Leaderboard claims
    lb = json.loads(Path("docs/results/leaderboard.json").read_text(encoding="utf-8"))
    models = {m["model_name"]: m for m in lb["models"]}

    tt = models["recsys_two_tower_clean"]["test_metrics"]["ndcg_at_10"]
    rk = models["recsys_two_tower_ranker_clean"]["test_metrics"]["ndcg_at_10"]
    knn = models["recsys_itemknn"]["test_metrics"]["ndcg_at_10"]
    als = models["recsys_als"]["test_metrics"]["ndcg_at_10"]
    pop = models["recsys_popularity"]["test_metrics"]["ndcg_at_10"]

    assert f"{tt:.4f}" in claims_text, f"Two-Tower test {tt:.4f} missing from claims"
    assert f"{rk:.4f}" in claims_text, f"Ranker test {rk:.4f} missing from claims"
    assert f"{knn:.4f}" in claims_text, f"Item-kNN test {knn:.4f} missing from claims"
    assert f"{als:.4f}" in claims_text, f"ALS test {als:.4f} missing from claims"
    assert f"{pop:.4f}" in claims_text, f"Popularity test {pop:.4f} missing from claims"

    # 2. Ablations claim 3: must be validation numbers
    ab = json.loads(Path("docs/results/ablations.json").read_text(encoding="utf-8"))
    variants = {v["variant"]: v for v in ab["variants"]}
    diff_text = variants["no_text"]["paired_ndcg_diff_vs_full"]
    diff_tags = variants["no_tags"]["paired_ndcg_diff_vs_full"]
    diff_num = variants["no_numeric"]["paired_ndcg_diff_vs_full"]
    diff_id = variants["no_id"]["paired_ndcg_diff_vs_full"]

    assert f"{diff_text:.4f}" in claims_text, f"Text ablation {diff_text:.4f} missing from claims"
    assert f"{diff_tags:.4f}" in claims_text, f"Tags ablation {diff_tags:.4f} missing from claims"
    assert f"{diff_num:.4f}" in claims_text, f"Numeric ablation {diff_num:.4f} missing from claims"
    assert f"{diff_id:+.4f}" in claims_text or f"{diff_id:.4f}" in claims_text, f"ID ablation {diff_id:.4f} missing from claims"
    assert "validation split" in claims_text.lower(), "Claim 3 must state 'validation split'"

    # 3. Ranker ablations claim 4: must state validation
    rab = json.loads(Path("docs/results/ranker_ablations.json").read_text(encoding="utf-8"))
    r_abls = {a["ablated_group"]: a for a in rab["ablations"]}
    diff_src = r_abls["without_source_scores"]["val_ndcg_diff_vs_full"]
    diff_price = r_abls["without_price_features"]["val_ndcg_diff_vs_full"]
    diff_qual = r_abls["without_item_quality"]["val_ndcg_diff_vs_full"]

    assert f"{diff_src:.4f}" in claims_text, f"Source scores ablation {diff_src:.4f} missing from claims"
    assert f"{diff_price:.4f}" in claims_text, f"Price features ablation {diff_price:.4f} missing from claims"
    assert f"{diff_qual:.4f}" in claims_text, f"Item quality ablation {diff_qual:.4f} missing from claims"

    # 4. SASRec claim 5
    sas = json.loads(Path("docs/results/sasrec_benchmark.json").read_text(encoding="utf-8"))
    p95_lat = sas["latency_benchmark"]["p95_latency_ms"]
    assert f"{p95_lat:.2f}" in claims_text, f"SASRec p95 latency {p95_lat:.2f} missing from claims"

    # 5. Cold start claim 6
    cs = json.loads(Path("docs/results/cold_start.json").read_text(encoding="utf-8"))
    pooled_a = cs["item_holdout_study"]["pooled"]["group_a_cold_target"]["models"]["routed_hybrid_zero_only"]
    cs_ndcg = pooled_a["ndcg_at_10"]
    assert f"{cs_ndcg:.4f}" in claims_text, f"Cold start zero-only NDCG {cs_ndcg:.4f} missing from claims"
    assert "weak" in claims_text.lower(), "Cold start claim must state grade WEAK"


def test_current_authoritative_state_consistency():
    report_path = Path("docs/results/SPRINT_REPORT.md")
    assert report_path.exists(), "docs/results/SPRINT_REPORT.md not found"
    report_text = report_path.read_text(encoding="utf-8")

    assert "# Current Authoritative State" in report_text, "Missing '# Current Authoritative State' header"
    auth_state_text = report_text.split("# Current Authoritative State")[1].split("# History (superseded, do not cite)")[0]

    lb = json.loads(Path("docs/results/leaderboard.json").read_text(encoding="utf-8"))
    models = {m["model_name"]: m for m in lb["models"]}

    for m_name, bm in models.items():
        if m_name in ["recsys_two_tower_clean", "recsys_two_tower_ranker_clean", "recsys_itemknn", "recsys_als", "recsys_popularity", "recsys_content"]:
            t_nd = bm["test_metrics"]["ndcg_at_10"]
            assert f"{t_nd:.4f}" in auth_state_text, f"Expected {m_name} test NDCG {t_nd:.4f} in Current Authoritative State"


def test_stale_strings_lint():
    """Linter ensuring superseded and refuted strings never appear outside History or archived files."""
    forbidden_tokens = ["22.2x", "+62", "hand-written"]
    superseded_ablation_nums = ["-0.0259", "-0.0148", "-0.0146", "-0.0132"]

    # Files to check strictly
    files_to_check = [
        Path("README.md"),
        Path("docs/claims-and-evidence.md"),
    ]

    for p in files_to_check:
        if p.exists():
            text = p.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in text, f"Forbidden stale token '{token}' found in {p}!"
            for num in superseded_ablation_nums:
                # Disallow old ablation numbers if labelled as test
                if num in text:
                    assert "test" not in text.lower(), f"Old ablation number {num} must not be labelled as test in {p}!"

    # For SPRINT_REPORT.md, check text strictly before '# History (superseded, do not cite)'
    sprint_report = Path("docs/results/SPRINT_REPORT.md")
    if sprint_report.exists():
        content = sprint_report.read_text(encoding="utf-8")
        assert "# History (superseded, do not cite)" in content, "Missing History section separator in SPRINT_REPORT.md"
        before_history = content.split("# History (superseded, do not cite)")[0]

        for token in forbidden_tokens:
            assert token not in before_history, (
                f"Forbidden stale token '{token}' found in SPRINT_REPORT.md outside History!"
            )
        for num in superseded_ablation_nums:
            assert num not in before_history, (
                f"Old budget-mismatch ablation figure '{num}' found in Current Authoritative State!"
            )

