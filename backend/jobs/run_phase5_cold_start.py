"""
Job: Execute Phase E Routed Cold-Start Multi-Cohort Study.

Resolves Issue I5:
  1. Fixes score scale mismatch via per-user z-score matching calibration in RoutedHybridRecommender.
  2. Evaluates across 3 distinct withheld-item cohorts (seeds 42, 43, 44), 300 items each.
  3. Verifies that withheld item sets and evaluated user sets differ across seeds.
  4. Evaluates Random, Content-Only, Two-Tower (clean architecture, neutralized), and Calibrated Routed Hybrid (N=5).
  5. Computes per-cohort and pooled bootstrap CIs and performs evidence strength assessment.
  6. Updates docs/results/cold_start.json and backend/data/recsys/reports/cold_start.json.

Protects test split by evaluating on the 5,000 validation users.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from recsys.coldstart import (  # noqa: E402
    evaluate_multi_cohort_holdout_study,
    evaluate_simulated_new_items_study,
    evaluate_sparse_targets_study,
)
from recsys.data import RecsysData  # noqa: E402
from recsys.evaluate import get_evaluation_user_split  # noqa: E402
from recsys.models.als import ALSRecommender  # noqa: E402
from recsys.models.content import ContentRecommender  # noqa: E402
from recsys.models.popularity import PopularityRecommender  # noqa: E402
from recsys.models.two_tower import TwoTowerModel  # noqa: E402


def run_phase5_cold_start(
    data_dir: Path | str = "backend/data/recsys",
    docs_results_dir: Path | str = "docs/results",
    cohort_seeds: list[int] | None = None,
    cold_threshold: int = 5,
    split: str = "val",
    n_boot: int = 1000,
) -> dict[str, Any]:
    if cohort_seeds is None:
        cohort_seeds = [42, 43, 44]

    data_path = Path(data_dir)
    docs_path = Path(docs_results_dir)
    reports_dir = data_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading RecsysData from {data_path}...")
    data = RecsysData.load(data_path)

    val_users, _ = get_evaluation_user_split(
        data, n_eval_users=5000, n_tuning_users=5000, seed=42, persistence_dir=data_path
    )

    # Load Clean Champion Two-Tower Model (Seed 44)
    tt_dir = data_path / "artifacts" / "recsys_two_tower_clean" / "seed_44"
    print(f"Loading Clean Two-Tower Champion from {tt_dir}...")
    tt_model = TwoTowerModel.load(tt_dir, data=data)

    # Load / Fit Content model
    cnt_dir = data_path / "artifacts" / "recsys_content"
    if cnt_dir.exists() and any(cnt_dir.iterdir()):
        latest_cnt = sorted([d for d in cnt_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[0]
        cnt_model = ContentRecommender.load(latest_cnt)
    else:
        cnt_model = ContentRecommender(weight_tags=0.5)
        cnt_model.fit(data)

    # Baselines for sparse & simulated studies
    pop_model = PopularityRecommender()
    pop_model.fit(data)

    als_dir = data_path / "artifacts" / "recsys_als"
    if als_dir.exists() and any(als_dir.iterdir()):
        latest_als = sorted([d for d in als_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[0]
        als_model = ALSRecommender.load(latest_als)
    else:
        als_model = ALSRecommender(factors=64, iterations=15, regularization=0.05, seed=42)
        als_model.fit(data)

    print("\n" + "=" * 70)
    print("STEP 1: Run Multi-Cohort True Item Holdout Study")
    print(f"Cohorts: {cohort_seeds}, Threshold N: {cold_threshold}, Split: {split}")
    print("=" * 70)

    holdout_res = evaluate_multi_cohort_holdout_study(
        data=data,
        primary_model=tt_model,
        cold_model=cnt_model,
        eval_users=val_users,
        cohort_seeds=cohort_seeds,
        fraction=0.10,
        cold_threshold=cold_threshold,
        split=split,
        n_boot=n_boot,
        cohort_size=200,
        seed=42,
    )

    print(f"\n{holdout_res['title']}")
    print(f"Total evaluated users: {holdout_res['n_eval_users']}")
    print(f"Total unique withheld items: {holdout_res['total_unique_items_withheld']}")
    grade = holdout_res.get("evidence_grade", holdout_res.get("evidence_status", "UNKNOWN"))
    print(f"Evidence status / grade: {grade.upper()}")
    print(f"Evidence statement: {holdout_res['evidence_statement']}\n")

    print(f"{'Model':<20s} | {'NDCG@10 (95% CI)':<28s} | {'HR@10 (95% CI)':<26s} | {'Mean Rank (95% CI)':<24s}")
    print("-" * 105)
    for m_name, s in holdout_res["models"].items():
        nd = s.get("ndcg_at_10", 0.0)
        nd_lo = s.get("ndcg_at_10_lo95", nd)
        nd_hi = s.get("ndcg_at_10_hi95", nd)
        hr = s.get("hit_rate_at_10", 0.0)
        hr_lo = s.get("hit_rate_at_10_lo95", hr)
        hr_hi = s.get("hit_rate_at_10_hi95", hr)
        mr = s.get("mean_rank", 0.0)
        mr_lo = s.get("mean_rank_lo95", mr)
        mr_hi = s.get("mean_rank_hi95", mr)
        nd_str = f"{nd:.5f} [{nd_lo:.5f}, {nd_hi:.5f}]"
        hr_str = f"{hr:.4f} [{hr_lo:.4f}, {hr_hi:.4f}]"
        mr_str = f"{mr:.1f} [{mr_lo:.1f}, {mr_hi:.1f}]"
        print(f"{m_name:<20s} | {nd_str:<28s} | {hr_str:<26s} | {mr_str:<24s}")

    print("\nPer-Cohort Details:")
    for c_key, c_val in holdout_res["per_cohort"].items():
        print(f"  {c_key}: n_users={c_val['n_users']}, sample items={c_val['withheld_items_sample'][:5]}...")
        for m_name, s in c_val["models"].items():
            nd = s.get("ndcg_at_10", 0.0)
            hr = s.get("hit_rate_at_10", 0.0)
            mr = s.get("mean_rank", 0.0)
            print(f"    {m_name:<16s}: NDCG@10 = {nd:.5f}, HR@10 = {hr:.4f}, Mean Rank = {mr:.1f}")

    print("\n" + "=" * 70)
    print("STEP 2: Run Auxiliary Diagnostic Studies on Validation Split")
    print("=" * 70)

    sparse_res = evaluate_sparse_targets_study(
        data=data,
        models={
            "popularity": pop_model,
            "content": cnt_model,
            "als": als_model,
            "two_tower": tt_model,
        },
        eval_users=val_users,
        max_train_positives=10,
        seed=42,
        n_boot=n_boot,
    )

    sim_res = evaluate_simulated_new_items_study(
        data=data,
        models={
            "popularity": pop_model,
            "als": als_model,
            "content": cnt_model,
            "two_tower": tt_model,
        },
        eval_users=val_users,
        fraction=0.10,
        seed=42,
        variant="c2",
        n_boot=n_boot,
    )

    # Build Unified Payload
    payload = {
        "metadata": {
            "report_key": "cold_start",
            "dataset_version": data.dataset_version,
            "n_eval_users": len(val_users),
            "split": split,
            "seed": 42,
            "cohort_seeds": cohort_seeds,
            "cold_threshold": cold_threshold,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "item_holdout_study": holdout_res,
        "sparse_targets_study": sparse_res,
        "simulated_new_items_study": sim_res,
    }

    # Save to both docs/results/cold_start.json and backend/data/recsys/reports/cold_start.json
    for out_path in [docs_path / "cold_start.json", reports_dir / "cold_start.json"]:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved cold-start report to {out_path}")

    return payload


if __name__ == "__main__":
    run_phase5_cold_start()
