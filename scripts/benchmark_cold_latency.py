"""
Benchmark cold latency for recommender rankers with candidate cache disabled.
Runs explain=False on 100 cold, distinct user histories for:
- recsys_two_tower_ranker_clean (Old Ranker)
- recsys_two_tower_ranker_clean_v2 (Ranker V2)
- recsys_two_tower_ranker_v2_trainstats (Ranker V2 TrainStats)
"""
import sys
from pathlib import Path

# Add backend and root to sys.path
sys.path.insert(0, str(Path("backend").resolve()))
sys.path.insert(0, str(Path(".").resolve()))

import time
import numpy as np
from recsys.inference import RecommenderService
from recsys.data import RecsysData
from recsys.evaluate import _build_user_interaction_maps, get_evaluation_user_split
import recsys.ranker as ranker_module


def main():
    print("=" * 90)
    print("BENCHMARK: COLD SERVING LATENCY (Candidate Cache Disabled, explain=False, N=100 Distinct Users)")
    print("=" * 90)

    real_data_dir = Path("backend/data/recsys")
    if not (real_data_dir / "data_manifest.json").exists():
        real_data_dir = Path("data/recsys")

    service = RecommenderService.get_instance(data_dir=real_data_dir)
    service.initialize()

    data = RecsysData.load(real_data_dir)
    user_train_pos, _ = _build_user_interaction_maps(data)

    eval_users, _ = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=real_data_dir,
    )

    # Select 100 distinct valid users with at least 3 interactions
    rng = np.random.default_rng(42)
    valid_users = [u for u in eval_users if len(user_train_pos.get(int(u), [])) >= 3]
    selected_users = rng.choice(valid_users, size=100, replace=False)

    user_histories = []
    for u in selected_users:
        item_indices = user_train_pos[int(u)]
        app_ids = [int(service.idx_to_app_id[idx]) for idx in item_indices if idx in service.idx_to_app_id][:50]
        if app_ids:
            user_histories.append((int(u), app_ids))

    print(f"Loaded {len(user_histories)} cold distinct user histories.")
    print(f"Average history length: {np.mean([len(h[1]) for h in user_histories]):.1f} items (min: {min(len(h[1]) for h in user_histories)}, max: {max(len(h[1]) for h in user_histories)})\n")

    models_to_test = [
        ("recsys_two_tower_ranker_clean", "Old Ranker (all-time stats, non-deployable)"),
        ("recsys_two_tower_ranker_clean_v2", "Ranker V2 (clean stage 1, all-time item stats)"),
        ("recsys_two_tower_ranker_v2_trainstats", "Ranker V2 TrainStats (clean stage 1 + train-only stats)"),
    ]

    results = {}

    for model_key, label in models_to_test:
        if model_key not in service.models:
            print(f"WARNING: {model_key} not in service.models! Skipping.")
            continue

        latencies = []
        # Warmup single call (to ensure libraries, JIT/OpenMP initialized)
        ranker_module._SCORE_CAND_CACHE.clear()
        _ = service.recommend(liked_app_ids=user_histories[0][1], model=model_key, k=10, explain=False)

        print(f"Running 100 cold requests for: {model_key} ({label})...")
        for idx, (uid, hist) in enumerate(user_histories):
            # STRICT REQUIREMENT: Candidate cache disabled on cold histories
            ranker_module._SCORE_CAND_CACHE.clear()

            t0 = time.perf_counter()
            resp = service.recommend(liked_app_ids=hist, model=model_key, k=10, explain=False)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)

        latencies = np.array(latencies)
        results[model_key] = {
            "label": label,
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "samples": latencies.tolist(),
        }

    print("\n" + "=" * 90)
    print("LATENCY SUMMARY TABLE (milliseconds)")
    print("=" * 90)
    print(f"{'Model':<38} | {'Mean':>7} | {'P50':>7} | {'P90':>7} | {'P95':>7} | {'P99':>7} | {'Max':>7}")
    print("-" * 90)
    for model_key, r in results.items():
        print(f"{model_key:<38} | {r['mean']:>7.2f} | {r['p50']:>7.2f} | {r['p90']:>7.2f} | {r['p95']:>7.2f} | {r['p99']:>7.2f} | {r['max']:>7.2f}")
    print("=" * 90)

    # Comparison vs Old Ranker
    old_res = results.get("recsys_two_tower_ranker_clean")
    if old_res:
        print("\nCOMPARISON VS OLD RANKER:")
        for mk in ["recsys_two_tower_ranker_clean_v2", "recsys_two_tower_ranker_v2_trainstats"]:
            if mk in results:
                diff_mean = results[mk]["mean"] - old_res["mean"]
                diff_p50 = results[mk]["p50"] - old_res["p50"]
                diff_p95 = results[mk]["p95"] - old_res["p95"]
                print(f"  {mk} vs Old Ranker:")
                print(f"    Mean diff: {diff_mean:+.2f} ms ({diff_mean/old_res['mean']*100:+.1f}%)")
                print(f"    P50 diff:  {diff_p50:+.2f} ms ({diff_p50/old_res['p50']*100:+.1f}%)")
                print(f"    P95 diff:  {diff_p95:+.2f} ms ({diff_p95/old_res['p95']*100:+.1f}%)")

    # Save raw log output
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_file = logs_dir / "cold_latency_benchmark.log"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("COLD LATENCY BENCHMARK (100 distinct histories, cache disabled, explain=False)\n")
        f.write("=" * 90 + "\n")
        for mk, r in results.items():
            f.write(f"\nModel: {mk} ({r['label']})\n")
            f.write(f"  Mean: {r['mean']:.2f} ms, Std: {r['std']:.2f} ms\n")
            f.write(f"  P50:  {r['p50']:.2f} ms, P90: {r['p90']:.2f} ms, P95: {r['p95']:.2f} ms, P99: {r['p99']:.2f} ms\n")
            f.write(f"  Min:  {r['min']:.2f} ms, Max: {r['max']:.2f} ms\n")
            f.write(f"  Raw latencies: {[round(x, 2) for x in r['samples']]}\n")
    print(f"\nRaw latency results written to: {out_file}")


if __name__ == "__main__":
    main()
