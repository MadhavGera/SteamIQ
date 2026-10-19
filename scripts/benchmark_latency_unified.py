"""
Unified Latency Benchmark for SteamIQ Recommender Models.

Runs a standardized 1,000-request single-user latency benchmark across all
production-candidate models on identical hardware with candidate cache disabled:
  1. Two-Tower Clean Champion (Stage 1 Retriever)
  2. SASRec Sequential Challenger (Stage 1 Retriever)
  3. Two-Tower Ranker V2 TrainStats (Stage 2 Production Ranker)
  4. End-to-End Pipeline (Retrieval + 25 Tabular Features + LightGBM Scoring)

Records:
  - Mean, Std, Min, Max
  - P50 (Median), P90, P95, P99
  - Full hardware telemetry (CPU, RAM, OS, versions)
Serializes to docs/results/latency_benchmark.json.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import psutil
import torch

# Add backend to sys.path
sys.path.insert(0, str(Path("backend").resolve()))

from recsys.data import RecsysData
from recsys.evaluate import _build_user_interaction_maps, get_evaluation_user_split
from recsys.inference import RecommenderService
from recsys.models.sequential import SASRecModel
from recsys.models.two_tower import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steamiq.latency_benchmark")


def get_cpu_info() -> str:
    """Gets detailed CPU name on Windows/Linux."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(cpu_name).strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_hardware_telemetry() -> dict[str, Any]:
    return {
        "cpu_model": get_cpu_info(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightgbm_version": lgb.__version__,
        "numpy_version": np.__version__,
    }


def compute_latency_stats(latencies: list[float], n_requests: int) -> dict[str, float]:
    arr = np.array(latencies, dtype=np.float64)
    return {
        "n_requests": n_requests,
        "batch_size": 1,
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }


def run_latency_benchmark(
    n_requests: int = 1000,
    warmup_requests: int = 50,
    seed: int = 42,
    data_dir: Path | str = "backend/data/recsys",
    docs_dir: Path | str = "docs/results",
) -> dict[str, Any]:
    data_path = Path(data_dir)
    docs_path = Path(docs_dir)
    reports_dir = data_path / "reports"
    docs_path.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = RecsysData.load(data_path)
    user_train_pos, _ = _build_user_interaction_maps(data)
    eval_users, _ = get_evaluation_user_split(
        data, n_eval_users=5000, n_tuning_users=5000, seed=seed, persistence_dir=data_path
    )

    rng = np.random.default_rng(seed)
    valid_eval_users = [int(u) for u in eval_users if len(user_train_pos.get(int(u), [])) >= 1]
    sampled_users = rng.choice(valid_eval_users, size=n_requests, replace=True)
    sampled_histories = [np.array(user_train_pos[u], dtype=np.int64) for u in sampled_users]

    logger.info("Sampled %d request histories (median length: %d)", n_requests, int(np.median([len(h) for h in sampled_histories])))

    results: dict[str, Any] = {}

    # 1. Benchmark Two-Tower Clean Champion (Seed 44)
    tt_path = data_path / "artifacts" / "recsys_two_tower_clean" / "seed_44"
    logger.info("Loading Two-Tower Champion from %s...", tt_path)
    tt_model = TwoTowerModel.load(tt_path, data=data)
    tt_model.precompute_item_vectors()

    logger.info("Warming up Two-Tower (%d requests)...", warmup_requests)
    for i in range(warmup_requests):
        _ = tt_model.score([sampled_histories[i % len(sampled_histories)]])

    logger.info("Benchmarking Two-Tower (%d requests, batch_size=1)...", n_requests)
    tt_latencies: list[float] = []
    for h in sampled_histories:
        t0 = time.perf_counter_ns()
        _ = tt_model.score([h])
        tt_latencies.append((time.perf_counter_ns() - t0) / 1e6)

    results["recsys_two_tower_clean"] = {
        "model_name": "recsys_two_tower_clean",
        "description": "Two-Tower Neural Retriever (Champion Seed 44)",
        "role": "retriever_stage_1",
        "sla_target_ms": 50.0,
        "sla_met": bool(np.percentile(tt_latencies, 95) < 50.0),
        **compute_latency_stats(tt_latencies, n_requests),
    }

    # 2. Benchmark SASRec Sequential Challenger (Champion Seed)
    bench_file = docs_path / "sasrec_benchmark.json"
    champ_seed = 42
    if bench_file.exists():
        with open(bench_file, encoding="utf-8") as f:
            bench_d = json.load(f)
        champ_seed = bench_d.get("metadata", {}).get("champion_seed", 42)

    sas_path = data_path / "artifacts" / "recsys_sasrec" / f"seed_{champ_seed}"
    if not sas_path.exists():
        sas_path = data_path / "artifacts" / "recsys_sasrec" / "seed_42"

    if sas_path.exists():
        logger.info("Loading SASRec Challenger (Champion Seed %s) from %s...", champ_seed, sas_path)
        sas_model = SASRecModel.load(sas_path, data=data)
        sas_model.precompute_item_vectors(data)

        logger.info("Warming up SASRec (%d requests)...", warmup_requests)
        for i in range(warmup_requests):
            _ = sas_model.score([sampled_histories[i % len(sampled_histories)]])

        logger.info("Benchmarking SASRec (%d requests, batch_size=1)...", n_requests)
        sas_latencies: list[float] = []
        for h in sampled_histories:
            t0 = time.perf_counter_ns()
            _ = sas_model.score([h])
            sas_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        results["recsys_sasrec"] = {
            "model_name": "recsys_sasrec",
            "description": f"SASRec Sequential Transformer Retriever (Champion Seed {champ_seed})",
            "role": "retriever_stage_1",
            "sla_target_ms": 50.0,
            "sla_met": bool(np.percentile(sas_latencies, 95) < 50.0),
            **compute_latency_stats(sas_latencies, n_requests),
        }

    # 3. Benchmark Two-Stage Production Ranker (Two-Tower Ranker Clean V2 TrainStats)
    # Using RecommenderService for realistic serving path
    logger.info("Initializing RecommenderService...")
    service = RecommenderService.get_instance(data_dir=data_path)
    service.initialize()

    ranker_key = "recsys_two_tower_ranker_v2_trainstats"
    if ranker_key in service.models:
        ranker_model = service.models[ranker_key]

        logger.info("Warming up Ranker V2 TrainStats (%d requests)...", warmup_requests)
        for i in range(warmup_requests):
            _ = ranker_model.score([sampled_histories[i % len(sampled_histories)]])

        logger.info("Benchmarking Ranker V2 TrainStats (%d requests, batch_size=1)...", n_requests)
        ranker_latencies: list[float] = []
        for h in sampled_histories:
            t0 = time.perf_counter_ns()
            _ = ranker_model.score([h])
            ranker_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        results[ranker_key] = {
            "model_name": ranker_key,
            "description": "Two-Tower Ranker V2 TrainStats (Production Ranker)",
            "role": "ranker_stage_2",
            "sla_target_ms": 50.0,
            "sla_met": bool(np.percentile(ranker_latencies, 95) < 50.0),
            **compute_latency_stats(ranker_latencies, n_requests),
        }

    # 4. Benchmark Full End-to-End Pipeline via service.recommend (candidate cache disabled)
    logger.info("Benchmarking End-to-End Recommendation Request (%d requests)...", min(n_requests, 500))
    e2e_latencies: list[float] = []
    # Test on valid user app_id sequences
    app_id_histories = []
    for u in sampled_users[:500]:
        item_indices = user_train_pos[u]
        app_ids = [int(service.idx_to_app_id[idx]) for idx in item_indices if idx in service.idx_to_app_id][:50]
        if app_ids:
            app_id_histories.append((u, app_ids))

    for u, app_ids in app_id_histories[:warmup_requests]:
        _ = service.recommend(liked_app_ids=app_ids, model=ranker_key, k=10, explain=False)

    for u, app_ids in app_id_histories:
        t0 = time.perf_counter_ns()
        _ = service.recommend(liked_app_ids=app_ids, model=ranker_key, k=10, explain=False)
        e2e_latencies.append((time.perf_counter_ns() - t0) / 1e6)

    results["end_to_end_serving_pipeline"] = {
        "model_name": "end_to_end_serving_pipeline",
        "description": "Full End-to-End Serving (Stage 1 Retrieval + Feature Join + Stage 2 LightGBM + Top-10 Format)",
        "role": "end_to_end_serving",
        "sla_target_ms": 50.0,
        "sla_met": bool(np.percentile(e2e_latencies, 95) < 50.0),
        **compute_latency_stats(e2e_latencies, len(e2e_latencies)),
    }

    hardware = get_hardware_telemetry()

    payload = {
        "metadata": {
            "report_key": "latency_benchmark",
            "n_requests": n_requests,
            "batch_size": 1,
            "hardware": hardware,
            "sla_constraint_ms": 50.0,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "models": results,
    }

    for p in [docs_path / "latency_benchmark.json", reports_dir / "latency_benchmark.json"]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved unified latency report to %s", p)

    print("\n" + "=" * 90)
    print("UNIFIED LATENCY BENCHMARK RESULTS (N=1,000, batch_size=1, cache=disabled)")
    print(f"Hardware: {hardware['cpu_model']} | {hardware['cpu_physical_cores']}C/{hardware['cpu_logical_cores']}T | {hardware['ram_total_gb']} GB RAM")
    print("=" * 90)
    print(f"{'Model / Pipeline':<36s} | {'Mean (ms)':<10s} | {'P50 (ms)':<10s} | {'P95 (ms)':<10s} | {'P99 (ms)':<10s} | {'SLA (<50ms)':<12s}")
    print("-" * 90)
    for k, v in results.items():
        print(f"{k:<36s} | {v['mean_ms']:<10.2f} | {v['p50_ms']:<10.2f} | {v['p95_ms']:<10.2f} | {v['p99_ms']:<10.2f} | {'PASSED' if v['sla_met'] else 'FAILED':<12s}")
    print("=" * 90 + "\n")

    return payload


if __name__ == "__main__":
    run_latency_benchmark()
