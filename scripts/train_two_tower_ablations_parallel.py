"""
Parallel training script for Two-Tower ablations (Phase J full budget).
Trains 4 variants (no_text, no_tags, no_numeric, no_id) across seeds (42, 43, 44)
with exact full-budget configuration:
  epochs=15, batch_size=2048, lr=2e-3, patience=5, pairs_per_epoch=None (all pairs)
"""
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path("backend").resolve()))

import torch
from recsys.data import RecsysData
from recsys.evaluate import get_evaluation_user_split
from recsys.models.two_tower import TwoTowerConfig
from recsys.train_two_tower import train_two_tower_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steamiq.train_ablations_parallel")


VARIANTS = {
    "no_text": {"use_text": False},
    "no_tags": {"use_tags": False},
    "no_numeric": {"use_numeric": False},
    "no_id": {"use_id": False},
}
SEEDS = [42, 43, 44]


def train_single_ablation(var_key: str, seed: int, data_path_str: str = "backend/data/recsys") -> str:
    data_path = Path(data_path_str)
    s_dir = data_path / "artifacts" / f"recsys_two_tower_{var_key}" / f"seed_{seed}"
    curves_file = s_dir / "training_curves.json"
    ckpt_file = s_dir / "two_tower.pt"

    if ckpt_file.exists() and curves_file.exists():
        return f"[{var_key} seed {seed}] Already exists, skipping."

    # Limit torch threads per process so 2 concurrent workers use ~8 cores
    torch.set_num_threads(4)

    t0 = time.perf_counter()
    data = RecsysData.load(data_path)
    _, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=data_path,
    )

    overrides = VARIANTS[var_key]
    cfg = TwoTowerConfig(
        n_items=data.n_items,
        text_dim=384,
        tag_dim=300,
        numeric_dim=6,
        id_dim=64,
        text_proj_dim=64,
        tag_proj_dim=32,
        numeric_proj_dim=16,
        hidden_dim=256,
        output_dim=64,
        pool="attn",
        id_dropout=0.25,
        proxy_dropout=0.25,
        no_pop_proxies=True,
        **overrides,
    )

    logger.info("Starting [%s seed %d] Full Budget: 15 epochs, all pairs, patience 5...", var_key, seed)
    model, curves = train_two_tower_model(
        data=data,
        tuning_users=tuning_users,
        cfg=cfg,
        epochs=15,
        batch_size=2048,
        lr=2e-3,
        patience=5,
        pairs_per_epoch=None,
        seed=seed,
    )

    s_dir.mkdir(parents=True, exist_ok=True)
    model.save(s_dir)
    with open(curves_file, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    elapsed = time.perf_counter() - t0
    best_ep = curves.get("best_epoch", -1)
    best_ndcg = curves.get("best_tuning_val_ndcg_at_10", 0.0)
    return f"[{var_key} seed {seed}] Finished in {elapsed:.1f}s (best epoch {best_ep}, tuning NDCG {best_ndcg:.4f})"


def main():
    tasks = []
    for var_key in VARIANTS:
        for s in SEEDS:
            tasks.append((var_key, s))

    logger.info("Queued %d ablation training tasks across 2 worker processes...", len(tasks))
    t_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_map = {
            executor.submit(train_single_ablation, var, s): (var, s)
            for var, s in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            var, s = future_map[future]
            try:
                res = future.result()
                logger.info(">>> TASK COMPLETE: %s", res)
            except Exception as e:
                logger.error(">>> TASK FAILED [%s seed %d]: %s", var, s, e, exc_info=True)

    total_time = time.perf_counter() - t_start
    logger.info("All ablation training tasks completed in %.1f seconds (%.2f minutes).", total_time, total_time / 60)


if __name__ == "__main__":
    main()
