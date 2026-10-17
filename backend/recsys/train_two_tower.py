"""
Training engine for the PyTorch Two-Tower Hybrid Retrieval Model.

Implements:
  - Chronological next-item pair generation from user training histories.
  - In-batch sampled softmax loss with logQ popularity correction and duplicate item false-negative masking.
  - Cosine annealing learning rate scheduling with gradient clipping.
  - Epoch evaluation on disjoint tuning users with early stopping and checkpointing.
  - Training curves persistence and MLflow metric logging.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from recsys.data import RecsysData
from recsys.evaluate import evaluate_model
from recsys.models.two_tower import (
    TwoTowerConfig,
    TwoTowerModel,
    TwoTowerModule,
    extract_standardized_numeric_features,
)

logger = logging.getLogger("steamiq.recsys.train_two_tower")


class NextItemPairDataset(Dataset):
    """
    PyTorch Dataset storing next-item interaction pairs (history, target).
    """

    def __init__(
        self,
        histories: np.ndarray,  # [N_pairs, max_history]
        masks: np.ndarray,      # [N_pairs, max_history]
        targets: np.ndarray,    # [N_pairs]
    ) -> None:
        self.histories = torch.from_numpy(histories)
        self.masks = torch.from_numpy(masks)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.histories[idx], self.masks[idx], self.targets[idx]


def build_next_item_training_pairs(
    data: RecsysData | pd.DataFrame,
    max_history: int = 50,
) -> tuple[NextItemPairDataset, np.ndarray]:
    """
    Builds chronological next-item training pairs from train_df.
    For each user sequence [i_0, i_1, ..., i_T]:
      For t >= 1:
        target = i_t
        history = [i_{max(0, t - max_history)}, ..., i_{t-1}]

    Returns:
        (dataset, item_pair_frequencies [n_items])
    """
    if isinstance(data, pd.DataFrame):
        train_df = data
        n_items = int(train_df["item_idx"].max()) + 1 if len(train_df) > 0 else 0
    else:
        train_df = data.train_df
        n_items = data.n_items

    train_sorted = train_df.sort_values(by=["user_idx", "ts"])
    user_sequences: dict[int, list[int]] = {}

    for u, item in zip(train_sorted["user_idx"].to_numpy(), train_sorted["item_idx"].to_numpy(), strict=True):
        if u not in user_sequences:
            user_sequences[u] = []
        user_sequences[u].append(int(item))

    hist_list: list[list[int]] = []
    target_list: list[int] = []

    for seq in user_sequences.values():
        if len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            target = seq[t]
            h = seq[max(0, t - max_history) : t]
            hist_list.append(h)
            target_list.append(target)

    n_pairs = len(target_list)
    logger.info("Generated %d chronological next-item training pairs", n_pairs)

    hist_arr = np.zeros((n_pairs, max_history), dtype=np.int64)
    mask_arr = np.zeros((n_pairs, max_history), dtype=bool)
    targets_arr = np.array(target_list, dtype=np.int64)

    for idx, h in enumerate(hist_list):
        length = len(h)
        hist_arr[idx, -length:] = h
        mask_arr[idx, -length:] = True

    # Compute item frequencies q_i = count_i / total_pairs
    item_counts = np.bincount(targets_arr, minlength=n_items).astype(np.float64)
    q_items = item_counts / max(1.0, float(n_pairs))

    dataset = NextItemPairDataset(hist_arr, mask_arr, targets_arr)
    return dataset, q_items.astype(np.float32)


def in_batch_sampled_softmax_loss(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor | None = None,
    target_ids: torch.Tensor | None = None,
    tau: torch.Tensor | float | None = None,
    log_q: torch.Tensor | None = None,
    return_logits: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Computes in-batch sampled softmax loss with logQ correction and false-negative masking.

    Args:
        user_vecs: User representations [B, 64]
        item_vecs: Target item representations [B, 64]
        target_ids: 1D LongTensor of target item indices [B]
        tau: Scalar temperature parameter
        log_q: 1D FloatTensor of log(q_i) popularity values [n_items]
        return_logits: If True, returns tuple (loss, logits)

    Returns:
        Scalar cross-entropy loss tensor, or (loss, logits) if return_logits=True
    """
    if item_vecs is None and "target_item_vecs" in kwargs:
        item_vecs = kwargs["target_item_vecs"]
    if target_ids is None and "target_item_indices" in kwargs:
        target_ids = kwargs["target_item_indices"]
    if tau is None and "log_tau" in kwargs:
        tau = torch.exp(kwargs["log_tau"])
    if tau is None:
        tau = torch.tensor(0.07, dtype=torch.float32, device=user_vecs.device)
    if log_q is None:
        log_q = torch.zeros(len(item_vecs) if item_vecs is not None else len(user_vecs), dtype=torch.float32, device=user_vecs.device)

    b = user_vecs.size(0)
    if isinstance(tau, float):
        tau = torch.tensor(tau, dtype=torch.float32, device=user_vecs.device)

    # 1. Cosine similarity logits scaled by temperature: [B, B]
    logits = torch.matmul(user_vecs, item_vecs.T) / tau

    # 2. LogQ Correction: subtract log(q_j) for candidate item j (columns)
    if log_q.ndim > 0 and len(log_q) > 0:
        batch_log_q = log_q[target_ids] if target_ids.max().item() < len(log_q) else log_q[:b]
        logits = logits - batch_log_q.unsqueeze(0)

    # 3. Mask False Negatives (duplicate item occurrences across different rows)
    matches = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)  # [B, B]
    diag_mask = torch.eye(b, dtype=torch.bool, device=user_vecs.device)
    false_negatives = matches & ~diag_mask

    if false_negatives.any():
        logits = logits.masked_fill(false_negatives, -1e9)

    # 4. Cross Entropy: target is the diagonal index [0, 1, ..., B-1]
    labels = torch.arange(b, dtype=torch.long, device=user_vecs.device)
    loss = F.cross_entropy(logits, labels)

    if return_logits or kwargs.get("return_logits", False):
        return loss, logits
    return loss


def train_two_tower_model(
    data: RecsysData | TwoTowerModel,
    tuning_users: np.ndarray | RecsysData | None = None,
    cfg: TwoTowerConfig | None = None,
    epochs: int = 15,
    batch_size: int = 2048,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    pairs_per_epoch: int | None = None,
    seed: int = 42,
    model: TwoTowerModel | None = None,
    val_user_indices: np.ndarray | None = None,
) -> tuple[TwoTowerModel, dict[str, Any]]:
    """
    Trains TwoTowerModel with early stopping on validation NDCG@10 of tuning users.

    Returns:
        (best_model, training_history_dict)
    """
    # Support positional call where model is first arg: train_two_tower_model(model, data, ...)
    if isinstance(data, TwoTowerModel):
        passed_model = data
        data = tuning_users  # second positional arg is data
        tuning_users = val_user_indices
        model = passed_model
        if cfg is None:
            cfg = model.config

    if tuning_users is None and val_user_indices is not None:
        tuning_users = val_user_indices

    torch.manual_seed(seed)
    np.random.seed(seed)

    if tuning_users is None:
        from recsys.evaluate import get_evaluation_user_split
        _, tuning_users = get_evaluation_user_split(data, seed=seed, persistence_dir=None)

    if cfg is None:
        if model is not None and model.config is not None:
            cfg = model.config
        else:
            cfg = TwoTowerConfig(n_items=data.n_items)
    elif cfg.n_items == 0:
        cfg.n_items = data.n_items

    # 1. Prepare Next-Item Training Dataset
    full_dataset, q_items = build_next_item_training_pairs(data, max_history=cfg.max_history)
    n_total_pairs = len(full_dataset)
    print(f"Training Two-Tower model on {n_total_pairs} pairs. Evaluating on {len(tuning_users)} tuning users...")

    # Compute log(q) tensor
    log_q_t = torch.from_numpy(np.log(np.maximum(q_items, 1e-12))).float()

    # 2. Extract item features into tensors
    n_items = data.n_items
    if data.item_text_emb is not None and len(data.item_text_emb) == n_items:
        text_t = torch.from_numpy(data.item_text_emb.astype(np.float32))
    else:
        text_t = torch.zeros((n_items, 384), dtype=torch.float32)

    if data.item_tags_sparse is not None and data.item_tags_sparse.shape[0] == n_items:
        if sp.issparse(data.item_tags_sparse):
            tag_dense = data.item_tags_sparse.toarray().astype(np.float32)
        else:
            tag_dense = np.asarray(data.item_tags_sparse, dtype=np.float32)
        tag_t = torch.from_numpy(tag_dense)
    else:
        tag_t = torch.zeros((n_items, 300), dtype=torch.float32)

    no_pop = bool(cfg.no_pop_proxies) if cfg else False
    num_arr = extract_standardized_numeric_features(data.items_df, no_pop_proxies=no_pop)
    num_t = torch.from_numpy(num_arr)

    # 3. Instantiate Module & Optimizer
    if model is not None and model.module is not None:
        module = model.module
        model_wrapper = model
    else:
        module = TwoTowerModule(cfg)
        model_wrapper = TwoTowerModel(cfg=cfg, module=module)

    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model_wrapper._item_features_tensors = {
        "all_ids": torch.arange(n_items, dtype=torch.long),
        "text": text_t,
        "tags": tag_t,
        "numeric": num_t,
    }

    best_val_ndcg = -1.0
    best_epoch = -1
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0

    history_log: list[dict[str, Any]] = []

    logger.info(
        "Beginning Two-Tower training for %d epochs (batch_size=%d, lr=%.4f, patience=%d)...",
        epochs,
        batch_size,
        lr,
        patience,
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        module.train()

        # Shuffle pairs or sample subset
        if pairs_per_epoch is not None and pairs_per_epoch < n_total_pairs:
            sample_idx = np.random.choice(n_total_pairs, size=pairs_per_epoch, replace=False)
            epoch_sub = torch.utils.data.Subset(full_dataset, sample_idx)
            loader = DataLoader(epoch_sub, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        epoch_loss = 0.0
        n_batches = 0

        for b_hist, b_mask, b_target in loader:
            optimizer.zero_grad()

            # Target item features
            t_text = text_t[b_target]
            t_tags = tag_t[b_target]
            t_num = num_t[b_target]

            # History item features
            h_text = text_t[b_hist]
            h_tags = tag_t[b_hist]
            h_num = num_t[b_hist]

            # Tower encodings
            target_vecs = module.encode_items(
                item_ids=b_target,
                text_embs=t_text,
                tag_features=t_tags,
                numeric_features=t_num,
                apply_proxy_dropout=(cfg.proxy_dropout > 0 if cfg else False),
            )

            user_vecs = module.encode_users(
                history_ids=b_hist,
                history_text=h_text,
                history_tags=h_tags,
                history_numeric=h_num,
                mask=b_mask,
            )

            loss = in_batch_sampled_softmax_loss(
                user_vecs=user_vecs,
                item_vecs=target_vecs,
                target_ids=b_target,
                tau=module.tau,
                log_q=log_q_t,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        current_tau = float(module.tau.item())
        epoch_sec = time.perf_counter() - epoch_start

        if epoch_sec > 600.0:
            logger.warning("Epoch took %.1fs (>10min). Consider reducing --pairs-per-epoch or batch size.", epoch_sec)

        # 4. Validation Evaluation on Tuning Users
        model_wrapper.cached_item_vecs = None  # refresh cache
        model_wrapper.precompute_item_vectors()

        val_res = evaluate_model(
            model=model_wrapper,
            data=data,
            split="val",
            k_list=(10, 20),
            user_indices=tuning_users,
            max_history=cfg.max_history,
            seed=seed,
            allow_small_user_set=(len(tuning_users) < 1000),
        )

        val_ndcg10 = val_res.metrics.get("ndcg_at_10", 0.0)
        val_hr10 = val_res.metrics.get("hit_rate_at_10", 0.0)

        logger.info(
            "Epoch %2d/%2d [%.1fs] Loss: %.4f | tau: %.4f | Val NDCG@10: %.4f | Val HR@10: %.4f",
            epoch,
            epochs,
            epoch_sec,
            avg_loss,
            current_tau,
            val_ndcg10,
            val_hr10,
        )

        # Log metrics to MLflow if run is active
        try:
            if mlflow.active_run():
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "tau": current_tau,
                    "tuning_val_ndcg_at_10": val_ndcg10,
                    "tuning_val_hr_at_10": val_hr10,
                    "epoch_seconds": epoch_sec,
                    "lr": scheduler.get_last_lr()[0],
                }, step=epoch)
        except Exception:
            pass

        lr_val = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else lr
        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "tau": current_tau,
            "val_ndcg_at_10": val_ndcg10,
            "val_hr_at_10": val_hr10,
            "learning_rate": lr_val,
            "epoch_sec": round(epoch_sec, 2),
            "wall_time": round(epoch_sec, 2),
            "eval_ndcg": val_ndcg10,
        }
        history_log.append(epoch_stats)

        # Checkpoint selection
        if val_ndcg10 > best_val_ndcg:
            best_val_ndcg = val_ndcg10
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in module.state_dict().items()}
            patience_counter = 0
            logger.info("  ★ New best checkpoint at epoch %d (Val NDCG@10: %.4f)", epoch, val_ndcg10)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        module.load_state_dict(best_state_dict)
        logger.info("Restored best model checkpoint from epoch %d (Val NDCG@10: %.4f)", best_epoch, best_val_ndcg)

    model_wrapper.cached_item_vecs = None
    model_wrapper.precompute_item_vectors()

    training_curves = {
        "best_epoch": best_epoch,
        "best_tuning_val_ndcg_at_10": best_val_ndcg,
        "epochs_completed": len(history_log),
        "history": history_log,
        "curves": history_log,
    }
    model_wrapper.training_curves = training_curves

    return model_wrapper, training_curves
