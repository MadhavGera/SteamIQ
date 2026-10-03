"""
PyTorch Two-Tower Hybrid Retrieval Model with Content Embeddings and Cold-Start ID Dropout.

Architecture:
  1. ItemFeatureEncoder (shared):
     - Item ID Embedding (dim 64) with learned 'unknown item' embedding for cold start.
     - MiniLM Text Projection: Linear(384 -> 64) + LayerNorm.
     - Tag Multi-Hot Projection: Linear(300 -> 32).
     - Standardized Numeric Projection: Linear(F -> 16).
     - Output: Raw representation concatenation (dim 176).
  2. ID Dropout:
     - Replaces item ID with unknown item embedding at training time with probability id_dropout (default 0.25).
  3. ItemTower:
     - Linear(176 -> 256) -> GELU -> Dropout(0.1) -> Linear(256 -> 64) -> L2 normalize.
  4. UserTower:
     - Sequential history encoding using shared ItemFeatureEncoder.
     - History pooling via learned attention (attn) or masked mean.
     - Linear(176 -> 256) -> GELU -> Linear(256 -> 64) -> L2 normalize.
  5. TwoTowerModel(BaseRecommender):
     - Wraps PyTorch module for fold-in scoring against precomputed item representations.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.data import RecsysData
from recsys.models.base import BaseRecommender

logger = logging.getLogger("steamiq.recsys.two_tower")


@dataclass
class TwoTowerConfig:
    n_items: int = 0
    text_dim: int = 384
    tag_dim: int = 300
    numeric_dim: int = 9
    id_dim: int = 64
    text_proj_dim: int = 64
    tag_proj_dim: int = 32
    numeric_proj_dim: int = 16
    hidden_dim: int = 256
    output_dim: int = 64
    pool: str = "attn"  # "attn" or "mean"
    id_dropout: float = 0.25
    use_id: bool = True
    use_text: bool = True
    use_tags: bool = True
    use_numeric: bool = True
    max_history: int = 50
    init_tau: float = 0.07

    @property
    def raw_item_dim(self) -> int:
        return (
            self.id_dim
            + self.text_proj_dim
            + self.tag_proj_dim
            + self.numeric_proj_dim
        )


def extract_standardized_numeric_features(items_df: pd.DataFrame) -> np.ndarray:
    """
    Extracts and standardizes 9 numerical/platform features from items_df.
    Features:
      0: log1p(price_final)
      1: positive_ratio / 100.0
      2: log1p(user_reviews)
      3: log1p(train_positives)
      4: win (0/1)
      5: mac (0/1)
      6: linux (0/1)
      7: steam_deck (0/1)
      8: normalized_release_year: (year - 2015) / 10.0
    """
    n = len(items_df)
    features = np.zeros((n, 9), dtype=np.float32)

    # 0. Price
    if "price_final" in items_df.columns:
        prices = items_df["price_final"].fillna(0.0).to_numpy(dtype=np.float32)
    elif "price" in items_df.columns:
        prices = items_df["price"].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        prices = np.zeros(n, dtype=np.float32)
    features[:, 0] = np.log1p(np.maximum(prices, 0.0))

    # 1. Positive review ratio
    if "positive_ratio" in items_df.columns:
        pos_ratio = items_df["positive_ratio"].fillna(50.0).to_numpy(dtype=np.float32)
    elif "review_score" in items_df.columns:
        pos_ratio = items_df["review_score"].fillna(50.0).to_numpy(dtype=np.float32)
    else:
        pos_ratio = np.full(n, 50.0, dtype=np.float32)
    features[:, 1] = pos_ratio / 100.0

    # 2. Total user reviews count
    if "user_reviews" in items_df.columns:
        reviews = items_df["user_reviews"].fillna(0.0).to_numpy(dtype=np.float32)
    elif "total_reviews" in items_df.columns:
        reviews = items_df["total_reviews"].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        reviews = np.zeros(n, dtype=np.float32)
    features[:, 2] = np.log1p(np.maximum(reviews, 0.0))

    # 3. Train positives count
    if "train_positives" in items_df.columns:
        pop = items_df["train_positives"].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        pop = np.zeros(n, dtype=np.float32)
    features[:, 3] = np.log1p(np.maximum(pop, 0.0))

    # 4-7. Platform indicators
    for idx, col in enumerate(["win", "mac", "linux", "steam_deck"], start=4):
        if col in items_df.columns:
            features[:, idx] = items_df[col].fillna(0).astype(np.float32)
        else:
            features[:, idx] = 1.0 if col == "win" else 0.0

    # 8. Release year
    if "date_release" in items_df.columns:
        dates = pd.to_datetime(items_df["date_release"], errors="coerce")
        years = dates.dt.year.fillna(2018).to_numpy(dtype=np.float32)
    else:
        years = np.full(n, 2018.0, dtype=np.float32)
    features[:, 8] = (years - 2015.0) / 10.0

    return features


class ItemFeatureEncoder(nn.Module):
    """
    Unified item feature encoder producing raw representation (dim 176).
    Shared between UserTower and ItemTower.
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_items = cfg.n_items
        self.unknown_id = cfg.n_items  # index reserved for unknown / masked item ID

        # 1. ID Embedding (n_items + 1 embeddings)
        self.id_embedding = nn.Embedding(cfg.n_items + 1, cfg.id_dim)

        # 2. Text Projection
        self.text_linear = nn.Linear(cfg.text_dim, cfg.text_proj_dim)
        self.text_norm = nn.LayerNorm(cfg.text_proj_dim)

        # 3. Tags Projection
        self.tag_linear = nn.Linear(cfg.tag_dim, cfg.tag_proj_dim)

        # 4. Numeric Projection
        self.numeric_linear = nn.Linear(cfg.numeric_dim, cfg.numeric_proj_dim)

    def forward(
        self,
        item_ids: torch.Tensor,
        text_embs: torch.Tensor | None = None,
        tag_features: torch.Tensor | None = None,
        numeric_features: torch.Tensor | None = None,
        mask_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: LongTensor of item indices [..., L]
            text_embs: FloatTensor of text embeddings [..., text_dim]
            tag_features: FloatTensor of tag multi-hot features [..., tag_dim]
            numeric_features: FloatTensor of numeric features [..., numeric_dim]
            mask_ids: Optional 1D LongTensor of item IDs that should use the unknown ID embedding

        Returns:
            Raw item representation tensor [..., raw_item_dim]
        """
        device = item_ids.device
        shape = item_ids.shape

        # ── 1. ID Representation with Dropout / Masking ──
        if self.cfg.use_id:
            effective_ids = torch.clamp(item_ids, 0, self.n_items)

            if self.training and self.cfg.id_dropout > 0:
                drop_mask = torch.rand(shape, device=device) < self.cfg.id_dropout
                effective_ids = torch.where(
                    drop_mask,
                    torch.tensor(self.unknown_id, device=device),
                    effective_ids,
                )
            elif mask_ids is not None and mask_ids.numel() > 0:
                mask = torch.isin(effective_ids, mask_ids)
                effective_ids = torch.where(
                    mask,
                    torch.tensor(self.unknown_id, device=device),
                    effective_ids,
                )

            id_repr = self.id_embedding(effective_ids)
        else:
            id_repr = torch.zeros(*shape, self.cfg.id_dim, device=device)

        # ── 2. Text Representation ──
        if self.cfg.use_text and text_embs is not None:
            text_repr = self.text_norm(self.text_linear(text_embs))
        else:
            text_repr = torch.zeros(*shape, self.cfg.text_proj_dim, device=device)

        # ── 3. Tag Representation ──
        if self.cfg.use_tags and tag_features is not None:
            tag_repr = self.tag_linear(tag_features)
        else:
            tag_repr = torch.zeros(*shape, self.cfg.tag_proj_dim, device=device)

        # ── 4. Numeric Representation ──
        if self.cfg.use_numeric and numeric_features is not None:
            num_repr = self.numeric_linear(numeric_features)
        else:
            num_repr = torch.zeros(*shape, self.cfg.numeric_proj_dim, device=device)

        # Concatenate branches -> raw item representation [..., 176]
        return torch.cat([id_repr, text_repr, tag_repr, num_repr], dim=-1)


class ItemTower(nn.Module):
    """
    Item Tower mapping raw representations to L2-normalized 64-d embeddings.
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.raw_item_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, raw_item_repr: torch.Tensor) -> torch.Tensor:
        out = self.net(raw_item_repr)
        return F.normalize(out, p=2, dim=-1)


class UserTower(nn.Module):
    """
    User Tower pooling sequential history items into L2-normalized 64-d embeddings.
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.raw_dim = cfg.raw_item_dim

        if cfg.pool == "attn":
            self.attn_w = nn.Linear(self.raw_dim, 64)
            self.attn_v = nn.Linear(64, 1, bias=False)

        self.net = nn.Sequential(
            nn.Linear(self.raw_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(
        self,
        history_raw_repr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            history_raw_repr: [B, L, raw_dim]
            mask: BoolTensor [B, L] where True indicates a valid non-padded item.

        Returns:
            L2-normalized user embedding [B, output_dim]
        """
        b, length, _ = history_raw_repr.shape
        mask_f = mask.unsqueeze(-1).float()  # [B, L, 1]
        history_raw_repr = history_raw_repr * mask_f

        if self.cfg.pool == "attn":
            # Self-attention pooling: w^T tanh(W h)
            scores = self.attn_v(torch.tanh(self.attn_w(history_raw_repr)))  # [B, L, 1]
            # Mask out padding items
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            weights = F.softmax(scores, dim=1)  # [B, L, 1]
            weights = weights * mask_f
            w_sum = torch.sum(weights, dim=1, keepdim=True)
            weights = weights / torch.clamp(w_sum, min=1e-8)
            weights = torch.nan_to_num(weights, nan=0.0)
            pooled = torch.sum(weights * history_raw_repr, dim=1)  # [B, raw_dim]
        else:
            # Masked mean pooling
            sum_h = torch.sum(history_raw_repr, dim=1)
            counts = torch.clamp(torch.sum(mask_f, dim=1), min=1.0)
            pooled = sum_h / counts  # [B, raw_dim]

        out = self.net(pooled)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModule(nn.Module):
    """
    Complete Two-Tower PyTorch network integrating encoder, item tower, and user tower.
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = ItemFeatureEncoder(cfg)
        self.item_tower = ItemTower(cfg)
        self.user_tower = UserTower(cfg)

        # Learnable temperature parameter initialized to math.log(init_tau)
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.init_tau), dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau), min=0.01, max=0.5)

    def encode_items(
        self,
        item_ids: torch.Tensor,
        text_embs: torch.Tensor | None = None,
        tag_features: torch.Tensor | None = None,
        numeric_features: torch.Tensor | None = None,
        mask_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw_repr = self.feature_encoder(
            item_ids=item_ids,
            text_embs=text_embs,
            tag_features=tag_features,
            numeric_features=numeric_features,
            mask_ids=mask_ids,
        )
        return self.item_tower(raw_repr)

    def encode_users(
        self,
        history_ids: torch.Tensor,
        history_text: torch.Tensor | None = None,
        history_tags: torch.Tensor | None = None,
        history_numeric: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = history_ids >= 0
            history_ids = torch.clamp(history_ids, min=0)

        raw_repr = self.feature_encoder(
            item_ids=history_ids,
            text_embs=history_text,
            tag_features=history_tags,
            numeric_features=history_numeric,
        )
        return self.user_tower(raw_repr, mask)

    def encode_user(
        self,
        history_ids: torch.Tensor,
        history_text: torch.Tensor | None = None,
        history_tags: torch.Tensor | None = None,
        history_numeric: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for encode_users."""
        return self.encode_users(
            history_ids=history_ids,
            history_text=history_text,
            history_tags=history_tags,
            history_numeric=history_numeric,
            mask=mask,
        )


class TwoTowerModel(BaseRecommender):
    """
    Production-ready Two-Tower Recommender System complying with BaseRecommender.
    """

    name: str = "recsys_two_tower"

    def __init__(
        self,
        cfg: TwoTowerConfig | None = None,
        module: TwoTowerModule | None = None,
        item_vecs: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.module = module
        self.cached_item_vecs = item_vecs
        self._item_features_tensors: dict[str, torch.Tensor] = {}

    @property
    def config(self) -> TwoTowerConfig | None:
        return self.cfg

    @property
    def item_vecs(self) -> np.ndarray | None:
        return self.cached_item_vecs

    def fit(self, data: RecsysData) -> None:
        """
        Fitting via external train_two_tower routine or standalone initialization.
        Initializes tensors and precomputes item representations.
        """
        if self.cfg is None:
            self.cfg = TwoTowerConfig(n_items=data.n_items)
        elif self.cfg.n_items == 0:
            self.cfg.n_items = data.n_items

        if self.module is None:
            self.module = TwoTowerModule(self.cfg)

        self._prepare_item_feature_tensors(data)
        self.precompute_item_vectors()

    def _prepare_item_feature_tensors(self, data: RecsysData) -> None:
        """Precomputes and caches PyTorch item feature tensors."""
        n_items = data.n_items

        # 1. Text embeddings [n_items, 384]
        if data.item_text_emb is not None and len(data.item_text_emb) == n_items:
            t_embs = torch.from_numpy(data.item_text_emb.astype(np.float32))
        else:
            t_embs = torch.zeros((n_items, 384), dtype=torch.float32)

        # 2. Tag sparse matrix -> dense tensor [n_items, 300]
        if data.item_tags_sparse is not None and data.item_tags_sparse.shape[0] == n_items:
            if sp.issparse(data.item_tags_sparse):
                tag_dense = data.item_tags_sparse.toarray().astype(np.float32)
            else:
                tag_dense = np.asarray(data.item_tags_sparse, dtype=np.float32)
            tag_t = torch.from_numpy(tag_dense)
        else:
            tag_t = torch.zeros((n_items, 300), dtype=torch.float32)

        # 3. Numeric features [n_items, 9]
        num_arr = extract_standardized_numeric_features(data.items_df)
        num_t = torch.from_numpy(num_arr)

        self._item_features_tensors = {
            "all_ids": torch.arange(n_items, dtype=torch.long),
            "text": t_embs,
            "tags": tag_t,
            "numeric": num_t,
        }

    def precompute_item_vectors(self, mask_ids: list[int] | None = None) -> np.ndarray:
        """
        Precomputes and caches L2-normalized representations for all catalog items.
        """
        if self.module is None:
            raise RuntimeError("Module has not been initialized.")

        self.module.eval()
        with torch.no_grad():
            t_dict = self._item_features_tensors
            mask_t = torch.tensor(mask_ids, dtype=torch.long) if mask_ids else None

            item_vecs = self.module.encode_items(
                item_ids=t_dict["all_ids"],
                text_embs=t_dict["text"],
                tag_features=t_dict["tags"],
                numeric_features=t_dict["numeric"],
                mask_ids=mask_t,
            )
            vecs_np = item_vecs.cpu().numpy().astype(np.float32)
            if mask_ids is None:
                self.cached_item_vecs = vecs_np
            return vecs_np

    def item_vectors(self, mask_ids: list[int] | None = None) -> np.ndarray:
        """
        Returns item representations. If mask_ids is provided, simulates cold-start items.
        """
        if mask_ids is not None and len(mask_ids) > 0:
            return self.precompute_item_vectors(mask_ids=mask_ids)
        if self.cached_item_vecs is None:
            return self.precompute_item_vectors()
        return self.cached_item_vecs

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Scores all catalog items for a batch of user interaction histories.
        """
        if self.module is None:
            raise RuntimeError("Model has not been fitted or loaded.")

        b = len(histories)
        if b == 0:
            n_items = self.cfg.n_items if self.cfg else 0
            return np.empty((0, n_items), dtype=np.float32)

        if self.cached_item_vecs is None:
            self.precompute_item_vectors()

        self.module.eval()
        max_h = self.cfg.max_history if self.cfg else 50
        n_items = len(self.cached_item_vecs) if self.cached_item_vecs is not None else (self.cfg.n_items if (self.cfg and self.cfg.n_items > 0) else 0)

        # Pad histories into tensor [B, max_h]
        hist_arr = np.zeros((b, max_h), dtype=np.int64)
        mask_arr = np.zeros((b, max_h), dtype=bool)

        for row_idx, h in enumerate(histories):
            if len(h) > 0:
                valid = h[(h >= 0) & (h < n_items)]
                if len(valid) > 0:
                    tail = valid[-max_h:]
                    hist_arr[row_idx, -len(tail):] = tail
                    mask_arr[row_idx, -len(tail):] = True

        hist_t = torch.from_numpy(hist_arr)
        mask_t = torch.from_numpy(mask_arr)

        # Lookup feature tensors for history items
        t_text = self._item_features_tensors["text"][hist_t]
        t_tags = self._item_features_tensors["tags"][hist_t]
        t_num = self._item_features_tensors["numeric"][hist_t]

        with torch.no_grad():
            user_vecs = self.module.encode_users(
                history_ids=hist_t,
                history_text=t_text,
                history_tags=t_tags,
                history_numeric=t_num,
                mask=mask_t,
            ).cpu().numpy().astype(np.float32)

        # Dot product: [B, 64] @ [64, n_items] -> [B, n_items]
        scores = user_vecs @ self.cached_item_vecs.T
        np.nan_to_num(scores, copy=False, nan=0.0)
        return scores.astype(np.float32)

    def save(self, model_dir: Path | str) -> None:
        """
        Serializes TwoTower weights, config, item feature tensors, and precomputed item vectors to model_dir.
        """
        if self.module is None or self.cfg is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. State dict
        torch.save(self.module.state_dict(), path / "two_tower.pt")

        # 2. Config JSON
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        # 3. Cached item vectors
        if self.cached_item_vecs is None:
            self.precompute_item_vectors()
        np.save(path / "item_vecs.npy", self.cached_item_vecs)

        # 4. Item feature tensors
        if self._item_features_tensors:
            torch.save(self._item_features_tensors, path / "item_features.pt")

    @classmethod
    def load(cls, model_dir: Path | str, data: RecsysData | None = None) -> TwoTowerModel:
        """
        Deserializes TwoTowerModel from model_dir.
        """
        path = Path(model_dir)
        cfg_file = path / "config.json"
        weights_file = path / "two_tower.pt"
        item_vecs_file = path / "item_vecs.npy"
        features_file = path / "item_features.pt"

        if not cfg_file.exists() or not weights_file.exists():
            raise FileNotFoundError(f"Missing model files in {path}")

        with open(cfg_file, encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg = TwoTowerConfig(**cfg_dict)
        module = TwoTowerModule(cfg)
        module.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        module.eval()

        item_vecs = np.load(item_vecs_file) if item_vecs_file.exists() else None

        model = cls(cfg=cfg, module=module, item_vecs=item_vecs)
        if features_file.exists():
            model._item_features_tensors = torch.load(features_file, map_location="cpu", weights_only=True)
        elif data is not None:
            model._prepare_item_feature_tensors(data)
        else:
            n = cfg.n_items
            model._item_features_tensors = {
                "all_ids": torch.arange(n, dtype=torch.long),
                "text": torch.zeros((n, cfg.text_dim), dtype=torch.float32),
                "tags": torch.zeros((n, cfg.tag_dim), dtype=torch.float32),
                "numeric": torch.zeros((n, cfg.numeric_dim), dtype=torch.float32),
            }

        return model

    def hyperparameters(self) -> dict[str, Any]:
        if self.cfg is None:
            return {}
        return {
            "hidden_dim": self.cfg.hidden_dim,
            "output_dim": self.cfg.output_dim,
            "pool": self.cfg.pool,
            "id_dropout": self.cfg.id_dropout,
            "use_id": self.cfg.use_id,
            "use_text": self.cfg.use_text,
            "use_tags": self.cfg.use_tags,
            "use_numeric": self.cfg.use_numeric,
            "max_history": self.cfg.max_history,
        }
