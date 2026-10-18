"""
Lightweight SASRec (Self-Attentive Sequential Recommendation) Model.

Architecture:
  - Causal multi-head self-attention over chronologically ordered positive interactions.
  - 2 TransformerEncoder layers, 2 attention heads, d_model = 64, max_seq_len = 50, dropout = 0.2.
  - Learned item ID embedding (vocab = n_items + 1, padding_idx = 0).
  - Optional projected MiniLM text embedding (Linear(384 -> 64) + LayerNorm).
  - Learned positional embedding (50 x 64).
  - Right-aligned input sequences with causal attention mask.
  - In-batch sampled softmax with logQ popularity correction.
  - Precomputable item vectors for sub-10ms online inference.

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
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.data import RecsysData
from recsys.models.base import BaseRecommender

logger = logging.getLogger("steamiq.recsys.sequential")


@dataclass
class SASRecConfig:
    n_items: int = 0
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    max_len: int = 50
    dropout: float = 0.2
    dim_feedforward: int = 256
    use_text: bool = False
    text_dim: int = 384
    init_tau: float = 0.07

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SASRecConfig:
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class SASRecModule(nn.Module):
    """
    PyTorch core module for SASRec.
    """

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.n_items = config.n_items
        self.d_model = config.d_model
        self.max_len = config.max_len

        # Token 0 is padding, 1..n_items are games
        self.item_embedding = nn.Embedding(
            config.n_items + 1, config.d_model, padding_idx=0
        )
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)

        if config.use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(config.text_dim, config.d_model),
                nn.LayerNorm(config.d_model),
            )
        else:
            self.text_proj = None

        self.emb_layernorm = nn.LayerNorm(config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.log_tau = nn.Parameter(torch.tensor(math.log(config.init_tau), dtype=torch.float32))

        # Register causal mask buffer
        causal = torch.triu(torch.full((config.max_len, config.max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def encode_history(
        self,
        seq_ids: torch.Tensor,
        text_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes user interaction sequences into user vectors.

        Args:
            seq_ids: LongTensor [B, max_len], right-aligned, 0 = padding, 1..N = items.
            text_embs: Optional tensor of text embeddings for items in sequence.

        Returns:
            user_vectors: FloatTensor [B, d_model], L2-normalized.
        """
        device = seq_ids.device
        item_emb = self.item_embedding(seq_ids)

        if self.text_proj is not None and text_embs is not None:
            text_proj_emb = self.text_proj(text_embs)
            item_emb = item_emb + text_proj_emb

        pos_ids = torch.arange(self.max_len, device=device).unsqueeze(0)
        x = self.emb_layernorm(item_emb + self.pos_embedding(pos_ids))
        x = self.emb_dropout(x)

        causal_mask = self.causal_mask[: self.max_len, : self.max_len]
        out = self.encoder(x, mask=causal_mask)

        # Extract sequence representation from the rightmost (last) position
        user_vec = out[:, -1, :]
        return F.normalize(user_vec, p=2, dim=-1)

    def encode_target(
        self,
        target_ids: torch.Tensor,
        text_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes target item indices [B] into target vectors [B, d_model].
        target_ids are 0-indexed catalog IDs [0..n_items-1].
        """
        tokens = target_ids + 1
        item_emb = self.item_embedding(tokens)
        if self.text_proj is not None and text_embs is not None:
            item_emb = item_emb + self.text_proj(text_embs)
        return F.normalize(item_emb, p=2, dim=-1)

    def all_item_vectors(self, all_text_embs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encodes all catalog items [1..n_items] into matrix [n_items, d_model].
        """
        device = self.item_embedding.weight.device
        all_tokens = torch.arange(1, self.n_items + 1, device=device)
        item_embs = self.item_embedding(all_tokens)
        if self.text_proj is not None and all_text_embs is not None:
            item_embs = item_embs + self.text_proj(all_text_embs)
        return F.normalize(item_embs, p=2, dim=-1)


class SASRecModel(BaseRecommender):
    """
    SASRec Sequential Recommender wrapping SASRecModule for training and fold-in scoring.
    """

    name: str = "recsys_sasrec"

    def __init__(self, config: SASRecConfig | None = None) -> None:
        self.config = config or SASRecConfig()
        self.module: SASRecModule | None = None
        self._item_vectors: np.ndarray | None = None
        self._text_embs_tensor: torch.Tensor | None = None
        self._text_table: torch.Tensor | None = None
        self.device = torch.device("cpu")
        self.training_time_sec: float = 0.0
        self.epoch_sec_list: list[float] = []

    def fit(self, data: RecsysData) -> None:
        """
        Fits SASRec from RecsysData. (For multi-epoch training, see train_sasrec).
        """
        self.config.n_items = int(data.n_items)
        self.module = SASRecModule(self.config).to(self.device)
        self.precompute_item_vectors(data)

    def precompute_item_vectors(self, data: RecsysData | None = None) -> np.ndarray:
        """
        Precomputes and caches L2-normalized representations for all catalog items.
        """
        if self.module is None:
            raise RuntimeError("SASRecModule is not initialized.")
        if data is not None and data.item_text_emb is not None and self.config.use_text:
            self._text_embs_tensor = torch.from_numpy(data.item_text_emb.astype(np.float32)).to(self.device)
            self._text_table = torch.zeros(
                (data.n_items + 1, data.item_text_emb.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            self._text_table[1:] = self._text_embs_tensor
        self.module.eval()
        with torch.no_grad():
            vecs = self.module.all_item_vectors(self._text_embs_tensor)
            self._item_vectors = vecs.cpu().numpy().astype(np.float32)
        return self._item_vectors

    def score(
        self,
        histories: list[np.ndarray],
        item_vecs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Scores all catalog items for a batch of user histories.

        Args:
            histories: List of 1D numpy arrays with user chronological positive items.
            item_vecs: Optional precomputed item vectors [n_items, d_model].

        Returns:
            scores: [len(histories), n_items] float32 array.
        """
        n_users = len(histories)
        if n_users == 0:
            return np.empty((0, self.config.n_items), dtype=np.float32)

        if self.module is None:
            raise RuntimeError("SASRecModel must be initialized before scoring.")

        max_len = self.config.max_len
        seq_arr = np.zeros((n_users, max_len), dtype=np.int64)

        for u, h in enumerate(histories):
            if len(h) > 0:
                h_recent = h[-max_len:]
                seq_arr[u, -len(h_recent) :] = h_recent + 1

        self.module.eval()
        with torch.no_grad():
            seq_tensor = torch.from_numpy(seq_arr).to(self.device)
            text_embs = None
            if self.config.use_text and self._text_table is not None:
                text_embs = F.embedding(seq_tensor, self._text_table)
            user_vecs = self.module.encode_history(seq_tensor, text_embs=text_embs).cpu().numpy()

        items = item_vecs if item_vecs is not None else self._item_vectors
        if items is None:
            items = self.precompute_item_vectors()

        # Dot product scoring against all catalog items: [n_users, n_items]
        scores = np.matmul(user_vecs, items.T)
        return scores.astype(np.float32)

    def parameter_count(self) -> int:
        """Returns total trainable parameter count."""
        if self.module is None:
            return 0
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def save(self, model_dir: Path | str) -> None:
        """Saves model weights and config to directory."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        if self.module is not None:
            torch.save(self.module.state_dict(), path / "sasrec.pt")

        if self._item_vectors is not None:
            np.save(path / "item_vectors.npy", self._item_vectors)

        meta = {
            "training_time_sec": self.training_time_sec,
            "epoch_sec_list": self.epoch_sec_list,
            "parameter_count": self.parameter_count(),
        }
        with open(path / "metrics_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str, data: RecsysData | None = None) -> SASRecModel:
        """Loads model from directory."""
        path = Path(model_dir)
        with open(path / "config.json", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = SASRecConfig.from_dict(cfg_dict)
        instance = cls(config)
        instance.module = SASRecModule(config).to(instance.device)

        weights_file = path / "sasrec.pt"
        if weights_file.exists():
            state_dict = torch.load(weights_file, map_location=instance.device)
            instance.module.load_state_dict(state_dict)

        if data is not None and data.item_text_emb is not None and instance.config.use_text:
            instance._text_embs_tensor = torch.from_numpy(data.item_text_emb.astype(np.float32)).to(instance.device)
            instance._text_table = torch.zeros(
                (data.n_items + 1, data.item_text_emb.shape[1]),
                dtype=torch.float32,
                device=instance.device,
            )
            instance._text_table[1:] = instance._text_embs_tensor

        vecs_file = path / "item_vectors.npy"
        if vecs_file.exists():
            instance._item_vectors = np.load(vecs_file)
        elif data is not None:
            instance.precompute_item_vectors(data)

        meta_file = path / "metrics_meta.json"
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            instance.training_time_sec = meta.get("training_time_sec", 0.0)
            instance.epoch_sec_list = meta.get("epoch_sec_list", [])

        return instance

    def hyperparameters(self) -> dict[str, Any]:
        """Returns hyperparameter dictionary."""
        return {
            "name": self.name,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "max_len": self.config.max_len,
            "dropout": self.config.dropout,
            "use_text": self.config.use_text,
            "dim_feedforward": self.config.dim_feedforward,
            "parameter_count": self.parameter_count(),
            "training_time_sec": self.training_time_sec,
        }
