"""
Online Recommender Inference Service — Phase 6.

ADR 0005: The Single Authorized Exception to ADR 0001 Decision 2 (The Golden Rule).
Executes bounded, fixed-size in-memory recommender scoring and candidate reranking at request time.
Constraints:
  1. Zero training at request time.
  2. Zero database writes.
  3. Zero raw_* or feature_* table queries.
  4. O(n_items x 64) computational budget; P95 < 1.0 s on CPU.
  5. Immutability: pre-trained artifacts loaded once into memory.
  6. Provenance metadata returned with every response.
  7. Honest 503 MODEL_NOT_READY when model artifacts are absent.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import settings
from core.errors import GameNotFoundError, ModelNotReadyError, ValidationError
from recsys.candidates import DEFAULT_CANDIDATE_COUNTS, generate_candidates
from recsys.data import RecsysData
from recsys.explain import FEATURE_CONCEPTS, FEATURE_TEMPLATES
from recsys.models.als import ALSRecommender
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerModel
from recsys.ranker import RankerModel
from recsys.ranker_features import ItemFeatureCache, UserFeatureCache, build_features

logger = logging.getLogger("steamiq.recsys.inference")

MODEL_ALIASES: dict[str, str] = {
    "popularity": "recsys_popularity",
    "recsys_popularity": "recsys_popularity",
    "content": "recsys_content",
    "recsys_content": "recsys_content",
    "itemknn": "recsys_itemknn",
    "item_knn": "recsys_itemknn",
    "recsys_itemknn": "recsys_itemknn",
    "als": "recsys_als",
    "recsys_als": "recsys_als",
    "two_tower": "recsys_two_tower_clean",
    "two_tower_clean": "recsys_two_tower_clean",
    "recsys_two_tower_clean": "recsys_two_tower_clean",
    "recsys_two_tower": "recsys_two_tower_clean",
    "ranker": "recsys_two_tower_ranker_v2_trainstats",
    "recsys_two_tower_ranker_clean": "recsys_two_tower_ranker_clean",
    "recsys_two_tower_ranker": "recsys_two_tower_ranker_clean",
    "ranker_v2": "recsys_two_tower_ranker_clean_v2",
    "ranker_clean_v2": "recsys_two_tower_ranker_clean_v2",
    "two_tower_ranker_clean_v2": "recsys_two_tower_ranker_clean_v2",
    "recsys_two_tower_ranker_clean_v2": "recsys_two_tower_ranker_clean_v2",
    "ranker_v2_trainstats": "recsys_two_tower_ranker_v2_trainstats",
    "recsys_two_tower_ranker_v2_trainstats": "recsys_two_tower_ranker_v2_trainstats",
}

DEPLOYABLE_MODELS: list[str] = [
    "recsys_popularity",
    "recsys_content",
    "recsys_itemknn",
    "recsys_als",
    "recsys_two_tower_clean",
    "recsys_two_tower_ranker_clean",
    "recsys_two_tower_ranker_clean_v2",
    "recsys_two_tower_ranker_v2_trainstats",
]

DEFAULT_MODEL_NAME = "recsys_two_tower_clean"


class RecommenderService:
    """
    In-memory, thread-safe recommender inference service.
    Loads serving catalog metadata and registered model artifacts once into memory.
    """

    _instance: RecommenderService | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = (
            Path(data_dir) if data_dir else Path(settings.resolved_recsys_data_dir)
        )
        self.data: RecsysData | None = None
        self.item_cache: ItemFeatureCache | None = None
        self.user_cache: UserFeatureCache | None = None

        # Catalog indices and metadata
        self.n_items: int = 0
        self.universe_size: int = 0
        self.app_id_to_idx: dict[int, int] = {}
        self.idx_to_app_id: dict[int, int] = {}
        self.items_metadata: dict[int, dict[str, Any]] = {}
        self.items_df: pd.DataFrame | None = None

        # Model artifacts
        self.models: dict[str, BaseRecommender] = {}
        self.model_runs_meta: dict[str, dict[str, Any]] = {}
        self.tt_item_vectors: np.ndarray | None = None  # [N, 64] L2-normalized

        # State flags
        self.is_initialized: bool = False
        self._init_lock: threading.Lock = threading.Lock()
        self.ranker_is_preview: bool = False

    @classmethod
    def get_instance(cls, data_dir: Path | str | None = None) -> RecommenderService:
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def initialize(self) -> None:
        """
        Loads item catalog and registered deployable model artifacts into memory.
        Safe for repeated calls (guarded by lock).
        """
        if self.is_initialized:
            return

        with self._init_lock:
            if self.is_initialized:
                return

            t0 = time.perf_counter()
            logger.info("Initializing RecommenderService from %s...", self.data_dir)

            if not self.data_dir.exists():
                logger.warning("Recsys data directory %s does not exist.", self.data_dir)
                return

            try:
                self.data = RecsysData.load(self.data_dir)
            except Exception as e:
                logger.warning("Failed to load RecsysData from %s: %s", self.data_dir, e)
                return

            # 1. Load and cache item universe
            if self.data.items_df is not None and not self.data.items_df.empty:
                df = self.data.items_df
                self.items_df = df
                self.n_items = len(df)
                self.universe_size = len(df)

                for _, r in df.iterrows():
                    app_id = int(r["app_id"])
                    item_idx = int(r["item_idx"])
                    self.app_id_to_idx[app_id] = item_idx
                    self.idx_to_app_id[item_idx] = app_id

                    tags_raw = r.get("tags", [])
                    if isinstance(tags_raw, np.ndarray | list):
                        tags_clean = [str(t).strip() for t in tags_raw if str(t).strip()]
                    else:
                        tags_clean = []

                    self.items_metadata[item_idx] = {
                        "app_id": app_id,
                        "item_idx": item_idx,
                        "title": str(r.get("title") or r.get("name") or f"Game #{app_id}"),
                        "date_release": str(r["date_release"]) if pd.notna(r.get("date_release")) else None,
                        "price_final": float(r["price_final"]) if pd.notna(r.get("price_final")) else 0.0,
                        "positive_ratio": int(r["positive_ratio"]) if pd.notna(r.get("positive_ratio")) else None,
                        "user_reviews": int(r["user_reviews"]) if pd.notna(r.get("user_reviews")) else None,
                        "tags": tags_clean,
                        "description_short": str(r["description_short"]) if pd.notna(r.get("description_short")) else None,
                        "popularity_rank": int(r.get("popularity_rank", item_idx + 1)),
                        "win": bool(r.get("win", False)),
                        "mac": bool(r.get("mac", False)),
                        "linux": bool(r.get("linux", False)),
                        "steam_deck": bool(r.get("steam_deck", False)),
                        "header_image_url": str(r.get("header_image_url") or ""),
                    }

            self.item_cache = ItemFeatureCache(self.data)
            self.user_cache = UserFeatureCache(self.data)

            # 2. Load latest model artifacts resolved via model_runs
            artifacts_base = self.data_dir / "artifacts"
            stage1_instances: dict[str, BaseRecommender] = {}

            # Read fallback metrics from leaderboard.json in case DB is unreachable
            lb_fallback: dict[str, dict[str, Any]] = {}
            for lb_p in [
                self.data_dir / "reports" / "leaderboard.json",
                Path("docs/results/leaderboard.json"),
                self.data_dir / "artifacts" / "recsys_leaderboard" / "leaderboard.json",
                Path("artifacts/recsys_leaderboard/leaderboard.json"),
            ]:
                if lb_p.exists():
                    try:
                        import json
                        raw_lb = json.loads(lb_p.read_text(encoding="utf-8"))
                        entries = raw_lb.get("models", raw_lb) if isinstance(raw_lb, dict) else raw_lb
                        if isinstance(entries, list):
                            for e in entries:
                                if isinstance(e, dict) and "model_name" in e:
                                    val_n = e.get("val_metrics", {}).get("ndcg_at_10") if isinstance(e.get("val_metrics"), dict) else e.get("val_ndcg_at_10", 0.0)
                                    test_n = e.get("test_metrics", {}).get("ndcg_at_10") if isinstance(e.get("test_metrics"), dict) else e.get("test_ndcg_at_10", 0.0)
                                    e_copy = dict(e)
                                    e_copy["val_ndcg_at_10"] = float(val_n) if val_n is not None else 0.0
                                    e_copy["test_ndcg_at_10"] = float(test_n) if test_n is not None else 0.0
                                    lb_fallback[e["model_name"]] = e_copy
                            break
                    except Exception:
                        pass

            db_runs_map: dict[str, Any] = {}
            try:
                from recsys.registry import get_latest_recsys_runs_sync
                latest_db_runs = get_latest_recsys_runs_sync(dataset_version=self.data.dataset_version if self.data else None)
                for r in latest_db_runs:
                    db_runs_map[r.model_name] = r
            except Exception as e:
                logger.debug("Database model_runs lookup fallback: %s", e)

            def resolve_model_artifact(model_name: str) -> tuple[Path | None, dict[str, Any]]:
                meta: dict[str, Any] = {}
                p: Path | None = None
                if model_name in db_runs_map:
                    run_obj = db_runs_map[model_name]
                    candidate_p = Path(run_obj.artifact_path)
                    if candidate_p.exists():
                        p = candidate_p
                        m = run_obj.metrics or {}
                        val_n = m.get("val_metrics", {}).get("ndcg_at_10") if isinstance(m.get("val_metrics"), dict) else m.get("val_ndcg_at_10", 0.0)
                        test_n = m.get("test_metrics", {}).get("ndcg_at_10") if isinstance(m.get("test_metrics"), dict) else m.get("test_ndcg_at_10", 0.0)
                        meta = {
                            "artifact_path": str(p),
                            "model_run_id": str(run_obj.id),
                            "stage": run_obj.stage,
                            "val_ndcg_at_10": float(val_n) if val_n is not None else 0.0,
                            "test_ndcg_at_10": float(test_n) if test_n is not None else 0.0,
                        }

                if p is None:
                    # Fallback to filesystem if not in db
                    art_dir = artifacts_base / model_name
                    if art_dir.exists():
                        cand_dirs = sorted([d for d in art_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                        if cand_dirs:
                            p = cand_dirs[0]
                            meta = {"artifact_path": str(p), "stage": "candidate"}

                # If missing model_run_id or val_ndcg_at_10 <= 0, enrich from leaderboard fallback
                if p is not None:
                    lb_entry = lb_fallback.get(model_name) or (
                        lb_fallback.get("recsys_two_tower_ranker_v2_trainstats", {})
                        if model_name == "recsys_two_tower_ranker_clean_v2"
                        else {}
                    )
                    if not meta.get("model_run_id"):
                        meta["model_run_id"] = lb_entry.get("model_run_id") or p.name
                    if not meta.get("val_ndcg_at_10") or meta["val_ndcg_at_10"] <= 0.0:
                        meta["val_ndcg_at_10"] = float(lb_entry.get("val_ndcg_at_10", 0.0))
                    if not meta.get("test_ndcg_at_10") or meta["test_ndcg_at_10"] <= 0.0:
                        meta["test_ndcg_at_10"] = float(lb_entry.get("test_ndcg_at_10_mean", lb_entry.get("test_ndcg_at_10", 0.0)))
                return p, meta

            # Load stage 1 models
            # (a) Popularity
            pop_dir, pop_meta = resolve_model_artifact("recsys_popularity")
            if pop_dir and pop_dir.exists():
                try:
                    self.models["recsys_popularity"] = PopularityRecommender.load(pop_dir)
                    self.model_runs_meta["recsys_popularity"] = pop_meta
                except Exception as e:
                    logger.warning("Could not load Popularity model: %s", e)

            # (b) Item-kNN
            knn_dir, knn_meta = resolve_model_artifact("recsys_itemknn")
            if knn_dir and knn_dir.exists():
                try:
                    m = ItemKNNRecommender.load(knn_dir)
                    self.models["recsys_itemknn"] = m
                    stage1_instances["itemknn"] = m
                    self.model_runs_meta["recsys_itemknn"] = knn_meta
                except Exception as e:
                    logger.warning("Could not load Item-kNN model: %s", e)

            # (c) ALS
            als_dir, als_meta = resolve_model_artifact("recsys_als")
            if als_dir and als_dir.exists():
                try:
                    m = ALSRecommender.load(als_dir)
                    self.models["recsys_als"] = m
                    stage1_instances["als"] = m
                    self.model_runs_meta["recsys_als"] = als_meta
                except Exception as e:
                    logger.warning("Could not load ALS model: %s", e)

            # (d) Content
            cnt_dir, cnt_meta = resolve_model_artifact("recsys_content")
            if cnt_dir and cnt_dir.exists():
                try:
                    m = ContentRecommender.load(cnt_dir)
                    self.models["recsys_content"] = m
                    stage1_instances["content"] = m
                    self.model_runs_meta["recsys_content"] = cnt_meta
                except Exception as e:
                    logger.warning("Could not load Content model: %s", e)

            # (e) Clean Two-Tower
            tt_dir, tt_meta = resolve_model_artifact("recsys_two_tower_clean")
            if not tt_dir:
                tt_dir, tt_meta = resolve_model_artifact("recsys_two_tower")
            if tt_dir and tt_dir.exists():
                try:
                    m = TwoTowerModel.load(tt_dir, data=self.data)
                    self.models["recsys_two_tower_clean"] = m
                    stage1_instances["two_tower"] = m
                    self.tt_item_vectors = m.precompute_item_vectors()
                    self.model_runs_meta["recsys_two_tower_clean"] = tt_meta
                except Exception as e:
                    logger.warning("Could not load Two-Tower model: %s", e)

            # (f) Clean Ranker
            ranker_dir, ranker_meta = resolve_model_artifact("recsys_two_tower_ranker_clean")
            if not ranker_dir:
                ranker_dir, ranker_meta = resolve_model_artifact("recsys_two_tower_ranker")
            if ranker_dir and ranker_dir.exists() and "two_tower" in stage1_instances:
                try:
                    m = RankerModel.load(ranker_dir, data=self.data, models=stage1_instances)
                    self.models["recsys_two_tower_ranker_clean"] = m
                    self.model_runs_meta["recsys_two_tower_ranker_clean"] = ranker_meta
                except Exception as e:
                    logger.warning("Could not load Ranker model: %s", e)

            # (g) Clean Ranker V2 (Strictly Leak-Free Two-Tower Stage 1)
            ranker_v2_dir, ranker_v2_meta = resolve_model_artifact("recsys_two_tower_ranker_clean_v2")
            if ranker_v2_dir and ranker_v2_dir.exists() and "two_tower" in stage1_instances:
                try:
                    m_v2 = RankerModel.load(ranker_v2_dir, data=self.data, models=stage1_instances)
                    self.models["recsys_two_tower_ranker_clean_v2"] = m_v2
                    self.model_runs_meta["recsys_two_tower_ranker_clean_v2"] = ranker_v2_meta
                except Exception as e:
                    logger.warning("Could not load Ranker V2 model: %s", e)

            # (h) Clean Ranker V2 TrainStats
            ranker_ts_dir, ranker_ts_meta = resolve_model_artifact("recsys_two_tower_ranker_v2_trainstats")
            if ranker_ts_dir and ranker_ts_dir.exists() and "two_tower" in stage1_instances:
                try:
                    m_ts = RankerModel.load(ranker_ts_dir, data=self.data, models=stage1_instances)
                    self.models["recsys_two_tower_ranker_v2_trainstats"] = m_ts
                    self.model_runs_meta["recsys_two_tower_ranker_v2_trainstats"] = ranker_ts_meta
                except Exception as e:
                    logger.warning("Could not load Ranker V2 TrainStats model: %s", e)

            self.is_initialized = True
            init_time = (time.perf_counter() - t0) * 1000
            logger.info(
                "RecommenderService initialized in %.1f ms. Loaded %d models: %s",
                init_time,
                len(self.models),
                list(self.models.keys()),
            )

    def is_ready(self) -> bool:
        """Returns True if the service is initialized and the default model is loaded."""
        if not self.is_initialized:
            self.initialize()
        return self.is_initialized and (DEFAULT_MODEL_NAME in self.models)

    def resolve_model_name(self, model_input: str | None) -> str:
        """Resolves user model alias to canonical deployable model name."""
        if not model_input:
            return DEFAULT_MODEL_NAME
        clean = model_input.strip().lower()
        if clean in MODEL_ALIASES:
            return MODEL_ALIASES[clean]
        valid_options = sorted(list(set(MODEL_ALIASES.keys())))
        raise ValidationError(
            f"Unknown model '{model_input}'. Valid deployable models: {valid_options}"
        )

    def recommend(
        self,
        liked_app_ids: list[int],
        model: str | None = None,
        k: int = 20,
        explain: bool = True,
    ) -> dict[str, Any]:
        """
        Computes personalized recommendations for user-provided library games.

        Args:
            liked_app_ids: List of Steam App IDs the user played / liked.
            model: Optional model identifier or alias.
            k: Number of recommendations to return (1..50).
            explain: Whether to synthesize explainability reasons.

        Returns:
            Structured recommendation dictionary.
        """
        t0 = time.perf_counter()

        if not self.is_initialized:
            self.initialize()

        # 1. Validate inputs
        if not liked_app_ids:
            raise ValidationError("liked_app_ids list cannot be empty.")

        if len(liked_app_ids) > 50:
            raise ValidationError("liked_app_ids cannot exceed 50 items (max library size).")

        k_clamped = max(1, min(50, int(k)))
        canonical_model = self.resolve_model_name(model)

        if canonical_model not in self.models:
            raise ModelNotReadyError(
                f"Model '{canonical_model}' is not ready or artifact is not loaded in memory.",
                details={"model": canonical_model, "available_models": list(self.models.keys())},
            )

        model_inst = self.models[canonical_model]

        # 2. Map app_ids to catalog indices
        known_indices: list[int] = []
        unknown_app_ids: list[int] = []
        for aid in liked_app_ids:
            if aid in self.app_id_to_idx:
                known_indices.append(self.app_id_to_idx[aid])
            else:
                unknown_app_ids.append(aid)

        if not known_indices:
            raise ValidationError(
                f"None of the provided app_ids exist in the {self.universe_size}-game universe. "
                f"Unknown IDs: {unknown_app_ids}"
            )

        warning: str | None = None
        if len(known_indices) < 3:
            warning = "Fewer than 3 known library games provided. Add more games for better personalization."

        # Cap library fold-in history to 50 items
        foldin_history = np.asarray(known_indices[:50], dtype=np.int64)

        # 3. Score through BaseRecommender
        scores = model_inst.score([foldin_history])[0].copy()

        # 4. Mask liked items so they are never recommended
        scores[known_indices] = -np.inf

        # 5. Top-k items
        top_k_indices = np.argsort(-scores)[:k_clamped]

        # 6. Synthesize explanations
        recommended_items: list[dict[str, Any]] = []

        # Precompute candidate features / SHAP if ranker and explain=True
        ranker_shap_map: dict[int, list[tuple[str, float]]] = {}
        if explain and "ranker" in canonical_model and isinstance(model_inst, RankerModel):
            try:
                # Generate candidate pool & features for this single user
                cand_res = generate_candidates(
                    histories=[foldin_history],
                    models=model_inst.models,
                    n_each=DEFAULT_CANDIDATE_COUNTS,
                    max_candidates=model_inst.config.max_candidates,
                )
                feat_df = build_features(
                    histories=[foldin_history],
                    candidates=cand_res,
                    data=self.data,
                    item_cache=self.item_cache,
                    user_cache=self.user_cache,
                )
                if not feat_df.empty and model_inst.ranker is not None:
                    active_cols = model_inst.feature_names
                    X_mat = feat_df[active_cols].to_numpy(dtype=np.float32)
                    c_items = cand_res.user_candidates[0]
                    # Compute fast feature importances / SHAP approximation from tree weights
                    booster = model_inst.ranker.booster_ if hasattr(model_inst.ranker, "booster_") else model_inst.ranker
                    if hasattr(booster, "predict"):
                        # LightGBM contrib shape: [n_candidates, n_features + 1]
                        contribs = booster.predict(X_mat, pred_contrib=True)
                        for row_idx, item_id in enumerate(c_items):
                            c_vals = contribs[row_idx, :-1]
                            ranker_shap_map[item_id] = list(zip(active_cols, c_vals, strict=False))
            except Exception as e:
                logger.debug("SHAP explanation synthesis fallback: %s", e)

        for rank_num, item_idx in enumerate(top_k_indices, start=1):
            i_int = int(item_idx)
            raw_meta = self.items_metadata.get(i_int, {})
            app_id = raw_meta.get("app_id", self.idx_to_app_id.get(i_int, 0))
            score_val = float(scores[i_int])

            because_you_liked: list[dict[str, Any]] = []
            reasons: list[str] = []

            if explain:
                if "ranker" in canonical_model:
                    # SHAP reasons + vector cosine anchors
                    shap_list = ranker_shap_map.get(i_int, [])
                    pos_features = sorted([f for f in shap_list if f[1] > 0 and f[0] in FEATURE_TEMPLATES], key=lambda x: x[1], reverse=True)
                    seen_concepts: set[str] = set()
                    for feat_name, _ in pos_features:
                        concept = FEATURE_CONCEPTS.get(feat_name, feat_name)
                        if concept in seen_concepts:
                            continue
                        seen_concepts.add(concept)
                        t_str = FEATURE_TEMPLATES[feat_name]
                        reasons.append(t_str)
                        if len(reasons) >= 3:
                            break
                    if not reasons:
                        reasons.append("High semantic and tag similarity with games you liked/reviewed")

                    # Cosine anchors
                    because_you_liked = self._compute_cosine_anchors(i_int, foldin_history, top_n=2, min_similarity=0.5)

                elif canonical_model == "recsys_two_tower_clean":
                    # Two-Tower: because you liked from item vector space
                    because_you_liked = self._compute_cosine_anchors(i_int, foldin_history, top_n=2, min_similarity=0.5)
                    reasons = []

                elif canonical_model == "recsys_itemknn":
                    # Item-kNN: history items with highest similarity matrix weight
                    knn_model = self.models.get("recsys_itemknn")
                    if isinstance(knn_model, ItemKNNRecommender) and knn_model.similarity_matrix is not None:
                        sim_mat = knn_model.similarity_matrix
                        contributions = []
                        for h_it in foldin_history:
                            h_int = int(h_it)
                            weight = float(sim_mat[h_int, i_int]) if h_int < sim_mat.shape[0] and i_int < sim_mat.shape[1] else 0.0
                            if weight >= 0.5:
                                h_meta = self.items_metadata.get(h_int, {})
                                contributions.append({
                                    "app_id": h_meta.get("app_id", 0),
                                    "title": h_meta.get("title", f"Game #{h_int}"),
                                    "similarity": weight,
                                })
                        contributions.sort(key=lambda x: x["similarity"], reverse=True)
                        because_you_liked = contributions[:2]
                    reasons = []
                else:
                    # ALS, Content, Popularity: no invented reasons
                    because_you_liked = []
                    reasons = []

            item_dict = {
                "rank": rank_num,
                "app_id": app_id,
                "item_idx": i_int,
                "title": raw_meta.get("title", f"Game #{app_id}"),
                "header_image_url": raw_meta.get("header_image_url", ""),
                "score": score_val,
                "price_final": raw_meta.get("price_final"),
                "positive_ratio": raw_meta.get("positive_ratio"),
                "user_reviews": raw_meta.get("user_reviews"),
                "tags": raw_meta.get("tags", []),
                "description_short": raw_meta.get("description_short"),
                "popularity_rank": raw_meta.get("popularity_rank", i_int + 1),
                "win": raw_meta.get("win", False),
                "mac": raw_meta.get("mac", False),
                "linux": raw_meta.get("linux", False),
                "steam_deck": raw_meta.get("steam_deck", False),
                "because_you_liked": because_you_liked,
                "reasons": reasons,
            }
            recommended_items.append(item_dict)

        took_ms = (time.perf_counter() - t0) * 1000
        meta = self.model_runs_meta.get(canonical_model, {})

        return {
            "items": recommended_items,
            "total_items": len(recommended_items),
            "model_name": canonical_model,
            "model_run_id": meta.get("model_run_id"),
            "val_ndcg_at_10": meta.get("val_ndcg_at_10", 0.0),
            "dataset_version": self.data.dataset_version if self.data else "c662b8232dc0",
            "universe_size": self.universe_size,
            "unknown_app_ids": unknown_app_ids,
            "warning": warning,
            "took_ms": float(took_ms),
        }

    def _compute_cosine_anchors(
        self,
        target_idx: int,
        history: np.ndarray,
        top_n: int = 2,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Finds the top N items in history with highest cosine similarity to target_idx exceeding min_similarity."""
        if len(history) == 0:
            return []

        if self.tt_item_vectors is not None and target_idx < len(self.tt_item_vectors):
            t_vec = self.tt_item_vectors[target_idx]
            h_vecs = self.tt_item_vectors[history]
            sims = np.dot(h_vecs, t_vec)
        elif self.data and self.data.item_text_emb is not None and target_idx < len(self.data.item_text_emb):
            t_vec = self.data.item_text_emb[target_idx]
            t_norm = np.linalg.norm(t_vec) or 1.0
            h_vecs = self.data.item_text_emb[history]
            h_norms = np.linalg.norm(h_vecs, axis=1, keepdims=True)
            h_norms[h_norms == 0] = 1.0
            sims = np.dot(h_vecs / h_norms, t_vec / t_norm)
        else:
            return []

        top_k = min(top_n, len(history))
        top_h_pos = np.argsort(-sims)[:top_k]

        anchors: list[dict[str, Any]] = []
        for pos in top_h_pos:
            sim_val = float(sims[pos])
            if sim_val < min_similarity:
                continue
            h_idx = int(history[pos])
            h_meta = self.items_metadata.get(h_idx, {})
            anchors.append({
                "app_id": h_meta.get("app_id", self.idx_to_app_id.get(h_idx, 0)),
                "title": h_meta.get("title", f"Game #{h_idx}"),
                "similarity": sim_val,
            })
        return anchors

    def neighbors(self, app_id: int, k: int = 10) -> list[dict[str, Any]]:
        """
        Computes cosine nearest neighbors for a game in the Two-Tower item vector space.
        """
        if not self.is_initialized:
            self.initialize()

        if app_id not in self.app_id_to_idx:
            raise GameNotFoundError(f"Game with app_id={app_id} not found in the {self.universe_size}-game universe.")

        query_idx = self.app_id_to_idx[app_id]
        if self.tt_item_vectors is None or query_idx >= len(self.tt_item_vectors):
            raise ModelNotReadyError("Two-Tower item embedding vectors are not loaded.")

        q_vec = self.tt_item_vectors[query_idx]
        sims = np.dot(self.tt_item_vectors, q_vec).copy()
        sims[query_idx] = -1.0  # mask self

        k_clamped = max(1, min(50, int(k)))
        top_indices = np.argsort(-sims)[:k_clamped]

        results: list[dict[str, Any]] = []
        for rank_num, idx in enumerate(top_indices, start=1):
            i_int = int(idx)
            meta = self.items_metadata.get(i_int, {})
            results.append({
                "rank": rank_num,
                "app_id": meta.get("app_id", self.idx_to_app_id.get(i_int, 0)),
                "title": meta.get("title", f"Game #{i_int}"),
                "header_image_url": meta.get("header_image_url", ""),
                "similarity": float(sims[i_int]),
                "price_final": meta.get("price_final"),
                "positive_ratio": meta.get("positive_ratio"),
                "user_reviews": meta.get("user_reviews"),
                "tags": meta.get("tags", []),
                "popularity_rank": meta.get("popularity_rank", i_int + 1),
            })
        return results

    def search_items(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """
        Case-insensitive title search over the in-memory catalog, ordered by popularity rank.
        """
        if not self.is_initialized:
            self.initialize()

        limit_clamped = max(1, min(100, int(limit)))
        q_clean = query.strip().lower()

        matched: list[dict[str, Any]] = []
        for meta in self.items_metadata.values():
            title = meta.get("title", "").lower()
            if not q_clean or q_clean in title:
                matched.append(meta)

        matched.sort(key=lambda x: x.get("popularity_rank", 999999))
        return matched[:limit_clamped]

    def get_models_info(self) -> list[dict[str, Any]]:
        """Returns metadata for all deployable recommender models."""
        if not self.is_initialized:
            self.initialize()

        info_list: list[dict[str, Any]] = []
        for fam in DEPLOYABLE_MODELS:
            is_ready = fam in self.models
            meta = self.model_runs_meta.get(fam, {})
            is_champ = fam == DEFAULT_MODEL_NAME
            stage = "production" if is_champ else meta.get("stage", "candidate")

        leaderboard_path = self.data_dir / "reports" / "leaderboard.json"
        lb_models_map: dict[str, Any] = {}
        if leaderboard_path.exists():
            try:
                import json
                with open(leaderboard_path, encoding="utf-8") as f:
                    lb_data = json.load(f)
                    for m in lb_data.get("models", []):
                        lb_models_map[m.get("model_name", "")] = m
            except Exception as e:
                logger.debug("Could not read leaderboard.json: %s", e)

        info_list: list[dict[str, Any]] = []
        for fam in DEPLOYABLE_MODELS:
            is_ready = fam in self.models
            meta = self.model_runs_meta.get(fam, {})
            is_champ = fam == DEFAULT_MODEL_NAME
            stage = "production" if is_champ else meta.get("stage", "candidate")

            lb_entry = lb_models_map.get(fam, {})
            test_m = lb_entry.get("test_metrics", {})
            val_m = lb_entry.get("val_metrics", {})

            if fam == "recsys_two_tower_ranker_clean":
                deployable = False
            elif fam == "recsys_two_tower_ranker_clean_v2":
                # v2's DB row has stage="candidate" (the default from log_recsys_run).
                # We override to "reference" here because v2 uses all-time-stats
                # stage-1 retrieval and is kept only as a diagnostic upper bound.
                # Artifact loading is unaffected: get_latest_recsys_runs has no
                # stage filter. promote_champion excludes v2 by the experiment-name
                # filter, so it is never eligible for promotion regardless of DB stage.
                deployable = False
                stage = "reference"
            else:
                deployable = True

            info_list.append({
                "model_name": fam,
                "display_name": fam.replace("recsys_", "").replace("_clean", "").replace("_", " ").title(),
                "deployable": deployable,
                "stage": stage,
                "model_run_id": meta.get("model_run_id") or lb_entry.get("model_run_id"),
                "val_ndcg_at_10": meta.get("val_ndcg_at_10") or val_m.get("ndcg_at_10", 0.0),
                "test_ndcg_at_10_mean": meta.get("test_ndcg_at_10") or test_m.get("ndcg_at_10", 0.0),
                "test_hr_at_10_mean": test_m.get("hit_rate_at_10", 0.0),
                "n_seeds": lb_entry.get("n_seeds", 1),
                "is_ready": is_ready,
                "is_preview": self.ranker_is_preview if "ranker" in fam else False,
            })
        return info_list


def get_recommender_service() -> RecommenderService:
    """Dependency injection helper for FastAPI handlers."""
    svc = RecommenderService.get_instance()
    if not svc.is_initialized:
        svc.initialize()
    return svc
