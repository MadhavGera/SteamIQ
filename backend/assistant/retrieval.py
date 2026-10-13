"""
Extractive Passage Retrieval Fallback (RAG without free-text generation) — Phase 6.

Indexes pre-materialized review intelligence passages (summary, complaints, loved features)
embedded with MiniLM. Returns top-k extractive passages with source game and field.

Evaluates:
  - Recall@1, Recall@3, Recall@5, and MRR.
  - Compares Dense MiniLM cosine retrieval against a TF-IDF cosine baseline.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("steamiq.assistant.retrieval")


class PassageIndex:
    """
    Extractive passage index for fallback retrieval.
    Stores and indexes passages from mart_review_intelligence.
    """

    def __init__(
        self,
        passages: list[dict[str, Any]] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.passages: list[dict[str, Any]] = passages or []
        self.model_name = model_name
        self._embedder = None
        self.dense_embeddings: np.ndarray | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        if self.passages:
            self._build_indices()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name, device="cpu")
        return self._embedder

    def _build_indices(self) -> None:
        texts = [p["text"] for p in self.passages]
        if not texts:
            return

        # 1. Build TF-IDF baseline index
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            lowercase=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # 2. Build Dense MiniLM index
        logger.info("Encoding %d passages with MiniLM for extractive retrieval...", len(texts))
        self.dense_embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        method: str = "dense",
    ) -> list[dict[str, Any]]:
        """
        Retrieves top-k passages for query using either 'dense' or 'tfidf'.
        Returns list of dicts with passage details and similarity score.
        """
        if not self.passages:
            return []

        k = min(k, len(self.passages))

        if method == "dense" and self.dense_embeddings is not None:
            query_vec = self.embedder.encode(
                [query],
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0].astype(np.float32)

            scores = np.dot(self.dense_embeddings, query_vec)
            top_indices = np.argsort(-scores)[:k]

            results = []
            for idx in top_indices:
                p = dict(self.passages[idx])
                p["score"] = round(float(scores[idx]), 4)
                results.append(p)
            return results

        elif method == "tfidf" and self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            q_vec = self.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(-sims)[:k]

            results = []
            for idx in top_indices:
                p = dict(self.passages[idx])
                p["score"] = round(float(sims[idx]), 4)
                results.append(p)
            return results

        return []

    def evaluate(
        self,
        gold_queries: list[dict[str, Any]],
        k_values: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates retrieval Recall@k and MRR on gold (query, expected_app_id, expected_field) pairs.
        Compares dense MiniLM against TF-IDF baseline.
        """
        k_vals = k_values or [1, 3, 5]
        max_k = max(k_vals)

        if not gold_queries or not self.passages:
            return {
                "dense": {f"recall_at_{k}": 0.0 for k in k_vals} | {"mrr": 0.0},
                "tfidf": {f"recall_at_{k}": 0.0 for k in k_vals} | {"mrr": 0.0},
            }

        def eval_method(method: str) -> tuple[dict[str, Any], dict[int, np.ndarray], np.ndarray]:
            n_queries = len(gold_queries)
            hit_k_vecs = {k: np.zeros(n_queries, dtype=float) for k in k_vals}
            rr_vec = np.zeros(n_queries, dtype=float)

            for i, item in enumerate(gold_queries):
                q = item["query"]
                exp_app_id = item.get("expected_app_id")
                exp_field = item.get("expected_field")

                retrieved = self.retrieve(q, k=max_k, method=method)

                hit_rank = 0
                for rank, p in enumerate(retrieved, start=1):
                    # Match if app_id matches (and field if specified)
                    app_match = (exp_app_id is None) or (p.get("app_id") == exp_app_id)
                    field_match = (exp_field is None) or (p.get("field") == exp_field)

                    if app_match and field_match:
                        hit_rank = rank
                        break

                if hit_rank > 0:
                    rr_vec[i] = 1.0 / hit_rank
                    for k in k_vals:
                        if hit_rank <= k:
                            hit_k_vecs[k][i] = 1.0

            def _boot(arr: np.ndarray) -> list[float]:
                if len(arr) == 0:
                    return [0.0, 0.0]
                rng = np.random.default_rng(42)
                means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(1000)]
                return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

            metrics: dict[str, Any] = {
                "n_queries": n_queries,
            }
            for k in k_vals:
                metrics[f"recall_at_{k}"] = round(float(np.mean(hit_k_vecs[k])), 4)
                metrics[f"recall_at_{k}_ci"] = _boot(hit_k_vecs[k])

            metrics["mrr"] = round(float(np.mean(rr_vec)), 4)
            metrics["mrr_ci"] = _boot(rr_vec)
            return metrics, hit_k_vecs, rr_vec

        dense_metrics, dense_hits, dense_rr = eval_method("dense")
        tfidf_metrics, tfidf_hits, tfidf_rr = eval_method("tfidf")

        def _boot_diff(arr: np.ndarray) -> list[float]:
            if len(arr) == 0:
                return [0.0, 0.0]
            rng = np.random.default_rng(42)
            means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(1000)]
            return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

        paired_diffs: dict[str, Any] = {}
        for k in k_vals:
            diff_hit = dense_hits[k] - tfidf_hits[k]
            paired_diffs[f"delta_recall_at_{k}"] = round(float(np.mean(diff_hit)), 4)
            paired_diffs[f"delta_recall_at_{k}_ci"] = _boot_diff(diff_hit)

        diff_rr = dense_rr - tfidf_rr
        paired_diffs["delta_mrr"] = round(float(np.mean(diff_rr)), 4)
        paired_diffs["delta_mrr_ci"] = _boot_diff(diff_rr)
        dense_beats_tfidf = bool(paired_diffs["delta_mrr_ci"][0] > 0.0)
        paired_diffs["dense_beats_tfidf"] = dense_beats_tfidf
        paired_diffs["claim"] = (
            "Dense MiniLM significantly outperforms TF-IDF (95% CI excludes 0)"
            if dense_beats_tfidf
            else "No evidence that dense retrieval outperforms TF-IDF (95% CI includes 0)"
        )

        user_idxs = [i for i, g in enumerate(gold_queries) if g.get("written_by") == "user"]
        user_subset_metrics = None
        if user_idxs:
            u_dense_rr = dense_rr[user_idxs]
            u_tfidf_rr = tfidf_rr[user_idxs]
            u_diff_rr = u_dense_rr - u_tfidf_rr
            user_subset_metrics = {
                "n_queries": len(user_idxs),
                "dense_mrr": round(float(np.mean(u_dense_rr)), 4),
                "dense_mrr_ci": _boot_diff(u_dense_rr),
                "tfidf_mrr": round(float(np.mean(u_tfidf_rr)), 4),
                "tfidf_mrr_ci": _boot_diff(u_tfidf_rr),
                "delta_mrr": round(float(np.mean(u_diff_rr)), 4),
                "delta_mrr_ci": _boot_diff(u_diff_rr),
            }

        return {
            "n_queries": len(gold_queries),
            "dense": dense_metrics,
            "tfidf": tfidf_metrics,
            "paired_difference": paired_diffs,
            "user_written_subset": user_subset_metrics,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "passages.json").write_text(json.dumps(self.passages, indent=2), encoding="utf-8")
        if self.dense_embeddings is not None:
            np.save(str(p / "dense_embeddings.npy"), self.dense_embeddings)
        if self.tfidf_vectorizer is not None:
            with (p / "tfidf_vectorizer.pkl").open("wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

    @classmethod
    def load(cls, path: Path | str) -> PassageIndex:
        p = Path(path)
        passages = json.loads((p / "passages.json").read_text(encoding="utf-8"))
        index = cls(passages=passages)

        dense_p = p / "dense_embeddings.npy"
        if dense_p.exists():
            index.dense_embeddings = np.load(str(dense_p))

        tfidf_p = p / "tfidf_vectorizer.pkl"
        if tfidf_p.exists():
            with tfidf_p.open("rb") as f:
                index.tfidf_vectorizer = pickle.load(f)
            texts = [item["text"] for item in index.passages]
            index.tfidf_matrix = index.tfidf_vectorizer.transform(texts)

        return index
