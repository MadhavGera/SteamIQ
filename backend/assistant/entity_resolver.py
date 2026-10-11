"""
Entity Resolver for Ask SteamIQ — Phase 6.

Resolves free-form game names and mentions in user queries to Steam app_id using:
  (a) Exact / normalized case-insensitive title matching
  (b) Fuzzy string matching (character & token overlap)
  (c) MiniLM title-embedding nearest neighbor search

Evaluates:
  - Top-1 accuracy on known games.
  - Unknown-game rejection precision and recall on queries referencing non-existent titles.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("steamiq.assistant.entity_resolver")


def normalize_title(title: str) -> str:
    """Normalize game title for robust matching: lowercased, punctuation stripped."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class EntityResolver:
    """
    In-memory game title resolver.
    Loads title catalog snapshot at initialization.
    """

    def __init__(
        self,
        catalog: list[dict[str, Any]] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.catalog = catalog or []
        self.model_name = model_name
        self._embedder = None
        self.title_embeddings: np.ndarray | None = None

        # Build lookup indices
        self.app_id_to_title: dict[int, str] = {}
        self.exact_map: dict[str, int] = {}
        self.normalized_map: dict[str, int] = {}
        self.titles_list: list[str] = []
        self.app_ids_list: list[int] = []

        if self.catalog:
            self._build_indices()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name, device="cpu")
        return self._embedder

    def _build_indices(self) -> None:
        for item in self.catalog:
            app_id = int(item["app_id"])
            title = str(item["title"]).strip()
            self.app_id_to_title[app_id] = title
            self.exact_map[title.lower()] = app_id

            norm = normalize_title(title)
            if norm:
                self.normalized_map[norm] = app_id

            self.titles_list.append(title)
            self.app_ids_list.append(app_id)

    def precompute_embeddings(self) -> None:
        """Precomputes normalized 384-d vectors for all catalog titles."""
        if not self.titles_list:
            return
        logger.info("Precomputing entity resolver title embeddings for %d games...", len(self.titles_list))
        self.title_embeddings = self.embedder.encode(
            self.titles_list,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def resolve(
        self,
        candidate_text: str,
        fuzzy_threshold: float = 0.72,
        embedding_threshold: float = 0.68,
    ) -> tuple[int | None, str | None, float, str]:
        """
        Resolves a game name string to (app_id, resolved_title, confidence, method).
        Methods tried sequentially: exact -> normalized -> fuzzy -> embedding.
        If no match passes thresholds, returns (None, None, 0.0, "unresolved").
        """
        text = candidate_text.strip()
        if not text:
            return None, None, 0.0, "unresolved"

        lower_text = text.lower()
        # 1. Exact match
        if lower_text in self.exact_map:
            app_id = self.exact_map[lower_text]
            return app_id, self.app_id_to_title[app_id], 1.0, "exact"

        # 2. Normalized match
        norm_text = normalize_title(text)
        if norm_text in self.normalized_map:
            app_id = self.normalized_map[norm_text]
            return app_id, self.app_id_to_title[app_id], 0.98, "normalized"

        # 3. Fuzzy match via sequence matcher
        best_fuzzy_match = None
        best_fuzzy_score = 0.0
        for norm_cand, app_id in self.normalized_map.items():
            ratio = difflib.SequenceMatcher(None, norm_text, norm_cand).ratio()
            if ratio > best_fuzzy_score:
                best_fuzzy_score = ratio
                best_fuzzy_match = app_id

        if best_fuzzy_match is not None and best_fuzzy_score >= fuzzy_threshold:
            return (
                best_fuzzy_match,
                self.app_id_to_title[best_fuzzy_match],
                round(float(best_fuzzy_score), 4),
                "fuzzy",
            )

        # 4. Dense semantic embedding match
        if self.title_embeddings is not None and len(self.titles_list) > 0:
            query_vec = self.embedder.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0].astype(np.float32)

            similarities = np.dot(self.title_embeddings, query_vec)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= embedding_threshold:
                app_id = self.app_ids_list[best_idx]
                return (
                    app_id,
                    self.titles_list[best_idx],
                    round(best_sim, 4),
                    "embedding",
                )

        return None, None, 0.0, "unresolved"

    def extract_and_resolve(
        self,
        query: str,
        fuzzy_threshold: float = 0.75,
    ) -> tuple[int | None, str | None, float, str]:
        """
        Extracts game entity from full query text and resolves to app_id.
        First tests substring matching for titles (sorted descending by title length).
        If substring found, verifies and resolves. Otherwise tests resolution on query text.
        """
        norm_query = normalize_title(query)

        # Check for direct title substring in normalized query (longest titles first)
        sorted_candidates = sorted(
            self.normalized_map.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for norm_title, app_id in sorted_candidates:
            if len(norm_title) >= 4 and f" {norm_title} " in f" {norm_query} ":
                return app_id, self.app_id_to_title[app_id], 0.99, "substring"

        # Check direct resolution on full query or extracted phrases
        return self.resolve(query, fuzzy_threshold=fuzzy_threshold)

    def evaluate(
        self,
        labeled_queries: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Evaluates top-1 accuracy on labeled queries and unknown-game rejection.
        labeled_queries format:
          [{"query": "...", "expected_app_id": 1145360 or None}]
        """
        if not labeled_queries:
            return {
                "top1_accuracy": 0.0,
                "unknown_rejection_precision": 0.0,
                "unknown_rejection_recall": 0.0,
                "unknown_rejection_f1": 0.0,
            }

        known_correct = 0
        known_total = 0

        # Confusion counts for unknown rejection
        # Positive = game is unknown (expected_app_id is None)
        # Negative = game is known (expected_app_id is int)
        tp = 0  # Expected unknown, resolver returned None
        fp = 0  # Expected known, resolver returned None (false rejection)
        fn = 0  # Expected unknown, resolver resolved to some app_id (false acceptance)
        tn = 0  # Expected known, resolver resolved to expected_app_id

        for item in labeled_queries:
            text = item.get("text") or item.get("query", "")
            exp_id = item.get("expected_app_id")

            resolved_id, _, _, _ = self.extract_and_resolve(text)

            if exp_id is not None:
                known_total += 1
                if resolved_id == exp_id:
                    known_correct += 1
                    tn += 1
                else:
                    fp += 1
            else:
                if resolved_id is None:
                    tp += 1
                else:
                    fn += 1

        top1_acc = float(known_correct / known_total) if known_total > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "top1_accuracy": round(top1_acc, 4),
            "unknown_rejection_precision": round(prec, 4),
            "unknown_rejection_recall": round(rec, 4),
            "unknown_rejection_f1": round(f1, 4),
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "catalog.json").write_text(json.dumps(self.catalog, indent=2), encoding="utf-8")
        if self.title_embeddings is not None:
            np.save(str(p / "title_embeddings.npy"), self.title_embeddings)

    @classmethod
    def load(cls, path: Path | str) -> EntityResolver:
        p = Path(path)
        cat = json.loads((p / "catalog.json").read_text(encoding="utf-8"))
        resolver = cls(catalog=cat)
        emb_path = p / "title_embeddings.npy"
        if emb_path.exists():
            resolver.title_embeddings = np.load(str(emb_path))
        return resolver
