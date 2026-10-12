"""
Intent Router models behind a unified interface — Phase 6.

Interface:
  fit(texts: list[str], labels: list[str]) -> BaseIntentRouter
  predict_proba(texts: list[str]) -> np.ndarray  # [n_samples, n_classes]
  predict(texts: list[str]) -> list[str]

Implementations:
  1. KeywordBaselineRouter: deterministic dictionary keyword matcher.
  2. TfidfLogisticRouter: TF-IDF n-grams + Logistic Regression.
  3. MiniLMLogisticRouter: all-MiniLM-L6-v2 sentence embeddings + Logistic Regression (Headline model).
  4. MiniLMKnnRouter: all-MiniLM-L6-v2 sentence embeddings + kNN distance weighting.

Constraints:
  - Pure Python + scikit-learn + sentence-transformers. Zero FastAPI or DB imports.
  - CPU inference only.
"""
from __future__ import annotations

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from assistant.intents import ALL_INTENTS, Intent

logger = logging.getLogger("steamiq.assistant.router")


class BaseIntentRouter(ABC):
    """Abstract base interface for all intent routers."""

    def __init__(self, classes: list[str] | None = None) -> None:
        self.classes_: list[str] = classes or list(ALL_INTENTS)
        self.is_fitted: bool = False

    def _align_probabilities(self, raw_probs: np.ndarray, model_classes: list[str]) -> np.ndarray:
        """Align raw model probabilities to canonical self.classes_ column order."""
        if list(model_classes) == list(self.classes_):
            return raw_probs
        n_samples = raw_probs.shape[0]
        aligned = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        model_cls_to_idx = {c: i for i, c in enumerate(model_classes)}
        for j, target_cls in enumerate(self.classes_):
            if target_cls in model_cls_to_idx:
                aligned[:, j] = raw_probs[:, model_cls_to_idx[target_cls]]
        row_sums = aligned.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()
        if np.any(zero_rows):
            aligned[zero_rows, :] = 1.0 / len(self.classes_)
            row_sums[zero_rows] = 1.0
        return aligned / row_sums

    @abstractmethod
    def fit(self, texts: list[str], labels: list[str]) -> BaseIntentRouter:
        """Fit router on texts and corresponding intent labels."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Returns [N, n_classes] array of predicted class probabilities."""
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[str]:
        """Predicts the highest-probability intent for each text."""
        if not texts:
            return []
        probs = self.predict_proba(texts)
        best_indices = np.argmax(probs, axis=1)
        return [self.classes_[idx] for idx in best_indices]

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist model artifacts to directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> BaseIntentRouter:
        """Load model artifacts from directory."""
        raise NotImplementedError


# ─── 1. Keyword-Rule Baseline ────────────────────────────────────────────────

class KeywordBaselineRouter(BaseIntentRouter):
    """
    Rule-based dictionary keyword baseline.
    Matches query substrings against per-intent token bags.
    """

    def __init__(self, classes: list[str] | None = None) -> None:
        super().__init__(classes)
        self.keyword_map: dict[str, list[str]] = {}

    def fit(self, texts: list[str], labels: list[str]) -> KeywordBaselineRouter:
        # Load default keywords from schema_metadata.json or initialize from classes
        schema_path = Path(__file__).resolve().parent / "schema_metadata.json"
        if schema_path.exists():
            try:
                schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
                for intent_key, intent_def in schema_data.get("intents", {}).items():
                    self.keyword_map[intent_key] = [
                        k.lower() for k in intent_def.get("keywords", [])
                    ]
            except Exception as e:
                logger.warning("Could not load keywords from schema: %s", e)

        # Ensure all classes have a list
        for c in self.classes_:
            if c not in self.keyword_map:
                self.keyword_map[c] = [c.replace("_", " ")]

        self.is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.classes_)), dtype=np.float64)

        n_samples = len(texts)
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i, text in enumerate(texts):
            lower_text = text.lower()
            matched_any = False
            for j, cls_name in enumerate(self.classes_):
                keywords = self.keyword_map.get(cls_name, [])
                for kw in keywords:
                    if kw in lower_text:
                        # Longer keyword matches get higher weight
                        scores[i, j] += len(kw.split()) + 1.0
                        matched_any = True

            if not matched_any:
                # Default unknown/unmatched queries to out_of_scope
                if Intent.OUT_OF_SCOPE.value in self.classes_:
                    oos_idx = self.classes_.index(Intent.OUT_OF_SCOPE.value)
                    scores[i, oos_idx] = 1.0
                else:
                    scores[i, :] = 1.0 / n_classes

        # Softmax / normalize row-wise
        row_sums = scores.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()
        if np.any(zero_rows):
            scores[zero_rows, :] = 1.0 / n_classes
            row_sums[zero_rows] = 1.0

        return scores / row_sums

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        state = {
            "classes": self.classes_,
            "keyword_map": self.keyword_map,
        }
        (p / "keyword_model.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> KeywordBaselineRouter:
        p = Path(path)
        state = json.loads((p / "keyword_model.json").read_text(encoding="utf-8"))
        router = cls(classes=state["classes"])
        router.keyword_map = state["keyword_map"]
        router.is_fitted = True
        return router


# ─── 2. TF-IDF + Logistic Regression Baseline ────────────────────────────────

class TfidfLogisticRouter(BaseIntentRouter):
    """
    Sparse statistical baseline: Word & char n-grams + Logistic Regression.
    """

    def __init__(self, classes: list[str] | None = None) -> None:
        super().__init__(classes)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            lowercase=True,
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    def fit(self, texts: list[str], labels: list[str]) -> TfidfLogisticRouter:
        x_tfidf = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x_tfidf, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        x_tfidf = self.vectorizer.transform(texts)
        raw_probs = self.classifier.predict_proba(x_tfidf)
        return self._align_probabilities(raw_probs, list(self.classifier.classes_))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "tfidf_model.pkl").open("wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, f)

    @classmethod
    def load(cls, path: Path | str) -> TfidfLogisticRouter:
        p = Path(path)
        with (p / "tfidf_model.pkl").open("rb") as f:
            data = pickle.load(f)
        router = cls()
        router.vectorizer = data["vectorizer"]
        router.classifier = data["classifier"]
        router.classes_ = list(router.classifier.classes_)
        router.is_fitted = True
        return router


# ─── 3. MiniLM Embeddings + Logistic Regression (Headline Model) ──────────────

class MiniLMLogisticRouter(BaseIntentRouter):
    """
    Dense semantic router: all-MiniLM-L6-v2 sentence embeddings + calibrated Logistic head.
    CPU inference: computes 384-d normalized embeddings then linear scoring.
    """

    def __init__(
        self,
        classes: list[str] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(classes)
        self.model_name = model_name
        self._embedder = None
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name, device="cpu")
        return self._embedder

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to 384-dim normalized numpy vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def fit(self, texts: list[str], labels: list[str]) -> MiniLMLogisticRouter:
        features = self.encode(texts)
        self.classifier.fit(features, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        features = self.encode(texts)
        raw_probs = self.classifier.predict_proba(features)
        return self._align_probabilities(raw_probs, list(self.classifier.classes_))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        metadata = {
            "classes": self.classes_,
            "model_name": self.model_name,
        }
        (p / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (p / "classifier.pkl").open("wb") as f:
            pickle.dump(self.classifier, f)

    @classmethod
    def load(cls, path: Path | str) -> MiniLMLogisticRouter:
        p = Path(path)
        meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        router = cls(classes=meta["classes"], model_name=meta.get("model_name", "all-MiniLM-L6-v2"))
        with (p / "classifier.pkl").open("rb") as f:
            router.classifier = pickle.load(f)
        router.is_fitted = True
        return router


# ─── 4. MiniLM Embeddings + kNN Baseline ─────────────────────────────────────

class MiniLMKnnRouter(BaseIntentRouter):
    """
    Non-parametric semantic baseline: all-MiniLM-L6-v2 embeddings + distance-weighted kNN.
    """

    def __init__(
        self,
        classes: list[str] | None = None,
        n_neighbors: int = 5,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(classes)
        self.n_neighbors = n_neighbors
        self.model_name = model_name
        self._embedder = None
        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="cosine",
        )

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name, device="cpu")
        return self._embedder

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def fit(self, texts: list[str], labels: list[str]) -> MiniLMKnnRouter:
        features = self.encode(texts)
        self.classifier.fit(features, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        features = self.encode(texts)
        raw_probs = self.classifier.predict_proba(features)
        return self._align_probabilities(raw_probs, list(self.classifier.classes_))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "classes": self.classes_,
            "n_neighbors": self.n_neighbors,
            "model_name": self.model_name,
        }
        (p / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (p / "knn_model.pkl").open("wb") as f:
            pickle.dump(self.classifier, f)

    @classmethod
    def load(cls, path: Path | str) -> MiniLMKnnRouter:
        p = Path(path)
        meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        router = cls(
            classes=meta["classes"],
            n_neighbors=meta.get("n_neighbors", 5),
            model_name=meta.get("model_name", "all-MiniLM-L6-v2"),
        )
        with (p / "knn_model.pkl").open("rb") as f:
            router.classifier = pickle.load(f)
        router.is_fitted = True
        return router
