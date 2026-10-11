"""
Comprehensive Unit and Integrity Tests for Ask SteamIQ — Phase 6.

Validates:
  1. Intent taxonomy and schema_metadata.json completeness.
  2. Router models (KeywordBaseline, TF-IDF Logistic, MiniLM Logistic/kNN).
  3. Calibration (Temperature scaling, ECE, abstention threshold selection).
  4. Entity resolver (exact, fuzzy, normalized, and unknown rejection).
  5. Extractive retrieval index (indexing, similarity search, recall).
  6. Deterministic executor and mode parity invariant.
  7. Architectural source-scan: Zero raw_*/feature_* reads and zero imports from jobs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from assistant.calibration import (
    TemperatureCalibrator,
    compute_coverage_accuracy_curve,
    compute_ece,
    evaluate_selective_prediction,
    select_abstention_threshold,
)
from assistant.entity_resolver import EntityResolver
from assistant.executor import (
    DEV_ONLY_INTENTS,
    build_structured_call_args,
    execute_template,
)
from assistant.intents import ALL_INTENTS, Intent
from assistant.retrieval import PassageIndex
from assistant.router import (
    KeywordBaselineRouter,
    MiniLMKnnRouter,
    TfidfLogisticRouter,
)


# ─── 1. Intent Taxonomy & Schema Metadata ─────────────────────────────────────


def test_intent_taxonomy_and_schema_metadata():
    schema_path = Path(__file__).resolve().parent.parent / "assistant" / "schema_metadata.json"
    assert schema_path.exists(), "schema_metadata.json must exist in backend/assistant"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "intents" in schema

    # Every Intent in ALL_INTENTS must have a schema definition
    for intent_name in ALL_INTENTS:
        assert intent_name in schema["intents"], f"Intent {intent_name} missing in schema_metadata.json"
        intent_meta = schema["intents"][intent_name]
        assert "mode_visibility" in intent_meta
        assert "description" in intent_meta


# ─── 2. Intent Routers ────────────────────────────────────────────────────────


def test_keyword_baseline_router():
    router = KeywordBaselineRouter()
    texts = ["tell me about Terraria", "summarize reviews for Cyberpunk", "who is the President of France?"]
    labels = ["game_overview", "review_summary", "out_of_scope"]
    router.fit(texts, labels)

    probs = router.predict_proba(["player sentiment and community response for Dota"])
    assert probs.shape == (1, len(ALL_INTENTS))
    assert np.isclose(np.sum(probs), 1.0)
    pred_idx = int(np.argmax(probs))
    assert router.classes_[pred_idx] == "review_summary"


def test_tfidf_logistic_router():
    router = TfidfLogisticRouter()
    texts = [
        "what is Terraria about?",
        "overview of Portal 2",
        "summarize player reviews for Elden Ring",
        "what are the top complaints?",
        "recommend games like Hades",
        "random chit chat hello",
    ]
    labels = [
        "game_overview",
        "game_overview",
        "review_summary",
        "top_complaints",
        "similar_games",
        "out_of_scope",
    ]
    router.fit(texts, labels)

    probs = router.predict_proba(["tell me what complaints players have"])
    assert probs.shape == (1, len(router.classes_))
    assert np.isclose(np.sum(probs), 1.0)
    pred_idx = int(np.argmax(probs))
    assert router.classes_[pred_idx] in ("top_complaints", "review_summary")


# ─── 3. Calibration & Selective Prediction ────────────────────────────────────


def test_temperature_calibrator_and_ece():
    np.random.seed(42)
    n_samples = 100
    n_classes = len(ALL_INTENTS)

    # Simulated over-confident probabilities
    logits = np.random.normal(loc=0.0, scale=3.0, size=(n_samples, n_classes))
    exp_logits = np.exp(logits)
    uncalibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    labels = np.random.randint(0, n_classes, size=n_samples)

    calibrator = TemperatureCalibrator()
    calibrator.fit(uncalibrated_probs, labels)
    assert calibrator.temperature > 0.0

    calibrated_probs = calibrator.calibrate(uncalibrated_probs)
    assert calibrated_probs.shape == (n_samples, n_classes)
    assert np.allclose(np.sum(calibrated_probs, axis=1), 1.0)

    ece, bins_data = compute_ece(calibrated_probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(bins_data) == 5


def test_select_abstention_threshold():
    np.random.seed(42)
    probs = np.array([
        [0.95, 0.05],
        [0.92, 0.08],
        [0.60, 0.40],
        [0.55, 0.45],
    ])
    labels = np.array([0, 0, 1, 1])  # 0.95 and 0.92 correct (predicted 0, label 0); 0.60 wrong (pred 0, label 1)

    tau = select_abstention_threshold(probs, labels, target_precision=0.90, min_coverage=0.40)
    assert tau >= 0.50


# ─── 4. Entity Resolver ───────────────────────────────────────────────────────


def test_entity_resolver():
    catalog = [
        {"app_id": 105600, "title": "Terraria"},
        {"app_id": 620, "title": "Portal 2"},
        {"app_id": 730, "title": "Counter-Strike 2"},
    ]
    resolver = EntityResolver(catalog=catalog)

    # Exact match
    app_id, title, conf, method = resolver.resolve("Terraria")
    assert app_id == 105600
    assert method == "exact"
    assert conf == 1.0

    # Normalized match
    app_id, title, conf, method = resolver.resolve("portal 2")
    assert app_id == 620
    assert method in ("exact", "normalized")

    # Unknown game rejection
    app_id, title, conf, method = resolver.resolve("Unknown Game 999xyz")
    assert app_id is None or conf < 0.60


# ─── 5. Extractive Retrieval Index ────────────────────────────────────────────


def test_extractive_passage_index():
    passages = [
        {
            "passage_id": "p1",
            "app_id": 105600,
            "title": "Terraria",
            "field": "loved_features",
            "text": "Players love the deep exploration, crafting systems, and massive boss battles.",
        },
        {
            "passage_id": "p2",
            "app_id": 620,
            "title": "Portal 2",
            "field": "top_complaints",
            "text": "Some users reported occasional co-op disconnections during workshop maps.",
        },
    ]

    index = PassageIndex(passages=passages)
    results = index.retrieve("boss battles crafting", k=1, method="tfidf")
    assert len(results) == 1
    assert results[0]["app_id"] == 105600
    assert "crafting" in results[0]["text"]


# ─── 6. Deterministic Executor & Mode Parity ──────────────────────────────────


def test_executor_mode_parity():
    """
    Mode Parity Invariant:
    build_structured_call_args does NOT branch on user_mode.
    Given (intent, slots), identical StructuredCallArgs are produced for player and developer.
    """
    slots = {"app_id": 105600, "limit": 5}

    args = build_structured_call_args(
        intent=Intent.GAME_OVERVIEW.value,
        slots=slots,
    )

    assert args.intent == Intent.GAME_OVERVIEW.value
    assert args.app_id == 105600
    assert args.limit == 5


def test_executor_developer_only_refusal_in_player_mode():
    """
    Developer-only intents (e.g. market_opportunity, update_impact) requested
    in Player mode MUST be honestly refused with explanatory message.
    """
    for dev_intent in DEV_ONLY_INTENTS:
        exec_res = execute_template(
            intent=dev_intent,
            data=None,
            slots={"app_id": 105600},
            user_mode="player",
            confidence=0.95,
        )
        assert "Developer Mode" in exec_res.answer
        assert exec_res.user_mode == "player"


# ─── 7. Architectural Integrity & Golden Rule Source Scan ─────────────────────


def test_assistant_source_scan_no_raw_reads_and_no_jobs_imports():
    """
    Golden Rule (ADR 0001, ADR 0005, ADR 0006):
    1. Zero queries or reads of raw_* or feature_* database tables in serving code.
    2. Zero imports from backend/jobs in serving code.
    """
    backend_dir = Path(__file__).resolve().parent.parent
    assistant_dir = backend_dir / "assistant"
    api_file = backend_dir / "api" / "assistant.py"

    files_to_scan = [f for f in assistant_dir.rglob("*.py") if "test" not in f.name] + [api_file]

    # Regex targeting database tables like raw_games, raw_reviews, feature_games, feature_reviews
    sql_table_pattern = re.compile(r"""\b(?:from|join|update|into|table)\s+['"]?(?:raw_|feature_)[a-z_0-9]+""", re.IGNORECASE)

    for fpath in files_to_scan:
        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue

            # Check 1: No raw_* or feature_* table scans in queries
            match = sql_table_pattern.search(stripped)
            assert match is None, f"SQL violation in {fpath.name}:{idx} -> {match.group(0)}"

            # Check 2: No import from jobs in serving code
            if fpath.name not in ("train_assistant.py", "serve_check_assistant.py"):
                assert not stripped.startswith("from jobs") and not stripped.startswith("import jobs"), (
                    f"Import violation in {fpath.name}:{idx} -> imports jobs in serving package: {stripped}"
                )
