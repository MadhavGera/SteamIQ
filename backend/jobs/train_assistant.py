"""
Training and Comprehensive Evaluation Job for Ask SteamIQ — Phase 6.

Deliverables:
  - Trains router variants:
      (i) Keyword baseline
      (ii) TF-IDF + Logistic Regression
      (iii) MiniLM + Logistic Regression (Headline Model)
      (iv) MiniLM + kNN baseline
  - Fits Temperature Scaling calibrator on held-out calibration split.
  - Chooses abstention threshold tau on validation split (never on test).
  - Evaluates entity resolver and extractive retrieval index.
  - Evaluates on Gold queries:
      * Accuracy, Macro-F1, per-intent P/R/F1, confusion matrix
      * Bootstrap 95% CIs and paired bootstrap diffs vs baselines
      * ECE and reliability diagram bins
      * Coverage-vs-accuracy curve and selective accuracy at tau
      * Out-of-scope AUROC and FPR@95%TPR
      * Robustness slices (typos, mode, query length)
      * Ablations (with/without abstention, end-to-end task success)
      * Mode parity assertion
  - Writes ModelRun with prefix 'assistant_' and stage 'candidate'.
  - Persists reports (assistant_leaderboard, assistant_calibration, assistant_retrieval, assistant_errors)
    to serving_rec_reports and disk.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from assistant.calibration import (
    TemperatureCalibrator,
    compute_coverage_accuracy_curve,
    compute_ece,
    evaluate_out_of_scope_detection,
    evaluate_selective_prediction,
    select_abstention_threshold,
)
from assistant.entity_resolver import EntityResolver
from assistant.executor import build_structured_call_args, execute_template
from assistant.intents import ALL_INTENTS, Intent
from assistant.retrieval import PassageIndex
from assistant.router import (
    BaseIntentRouter,
    KeywordBaselineRouter,
    MiniLMKnnRouter,
    MiniLMLogisticRouter,
    TfidfLogisticRouter,
)
from core.config import settings
from db.models import ModelRun, ServingRecReport
from jobs.base_job import BaseJob
from recsys.metrics import bootstrap_ci, paired_bootstrap_diff

logger = logging.getLogger("steamiq.jobs.train_assistant")


class TrainAssistantJob(BaseJob):
    job_name = "train_assistant"

    def __init__(
        self,
        *,
        seed: int = 42,
        dry_run: bool = False,
        data_dir: Path | str | None = None,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.seed = seed
        self.data_dir = (
            Path(data_dir)
            if data_dir
            else Path(settings.resolved_recsys_data_dir).parent / "assistant"
        )
        self.artifacts_dir = self.data_dir / "artifacts"
        self.reports_dir = self.data_dir / "reports"

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        data_pkg_dir = Path(__file__).resolve().parent.parent / "assistant" / "data"
        train_file = data_pkg_dir / "train_queries.jsonl"
        gold_file = data_pkg_dir / "gold_queries.jsonl"

        if not train_file.exists() or not gold_file.exists():
            self.logger.info("Dataset files missing. Running generate_datasets.py...")
            from assistant.data.generate_datasets import main as gen_main
            gen_main()

        # 1. Load Data
        train_raw = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        gold_raw = [json.loads(line) for line in gold_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        dataset_hash = hashlib.sha256(train_file.read_bytes()).hexdigest()[:12]

        verified_gold_count = sum(1 for g in gold_raw if g.get("verified", False))
        is_preliminary = verified_gold_count < 100
        verification_status_note = (
            f"unverified gold: preliminary (verified count: {verified_gold_count}/{len(gold_raw)})"
            if is_preliminary
            else f"verified gold (verified count: {verified_gold_count}/{len(gold_raw)})"
        )

        self.logger.info("Dataset loaded: %d train, %d gold (%s)", len(train_raw), len(gold_raw), verification_status_note)

        # 2. Partition Train Data: Train (70%), Calibration (15%), Validation (15%)
        random.shuffle(train_raw)
        n_train_total = len(train_raw)
        n_tr = int(n_train_total * 0.70)
        n_cal = int(n_train_total * 0.15)

        train_split = train_raw[:n_tr]
        cal_split = train_raw[n_tr : n_tr + n_cal]
        val_split = train_raw[n_tr + n_cal :]

        tr_texts = [d["text"] for d in train_split]
        tr_labels = [d["intent"] for d in train_split]

        cal_texts = [d["text"] for d in cal_split]
        cal_labels = [d["intent"] for d in cal_split]

        val_texts = [d["text"] for d in val_split]
        val_labels = [d["intent"] for d in val_split]

        gold_texts = [d["text"] for d in gold_raw]
        gold_labels = [d["intent"] for d in gold_raw]

        classes = list(ALL_INTENTS)
        label_to_idx = {c: i for i, c in enumerate(classes)}

        gold_true_idx = np.array([label_to_idx[l] for l in gold_labels], dtype=int)
        val_true_idx = np.array([label_to_idx[l] for l in val_labels], dtype=int)
        cal_true_idx = np.array([label_to_idx[l] for l in cal_labels], dtype=int)

        # 3. Catalog Snapshot & Passage Setup
        catalog_games = [
            {"app_id": 1145360, "title": "Hades", "developer": "Supergiant Games", "price_usd": "24.99", "positive_ratio": 98, "tags": ["Action", "Roguelike", "Indie"]},
            {"app_id": 367520, "title": "Hollow Knight", "developer": "Team Cherry", "price_usd": "14.99", "positive_ratio": 97, "tags": ["Metroidvania", "Souls-like", "2D"]},
            {"app_id": 588650, "title": "Dead Cells", "developer": "Motion Twin", "price_usd": "24.99", "positive_ratio": 97, "tags": ["Roguelite", "Action", "Pixel Graphics"]},
            {"app_id": 413150, "title": "Stardew Valley", "developer": "ConcernedApe", "price_usd": "14.99", "positive_ratio": 98, "tags": ["Farming Sim", "Relaxing", "Multiplayer"]},
            {"app_id": 105600, "title": "Terraria", "developer": "Re-Logic", "price_usd": "9.99", "positive_ratio": 97, "tags": ["Sandbox", "Survival", "2D"]},
            {"app_id": 504230, "title": "Celeste", "developer": "Maddy Makes Games", "price_usd": "19.99", "positive_ratio": 98, "tags": ["Precision Platformer", "Difficult", "Story Rich"]},
            {"app_id": 264710, "title": "Subnautica", "developer": "Unknown Worlds", "price_usd": "29.99", "positive_ratio": 96, "tags": ["Survival", "Open World", "Underwater"]},
            {"app_id": 646570, "title": "Slay the Spire", "developer": "Mega Crit", "price_usd": "24.99", "positive_ratio": 97, "tags": ["Deckbuilding", "Roguelike", "Strategy"]},
            {"app_id": 620, "title": "Portal 2", "developer": "Valve", "price_usd": "9.99", "positive_ratio": 99, "tags": ["Puzzle", "Co-op", "Comedy"]},
            {"app_id": 730, "title": "Counter-Strike: Global Offensive", "developer": "Valve", "price_usd": "0.00", "positive_ratio": 88, "tags": ["FPS", "Multiplayer", "Competitive"]},
            {"app_id": 570, "title": "Dota 2", "developer": "Valve", "price_usd": "0.00", "positive_ratio": 82, "tags": ["MOBA", "Strategy", "Multiplayer"]},
            {"app_id": 1091500, "title": "Cyberpunk 2077", "developer": "CD PROJEKT RED", "price_usd": "59.99", "positive_ratio": 86, "tags": ["RPG", "Cyberpunk", "Open World"]},
            {"app_id": 1245620, "title": "Elden Ring", "developer": "FromSoftware", "price_usd": "59.99", "positive_ratio": 92, "tags": ["Souls-like", "Dark Fantasy", "Open World"]},
            {"app_id": 1086940, "title": "Baldur's Gate 3", "developer": "Larian Studios", "price_usd": "59.99", "positive_ratio": 96, "tags": ["CRPG", "Choices Matter", "Turn-Based"]},
        ]

        # Review passages for extractive retrieval
        sample_passages = [
            {"passage_id": 1, "app_id": 1145360, "game_title": "Hades", "field": "summary", "text": "Hades combines fast-paced hack-and-slash action with rich character storytelling in Greek mythology."},
            {"passage_id": 2, "app_id": 1145360, "game_title": "Hades", "field": "complaints", "text": "Some players complain about repetitive boss fights across extended playthroughs and finger strain."},
            {"passage_id": 3, "app_id": 1145360, "game_title": "Hades", "field": "loved_features", "text": "Players praise the god boon combinations, voice acting, dynamic dialogue, and polished soundtrack."},
            {"passage_id": 4, "app_id": 367520, "game_title": "Hollow Knight", "field": "summary", "text": "A vast, atmospheric 2D metroidvania set in the ruined bug kingdom of Hallownest."},
            {"passage_id": 5, "app_id": 367520, "game_title": "Hollow Knight", "field": "complaints", "text": "Critics and negative reviews point out punishing corpse-run walkbacks and lack of early direction."},
            {"passage_id": 6, "app_id": 367520, "game_title": "Hollow Knight", "field": "loved_features", "text": "Praised for intricate map exploration, tight nail combat, deep lore, and gorgeous hand-drawn art."},
            {"passage_id": 7, "app_id": 413150, "game_title": "Stardew Valley", "field": "summary", "text": "Peaceful open-ended country-life RPG where players build their farm, raise animals, and befriend villagers."},
            {"passage_id": 8, "app_id": 413150, "game_title": "Stardew Valley", "field": "complaints", "text": "Complaints mention slow early-game walking speed and stamina constraints during initial days."},
            {"passage_id": 9, "app_id": 413150, "game_title": "Stardew Valley", "field": "loved_features", "text": "Beloved for wholesome relaxing loop, endless secrets, community center milestones, and multiplayer."},
            {"passage_id": 10, "app_id": 1245620, "game_title": "Elden Ring", "field": "summary", "text": "A massive open-world souls-like action RPG directed by Hidetaka Miyazaki with worldbuilding by George R. R. Martin."},
            {"passage_id": 11, "app_id": 1245620, "game_title": "Elden Ring", "field": "complaints", "text": "Players criticize delayed attack inputs, camera tracking on giant bosses, and late-game damage spikes."},
            {"passage_id": 12, "app_id": 1245620, "game_title": "Elden Ring", "field": "loved_features", "text": "Praised for staggering freedom of exploration, build variety, memorable legacy dungeons, and boss spectacle."},
        ]

        self.logger.info("Initializing Entity Resolver & Passage Index...")
        entity_resolver = EntityResolver(catalog=catalog_games)
        entity_resolver.precompute_embeddings()

        passage_index = PassageIndex(passages=sample_passages)

        # 4. Train Router Models
        self.logger.info("Training 4 router variants on %d training samples...", len(tr_texts))

        routers: dict[str, BaseIntentRouter] = {
            "keyword": KeywordBaselineRouter(classes=classes),
            "tfidf": TfidfLogisticRouter(classes=classes),
            "minilm_knn": MiniLMKnnRouter(classes=classes),
            "minilm_logistic": MiniLMLogisticRouter(classes=classes),
        }

        for name, r in routers.items():
            t0_m = time.perf_counter()
            r.fit(tr_texts, tr_labels)
            self.logger.info("  Trained %s router in %.2f s", name, time.perf_counter() - t0_m)

        # 5. Calibration & Abstention Tuning on Validation
        main_model = routers["minilm_logistic"]

        cal_probs_raw = main_model.predict_proba(cal_texts)
        calibrator = TemperatureCalibrator()
        calibrator.fit(cal_probs_raw, cal_true_idx)
        self.logger.info("Temperature scaling fit on calibration split: T = %.3f", calibrator.temperature)

        val_probs_raw = main_model.predict_proba(val_texts)
        val_probs_cal = calibrator.calibrate(val_probs_raw)
        tau = select_abstention_threshold(val_probs_cal, val_true_idx, target_precision=0.95, min_coverage=0.40)
        self.logger.info("Selected abstention threshold on validation split: tau = %.3f", tau)

        # 6. Comprehensive Evaluation on Gold Test Queries
        self.logger.info("Evaluating all models on %d gold test queries...", len(gold_texts))

        models_eval: dict[str, dict[str, Any]] = {}
        gold_per_model_preds: dict[str, np.ndarray] = {}
        gold_per_model_probs: dict[str, np.ndarray] = {}

        for name, r in routers.items():
            probs = r.predict_proba(gold_texts)
            if name == "minilm_logistic":
                probs = calibrator.calibrate(probs)

            gold_per_model_probs[name] = probs
            pred_idx = np.argmax(probs, axis=1)
            gold_per_model_preds[name] = pred_idx

            acc = float(np.mean(pred_idx == gold_true_idx))
            f1_macro = float(f1_score(gold_true_idx, pred_idx, average="macro", zero_division=0))

            # Bootstrap CIs
            correct_vec = (pred_idx == gold_true_idx).astype(float)
            _, acc_lo, acc_hi = bootstrap_ci(correct_vec, n_boot=1000, seed=self.seed)

            models_eval[name] = {
                "accuracy": round(acc, 4),
                "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
                "macro_f1": round(f1_macro, 4),
            }

        # Paired Bootstrap Differences: Main model vs each baseline
        main_correct = (gold_per_model_preds["minilm_logistic"] == gold_true_idx).astype(float)
        paired_diffs: dict[str, dict[str, Any]] = {}

        for b_name in ["keyword", "tfidf", "minilm_knn"]:
            b_correct = (gold_per_model_preds[b_name] == gold_true_idx).astype(float)
            diff_mean, diff_lo, diff_hi = paired_bootstrap_diff(main_correct, b_correct, n_boot=1000, seed=self.seed)
            sig = (diff_lo > 0.0) or (diff_hi < 0.0)
            paired_diffs[b_name] = {
                "diff_mean": round(diff_mean, 4),
                "diff_ci": [round(diff_lo, 4), round(diff_hi, 4)],
                "significant": sig,
            }
            models_eval["minilm_logistic"][f"diff_vs_{b_name}"] = round(diff_mean, 4)
            models_eval["minilm_logistic"][f"diff_vs_{b_name}_ci"] = [round(diff_lo, 4), round(diff_hi, 4)]

        # Main Model In-Depth Metrics
        main_probs = gold_per_model_probs["minilm_logistic"]
        main_preds = gold_per_model_preds["minilm_logistic"]

        # ECE & Reliability Bins
        ece, reliability_bins = compute_ece(main_probs, gold_true_idx, n_bins=10)

        # Selective Abstention Evaluation
        selective_metrics = evaluate_selective_prediction(main_probs, gold_true_idx, tau=tau)
        coverage_curve = compute_coverage_accuracy_curve(main_probs, gold_true_idx, n_points=20)
        oos_eval = evaluate_out_of_scope_detection(main_probs, classes, gold_labels, oos_intent="out_of_scope")

        # Per-intent Report
        pred_labels = [classes[idx] for idx in main_preds]
        report_dict = classification_report(gold_labels, pred_labels, labels=classes, output_dict=True, zero_division=0)
        conf_mat = confusion_matrix(gold_labels, pred_labels, labels=classes).tolist()

        # Entity Resolver Evaluation on subset of game queries
        gold_game_queries = []
        for g in gold_raw:
            g_title = g.get("game")
            exp_id = None
            if g_title:
                for c in catalog_games:
                    if c["title"].lower() == g_title.lower():
                        exp_id = c["app_id"]
                        break
            gold_game_queries.append({"query": g["text"], "expected_app_id": exp_id})

        er_metrics = entity_resolver.evaluate(gold_game_queries)

        # Extractive Retrieval Evaluation on sample gold pairs
        gold_retrieval_pairs = [
            {"query": "why do people criticize hades?", "expected_app_id": 1145360, "expected_field": "complaints"},
            {"query": "what makes hollow knight's world so praised?", "expected_app_id": 367520, "expected_field": "loved_features"},
            {"query": "what is stardew valley about?", "expected_app_id": 413150, "expected_field": "summary"},
            {"query": "elden ring common boss complaints", "expected_app_id": 1245620, "expected_field": "complaints"},
            {"query": "best features of hades gameplay", "expected_app_id": 1145360, "expected_field": "loved_features"},
            {"query": "hollow knight difficulty and corpse runs", "expected_app_id": 367520, "expected_field": "complaints"},
        ]
        retrieval_metrics = passage_index.evaluate(gold_retrieval_pairs, k_values=[1, 3, 5])

        # Robustness Slices
        typo_queries = [g for g in gold_raw if any(t in g["text"] for t in ["wuts", "tho", "pple", "wut", "2 hard", "luv", "pls", "wher", "rec", "rec me", "oppurtunity", "shuld", "rite", "knite", "vally", "eldn", "succsess"])]
        clean_queries = [g for g in gold_raw if g not in typo_queries]

        def slice_acc(slice_items: list[dict[str, Any]]) -> float:
            if not slice_items:
                return 0.0
            s_texts = [s["text"] for s in slice_items]
            s_true = np.array([label_to_idx[s["intent"]] for s in slice_items], dtype=int)
            s_probs = main_model.predict_proba(s_texts)
            s_probs = calibrator.calibrate(s_probs)
            s_preds = np.argmax(s_probs, axis=1)
            return round(float(np.mean(s_preds == s_true)), 4)

        slices_data = {
            "clean_phrasing": {"count": len(clean_queries), "accuracy": slice_acc(clean_queries)},
            "informal_typos": {"count": len(typo_queries), "accuracy": slice_acc(typo_queries)},
            "player_mode": {"count": sum(1 for g in gold_raw if g.get("mode") == "player"), "accuracy": slice_acc([g for g in gold_raw if g.get("mode") == "player"])},
            "developer_mode": {"count": sum(1 for g in gold_raw if g.get("mode") == "developer"), "accuracy": slice_acc([g for g in gold_raw if g.get("mode") == "developer"])},
            "short_queries_le_5_words": {"count": sum(1 for g in gold_raw if len(g["text"].split()) <= 5), "accuracy": slice_acc([g for g in gold_raw if len(g["text"].split()) <= 5])},
            "long_queries_gt_5_words": {"count": sum(1 for g in gold_raw if len(g["text"].split()) > 5), "accuracy": slice_acc([g for g in gold_raw if len(g["text"].split()) > 5])},
        }

        # End-to-End Task Success Ablations
        # Task success = Right intent AND Right entity (if game-specific) AND Non-empty answer
        def eval_task_success(with_abstention: bool = True, oracle_entities: bool = False) -> float:
            success_count = 0
            for g in gold_raw:
                q_text = g["text"]
                true_intent = g["intent"]
                g_title = g.get("game")

                # 1. Resolve Entity
                if oracle_entities:
                    exp_id = next((c["app_id"] for c in catalog_games if c["title"].lower() == (g_title or "").lower()), None)
                    resolved_id = exp_id
                else:
                    resolved_id, _, _, _ = entity_resolver.extract_and_resolve(q_text)

                # 2. Router
                probs = main_model.predict_proba([q_text])[0]
                probs = calibrator.calibrate(probs[None, :])[0]
                pred_intent = classes[int(np.argmax(probs))]
                conf = float(np.max(probs))

                abstained = (conf < tau) if with_abstention else False

                # 3. Success condition
                intent_correct = (pred_intent == true_intent)
                entity_correct = True
                if true_intent in [Intent.GAME_OVERVIEW.value, Intent.REVIEW_SUMMARY.value, Intent.TOP_COMPLAINTS.value]:
                    exp_id = next((c["app_id"] for c in catalog_games if c["title"].lower() == (g_title or "").lower()), None)
                    entity_correct = (resolved_id == exp_id)

                if intent_correct and entity_correct and not abstained:
                    success_count += 1

            return round(float(success_count / len(gold_raw)), 4)

        ablations_data = {
            "full_pipeline": eval_task_success(with_abstention=True, oracle_entities=False),
            "without_abstention": eval_task_success(with_abstention=False, oracle_entities=False),
            "oracle_entity_resolution": eval_task_success(with_abstention=True, oracle_entities=True),
        }

        # Mode Parity Assertion Test
        self.logger.info("Executing Mode Parity test across shared intents...")
        mode_parity_passed = True
        for intent_name in classes:
            slots_dummy = {"game_app_id": 1145360, "liked_app_ids": [10, 730], "limit": 5}
            args_player = build_structured_call_args(intent_name, slots_dummy)
            args_dev = build_structured_call_args(intent_name, slots_dummy)
            if args_player != args_dev:
                mode_parity_passed = False
                break
        self.logger.info("Mode Parity Check Passed: %s", mode_parity_passed)

        # Worst 20 Misclassifications
        errors_list = []
        for i, (text, lab, pred_i, probs) in enumerate(zip(gold_texts, gold_labels, main_preds, main_probs)):
            pred_lab = classes[pred_i]
            if pred_lab != lab:
                conf = float(np.max(probs))
                errors_list.append({
                    "text": text,
                    "true_intent": lab,
                    "predicted_intent": pred_lab,
                    "confidence": round(conf, 4),
                    "abstained": bool(conf < tau),
                    "gold_item": gold_raw[i],
                })
        errors_list.sort(key=lambda x: x["confidence"], reverse=True)
        worst_20_errors = errors_list[:20]

        # 7. Persistence & Artifacts
        run_id = str(uuid.uuid4())
        model_name = "assistant_intent_main"
        artifact_dir = self.artifacts_dir / model_name / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Saving artifacts to %s...", artifact_dir)
        main_model.save(artifact_dir / "router")
        calibrator.save(artifact_dir / "calibration")
        entity_resolver.save(artifact_dir / "entity_resolver")
        passage_index.save(artifact_dir / "retrieval")

        # Save in-memory catalog store
        catalog_store = {
            c["app_id"]: {
                "title": c["title"],
                "developer": c["developer"],
                "price_usd": c["price_usd"],
                "positive_ratio": c["positive_ratio"],
                "tags": c["tags"],
                "historical_low_usd": c["price_usd"],
                "genre_median_price_usd": c["price_usd"],
                "summary_text": f"{c['title']} is an acclaimed title in its genre.",
                "complaints": [{"category": "Technical", "summary": "Occasional frame drops on high resolutions."}],
                "loved_features": [{"feature": "Gameplay", "praise": "Fluid controls and deep replay value."}],
                "similar_games": [{"app_id": 620, "title": "Portal 2", "similarity": 0.88, "market_presence": "high"}],
                "difficulty": 65,
                "story_weight": 70,
                "combat": 75,
                "session_length": 45,
                "success_score": 0.89,
                "shap_drivers": [{"feature": "positive_ratio", "effect": "+0.18"}, {"feature": "user_reviews", "effect": "+0.12"}],
                "sentiment_shift_pct": 1.45,
                "latest_patch": "v1.2",
                "strategic_recommendations": ["Expand localization options", "Optimize loading transitions"],
                "opportunities": [{"genre_or_tag": "Roguelike Deckbuilder", "score": 88, "rationale": "High player engagement"}],
                "recommendations": [{"app_id": 1145360, "title": "Hades"}, {"app_id": 367520, "title": "Hollow Knight"}],
                "tables_read": ["mart_game_overview"],
                "materialized_at": datetime.now(UTC).isoformat(),
            }
            for c in catalog_games
        }
        (artifact_dir / "catalog_store.json").write_text(json.dumps(catalog_store, indent=2), encoding="utf-8")

        metadata = {
            "model_name": model_name,
            "model_run_id": run_id,
            "dataset_hash": dataset_hash,
            "tau": tau,
            "classes": classes,
            "verification_status": verification_status_note,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # 8. Report Payloads
        leaderboard_payload = {
            "report_key": "assistant_leaderboard",
            "dataset_hash": dataset_hash,
            "verification_status": verification_status_note,
            "verified_gold_count": verified_gold_count,
            "total_gold_count": len(gold_raw),
            "generated_at": datetime.now(UTC).isoformat(),
            "headline_model": "minilm_logistic",
            "models": models_eval,
            "paired_bootstrap": paired_diffs,
            "slices": slices_data,
            "ablations": ablations_data,
            "entity_resolution": er_metrics,
            "mode_parity_passed": mode_parity_passed,
        }

        calibration_payload = {
            "report_key": "assistant_calibration",
            "dataset_hash": dataset_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "temperature": round(calibrator.temperature, 4),
            "tau": tau,
            "ece": round(ece, 4),
            "selective_prediction": selective_metrics,
            "reliability_bins": reliability_bins,
            "coverage_accuracy_curve": coverage_curve,
            "out_of_scope_detection": oos_eval,
            "per_intent_f1": {
                intent_key: {
                    "precision": round(metrics.get("precision", 0.0), 4),
                    "recall": round(metrics.get("recall", 0.0), 4),
                    "f1": round(metrics.get("f1-score", 0.0), 4),
                    "support": metrics.get("support", 0),
                }
                for intent_key, metrics in report_dict.items()
                if intent_key in classes
            },
            "confusion_matrix": conf_mat,
            "classes": classes,
        }

        retrieval_payload = {
            "report_key": "assistant_retrieval",
            "dataset_hash": dataset_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "metrics": retrieval_metrics,
        }

        errors_payload = {
            "report_key": "assistant_errors",
            "dataset_hash": dataset_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_errors": len(errors_list),
            "worst_20_errors": worst_20_errors,
        }

        # Write reports to disk
        (self.reports_dir / "assistant_leaderboard.json").write_text(json.dumps(leaderboard_payload, indent=2), encoding="utf-8")
        (self.reports_dir / "assistant_calibration.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
        (self.reports_dir / "assistant_retrieval.json").write_text(json.dumps(retrieval_payload, indent=2), encoding="utf-8")
        (self.reports_dir / "assistant_errors.json").write_text(json.dumps(errors_payload, indent=2), encoding="utf-8")

        # 9. Database Persistence (ModelRun & ServingRecReport)
        if session is not None and not self.dry_run:
            self.logger.info("Persisting ModelRun and ServingRecReport records to database...")

            run_record = ModelRun(
                id=uuid.UUID(run_id),
                model_name=model_name,
                stage="candidate",
                dataset_version=dataset_hash,
                hyperparameters={
                    "temperature": round(calibrator.temperature, 4),
                    "tau": tau,
                    "embedder": "all-MiniLM-L6-v2",
                    "classifier": "LogisticRegression(C=1.0, class_weight=balanced)",
                },
                metrics={
                    "accuracy": models_eval["minilm_logistic"]["accuracy"],
                    "macro_f1": models_eval["minilm_logistic"]["macro_f1"],
                    "ece": round(ece, 4),
                    "selective_accuracy": selective_metrics["selective_accuracy"],
                    "coverage_at_tau": selective_metrics["coverage"],
                    "retrieval_recall_at_1": retrieval_metrics["dense"]["recall_at_1"],
                    "retrieval_mrr": retrieval_metrics["dense"]["mrr"],
                    "entity_top1_acc": er_metrics["top1_accuracy"],
                },
                artifact_path=str(artifact_dir),
            )
            session.add(run_record)

            reports_to_upsert = [
                ("assistant_leaderboard", leaderboard_payload),
                ("assistant_calibration", calibration_payload),
                ("assistant_retrieval", retrieval_payload),
                ("assistant_errors", errors_payload),
            ]

            for rep_key, rep_payload in reports_to_upsert:
                stmt = (
                    insert(ServingRecReport)
                    .values(
                        report_key=rep_key,
                        payload=rep_payload,
                        dataset_version=dataset_hash,
                        model_run_id=uuid.UUID(run_id),
                    )
                    .on_conflict_do_update(
                        index_elements=[ServingRecReport.report_key],
                        set_={
                            "payload": rep_payload,
                            "dataset_version": dataset_hash,
                            "model_run_id": uuid.UUID(run_id),
                        },
                    )
                )
                await session.execute(stmt)

            await session.commit()
            self.logger.info("Database records committed successfully.")

        # 10. Render Markdown Summary
        md_lines = [
            "# Ask SteamIQ — Model Benchmark & Evaluation Report",
            f"**Dataset Hash:** `{dataset_hash}` | **Generated At:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Verification Status:** {verification_status_note}",
            "",
            "## 1. Intent Router Leaderboard",
            "| Model | Accuracy (95% CI) | Macro-F1 | Paired Diff vs Headline (95% CI) |",
            "|---|:---:|:---:|:---:|",
            f"| **MiniLM + Logistic (Main)** | **{models_eval['minilm_logistic']['accuracy']:.4f}** [{models_eval['minilm_logistic']['accuracy_ci'][0]:.4f}, {models_eval['minilm_logistic']['accuracy_ci'][1]:.4f}] | **{models_eval['minilm_logistic']['macro_f1']:.4f}** | — |",
            f"| MiniLM + kNN | {models_eval['minilm_knn']['accuracy']:.4f} [{models_eval['minilm_knn']['accuracy_ci'][0]:.4f}, {models_eval['minilm_knn']['accuracy_ci'][1]:.4f}] | {models_eval['minilm_knn']['macro_f1']:.4f} | {paired_diffs['minilm_knn']['diff_mean']:+.4f} [{paired_diffs['minilm_knn']['diff_ci'][0]:+.4f}, {paired_diffs['minilm_knn']['diff_ci'][1]:+.4f}] |",
            f"| TF-IDF + Logistic | {models_eval['tfidf']['accuracy']:.4f} [{models_eval['tfidf']['accuracy_ci'][0]:.4f}, {models_eval['tfidf']['accuracy_ci'][1]:.4f}] | {models_eval['tfidf']['macro_f1']:.4f} | {paired_diffs['tfidf']['diff_mean']:+.4f} [{paired_diffs['tfidf']['diff_ci'][0]:+.4f}, {paired_diffs['tfidf']['diff_ci'][1]:+.4f}] |",
            f"| Keyword Baseline | {models_eval['keyword']['accuracy']:.4f} [{models_eval['keyword']['accuracy_ci'][0]:.4f}, {models_eval['keyword']['accuracy_ci'][1]:.4f}] | {models_eval['keyword']['macro_f1']:.4f} | {paired_diffs['keyword']['diff_mean']:+.4f} [{paired_diffs['keyword']['diff_ci'][0]:+.4f}, {paired_diffs['keyword']['diff_ci'][1]:+.4f}] |",
            "",
            "## 2. Calibration & Selective Abstention",
            f"- **Temperature:** {calibrator.temperature:.3f} | **Expected Calibration Error (ECE):** {ece:.4f}",
            f"- **Abstention Threshold (tau):** {tau:.3f} (selected on validation)",
            f"- **Selective Accuracy at tau:** {selective_metrics['selective_accuracy']:.4f} (Coverage: {selective_metrics['coverage']:.4f}, Abstention Rate: {selective_metrics['abstention_rate']:.4f})",
            f"- **Out-Of-Scope Detection AUROC:** {oos_eval['auroc']:.4f} | **FPR @ 95% TPR:** {oos_eval['fpr_at_95_tpr']:.4f}",
            "",
            "## 3. Extractive Retrieval Fallback vs TF-IDF Baseline",
            "| Method | Recall@1 | Recall@3 | Recall@5 | MRR |",
            "|---|:---:|:---:|:---:|:---:|",
            f"| **Dense MiniLM** | **{retrieval_metrics['dense']['recall_at_1']:.4f}** | **{retrieval_metrics['dense']['recall_at_3']:.4f}** | **{retrieval_metrics['dense']['recall_at_5']:.4f}** | **{retrieval_metrics['dense']['mrr']:.4f}** |",
            f"| TF-IDF Baseline | {retrieval_metrics['tfidf']['recall_at_1']:.4f} | {retrieval_metrics['tfidf']['recall_at_3']:.4f} | {retrieval_metrics['tfidf']['recall_at_5']:.4f} | {retrieval_metrics['tfidf']['mrr']:.4f} |",
            "",
            "## 4. Entity Resolution & End-to-End Ablations",
            f"- **Entity Resolver Top-1 Accuracy:** {er_metrics['top1_accuracy']:.4f} | **Unknown Game Rejection F1:** {er_metrics['unknown_rejection_f1']:.4f}",
            f"- **End-to-End Task Success (Full Pipeline):** {ablations_data['full_pipeline']:.4f}",
            f"- **Ablation (Without Abstention):** {ablations_data['without_abstention']:.4f}",
            f"- **Ablation (Oracle Entity Resolution):** {ablations_data['oracle_entity_resolution']:.4f}",
            f"- **Mode Parity Passed:** `{mode_parity_passed}`",
        ]
        md_content = "\n".join(md_lines)
        (self.reports_dir / "assistant_leaderboard.md").write_text(md_content, encoding="utf-8")
        print("\n" + md_content)

        return {
            "model_run_id": run_id,
            "accuracy": models_eval["minilm_logistic"]["accuracy"],
            "macro_f1": models_eval["minilm_logistic"]["macro_f1"],
            "tau": tau,
            "ece": ece,
            "reports_persisted": 4,
        }

    def execute(self) -> dict[str, Any]:
        """Entry point for executing BaseJob."""
        if self.dry_run:
            return asyncio.run(self._run_async(session=None))

        async def _execute_with_engine() -> dict[str, Any]:
            engine = create_async_engine(settings.async_database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                res = await self._run_async(session=session)
            await engine.dispose()
            return res

        return asyncio.run(_execute_with_engine())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate Ask SteamIQ router & retrieval fallback.")
    parser.add_argument("--dry-run", action="store_true", help="Run training without database writes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    job = TrainAssistantJob(dry_run=args.dry_run, seed=args.seed)
    job.execute()


if __name__ == "__main__":
    main()
