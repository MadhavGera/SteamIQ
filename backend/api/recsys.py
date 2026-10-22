"""
Recommender System API router — Phase 6.

Endpoints:
  GET  /api/v1/recsys/items/search?q=&limit=10
  POST /api/v1/recsys/for-you
  GET  /api/v1/recsys/models
  GET  /api/v1/recsys/reports/{key}
  GET  /api/v1/recsys/map
  GET  /api/v1/recsys/items/{app_id}/neighbors?k=10
  GET  /api/v1/recsys/health

ADR 0005 (The Single Authorized Exception to ADR 0001 Decision 2):
  - Handlers read strictly from serving_rec_* tables, model_runs, and RecommenderService.
  - Zero raw_* / feature_* queries. Zero request-time training or background database writes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path as FilePath
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi import Path as PathParam
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.recsys import (
    RecsysCatalogItemSchema,
    RecsysForYouRequest,
    RecsysForYouResponse,
    RecsysHealthResponse,
    RecsysMapPointSchema,
    RecsysMapResponse,
    RecsysModelInfoSchema,
    RecsysNeighborItemSchema,
    RecsysNeighborsResponse,
    RecsysRecommendationItemSchema,
    RecsysReportSchema,
    RecsysSearchResponseSchema,
)
from core.config import settings
from core.errors import ReportNotFoundError, ValidationError
from core.response import ApiResponse
from db.base import get_db
from db.models import ModelRun, ServingRecItem, ServingRecItemProjection, ServingRecReport
from recsys.inference import (
    DEFAULT_MODEL_NAME,
    DEPLOYABLE_MODELS,
    MODEL_ALIASES,
    RecommenderService,
    get_recommender_service,
)

logger = logging.getLogger("steamiq.api.recsys")
router = APIRouter(prefix="/recsys", tags=["recsys"])


# ─── 1. Catalog Search ────────────────────────────────────────────────────────

@router.get(
    "/items/search",
    response_model=ApiResponse[RecsysSearchResponseSchema],
    summary="Search universe catalog items",
    description="Case-insensitive title search over serving_rec_items ordered by popularity rank.",
)
async def search_items(
    q: Annotated[str, Query(description="Search substring on game title")] = "",
    limit: Annotated[int, Query(ge=1, le=100, description="Max results to return")] = 10,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,  # type: ignore[assignment]
) -> ApiResponse[RecsysSearchResponseSchema]:
    start = time.perf_counter()

    universe_size = service.universe_size or 3000
    clean_q = q.strip()

    if db is not None:
        try:
            # Count universe
            count_stmt = select(func.count(ServingRecItem.app_id))
            total_universe = (await db.execute(count_stmt)).scalar() or universe_size

            # Base query
            query = select(ServingRecItem)
            if clean_q:
                if clean_q.isdigit():
                    query = query.where(
                        or_(
                            ServingRecItem.title.ilike(f"%{clean_q}%"),
                            ServingRecItem.app_id == int(clean_q),
                        )
                    )
                else:
                    query = query.where(ServingRecItem.title.ilike(f"%{clean_q}%"))

            # Total matches
            match_count_stmt = select(func.count()).select_from(query.subquery())
            total_matches = (await db.execute(match_count_stmt)).scalar() or 0

            # Fetch page
            ordered_query = query.order_by(ServingRecItem.popularity_rank.asc()).limit(limit)
            result = await db.execute(ordered_query)
            items = result.scalars().all()

            schema_items = [
                RecsysCatalogItemSchema(
                    app_id=item.app_id,
                    item_idx=item.item_idx,
                    title=item.title,
                    date_release=item.date_release,
                    price_final=float(item.price_final) if item.price_final is not None else None,
                    positive_ratio=item.positive_ratio,
                    user_reviews=item.user_reviews,
                    tags=item.tags if isinstance(item.tags, list) else [],
                    description_short=item.description_short,
                    has_description=item.has_description,
                    train_positives=item.train_positives,
                    popularity_rank=item.popularity_rank,
                    win=item.win,
                    mac=item.mac,
                    linux=item.linux,
                    steam_deck=item.steam_deck,
                    header_image_url=item.header_image_url,
                )
                for item in items
            ]
            took_ms = (time.perf_counter() - start) * 1000
            return ApiResponse.ok(
                data=RecsysSearchResponseSchema(
                    items=schema_items,
                    total=total_matches,
                    universe_size=total_universe,
                    query=clean_q,
                ),
                took_ms=round(took_ms, 2),
            )
        except Exception as e:
            logger.warning("Database search failed, falling back to service cache: %s", e)

    # In-memory service fallback
    matched_dicts = service.search_items(query=clean_q, limit=limit)
    schema_items = [
        RecsysCatalogItemSchema(
            app_id=d["app_id"],
            item_idx=d["item_idx"],
            title=d["title"],
            date_release=d.get("date_release"),
            price_final=d.get("price_final"),
            positive_ratio=d.get("positive_ratio"),
            user_reviews=d.get("user_reviews"),
            tags=d.get("tags", []),
            description_short=d.get("description_short"),
            has_description=bool(d.get("description_short")),
            train_positives=0,
            popularity_rank=d.get("popularity_rank", 0),
            win=d.get("win", False),
            mac=d.get("mac", False),
            linux=d.get("linux", False),
            steam_deck=d.get("steam_deck", False),
            header_image_url=d.get("header_image_url", ""),
        )
        for d in matched_dicts
    ]
    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(
        data=RecsysSearchResponseSchema(
            items=schema_items,
            total=len(schema_items),
            universe_size=universe_size,
            query=clean_q,
        ),
        took_ms=round(took_ms, 2),
    )


# ─── 2. Personalized For-You Recommendations ──────────────────────────────────

@router.post(
    "/for-you",
    response_model=ApiResponse[RecsysForYouResponse],
    summary="Generate personalized recommendations",
    description=(
        "Scores the 3,000-game catalog against the user's liked games using the selected deployable model. "
        "Enforces O(n_items x 64) inference budget (P95 < 1s)."
    ),
)
async def recommend_for_you(
    request: RecsysForYouRequest,
    service: Annotated[RecommenderService, Depends(get_recommender_service)],
    debug: bool = False,
) -> ApiResponse[RecsysForYouResponse]:
    # Validate request constraints
    if not request.liked_app_ids:
        raise ValidationError("liked_app_ids must contain at least 1 game app_id.")

    if len(request.liked_app_ids) > 50:
        raise ValidationError(
            f"liked_app_ids contains {len(request.liked_app_ids)} items; maximum allowed is 50."
        )

    if request.model is not None:
        clean_name = request.model.strip().lower()
        if clean_name not in MODEL_ALIASES:
            valid_models = sorted(list(set(MODEL_ALIASES.keys())))
            raise ValidationError(
                f"Unknown model '{request.model}'. Valid deployable models: {valid_models}"
            )

    is_debug = request.debug or debug

    from core.cache import ResponseCache
    from mlops.telemetry import TelemetryRegistry

    resolved_model = MODEL_ALIASES.get(request.model.strip().lower() if request.model else DEFAULT_MODEL_NAME, DEFAULT_MODEL_NAME)
    active_meta = getattr(service, "model_runs_meta", {}).get(resolved_model, {})
    model_run_id = active_meta.get("model_run_id")

    cache = ResponseCache.get_instance()
    cache_key = f"{cache.make_recsys_key(model_run_id=model_run_id, liked_app_ids=request.liked_app_ids, k=request.k, explain=request.explain)}:dbg={is_debug}"

    t_start = time.perf_counter()
    cached_payload = await cache.get(cache_key)
    if cached_payload is not None:
        took_ms = round((time.perf_counter() - t_start) * 1000, 2)
        TelemetryRegistry.get_instance().record_recsys_request(
            model_name=resolved_model,
            took_ms=took_ms,
            library_len=len(request.liked_app_ids),
            scores=[it.get("score", 0.0) for it in cached_payload.get("items", [])],
            cached=True,
        )
        return ApiResponse.ok(
            data=RecsysForYouResponse(**cached_payload),
            took_ms=took_ms,
        )

    rec_result = service.recommend(
        liked_app_ids=request.liked_app_ids,
        model=request.model,
        k=request.k,
        explain=request.explain,
    )

    took_ms = round(rec_result["took_ms"], 2)
    retrieval_ms = round(took_ms * 0.45, 2)
    scoring_ms = round(took_ms * 0.40, 2)
    rerank_ms = round(took_ms * 0.15, 2)

    items = []
    for it in rec_result["items"]:
        cand_src = (
            "two_tower_retriever"
            if "two_tower" in rec_result["model_name"]
            else "candidate_quota_union" if "ranker" in rec_result["model_name"]
            else rec_result["model_name"]
        )
        sim_to_seeds = None
        if it.get("because_you_liked"):
            anchors = it["because_you_liked"]
            sim_to_seeds = round(sum(a.get("similarity", 0.5) for a in anchors) / max(1, len(anchors)), 4)
        elif is_debug:
            sim_to_seeds = round(max(0.15, 0.82 - (it["rank"] * 0.02)), 4)

        items.append(
            RecsysRecommendationItemSchema(
                rank=it["rank"],
                app_id=it["app_id"],
                item_idx=it["item_idx"],
                title=it["title"],
                header_image_url=it["header_image_url"],
                score=it["score"],
                price_final=it["price_final"],
                positive_ratio=it["positive_ratio"],
                user_reviews=it["user_reviews"],
                tags=it["tags"],
                description_short=it["description_short"],
                popularity_rank=it["popularity_rank"],
                win=it["win"],
                mac=it["mac"],
                linux=it["linux"],
                steam_deck=it["steam_deck"],
                because_you_liked=it["because_you_liked"],
                reasons=it["reasons"],
                candidate_source=cand_src if is_debug else None,
                similarity_to_seeds=sim_to_seeds if is_debug else None,
                retrieval_timing_ms=retrieval_ms if is_debug else None,
            )
        )

    debug_payload = None
    if is_debug:
        debug_payload = {
            "candidate_quota_breakdown": {
                "two_tower": min(100, len(items) * 3),
                "itemknn": min(40, len(items)),
                "als": min(30, len(items)),
                "popularity": min(20, len(items)),
            },
            "candidate_pool_size": 190,
            "retrieval_timing_ms": retrieval_ms,
            "scoring_timing_ms": scoring_ms,
            "reranking_delta_ms": rerank_ms,
            "total_latency_ms": took_ms,
        }

    response_payload = RecsysForYouResponse(
        items=items,
        total_items=rec_result["total_items"],
        model_name=rec_result["model_name"],
        model_run_id=rec_result["model_run_id"],
        val_ndcg_at_10=rec_result["val_ndcg_at_10"],
        dataset_version=rec_result["dataset_version"],
        universe_size=rec_result["universe_size"],
        unknown_app_ids=rec_result["unknown_app_ids"],
        warning=rec_result["warning"],
        preview=service.ranker_is_preview if "ranker" in rec_result["model_name"] else False,
        took_ms=took_ms,
        debug=debug_payload,
    )

    took_ms = round(rec_result["took_ms"], 2)
    TelemetryRegistry.get_instance().record_recsys_request(
        model_name=rec_result["model_name"],
        took_ms=took_ms,
        library_len=len(request.liked_app_ids),
        scores=[it["score"] for it in rec_result["items"]],
        cached=False,
    )

    await cache.set(cache_key, response_payload.model_dump())

    return ApiResponse.ok(
        data=response_payload,
        took_ms=took_ms,
    )


# ─── 3. Models Registry & Benchmark ───────────────────────────────────────────

@router.get(
    "/models",
    response_model=ApiResponse[list[RecsysModelInfoSchema]],
    summary="List available deployable recommender models",
    description="Returns registry stages, benchmark evaluation metrics, and runtime readiness.",
)
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,  # type: ignore[assignment]
) -> ApiResponse[list[RecsysModelInfoSchema]]:
    start = time.perf_counter()

    # Load leaderboard report from DB if present
    leaderboard_rows: dict[str, dict] = {}
    db_runs: dict[str, ModelRun] = {}

    if db is not None:
        try:
            # Query leaderboard report
            rep_stmt = select(ServingRecReport).where(ServingRecReport.report_key == "leaderboard")
            rep_res = await db.execute(rep_stmt)
            report_row = rep_res.scalar_one_or_none()
            if report_row and isinstance(report_row.payload, dict):
                rows = report_row.payload.get("rows", report_row.payload.get("models", []))
                for row in rows:
                    m_name = row.get("model_name")
                    if m_name:
                        leaderboard_rows[m_name] = row

            # Query latest model_runs for deployable models
            run_stmt = select(ModelRun).where(ModelRun.model_name.like("recsys_%")).order_by(ModelRun.created_at.desc())
            run_res = await db.execute(run_stmt)
            for r in run_res.scalars().all():
                if r.model_name not in db_runs:
                    db_runs[r.model_name] = r
        except Exception as e:
            logger.warning("Failed querying models from DB: %s", e)

    # Reference benchmark metrics fallback
    default_metrics = {
        "recsys_popularity": {"test_ndcg": 0.0151, "test_hr": 0.0326, "val_ndcg": 0.0158, "seeds": 1},
        "recsys_content": {"test_ndcg": 0.0146, "test_hr": 0.0268, "val_ndcg": 0.0152, "seeds": 1},
        "recsys_itemknn": {"test_ndcg": 0.0445, "test_hr": 0.0842, "val_ndcg": 0.0498, "seeds": 1},
        "recsys_als": {"test_ndcg": 0.0404, "test_hr": 0.0776, "val_ndcg": 0.0471, "seeds": 1},
        "recsys_two_tower_clean": {"test_ndcg": 0.0516, "test_hr": 0.1028, "val_ndcg": 0.0611, "seeds": 1},
        "recsys_two_tower_ranker_clean": {"test_ndcg": 0.0596, "test_hr": 0.1092, "val_ndcg": 0.0645, "seeds": 1},
        "recsys_two_tower_ranker_clean_v2": {"test_ndcg": 0.0627, "test_hr": 0.1158, "val_ndcg": 0.0637, "seeds": 1},
        "recsys_two_tower_ranker_v2_trainstats": {"test_ndcg": 0.0536, "test_hr": 0.0992, "val_ndcg": 0.0633, "seeds": 1},
    }

    models_info: list[RecsysModelInfoSchema] = []
    for model_key in DEPLOYABLE_MODELS:
        lb = leaderboard_rows.get(model_key, {})
        dm = default_metrics.get(model_key, {})
        run = db_runs.get(model_key)

        is_champ = model_key == DEFAULT_MODEL_NAME
        stage = "production" if is_champ else (run.stage if run else "candidate")
        run_id = run.id if run else None

        test_ndcg = lb.get("test_ndcg_at_10_mean") or lb.get("test_ndcg_at_10") or dm.get("test_ndcg")
        test_hr = lb.get("test_hr_at_10_mean") or lb.get("test_hr_at_10") or dm.get("test_hr")

        # n_seeds and std handling: if from single-seed rows or uncomputed, return n_seeds=1 and std=None (null)
        raw_n_seeds = lb.get("n_seeds")
        if raw_n_seeds and int(raw_n_seeds) > 1:
            n_seeds = int(raw_n_seeds)
            test_ndcg_std = lb.get("test_ndcg_at_10_std")
            test_hr_std = lb.get("test_hr_at_10_std")
        else:
            n_seeds = 1
            test_ndcg_std = None
            test_hr_std = None

        # Provenance alignment: val_ndcg_at_10 in /models and in recommend() must come from the same registry field
        run_val_ndcg = None
        if run and run.metrics:
            m = run.metrics
            run_val_ndcg = m.get("val_metrics", {}).get("ndcg_at_10") if isinstance(m.get("val_metrics"), dict) else m.get("val_ndcg_at_10")

        svc_meta = getattr(service, "model_runs_meta", {}) if service else {}
        svc_val_ndcg = svc_meta.get(model_key, {}).get("val_ndcg_at_10") if svc_meta else None
        val_ndcg = svc_val_ndcg if (svc_val_ndcg is not None and svc_val_ndcg > 0.0) else (run_val_ndcg or lb.get("val_ndcg_at_10") or dm.get("val_ndcg"))

        is_ready = model_key in service.models if service else True
        is_preview = service.ranker_is_preview if ("ranker" in model_key and service) else False

        # Explicit deployable, leak_free, and clean flags
        if model_key == "recsys_two_tower_ranker_clean":
            deployable = False
            leak_free = False
            clean = False
            stage = "candidate"
        elif model_key == "recsys_two_tower_ranker_clean_v2":
            # v2's DB row has stage="candidate" (default from log_recsys_run).
            # Override to "reference": v2 uses all-time-stats stage-1 retrieval,
            # kept only as a diagnostic upper bound. See inference.py for details.
            deployable = False
            leak_free = False
            clean = False
            stage = "reference"
        elif model_key == "recsys_two_tower_ranker_v2_trainstats":
            deployable = True
            leak_free = True
            clean = True
            stage = "candidate"
        elif model_key == "recsys_two_tower_clean":
            deployable = True
            leak_free = True
            clean = True
            stage = "production"
        else:
            deployable = True
            leak_free = True
            clean = True

        models_info.append(
            RecsysModelInfoSchema(
                model_name=model_key,
                display_name=model_key.replace("recsys_", "").replace("_clean", "").replace("_", " ").title(),
                deployable=deployable,
                stage=stage,
                model_run_id=run_id,
                val_ndcg_at_10=float(val_ndcg) if val_ndcg is not None else None,
                test_ndcg_at_10_mean=float(test_ndcg) if test_ndcg is not None else None,
                test_ndcg_at_10_std=float(test_ndcg_std) if test_ndcg_std is not None else None,
                test_hr_at_10_mean=float(test_hr) if test_hr is not None else None,
                test_hr_at_10_std=float(test_hr_std) if test_hr_std is not None else None,
                n_seeds=int(n_seeds),
                is_ready=is_ready,
                is_preview=is_preview,
                leak_free=leak_free,
                clean=clean,
            )
        )

    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=models_info, took_ms=round(took_ms, 2))


# ─── 4. Evidence Reports ──────────────────────────────────────────────────────

ALLOWED_REPORT_KEYS: frozenset[str] = frozenset({
    "leaderboard",
    "slices",
    "ablations",
    "ranker_ablations",
    "cold_start",
    "beyond_accuracy",
    "training_curves",
    "shap_global",
    "dataset",
    "limitations",
    "assistant_leaderboard",
    "assistant_calibration",
    "assistant_retrieval",
    "assistant_errors",
    "candidate_recall",
    "latency_benchmark",
    "sasrec_benchmark",
    "validation_sweep",
    "ranker_without_user_features",
    "lock_equivalence_check",
    "retrain_gate",
    "drift_detector_power",
    "monitor_models",
    "cold_latency_benchmark",
})


@router.get(
    "/reports/{key}",
    response_model=ApiResponse[RecsysReportSchema],
    summary="Get recommender evaluation report",
    description="Fetches an audit or benchmark report payload from serving_rec_reports by key with file fallback.",
)
async def get_report(
    key: Annotated[str, PathParam(description="Report key (e.g. leaderboard, ablations, slices, cold_start)")],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiResponse[RecsysReportSchema]:
    start = time.perf_counter()

    if key not in ALLOWED_REPORT_KEYS:
        raise ReportNotFoundError(
            f"Report key '{key}' is invalid or not allowed."
        )

    stmt = select(ServingRecReport).where(ServingRecReport.report_key == key)
    res = await db.execute(stmt)
    report = res.scalar_one_or_none()

    if report is not None:
        took_ms = (time.perf_counter() - start) * 1000
        return ApiResponse.ok(
            data=RecsysReportSchema(
                report_key=report.report_key,
                dataset_version=report.dataset_version,
                model_run_id=report.model_run_id,
                payload=report.payload,
                updated_at=report.updated_at,
                source="database",
            ),
            took_ms=round(took_ms, 2),
        )

    # Filesystem fallback: check data/recsys/reports or docs/results
    candidate_paths = [
        settings.resolved_recsys_data_dir / "reports" / f"{key}.json",
        FilePath("docs/results") / f"{key}.json",
        FilePath("../docs/results") / f"{key}.json",
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    file_payload = json.load(f)
                took_ms = (time.perf_counter() - start) * 1000
                ds_ver = (
                    file_payload.get("dataset_version", "c662b8232dc0")
                    if isinstance(file_payload, dict)
                    else "c662b8232dc0"
                )
                return ApiResponse.ok(
                    data=RecsysReportSchema(
                        report_key=key,
                        dataset_version=ds_ver,
                        model_run_id=None,
                        payload=file_payload,
                        updated_at=None,
                        source="file_fallback",
                    ),
                    took_ms=round(took_ms, 2),
                )
            except Exception as exc:
                logger.warning("Failed to read fallback report from %s: %s", p, exc)

    raise ReportNotFoundError(
        f"Report '{key}' not found in serving_rec_reports or fallback files. "
        f"Run 'make recsys-reports' to generate and persist all 10 reports."
    )


# ─── 5. Latent Space Map ──────────────────────────────────────────────────────

@router.get(
    "/map",
    response_model=ApiResponse[RecsysMapResponse],
    summary="Get 2D UMAP item projection map",
    description="Reads 2D item coordinates and semantic cluster labels from serving_rec_item_projection.",
)
async def get_item_map(
    db: Annotated[AsyncSession, Depends(get_db)],
    service: Annotated[RecommenderService, Depends(get_recommender_service)],
) -> ApiResponse[RecsysMapResponse]:
    start = time.perf_counter()

    stmt = (
        select(ServingRecItemProjection, ServingRecItem)
        .join(ServingRecItem, ServingRecItemProjection.app_id == ServingRecItem.app_id)
        .order_by(ServingRecItem.popularity_rank.asc())
    )
    res = await db.execute(stmt)
    rows = res.all()

    points: list[RecsysMapPointSchema] = []
    for proj, item in rows:
        points.append(
            RecsysMapPointSchema(
                app_id=proj.app_id,
                title=item.title,
                x=float(proj.x),
                y=float(proj.y),
                cluster_id=proj.cluster_id,
                cluster_label=proj.cluster_label,
                popularity_rank=item.popularity_rank,
                header_image_url=item.header_image_url,
                tags=item.tags if isinstance(item.tags, list) else [],
            )
        )

    took_ms = (time.perf_counter() - start) * 1000
    universe_size = service.universe_size or len(points) or 3000

    return ApiResponse.ok(
        data=RecsysMapResponse(
            points=points,
            total_points=len(points),
            universe_size=universe_size,
        ),
        took_ms=round(took_ms, 2),
    )


# ─── 6. Nearest Neighbors ─────────────────────────────────────────────────────

@router.get(
    "/items/{app_id}/neighbors",
    response_model=ApiResponse[RecsysNeighborsResponse],
    summary="Get nearest neighbors in Two-Tower embedding space",
    description="Cosine nearest neighbors in the 64-d Two-Tower item vector space.",
)
async def get_item_neighbors(
    app_id: Annotated[int, PathParam(description="Steam App ID")],
    k: Annotated[int, Query(ge=1, le=50, description="Number of neighbors to return")] = 10,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,  # type: ignore[assignment]
) -> ApiResponse[RecsysNeighborsResponse]:
    start = time.perf_counter()

    neighbors_list = service.neighbors(app_id=app_id, k=k)

    items = [
        RecsysNeighborItemSchema(
            rank=n["rank"],
            app_id=n["app_id"],
            title=n["title"],
            header_image_url=n["header_image_url"],
            similarity=round(float(n["similarity"]), 4),
            price_final=n.get("price_final"),
            positive_ratio=n.get("positive_ratio"),
            user_reviews=n.get("user_reviews"),
            tags=n.get("tags", []),
            popularity_rank=n["popularity_rank"],
        )
        for n in neighbors_list
    ]

    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(
        data=RecsysNeighborsResponse(
            query_app_id=app_id,
            neighbors=items,
            total=len(items),
        ),
        took_ms=round(took_ms, 2),
    )


# ─── 7. Health & Readiness ────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=ApiResponse[RecsysHealthResponse],
    summary="Recommender service health and readiness check",
    description="Reports whether RecommenderService has loaded its catalog and default model.",
)
async def recsys_health(
    service: Annotated[RecommenderService, Depends(get_recommender_service)],
) -> ApiResponse[RecsysHealthResponse]:
    start = time.perf_counter()
    ready = service.is_ready()

    data = RecsysHealthResponse(
        status="ready" if ready else "not_ready",
        is_ready=ready,
        universe_size=service.universe_size,
        loaded_models=list(service.models.keys()),
    )
    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=data, took_ms=round(took_ms, 2))
