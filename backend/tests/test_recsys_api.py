"""
Unit and integration tests for Recommender System Serving API — Phase 6.

Validates:
  1. Request validation (empty list, >50 games, invalid model names -> 422).
  2. MODEL_NOT_READY 503 response envelope.
  3. Response envelope shapes and field structures.
  4. Search ordering by popularity rank.
  5. Real service refusing inference without artifacts (raises ModelNotReadyError).
  6. Golden Rule static AST/source scan (zero job imports, zero raw/feature table access).
"""
from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from core.app import create_app
from core.errors import GameNotFoundError, ModelNotReadyError, ValidationError
from db.base import get_db
from db.models import ServingRecItem, ServingRecItemProjection, ServingRecReport
from recsys.inference import RecommenderService, get_recommender_service

# ─── Fake Recommender Service for Isolated API Testing ────────────────────────

class FakeRecommenderService:
    """Mock RecommenderService returning deterministic synthetic recs."""

    def __init__(self, is_ready: bool = True) -> None:
        self._ready = is_ready
        self.universe_size = 3000
        self.n_items = 3000
        self.ranker_is_preview = False
        self.models = {
            "recsys_popularity": object(),
            "recsys_content": object(),
            "recsys_itemknn": object(),
            "recsys_als": object(),
            "recsys_two_tower_clean": object(),
            "recsys_two_tower_ranker_clean": object(),
        } if is_ready else {}

        self.app_id_to_idx = {
            10: 0,
            20: 1,
            30: 2,
            40: 3,
            50: 4,
            1145360: 5,
            730: 6,
        }
        self.idx_to_app_id = {v: k for k, v in self.app_id_to_idx.items()}

        self.items_metadata = {
            0: {"app_id": 10, "item_idx": 0, "title": "Counter-Strike", "popularity_rank": 1, "tags": ["Action", "FPS"], "header_image_url": "http://img/10.jpg", "price_final": 9.99, "positive_ratio": 97, "user_reviews": 120000, "win": True, "mac": True, "linux": True, "steam_deck": True},
            1: {"app_id": 20, "item_idx": 1, "title": "Team Fortress Classic", "popularity_rank": 4, "tags": ["Action", "Hero Shooter"], "header_image_url": "http://img/20.jpg", "price_final": 4.99, "positive_ratio": 85, "user_reviews": 15000, "win": True, "mac": True, "linux": True, "steam_deck": False},
            2: {"app_id": 30, "item_idx": 2, "title": "Day of Defeat", "popularity_rank": 5, "tags": ["FPS", "WW2"], "header_image_url": "http://img/30.jpg", "price_final": 4.99, "positive_ratio": 90, "user_reviews": 8000, "win": True, "mac": True, "linux": True, "steam_deck": False},
            3: {"app_id": 40, "item_idx": 3, "title": "Deathmatch Classic", "popularity_rank": 6, "tags": ["Action", "Multiplayer"], "header_image_url": "http://img/40.jpg", "price_final": 4.99, "positive_ratio": 80, "user_reviews": 2000, "win": True, "mac": False, "linux": False, "steam_deck": False},
            4: {"app_id": 50, "item_idx": 4, "title": "Half-Life: Opposing Force", "popularity_rank": 3, "tags": ["Sci-Fi", "Classic"], "header_image_url": "http://img/50.jpg", "price_final": 4.99, "positive_ratio": 95, "user_reviews": 20000, "win": True, "mac": True, "linux": True, "steam_deck": True},
            5: {"app_id": 1145360, "item_idx": 5, "title": "Hollow Knight", "popularity_rank": 2, "tags": ["Metroidvania", "Souls-like"], "header_image_url": "http://img/1145360.jpg", "price_final": 14.99, "positive_ratio": 97, "user_reviews": 350000, "win": True, "mac": True, "linux": True, "steam_deck": True},
            6: {"app_id": 730, "item_idx": 6, "title": "Counter-Strike 2", "popularity_rank": 1, "tags": ["FPS", "Shooter", "Competitive"], "header_image_url": "http://img/730.jpg", "price_final": 0.0, "positive_ratio": 88, "user_reviews": 8000000, "win": True, "mac": False, "linux": True, "steam_deck": True},
        }

    def is_ready(self) -> bool:
        return self._ready

    def search_items(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        clean_q = query.strip().lower()
        matched = [
            m for m in self.items_metadata.values()
            if not clean_q or clean_q in m["title"].lower()
        ]
        matched.sort(key=lambda x: x["popularity_rank"])
        return matched[:limit]

    def recommend(
        self,
        liked_app_ids: list[int],
        model: str | None = None,
        k: int = 20,
        explain: bool = True,
    ) -> dict[str, Any]:
        if not self._ready:
            raise ModelNotReadyError("RecommenderService is not ready (artifacts missing).")

        if not liked_app_ids:
            raise ValidationError("liked_app_ids list cannot be empty.")
        if len(liked_app_ids) > 50:
            raise ValidationError("liked_app_ids cannot exceed 50 items.")

        canonical_model = "recsys_two_tower_clean"
        if model:
            from recsys.inference import MODEL_ALIASES
            clean = model.strip().lower()
            if clean not in MODEL_ALIASES:
                raise ValidationError(f"Unknown model '{model}'.")
            canonical_model = MODEL_ALIASES[clean]

        known = [aid for aid in liked_app_ids if aid in self.app_id_to_idx]
        unknown = [aid for aid in liked_app_ids if aid not in self.app_id_to_idx]
        if not known:
            raise ValidationError("None of the provided app_ids exist in the universe.")

        warning = "Fewer than 3 known library games provided." if len(known) < 3 else None

        items = []
        for rank_num, (_idx, meta) in enumerate(list(self.items_metadata.items())[:k], start=1):
            if meta["app_id"] in known:
                continue
            items.append({
                "rank": rank_num,
                "app_id": meta["app_id"],
                "item_idx": meta["item_idx"],
                "title": meta["title"],
                "header_image_url": meta["header_image_url"],
                "score": round(1.0 / (rank_num + 1), 4),
                "price_final": meta["price_final"],
                "positive_ratio": meta["positive_ratio"],
                "user_reviews": meta["user_reviews"],
                "tags": meta["tags"],
                "description_short": "A great game",
                "popularity_rank": meta["popularity_rank"],
                "win": meta["win"],
                "mac": meta["mac"],
                "linux": meta["linux"],
                "steam_deck": meta["steam_deck"],
                "because_you_liked": [{"app_id": known[0], "title": "Known Anchor", "similarity": 0.88}] if explain else [],
                "reasons": ["Top collaborative match"] if explain else [],
            })

        return {
            "items": items,
            "total_items": len(items),
            "model_name": canonical_model,
            "model_run_id": "00000000-0000-0000-0000-000000000001",
            "val_ndcg_at_10": 0.0611,
            "dataset_version": "c662b8232dc0",
            "universe_size": self.universe_size,
            "unknown_app_ids": unknown,
            "warning": warning,
            "took_ms": 12.5,
        }

    def neighbors(self, app_id: int, k: int = 10) -> list[dict[str, Any]]:
        if not self._ready:
            raise ModelNotReadyError("Model not ready.")
        if app_id not in self.app_id_to_idx:
            raise GameNotFoundError(f"Game with app_id={app_id} not found.")

        results = []
        for r, (_idx, meta) in enumerate(list(self.items_metadata.items())[:k], start=1):
            if meta["app_id"] == app_id:
                continue
            results.append({
                "rank": r,
                "app_id": meta["app_id"],
                "title": meta["title"],
                "header_image_url": meta["header_image_url"],
                "similarity": 0.95 - (r * 0.05),
                "price_final": meta["price_final"],
                "positive_ratio": meta["positive_ratio"],
                "user_reviews": meta["user_reviews"],
                "tags": meta["tags"],
                "popularity_rank": meta["popularity_rank"],
            })
        return results


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_service():
    return FakeRecommenderService(is_ready=True)


@pytest.fixture
def fake_unready_service():
    return FakeRecommenderService(is_ready=False)


@pytest.fixture
async def recsys_client(db_session: AsyncSession, fake_service: FakeRecommenderService):
    """FastAPI AsyncClient with fake RecommenderService and in-memory DB session."""
    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_recommender_service] = lambda: fake_service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ─── Helper for Populating Test Database ──────────────────────────────────────

async def _seed_test_recsys_db(session: AsyncSession) -> None:
    # 1. Serving items
    item1 = ServingRecItem(
        app_id=10,
        item_idx=0,
        title="Counter-Strike",
        date_release="2000-11-01",
        price_final=Decimal("9.99"),
        positive_ratio=97,
        user_reviews=120000,
        tags=["Action", "FPS"],
        description_short="Action FPS game",
        has_description=True,
        train_positives=500,
        popularity_rank=1,
        win=True,
        mac=True,
        linux=True,
        steam_deck=True,
        header_image_url="http://img/10.jpg",
    )
    item2 = ServingRecItem(
        app_id=1145360,
        item_idx=1,
        title="Hollow Knight",
        date_release="2017-02-24",
        price_final=Decimal("14.99"),
        positive_ratio=97,
        user_reviews=350000,
        tags=["Metroidvania", "Souls-like", "NSFW", "Hentai"],
        description_short="Challenging action game",
        has_description=True,
        train_positives=900,
        popularity_rank=2,
        win=True,
        mac=True,
        linux=True,
        steam_deck=True,
        header_image_url="http://img/1145360.jpg",
    )
    session.add_all([item1, item2])
    await session.commit()

    # 2. Projection
    proj1 = ServingRecItemProjection(
        app_id=10,
        x=1.23,
        y=-4.56,
        cluster_id=0,
        cluster_label="FPS, Action",
    )
    proj2 = ServingRecItemProjection(
        app_id=1145360,
        x=5.67,
        y=8.90,
        cluster_id=1,
        cluster_label="NSFW, Hentai",
    )
    session.add_all([proj1, proj2])

    # 3. Report
    rep = ServingRecReport(
        report_key="leaderboard",
        dataset_version="c662b8232dc0",
        payload={
            "rows": [
                {
                    "model_name": "recsys_two_tower_clean",
                    "test_ndcg_at_10_mean": 0.0516,
                    "test_ndcg_at_10_std": 0.0008,
                    "test_hr_at_10_mean": 0.1028,
                    "test_hr_at_10_std": 0.0012,
                    "val_ndcg_at_10": 0.0611,
                    "n_seeds": 3,
                }
            ]
        },
    )
    session.add(rep)
    await session.commit()


# ─── 1. Endpoint & Envelope Tests ─────────────────────────────────────────────

@pytest.mark.anyio
async def test_for_you_endpoint_success(recsys_client: AsyncClient, db_session: AsyncSession) -> None:
    await _seed_test_recsys_db(db_session)

    payload = {
        "liked_app_ids": [10, 20, 30, 40, 50],
        "model": "two_tower_clean",
        "k": 10,
        "explain": True,
    }
    resp = await recsys_client.post("/api/v1/recsys/for-you", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    assert "data" in body
    data = body["data"]

    assert data["model_name"] == "recsys_two_tower_clean"
    assert len(data["items"]) <= 10
    assert data["universe_size"] == 3000
    assert data["took_ms"] > 0

    first_item = data["items"][0]
    assert "rank" in first_item
    assert "app_id" in first_item
    assert "title" in first_item
    assert "score" in first_item
    assert "popularity_rank" in first_item
    assert "because_you_liked" in first_item
    assert "reasons" in first_item


@pytest.mark.anyio
async def test_for_you_validation_empty_list(recsys_client: AsyncClient) -> None:
    resp = await recsys_client.post("/api/v1/recsys/for-you", json={"liked_app_ids": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"


@pytest.mark.anyio
async def test_for_you_validation_too_many_items(recsys_client: AsyncClient) -> None:
    too_many = list(range(1, 55))
    resp = await recsys_client.post("/api/v1/recsys/for-you", json={"liked_app_ids": too_many})
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"


@pytest.mark.anyio
async def test_for_you_validation_invalid_model_name(recsys_client: AsyncClient) -> None:
    resp = await recsys_client.post(
        "/api/v1/recsys/for-you",
        json={"liked_app_ids": [10, 20], "model": "super_gpt_recommender"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert "Valid deployable models" in str(body["error"]["message"])


@pytest.mark.anyio
async def test_for_you_model_not_ready_503(
    db_session: AsyncSession, fake_unready_service: FakeRecommenderService
) -> None:
    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_recommender_service] = lambda: fake_unready_service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/recsys/for-you",
            json={"liked_app_ids": [10, 20]},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "MODEL_NOT_READY"


@pytest.mark.anyio
async def test_search_ordering_by_popularity(recsys_client: AsyncClient, db_session: AsyncSession) -> None:
    await _seed_test_recsys_db(db_session)

    resp = await recsys_client.get("/api/v1/recsys/items/search?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    items = body["data"]["items"]
    assert len(items) >= 2
    # Verify ordered by popularity rank ascending
    ranks = [it["popularity_rank"] for it in items]
    assert ranks == sorted(ranks)


@pytest.mark.anyio
async def test_models_endpoint(recsys_client: AsyncClient, db_session: AsyncSession) -> None:
    await _seed_test_recsys_db(db_session)

    resp = await recsys_client.get("/api/v1/recsys/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    models = body["data"]
    assert len(models) >= 6

    model_names = [m["model_name"] for m in models]
    assert "recsys_popularity" in model_names
    assert "recsys_two_tower_clean" in model_names
    assert "recsys_two_tower_ranker_clean" in model_names

    # Check clean production champion
    champ = next(m for m in models if m["model_name"] == "recsys_two_tower_clean")
    assert champ["deployable"] is True
    assert champ["stage"] == "production"
    assert champ["test_ndcg_at_10_mean"] is not None


@pytest.mark.anyio
async def test_reports_endpoint_found_and_not_found(
    recsys_client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_test_recsys_db(db_session)

    # 1. Existing report
    resp = await recsys_client.get("/api/v1/recsys/reports/leaderboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["report_key"] == "leaderboard"

    # 2. Non-existent report -> 404
    resp_missing = await recsys_client.get("/api/v1/recsys/reports/nonexistent_report")
    assert resp_missing.status_code == 404
    missing_body = resp_missing.json()
    assert missing_body["success"] is False
    assert missing_body["error"]["code"] == "report_not_found"


@pytest.mark.anyio
async def test_item_map_endpoint(recsys_client: AsyncClient, db_session: AsyncSession) -> None:
    await _seed_test_recsys_db(db_session)

    resp = await recsys_client.get("/api/v1/recsys/map")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "points" in body["data"]
    assert body["data"]["total_points"] >= 2
    pt = body["data"]["points"][0]
    assert "x" in pt
    assert "y" in pt
    assert "cluster_label" in pt

    # Verify NSFW/Hentai labels and tags are sanitized to clean, appropriate values
    pt2 = body["data"]["points"][1]
    assert pt2["cluster_label"] == "Anime, Visual Novel"
    assert "NSFW" not in pt2["tags"]
    assert "Hentai" not in pt2["tags"]
    assert "Metroidvania" in pt2["tags"]


@pytest.mark.anyio
async def test_neighbors_endpoint(recsys_client: AsyncClient) -> None:
    resp = await recsys_client.get("/api/v1/recsys/items/10/neighbors?k=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["query_app_id"] == 10
    assert len(body["data"]["neighbors"]) <= 5

    # Unknown app_id -> 404
    resp_404 = await recsys_client.get("/api/v1/recsys/items/9999999/neighbors")
    assert resp_404.status_code == 404
    assert resp_404.json()["error"]["code"] == "game_not_found"


@pytest.mark.anyio
async def test_recsys_health(recsys_client: AsyncClient) -> None:
    resp = await recsys_client.get("/api/v1/recsys/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ready"
    assert body["data"]["is_ready"] is True


# ─── 2. Real RecommenderService Refusal Without Artifacts ─────────────────────

def test_real_service_refuses_when_no_artifacts(tmp_path: Path) -> None:
    """Real service initialized with an empty directory safely raises ModelNotReadyError on recommend."""
    empty_dir = tmp_path / "empty_recsys"
    empty_dir.mkdir(parents=True, exist_ok=True)

    svc = RecommenderService(data_dir=empty_dir)
    svc.initialize()

    assert not svc.is_ready()

    # Calling recommend on an unready service raises ModelNotReadyError (503), never crash
    with pytest.raises(ModelNotReadyError) as exc_info:
        svc.recommend(liked_app_ids=[10, 20])

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "MODEL_NOT_READY"


# ─── 3. Golden Rule Static Source Code Scan Audit ─────────────────────────────

def test_golden_rule_api_recsys_code_audit() -> None:
    """
    Asserts that api/recsys.py strictly adheres to ADR 0001 / ADR 0005:
      - Never imports from jobs or backend.jobs
      - Never queries raw_* or feature_* or mart_* tables
      - Reads strictly serving_rec_* and model_runs
    """
    recsys_api_file = Path(__file__).parent.parent / "api" / "recsys.py"
    assert recsys_api_file.exists(), f"File {recsys_api_file} must exist."

    code = recsys_api_file.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(recsys_api_file))

    forbidden_module_prefixes = ("backend.jobs", "jobs")
    forbidden_symbols = {
        "RawGame", "RawReview", "RawNews",
        "FeatureGameMarket", "FeatureMarketFeature",
        "MartGameOverview", "MartGenreTrend", "MartOpportunity", "MartGameMatchProfile",
    }
    forbidden_sql_tables = ("raw_games", "raw_reviews", "feature_game", "feature_market", "mart_")

    # 1. AST check on imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_module_prefixes:
                    assert not (alias.name == prefix or alias.name.startswith(f"{prefix}.")), (
                        f"Golden Rule Violation in api/recsys.py: imported {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for prefix in forbidden_module_prefixes:
                assert not (mod_name == prefix or mod_name.startswith(f"{prefix}.")), (
                    f"Golden Rule Violation in api/recsys.py: from {mod_name} import ..."
                )

    # 2. AST check on identifiers and string constants (excluding docstrings)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_symbols, (
                f"Golden Rule Violation in api/recsys.py: references forbidden model '{node.id}'"
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Check if string constant is a SQL string referencing forbidden tables
            val_lower = node.value.lower()
            for tbl in forbidden_sql_tables:
                assert tbl not in val_lower, (
                    f"Golden Rule Violation in api/recsys.py: references forbidden table '{tbl}' in query"
                )
