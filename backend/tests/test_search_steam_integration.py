import pytest
from httpx import Response
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient

from api.schemas.games import GameSummarySchema
from api.games import search_games
from core.response import ApiResponse

# A mock for MartGameOverview to return from DB
class MockMartGame:
    def __init__(self, app_id, name, positive_reviews=1000):
        self.app_id = app_id
        self.name = name
        self.short_description = "Mock desc"
        self.header_image = "img"
        self.developer = "Dev"
        self.publisher = "Pub"
        self.release_date = "2020-01-01"
        self.is_free = False
        self.final_price_usd = 19.99
        self.discount_pct = 0
        self.positive_reviews = positive_reviews
        self.negative_reviews = 100
        self.owners_estimate = "1M - 2M"
        self.genres = []
        self.primary_genre = "Action"
        self.revenue_tier = "Gold"
        self.success_score = 90.0
        self.net_sentiment_pct = 90.0

@pytest.fixture
def mock_db():
    class MockResult:
        def __init__(self, items):
            self.items = items
        def scalars(self):
            class Scalars:
                def all(self_inner):
                    return self.items
            return Scalars()

    class MockDB:
        def __init__(self):
            self.call_count = 0
        async def execute(self, query):
            self.call_count += 1
            if self.call_count == 1:
                class CountResult:
                    def scalar_one(self):
                        return 1
                return CountResult()
            return MockResult([MockMartGame(10, "Elden Ring")])
    return MockDB()

@pytest.mark.asyncio
async def test_search_local_only_match(mock_db):
    with patch("api.games.fetch_steam_search", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = []
        
        response = await search_games(q="Elden", page=1, page_size=20, db=mock_db)
        assert response.success is True
        games = response.data.games
        
        assert len(games) == 1
        assert games[0].app_id == 10
        assert games[0].name == "Elden Ring"
        assert games[0].is_ingested is True

@pytest.mark.asyncio
async def test_search_steam_only_match(mock_db):
    # Setup DB to return nothing
    class EmptyDB:
        def __init__(self):
            self.call_count = 0
        async def execute(self, query):
            self.call_count += 1
            if self.call_count == 1:
                class CountResult:
                    def scalar_one(self):
                        return 0
                return CountResult()
            class MockResult:
                def scalars(self):
                    class Scalars:
                        def all(self_inner):
                            return []
                    return Scalars()
            return MockResult()

    with patch("api.games.fetch_steam_search", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [
            {"id": 1245620, "name": "Elden Ring", "tiny_image": "img_url"}
        ]
        
        response = await search_games(q="Elden", page=1, page_size=20, db=EmptyDB())
        assert response.success is True
        games = response.data.games
        
        assert len(games) == 1
        assert games[0].app_id == 1245620
        assert games[0].name == "Elden Ring"
        assert games[0].is_ingested is False
        assert games[0].header_image == "img_url"
        # Check honest fallback (null fields)
        assert games[0].developer is None
        assert games[0].final_price_usd is None

@pytest.mark.asyncio
async def test_search_dedup_match(mock_db):
    # DB returns app_id 10
    with patch("api.games.fetch_steam_search", new_callable=AsyncMock) as mock_fetch:
        # Steam API also returns app_id 10, plus app_id 20
        mock_fetch.return_value = [
            {"id": 10, "name": "Elden Ring (Steam)", "tiny_image": "img1"},
            {"id": 20, "name": "Elden Ring 2", "tiny_image": "img2"}
        ]
        
        response = await search_games(q="Elden", page=1, page_size=20, db=mock_db)
        assert response.success is True
        games = response.data.games
        
        assert len(games) == 2
        # First should be the local DB one (is_ingested = True, name = "Elden Ring")
        assert games[0].app_id == 10
        assert games[0].is_ingested is True
        assert games[0].name == "Elden Ring"
        
        # Second should be from Steam API (is_ingested = False)
        assert games[1].app_id == 20
        assert games[1].is_ingested is False
        assert games[1].name == "Elden Ring 2"

@pytest.mark.asyncio
async def test_search_steam_api_timeout_degrades_gracefully(mock_db):
    # We'll patch httpx.AsyncClient.get to raise a TimeoutException
    from httpx import TimeoutException
    
    with patch("httpx.AsyncClient.get", side_effect=TimeoutException("Timeout")):
        # clear cache to ensure we make the API call
        from api.games import _steam_search_cache
        _steam_search_cache.clear()

        response = await search_games(q="Elden", page=1, page_size=20, db=mock_db)
        assert response.success is True
        games = response.data.games
        
        # Should gracefully fallback to only local results
        assert len(games) == 1
        assert games[0].app_id == 10
        assert games[0].is_ingested is True
