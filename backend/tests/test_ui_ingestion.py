import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta, timezone

from db.models import IngestionJob
from services.ingestion_runner import run_full_pipeline_async

pytestmark = pytest.mark.asyncio

@patch("api.ingestion.httpx.AsyncClient.get")
@patch("api.ingestion.q.enqueue")
@patch("api.ingestion.redis_conn")
async def test_trigger_success_full_pipeline(mock_redis, mock_enqueue, mock_get, client: AsyncClient):
    mock_redis.get.return_value = None
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"123": {"success": True}}
    
    # We must patch the context manager since `async with AsyncClient()` is used
    async def mock_async_get(*args, **kwargs):
        return mock_resp
    
    # We mock the return value of httpx.AsyncClient.get
    mock_get.side_effect = mock_async_get
    
    response = await client.post("/api/v1/ingestion/trigger", json={"app_id": 123})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    mock_enqueue.assert_called_once()


@patch("api.ingestion.httpx.AsyncClient.get")
@patch("api.ingestion.q.enqueue")
@patch("api.ingestion.redis_conn")
async def test_trigger_duplicate_rejection(mock_redis, mock_enqueue, mock_get, client: AsyncClient, db_session: AsyncSession):
    mock_redis.get.return_value = None
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"123": {"success": True}}
    async def mock_async_get(*args, **kwargs):
        return mock_resp
    mock_get.side_effect = mock_async_get
    
    # Insert a running job manually
    job = IngestionJob(app_id=123, status="running")
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    
    response = await client.post("/api/v1/ingestion/trigger", json={"app_id": 123})
    assert response.status_code == 409
    assert "A job is already running" in response.json()["error"]["message"]

@patch("api.ingestion.httpx.AsyncClient.get")
@patch("api.ingestion.q.enqueue")
@patch("api.ingestion.redis_conn")
async def test_trigger_rate_limit(mock_redis, mock_enqueue, mock_get, client: AsyncClient):
    # Setup redis to say rate limit is hit
    mock_redis.get.return_value = "3"
    
    response = await client.post("/api/v1/ingestion/trigger", json={"app_id": 123})
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]["message"]


async def test_staleness_timeout(client: AsyncClient, db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    stale_time = now - timedelta(minutes=15)
    
    job = IngestionJob(app_id=123, status="running")
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    
    # Manually override updated_at
    job.updated_at = stale_time
    await db_session.commit()
    
    response = await client.get(f"/api/v1/ingestion/status/{job.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "timed out" in data["error_message"]
    
    # Check DB was actually updated
    await db_session.refresh(job)
    assert job.status == "failed"


@patch("services.ingestion_runner.IngestGamesJob.execute", new_callable=MagicMock)
@patch("services.ingestion_runner.AsyncSessionLocal")
async def test_worker_failure_surfacing(mock_session_maker, mock_ingest, db_session: AsyncSession):
    # Insert a job
    job = IngestionJob(app_id=123, status="pending")
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    
    # Mock AsyncSessionLocal to yield the test db_session
    class MockAsyncSessionContextManager:
        async def __aenter__(self):
            return db_session
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    mock_session_maker.return_value = MockAsyncSessionContextManager()
    
    # Simulate a crash
    mock_ingest.side_effect = ValueError("Simulated pipeline crash")
    
    # Run the worker function directly
    await run_full_pipeline_async(job.id, 123)
    
    # Check the job status in DB
    await db_session.refresh(job)
    assert job.status == "failed"
    assert "Simulated pipeline crash" in job.error_message
