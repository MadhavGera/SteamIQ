import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone, timedelta
from redis import Redis
from rq import Queue
from sqlalchemy.exc import IntegrityError

from core.config import settings
from db.base import get_db
from db.models import IngestionJob
from services.ingestion_runner import run_full_pipeline

router = APIRouter(tags=["ingestion"])

# Initialize Redis and RQ
redis_conn = Redis.from_url(settings.redis_url)
q = Queue(connection=redis_conn)

class TriggerRequest(BaseModel):
    app_id: int

@router.post("/trigger")
async def trigger_ingestion(req: TriggerRequest, request: Request, db: AsyncSession = Depends(get_db)):
    app_id = req.app_id
    
    # 1. Rate Limiting by IP
    client_ip = request.client.host if request.client else "unknown"
    rate_limit_key = f"rate_limit:ingest:{client_ip}"
    
    current_count = redis_conn.get(rate_limit_key)
    if current_count and int(current_count) >= 3:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        
    # 2. Validate against Steam API
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"https://store.steampowered.com/api/appdetails?appids={app_id}")
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="Steam API error.")
            
            data = resp.json()
            if str(app_id) not in data or not data[str(app_id)].get("success"):
                raise HTTPException(status_code=404, detail=f"App ID {app_id} not found on Steam.")
        except httpx.RequestError:
            raise HTTPException(status_code=502, detail="Failed to connect to Steam API.")
            
    redis_conn.incr(rate_limit_key)
    redis_conn.expire(rate_limit_key, 3600)  # 1 hour
    
    # 3. Create job (with dedup check)
    new_job = IngestionJob(app_id=app_id, status="pending")
    db.add(new_job)
    try:
        await db.commit()
        await db.refresh(new_job)
    except IntegrityError:
        await db.rollback()
        res = await db.execute(
            select(IngestionJob).where(IngestionJob.app_id == app_id, IngestionJob.status == 'running')
        )
        existing_job = res.scalar_one_or_none()
        if existing_job:
            return {"job_id": existing_job.id, "status": existing_job.status}
        raise HTTPException(status_code=409, detail="A job is already running for this app_id.")

    # 4. Enqueue RQ task
    q.enqueue(run_full_pipeline, new_job.id, app_id, job_timeout="1h")
    
    return {"job_id": new_job.id, "status": new_job.status}

@router.get("/status/{job_id}")
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    # Staleness check
    if job.status == "running":
        now = datetime.now(timezone.utc) if job.updated_at.tzinfo else datetime.now()
        if (now - job.updated_at) > timedelta(minutes=10):
            job.status = "failed"
            job.error_message = "Job timed out / worker died."
            job.updated_at = datetime.now()
            await db.commit()
            
    return {
        "job_id": job.id,
        "app_id": job.app_id,
        "status": job.status,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
