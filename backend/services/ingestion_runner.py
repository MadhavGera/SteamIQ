import asyncio
from datetime import datetime, timedelta
import logging
import os
from typing import Any
from redis import Redis
from rq import Queue, SimpleWorker, Worker
from rq.job import Job
from rq.registry import ScheduledJobRegistry
from sqlalchemy import select

from core.config import settings
from db.base import AsyncSessionLocal
from db.models import IngestionJob

from jobs.ingest_games import IngestGamesJob
from jobs.process_reviews import ProcessReviewsJob
from jobs.process_embeddings import ProcessEmbeddingsJob
from jobs.materialize_marts import MaterializeMartsJob
from jobs.materialize_match_profiles import MaterializeMatchProfilesJob
from jobs.materialize_update_impact import MaterializeUpdateImpactJob
from jobs.generate_recommendations import GenerateRecommendationsJob

logger = logging.getLogger(__name__)

RECURRING_PRICE_SNAPSHOT_JOB_ID = "recurring_daily_price_snapshots"
DEFAULT_SNAPSHOT_INTERVAL = timedelta(days=1)

async def run_full_pipeline_async(job_id: str, app_id: int):
    # 1. Update status to running
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
        job = res.scalar_one_or_none()
        if not job:
            logger.error(f"IngestionJob {job_id} not found!")
            return
        
        job.status = "running"
        job.updated_at = datetime.now()
        await db.commit()

    try:
        # 1. Ingest base game metadata and raw reviews (Synchronous)
        logger.info(f"[{job_id}] Running ingest_games...")
        await asyncio.to_thread(IngestGamesJob(app_id=app_id).execute)
        
        # 2. Process NLP topics on raw reviews (Async run method)
        logger.info(f"[{job_id}] Running process_reviews...")
        await ProcessReviewsJob().run(app_id=app_id)
        
        # 3. Generate vectors
        logger.info(f"[{job_id}] Running process_embeddings...")
        await ProcessEmbeddingsJob(app_id=app_id)._run_async()
        
        # 4. Materialize data marts for UI dashboard
        logger.info(f"[{job_id}] Running materialize_marts...")
        await MaterializeMartsJob(app_id=app_id)._run_async()
        
        # 5. Materialize match profile for Game Intelligence feature
        logger.info(f"[{job_id}] Running materialize_match_profiles...")
        async with AsyncSessionLocal() as session:
            await MaterializeMatchProfilesJob(app_id=app_id)._run_async(session)
            await session.commit()
        
        # 6. Materialize update impact
        logger.info(f"[{job_id}] Running materialize_update_impact...")
        await MaterializeUpdateImpactJob(app_id=app_id)._run_async()

        # 7. Generate recommendations
        logger.info(f"[{job_id}] Running generate_recommendations...")
        await GenerateRecommendationsJob(app_id=app_id)._run_async()

        # Mark success
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = res.scalar_one()
            job.status = "completed"
            job.updated_at = datetime.now()
            await db.commit()
            
        logger.info(f"Pipeline completed successfully for job {job_id}")

    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = res.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.updated_at = datetime.now()
                await db.commit()

def run_full_pipeline(job_id: str, app_id: int):
    """
    RQ Worker task that runs the full ingestion and ML pipeline for a given app_id.
    """
    logger.info(f"Starting async ingestion pipeline for job {job_id} (app_id={app_id})")
    asyncio.run(run_full_pipeline_async(job_id, app_id))

def run_recurring_price_snapshots() -> dict[str, Any]:
    """
    RQ Worker task: executes daily recurring price snapshot ingestion for all catalog games,
    and automatically re-enqueues the next daily cycle (24h).
    """
    logger.info("Executing scheduled recurring price snapshots for catalog games...")
    job = IngestGamesJob(snapshot_all_prices=True)
    result = job.execute()

    # Re-enqueue for next 24 hours to maintain continuous daily schedule
    try:
        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue(connection=redis_conn)
        register_scheduled_price_snapshots(queue)
    except Exception as e:
        logger.warning(f"Failed to re-enqueue recurring price snapshot: {e}")

    return result

def register_scheduled_price_snapshots(
    queue: Queue | None = None,
    interval: timedelta = DEFAULT_SNAPSHOT_INTERVAL,
) -> Job | None:
    """
    Registers the recurring daily price snapshot job with the RQ scheduler.
    If the job is already scheduled or queued, returns the existing job to avoid duplication.
    """
    if queue is None:
        redis_conn = Redis.from_url(settings.redis_url)
        queue = Queue(connection=redis_conn)

    registry = ScheduledJobRegistry(queue=queue)
    scheduled_ids = registry.get_job_ids()

    if RECURRING_PRICE_SNAPSHOT_JOB_ID in scheduled_ids:
        logger.info(
            f"Scheduled job '{RECURRING_PRICE_SNAPSHOT_JOB_ID}' already present in registry."
        )
        return queue.fetch_job(RECURRING_PRICE_SNAPSHOT_JOB_ID)

    if RECURRING_PRICE_SNAPSHOT_JOB_ID in queue.get_job_ids():
        logger.info(
            f"Job '{RECURRING_PRICE_SNAPSHOT_JOB_ID}' is already queued in '{queue.name}'."
        )
        return queue.fetch_job(RECURRING_PRICE_SNAPSHOT_JOB_ID)

    job = queue.enqueue_in(
        interval,
        run_recurring_price_snapshots,
        job_id=RECURRING_PRICE_SNAPSHOT_JOB_ID,
        job_timeout="2h",
        description="Daily recurring price snapshot for catalog games",
    )
    logger.info(
        f"Registered '{RECURRING_PRICE_SNAPSHOT_JOB_ID}' with scheduler (interval={interval}) on queue '{queue.name}'."
    )
    return job

def start_worker():
    """
    Entry point for the SteamIQ RQ worker service.
    Registers scheduled recurring jobs and runs the worker with scheduler enabled.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing SteamIQ RQ worker...")
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue(connection=redis_conn)

    # Register recurring jobs at worker startup
    job = register_scheduled_price_snapshots(queue)
    if job:
        logger.info(f"Registered recurring price snapshots job '{job.id}' on queue '{queue.name}'")

    worker_cls = SimpleWorker if os.name == "nt" else Worker
    worker = worker_cls([queue], connection=redis_conn)
    logger.info("Starting RQ worker with scheduler enabled...")
    worker.work(with_scheduler=True)

if __name__ == "__main__":
    start_worker()

