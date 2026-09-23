import asyncio
import logging
from datetime import datetime
from sqlalchemy import select

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
