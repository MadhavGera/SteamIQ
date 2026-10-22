"""
Job: 2D Projection and Clustering of Two-Tower Item Embeddings — Phase 6.

Subclasses BaseJob.
Takes champion Two-Tower 64-d item vectors, runs UMAP (n_neighbors=15, min_dist=0.1, metric='cosine', random_state=seed)
to 2D coordinates, runs KMeans (k=20, random_state=seed) on the 64-d vectors, computes cluster tag lifts (minimum 20 items),
and upserts serving_rec_item_projection.

Usage:
  python -m jobs.project_item_embeddings
  make recsys-project
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import umap
from sklearn.cluster import KMeans
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun, ServingRecItemProjection
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.models.two_tower import TwoTowerModel


class ProjectItemEmbeddingsJob(BaseJob):
    job_name = "project_item_embeddings"

    def __init__(
        self,
        *,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.profile = profile
        self.data_dir = data_dir
        self.seed = int(seed)

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recsys dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)

        # 1. Locate champion Two-Tower artifact
        tt_artifact_path: Path | None = None
        if session is not None:
            # Check model_runs for production clean two-tower first
            stmt = (
                select(ModelRun)
                .where(ModelRun.model_name.in_(["recsys_two_tower_clean", "recsys_two_tower"]))
                .where(ModelRun.stage.in_(["production", "candidate"]))
                .order_by(ModelRun.created_at.desc())
            )
            res = await session.execute(stmt)
            runs = res.scalars().all()
            for r in runs:
                if r.artifact_path and Path(r.artifact_path).exists():
                    tt_artifact_path = Path(r.artifact_path)
                    self.logger.info("Found Two-Tower artifact from DB run %s: %s", r.id, tt_artifact_path)
                    break

        if tt_artifact_path is None or not tt_artifact_path.exists():
            # Fallback to scanning disk for recsys_two_tower_clean or recsys_two_tower
            for fam in ["recsys_two_tower_clean", "recsys_two_tower"]:
                art_base = resolved_data_dir / "artifacts" / fam
                if art_base.exists():
                    cand_dirs = sorted(
                        [d for d in art_base.iterdir() if d.is_dir()],
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )
                    if cand_dirs:
                        tt_artifact_path = cand_dirs[0]
                        self.logger.info("Found Two-Tower artifact on disk: %s", tt_artifact_path)
                        break

        if tt_artifact_path is None or not tt_artifact_path.exists():
            raise FileNotFoundError(
                f"No Two-Tower model artifact found in {resolved_data_dir / 'artifacts'} or database."
            )

        # 2. Load Two-Tower model and precompute 64-d item vectors
        self.logger.info("Loading Two-Tower model from %s...", tt_artifact_path)
        tt_model = TwoTowerModel.load(tt_artifact_path, data=data)
        item_vectors = tt_model.precompute_item_vectors()  # Shape: [N, 64], L2-normalized
        n_items = item_vectors.shape[0]
        self.logger.info("Precomputed 64-d item vectors for %d universe items.", n_items)

        # 3. UMAP 2D Projection (n_neighbors=15, min_dist=0.1, metric='cosine', random_state=seed)
        self.logger.info("Running UMAP dimensionality reduction (64-D -> 2D, seed=%d)...", self.seed)
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=self.seed,
        )
        coords_2d = reducer.fit_transform(item_vectors)

        # 4. KMeans Clustering on 64-d vectors (k=20, random_state=seed)
        self.logger.info("Running KMeans clustering (k=20, seed=%d)...", self.seed)
        kmeans = KMeans(n_clusters=20, random_state=self.seed, n_init=10)
        cluster_ids = kmeans.fit_predict(item_vectors)

        # 5. Label each cluster with its 2 most distinctive tags (highest lift, min 20 universe items)
        items_df = data.items_df
        if items_df is None or items_df.empty:
            raise ValueError("RecsysData items_df is missing or empty.")

        INAPPROPRIATE_TAG_PATTERNS = ("nsfw", "hentai", "sexual content", "nudity", "erotic")

        all_tags: set[str] = set()
        for t_val in items_df["tags"]:
            if t_val is not None and isinstance(t_val, np.ndarray | list):
                for t in t_val:
                    t_str = str(t).strip()
                    if t_str and not any(pat in t_str.lower() for pat in INAPPROPRIATE_TAG_PATTERNS):
                        all_tags.add(t_str)

        total_universe = len(items_df)
        tag_universe_counts: dict[str, int] = {t: 0 for t in all_tags}
        for t_val in items_df["tags"]:
            if t_val is not None and isinstance(t_val, np.ndarray | list):
                for t in set(str(x).strip() for x in t_val if str(x).strip()):
                    if t in tag_universe_counts:
                        tag_universe_counts[t] += 1

        # Minimum 20 items in universe for distinctiveness stability
        valid_tags = {t for t, count in tag_universe_counts.items() if count >= 20}
        self.logger.info("Filtered %d universe tags (>= 20 items) from %d total tags.", len(valid_tags), len(all_tags))

        cluster_labels: dict[int, str] = {}
        for c in range(20):
            c_indices = np.where(cluster_ids == c)[0]
            c_size = len(c_indices)
            if c_size == 0:
                cluster_labels[c] = "General"
                continue

            c_tag_counts = {t: 0 for t in valid_tags}
            for idx in c_indices:
                t_val = items_df.iloc[idx]["tags"]
                if t_val is not None and isinstance(t_val, np.ndarray | list):
                    for t in set(str(x).strip() for x in t_val if str(x).strip()):
                        if t in valid_tags:
                            c_tag_counts[t] += 1

            tag_lifts = []
            for t in valid_tags:
                if c_tag_counts[t] > 0:
                    p_cluster = c_tag_counts[t] / c_size
                    p_universe = tag_universe_counts[t] / total_universe
                    lift = p_cluster / p_universe
                    tag_lifts.append((t, lift, c_tag_counts[t]))

            tag_lifts.sort(key=lambda x: (x[1], x[2]), reverse=True)
            top_2 = [t[0] for t in tag_lifts[:2]]
            raw_label = ", ".join(top_2) if top_2 else "General"
            if any(pat in raw_label.lower() for pat in INAPPROPRIATE_TAG_PATTERNS):
                raw_label = "Anime, Visual Novel"
            cluster_labels[c] = raw_label
            self.logger.info("  Cluster %2d (%3d items): %s", c, c_size, cluster_labels[c])

        # 6. Build records and upsert into serving_rec_item_projection
        records: list[dict[str, Any]] = []
        for i in range(len(items_df)):
            row = items_df.iloc[i]
            app_id = int(row["app_id"])
            c_id = int(cluster_ids[i])
            records.append({
                "app_id": app_id,
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "cluster_id": c_id,
                "cluster_label": cluster_labels.get(c_id, "General"),
            })

        self.logger.info("Prepared %d item projection records.", len(records))

        if not self.dry_run and session is not None:
            self.logger.info("Upserting %d records into serving_rec_item_projection...", len(records))
            for i in range(0, len(records), 1000):
                chunk = records[i : i + 1000]
                stmt = pg_insert(ServingRecItemProjection).values(chunk)
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=[ServingRecItemProjection.app_id],
                    set_={
                        "x": stmt.excluded.x,
                        "y": stmt.excluded.y,
                        "cluster_id": stmt.excluded.cluster_id,
                        "cluster_label": stmt.excluded.cluster_label,
                    },
                )
                await session.execute(upsert_stmt)
            await session.commit()
            self.logger.info("Successfully persisted 2D item projections to database.")
        else:
            self.logger.info("[Dry Run] Skipped database write for serving_rec_item_projection.")

        return {
            "n_items": len(records),
            "n_clusters": 20,
            "cluster_labels": cluster_labels,
            "dry_run": self.dry_run,
        }

    def execute(self) -> dict[str, Any]:
        """Entry point for BaseJob."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="2D UMAP projection and KMeans clustering for Two-Tower item embeddings.")
    parser.add_argument("--profile", type=str, default=None, help="Recsys profile (dev/full)")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP and KMeans (default 42)")
    parser.add_argument("--dry-run", action="store_true", help="Run projection and clustering without database writes")
    args = parser.parse_args()

    job = ProjectItemEmbeddingsJob(
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
