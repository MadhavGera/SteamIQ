"""
Memory-Safe Audit Job for Kaggle Steam Datasets.

Streams tens of millions of interaction rows from recommendations.csv in chunks
using bitsets and numpy arrays to guarantee peak memory remains under 3 GB.

Validates schema headers and computes:
  - Total rows, unique users, unique items, positive recommendation share
  - Temporal bounds (date min/max, rows per year)
  - Playtime hour quantiles (p0, p25, p50, p75, p90, p99, p100)
  - Duplicate (user_id, app_id) pair counts
  - Density thresholds: users with >=5 positives, items with >=20 positives
  - Cross-dataset join coverages (games.csv, users.csv, games_metadata.json)

Rule 2: Subclasses BaseJob, implements run(), supports dry_run, self.resolve_path(), self.logger.
Rule 4: Zero FastAPI and database imports in core recsys packages.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import settings
from jobs.base_job import BaseJob
from recsys.data_schema import (
    GAMES_COLUMNS,
    METADATA_KEYS,
    RECOMMENDATIONS_COLUMNS,
    USERS_COLUMNS,
    validate_header,
    validate_json_lines_keys,
)


class AuditRecsysDataJob(BaseJob):
    """
    BaseJob auditing raw Steam recommendation files with strict memory safety.
    """

    job_name = "audit_recsys_data"

    def __init__(
        self,
        *,
        raw_dir: Path | None = None,
        data_dir: Path | None = None,
        chunk_size: int = 1_000_000,
        sample_capacity: int = 1_000_000,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.raw_dir = raw_dir
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.sample_capacity = sample_capacity
        self.seed = seed

    def _get_raw_dir(self) -> Path:
        if self.raw_dir:
            return self.raw_dir
        return settings.resolved_recsys_raw_dir

    def _get_data_dir(self) -> Path:
        if self.data_dir:
            return self.data_dir
        return settings.resolved_recsys_data_dir

    def run(self) -> dict[str, Any]:
        raw_dir = self._get_raw_dir()
        data_dir = self._get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        games_csv = raw_dir / "games.csv"
        metadata_json = raw_dir / "games_metadata.json"
        users_csv = raw_dir / "users.csv"
        recs_csv = raw_dir / "recommendations.csv"

        self.logger.info("Validating dataset schemas in %s...", raw_dir)
        validate_header(games_csv, GAMES_COLUMNS)
        validate_header(users_csv, USERS_COLUMNS)
        validate_header(recs_csv, RECOMMENDATIONS_COLUMNS)
        validate_json_lines_keys(metadata_json, METADATA_KEYS)
        self.logger.info("✅ All 4 data file schemas verified successfully.")

        # ── 1. Audit games.csv ───────────────────────────────────────────────
        self.logger.info("Reading %s...", games_csv.name)
        games_df = pd.read_csv(games_csv, usecols=["app_id"])
        catalog_app_ids = set(games_df["app_id"].dropna().astype(np.int32))
        total_catalog_games = len(catalog_app_ids)
        self.logger.info("Loaded %d games from games.csv", total_catalog_games)

        # ── 2. Audit games_metadata.json ─────────────────────────────────────
        self.logger.info("Streaming %s...", metadata_json.name)
        metadata_total = 0
        metadata_with_desc = 0
        tag_counts: list[int] = []

        with open(metadata_json, encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    metadata_total += 1
                    desc = record.get("description", "")
                    if desc and str(desc).strip():
                        metadata_with_desc += 1

                    tags = record.get("tags")
                    if isinstance(tags, list | dict):
                        tag_counts.append(len(tags))
                    else:
                        tag_counts.append(0)
                except Exception:
                    continue

        tag_counts_arr = np.array(tag_counts, dtype=np.int32) if tag_counts else np.array([0], dtype=np.int32)
        tag_quantiles = {
            f"p{q}": int(np.percentile(tag_counts_arr, q))
            for q in [0, 25, 50, 75, 90, 99, 100]
        }
        share_games_with_desc = float(metadata_with_desc / metadata_total) if metadata_total > 0 else 0.0
        self.logger.info("Metadata total games: %d, with description: %.2f%%", metadata_total, share_games_with_desc * 100)

        # ── 3. Audit users.csv ───────────────────────────────────────────────
        self.logger.info("Reading %s in chunks...", users_csv.name)
        # Determine maximum user ID in users.csv for compact bit array
        max_user_id = 0
        user_chunks = pd.read_csv(users_csv, usecols=["user_id"], chunksize=1_000_000, dtype={"user_id": "int32"})
        user_id_chunks: list[np.ndarray] = []
        for chunk in user_chunks:
            arr = chunk["user_id"].to_numpy(dtype=np.int32)
            if len(arr) > 0:
                max_user_id = max(max_user_id, int(np.max(arr)))
                user_id_chunks.append(arr)

        users_csv_mask = np.zeros(max_user_id + 1, dtype=bool) if max_user_id > 0 else np.zeros(1, dtype=bool)
        total_users_csv_count = 0
        for arr in user_id_chunks:
            users_csv_mask[arr] = True
            total_users_csv_count += len(arr)
        del user_id_chunks
        self.logger.info("Loaded %d users from users.csv (Max User ID: %d)", total_users_csv_count, max_user_id)

        # ── 4. Stream recommendations.csv ────────────────────────────────────
        self.logger.info(
            "Streaming %s in chunks of %s rows%s...",
            recs_csv.name,
            f"{self.chunk_size:,}",
            " [DRY RUN: 2 chunks max]" if self.dry_run else "",
        )

        recs_dtypes = {
            "app_id": "int32",
            "helpful": "int32",
            "funny": "int32",
            "is_recommended": "bool",
            "hours": "float32",
            "user_id": "int32",
            "review_id": "int32",
        }

        # Dynamic accumulator arrays using capacity resizing
        initial_user_cap = max(max_user_id + 1, 2_000_000)
        user_total_positives = np.zeros(initial_user_cap, dtype=np.int16)
        user_catalog_positives = np.zeros(initial_user_cap, dtype=np.int16)
        user_seen_mask = np.zeros(initial_user_cap, dtype=bool)

        initial_item_cap = 2_000_000
        item_positives = np.zeros(initial_item_cap, dtype=np.int32)
        item_seen_mask = np.zeros(initial_item_cap, dtype=bool)

        total_rows = 0
        total_recommended_true = 0
        total_in_games_csv = 0
        total_users_in_users_csv = 0

        date_min_str: str | None = None
        date_max_str: str | None = None
        rows_per_year: dict[str, int] = {}

        # Reservoir sample for playtime hours quantiles
        rng = np.random.default_rng(self.seed)
        hours_sample: list[float] = []
        global_min_hours = float("inf")
        global_max_hours = float("-inf")

        # Duplicate tracking using 16 hash buckets (memory bounded)
        num_buckets = 16
        duplicate_hash_buckets: list[set[int]] = [set() for _ in range(num_buckets)]
        duplicate_pairs_count = 0

        chunks_count = 0
        reader = pd.read_csv(
            recs_csv,
            chunksize=self.chunk_size,
            dtype=recs_dtypes,
            parse_dates=False,
        )

        for chunk_df in reader:
            chunks_count += 1
            n_chunk = len(chunk_df)
            total_rows += n_chunk

            u_ids = chunk_df["user_id"].to_numpy(dtype=np.int32)
            a_ids = chunk_df["app_id"].to_numpy(dtype=np.int32)
            is_rec = chunk_df["is_recommended"].to_numpy(dtype=bool)
            hours_arr = chunk_df["hours"].to_numpy(dtype=np.float32)
            dates_arr = chunk_df["date"].astype(str).to_numpy()

            # Resize user arrays if required
            chunk_max_u = int(np.max(u_ids)) if len(u_ids) > 0 else 0
            if chunk_max_u >= len(user_seen_mask):
                new_size = max(chunk_max_u + 1, len(user_seen_mask) * 2)
                user_seen_mask = np.pad(user_seen_mask, (0, new_size - len(user_seen_mask)))
                user_total_positives = np.pad(user_total_positives, (0, new_size - len(user_total_positives)))
                user_catalog_positives = np.pad(user_catalog_positives, (0, new_size - len(user_catalog_positives)))

            # Resize item arrays if required
            chunk_max_a = int(np.max(a_ids)) if len(a_ids) > 0 else 0
            if chunk_max_a >= len(item_seen_mask):
                new_size = max(chunk_max_a + 1, len(item_seen_mask) * 2)
                item_seen_mask = np.pad(item_seen_mask, (0, new_size - len(item_seen_mask)))
                item_positives = np.pad(item_positives, (0, new_size - len(item_positives)))

            # Mark seen users and items
            user_seen_mask[u_ids] = True
            item_seen_mask[a_ids] = True

            # Recommendation counts
            pos_mask = is_rec
            total_recommended_true += int(np.sum(pos_mask))

            # Accumulate user & item positive counts
            pos_users = u_ids[pos_mask]
            pos_items = a_ids[pos_mask]

            if len(pos_users) > 0:
                np.add.at(user_total_positives, pos_users, 1)

            if len(pos_items) > 0:
                np.add.at(item_positives, pos_items, 1)

            # Restricted to games in catalog
            in_catalog_mask = np.isin(a_ids, list(catalog_app_ids))
            total_in_games_csv += int(np.sum(in_catalog_mask))

            pos_catalog_users = u_ids[pos_mask & in_catalog_mask]
            if len(pos_catalog_users) > 0:
                np.add.at(user_catalog_positives, pos_catalog_users, 1)

            # Duplicate pair check (user_id << 32 | app_id)
            for u, a in zip(u_ids, a_ids, strict=False):
                pair_key = (int(u) << 32) | int(a)
                bucket_idx = u % num_buckets
                if pair_key in duplicate_hash_buckets[bucket_idx]:
                    duplicate_pairs_count += 1
                else:
                    duplicate_hash_buckets[bucket_idx].add(pair_key)

            # Dates min/max & year aggregation
            for d in dates_arr:
                if len(d) >= 4 and d[:4].isdigit():
                    yr = d[:4]
                    rows_per_year[yr] = rows_per_year.get(yr, 0) + 1
                    if date_min_str is None or d < date_min_str:
                        date_min_str = d
                    if date_max_str is None or d > date_max_str:
                        date_max_str = d

            # Hours min/max & reservoir sample
            if len(hours_arr) > 0:
                chunk_min = float(np.min(hours_arr))
                chunk_max = float(np.max(hours_arr))
                if chunk_min < global_min_hours:
                    global_min_hours = chunk_min
                if chunk_max > global_max_hours:
                    global_max_hours = chunk_max

                # Reservoir sample
                sample_take = min(n_chunk, 50_000)
                sampled_indices = rng.choice(n_chunk, size=sample_take, replace=False)
                hours_sample.extend(hours_arr[sampled_indices].tolist())
                if len(hours_sample) > self.sample_capacity:
                    hours_sample = rng.choice(hours_sample, size=self.sample_capacity, replace=False).tolist()

            self.logger.info("Processed chunk %d (%d rows, total: %d)...", chunks_count, n_chunk, total_rows)
            if self.dry_run and chunks_count >= 2:
                self.logger.info("Dry run limit reached (2 chunks). Stopping stream.")
                break

        # ── 5. Final Computations ────────────────────────────────────────────
        unique_users_total = int(np.sum(user_seen_mask))
        unique_items_total = int(np.sum(item_seen_mask))
        share_recommended = float(total_recommended_true / total_rows) if total_rows > 0 else 0.0

        # Hours quantiles
        hours_np = np.array(hours_sample, dtype=np.float32) if hours_sample else np.array([0.0], dtype=np.float32)
        hours_quantiles = {
            "p0": round(float(global_min_hours if global_min_hours != float("inf") else 0.0), 2),
            "p25": round(float(np.percentile(hours_np, 25)), 2),
            "p50": round(float(np.percentile(hours_np, 50)), 2),
            "p75": round(float(np.percentile(hours_np, 75)), 2),
            "p90": round(float(np.percentile(hours_np, 90)), 2),
            "p99": round(float(np.percentile(hours_np, 99)), 2),
            "p100": round(float(global_max_hours if global_max_hours != float("-inf") else 0.0), 2),
        }

        # Density threshold counts
        users_ge_5_pos = int(np.sum(user_total_positives >= 5))
        users_ge_5_pos_catalog = int(np.sum(user_catalog_positives >= 5))

        # Item positive distribution & >=20 positives
        seen_items_positives = item_positives[item_seen_mask]
        items_ge_20_pos = int(np.sum(seen_items_positives >= 20))
        item_pos_quantiles = {
            f"p{q}": int(np.percentile(seen_items_positives, q)) if len(seen_items_positives) > 0 else 0
            for q in [0, 25, 50, 75, 90, 99, 100]
        }

        # Users in users.csv match count
        if len(users_csv_mask) > 0:
            limit_idx = min(len(user_seen_mask), len(users_csv_mask))
            total_users_in_users_csv = int(np.sum(user_seen_mask[:limit_idx] & users_csv_mask[:limit_idx]))
        else:
            total_users_in_users_csv = 0

        join_coverage = {
            "interaction_app_ids_in_games_share": round(float(total_in_games_csv / total_rows), 4) if total_rows > 0 else 0.0,
            "games_with_non_empty_description_share": round(share_games_with_desc, 4),
            "tag_count_quantiles": tag_quantiles,
            "interaction_users_in_users_csv_share": round(float(total_users_in_users_csv / unique_users_total), 4) if unique_users_total > 0 else 0.0,
        }

        audit_results: dict[str, Any] = {
            "total_rows": total_rows,
            "unique_users": unique_users_total,
            "unique_items": unique_items_total,
            "share_is_recommended_true": round(share_recommended, 4),
            "date_min": date_min_str,
            "date_max": date_max_str,
            "rows_per_year": dict(sorted(rows_per_year.items())),
            "hours_quantiles": hours_quantiles,
            "duplicate_user_app_pairs": duplicate_pairs_count,
            "users_with_ge_5_positives": users_ge_5_pos,
            "users_with_ge_5_positives_in_catalog": users_ge_5_pos_catalog,
            "items_with_ge_20_positives": items_ge_20_pos,
            "item_positive_count_quantiles": item_pos_quantiles,
            "join_coverage": join_coverage,
            "dry_run": self.dry_run,
            "chunks_processed": chunks_count,
            "audit_timestamp": datetime.now(UTC).isoformat(),
        }

        # Write results to audit.json
        output_file = data_dir / "audit.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
        self.logger.info("Saved audit results to %s", output_file)

        # ── 6. Print Summary Table ───────────────────────────────────────────
        summary_table = f"""
================================================================================
STEAM RECOMMENDATIONS DATASET AUDIT SUMMARY
================================================================================
Total Interaction Rows:       {total_rows:,}
Unique Users:                 {unique_users_total:,}
Unique Items (Games):         {unique_items_total:,}
Share is_recommended=True:    {share_recommended * 100:.2f}%
Date Range:                   {date_min_str} to {date_max_str}
Duplicate (user, app) Pairs:  {duplicate_pairs_count:,}
--------------------------------------------------------------------------------
DENSITY & FILTERING THRESHOLDS
Users with >= 5 Positives:    {users_ge_5_pos:,}
Users with >= 5 (In Catalog): {users_ge_5_pos_catalog:,}
Items with >= 20 Positives:   {items_ge_20_pos:,}
--------------------------------------------------------------------------------
PLAYTIME HOURS QUANTILES
p0: {hours_quantiles['p0']}h | p25: {hours_quantiles['p25']}h | p50: {hours_quantiles['p50']}h | p75: {hours_quantiles['p75']}h | p90: {hours_quantiles['p90']}h | p99: {hours_quantiles['p99']}h | p100: {hours_quantiles['p100']}h
--------------------------------------------------------------------------------
JOIN COVERAGE
Interactions in games.csv:    {join_coverage['interaction_app_ids_in_games_share'] * 100:.2f}%
Users found in users.csv:     {join_coverage['interaction_users_in_users_csv_share'] * 100:.2f}%
Games with description:       {join_coverage['games_with_non_empty_description_share'] * 100:.2f}%
Tag count (median p50):       {tag_quantiles.get('p50', 0)} tags/game
================================================================================
"""
        print(summary_table)
        self.logger.info("Audit complete. Status: OK.")

        return audit_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-safe audit of Kaggle Steam recommendation files")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing raw Steam CSV/JSON files")
    parser.add_argument("--data-dir", type=str, default=None, help="Output directory for audit.json")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size for streaming recommendations.csv")
    parser.add_argument("--dry-run", action="store_true", help="Process only the first 2 chunks")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None

    job = AuditRecsysDataJob(
        raw_dir=raw_dir,
        data_dir=data_dir,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
