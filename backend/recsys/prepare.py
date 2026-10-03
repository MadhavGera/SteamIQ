"""
Pure Python pipeline for preparing model-ready recommender datasets.

Converts raw CSV recommendations into memory-safe Parquet, enforces item universe & user
sampling, deduplicates interactions, and builds leakage-free temporal train/val/test splits.

Rule 4: Zero FastAPI and zero database imports.
Rule 7: Deterministic random seeds from RecsysConfig.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from recsys.config import RecsysConfig
from recsys.data_schema import (
    GAMES_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
    USERS_COLUMNS,
    validate_header,
)

logger = logging.getLogger("steamiq.recsys.prepare")


def _compute_file_fingerprint(file_path: Path) -> dict[str, Any]:
    """Computes basic filesystem fingerprint for cache invalidation."""
    stat = file_path.stat()
    return {
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def convert_csv_to_interactions_parquet(
    raw_recs_csv: Path | str,
    output_parquet: Path | str,
    chunk_size: int = 1_000_000,
    force: bool = False,
) -> Path:
    """
    STAGE 0: Converts recommendations.csv once to Parquet with ~1M rows per row group.
    Skips conversion if the output Parquet exists and the source file fingerprint matches.
    """
    csv_path = Path(raw_recs_csv)
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path = out_path.parent / ".interactions_all.fingerprint.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw recommendations CSV not found: {csv_path}")

    current_fingerprint = _compute_file_fingerprint(csv_path)

    if not force and out_path.exists() and fingerprint_path.exists():
        try:
            with open(fingerprint_path, encoding="utf-8") as f:
                saved_fingerprint = json.load(f)
            if (
                saved_fingerprint.get("file_size") == current_fingerprint["file_size"]
                and saved_fingerprint.get("mtime") == current_fingerprint["mtime"]
            ):
                logger.info(
                    "Cached Parquet %s matches fingerprint of %s. Skipping conversion.",
                    out_path.name,
                    csv_path.name,
                )
                return out_path
        except Exception as exc:
            logger.warning("Could not read fingerprint file (%s). Rebuilding Parquet...", exc)

    logger.info("Converting %s to Parquet %s...", csv_path.name, out_path.name)
    validate_header(csv_path, RECOMMENDATIONS_COLUMNS)

    # Schema definition for pyarrow table
    arrow_schema = pa.schema([
        ("app_id", pa.int32()),
        ("user_id", pa.int32()),
        ("is_recommended", pa.bool_()),
        ("hours", pa.float32()),
        ("date", pa.date32()),
        ("review_id", pa.int32()),
    ])

    dtypes = {
        "app_id": "int32",
        "user_id": "int32",
        "is_recommended": "bool",
        "hours": "float32",
        "review_id": "int32",
    }

    total_rows = 0
    with pq.ParquetWriter(out_path, arrow_schema, compression="snappy") as writer:
        for chunk_df in pd.read_csv(
            csv_path,
            usecols=["app_id", "user_id", "is_recommended", "hours", "date", "review_id"],
            dtype=dtypes,
            chunksize=chunk_size,
        ):
            # Parse dates cleanly to date32
            date_col = pd.to_datetime(chunk_df["date"], errors="coerce").dt.date
            chunk_df["date"] = date_col
            chunk_df["hours"] = chunk_df["hours"].fillna(0.0).astype("float32")

            # Create pyarrow table
            table = pa.Table.from_pandas(
                chunk_df,
                schema=arrow_schema,
                preserve_index=False,
            )
            writer.write_table(table)
            total_rows += len(chunk_df)

    # Save fingerprint
    fingerprint_data = {
        "file_size": current_fingerprint["file_size"],
        "mtime": current_fingerprint["mtime"],
        "total_rows": total_rows,
    }
    with open(fingerprint_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint_data, f, indent=2)

    logger.info("Successfully converted %d rows into %s", total_rows, out_path.name)
    return out_path


def prepare_interactions(
    raw_dir: Path | str,
    data_dir: Path | str,
    config: RecsysConfig,
) -> dict[str, Any]:
    """
    Main interaction preparation pipeline.

    Stages:
      0. Convert recommendations.csv -> interactions_all.parquet (cached).
      1. Pass 1: Count item positives for items in games.csv.
      2. Item Universe: Select top max_items with >= min_item_positives.
      3. Pass 2: Count user in-universe positives; sample up to max_users eligible users.
      4. Pass 3: Stream & filter rows for sampled users & universe items.
      5. Deduplication: Keep latest (date, review_id) for each (user, item).
      6. Temporal Split: Split user positives chronologically into train / val / test.
      7. Negatives: Assign split ('train' if before user val date, else 'holdout').
      8. Users & Mappings: Save users.parquet, mappings.parquet, and split parquets.
    """
    raw_path = Path(raw_dir)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    games_csv = raw_path / "games.csv"
    users_csv = raw_path / "users.csv"
    recs_csv = raw_path / "recommendations.csv"
    interactions_parquet = data_path / "interactions_all.parquet"

    validate_header(games_csv, GAMES_COLUMNS)
    validate_header(users_csv, USERS_COLUMNS)

    # Stage 0: Fast CSV -> Parquet conversion
    convert_csv_to_interactions_parquet(recs_csv, interactions_parquet)

    # ── 1. PASS 1: Item Positive Counts ──────────────────────────────────────
    logger.info("PASS 1: Reading valid catalog items from %s...", games_csv.name)
    games_df = pd.read_csv(games_csv, usecols=["app_id"])
    catalog_app_ids = set(games_df["app_id"].dropna().astype(np.int32))
    logger.info("Found %d catalog app_ids in %s", len(catalog_app_ids), games_csv.name)

    item_positive_counts: dict[int, int] = {}
    dataset = ds.dataset(interactions_parquet, format="parquet")

    logger.info("Streaming interactions_all.parquet for PASS 1 item positive aggregation...")
    for batch in dataset.to_batches(columns=["app_id", "is_recommended"], batch_size=1_000_000):
        app_arr = batch.column("app_id").to_numpy(zero_copy_only=False)
        rec_arr = batch.column("is_recommended").to_numpy(zero_copy_only=False)

        pos_mask = rec_arr
        if not np.any(pos_mask):
            continue

        pos_apps = app_arr[pos_mask]
        unique_apps, counts = np.unique(pos_apps, return_counts=True)
        for app_id, count in zip(unique_apps, counts, strict=False):
            app_id_int = int(app_id)
            if app_id_int in catalog_app_ids:
                item_positive_counts[app_id_int] = item_positive_counts.get(app_id_int, 0) + int(count)

    # ── 2. ITEM UNIVERSE ─────────────────────────────────────────────────────
    # Universe membership uses counts from all periods (a minor, accepted leak)
    eligible_items = [
        (app_id, count)
        for app_id, count in item_positive_counts.items()
        if count >= config.min_item_positives
    ]
    # Sort by positive count descending, tie-break by app_id ascending
    eligible_items.sort(key=lambda x: (-x[1], x[0]))

    universe_items = eligible_items[: config.max_items]
    universe_app_ids = {app_id for app_id, _ in universe_items}
    num_universe_items = len(universe_items)

    logger.info(
        "ITEM UNIVERSE: Selected %d items (target: %d, min positives: %d, eligible: %d)",
        num_universe_items,
        config.max_items,
        config.min_item_positives,
        len(eligible_items),
    )

    # Item indexing: item_idx contiguous 0..I-1 ordered by positive count descending
    item_to_idx: dict[int, int] = {app_id: idx for idx, (app_id, _) in enumerate(universe_items)}
    idx_to_item: dict[int, int] = {idx: app_id for app_id, idx in item_to_idx.items()}

    # ── 3. PASS 2: User In-Universe Positive Counts & Sampling ───────────────
    logger.info("PASS 2: Streaming interactions to compute user in-universe positive counts...")
    user_pos_counts: dict[int, int] = {}

    for batch in dataset.to_batches(columns=["user_id", "app_id", "is_recommended"], batch_size=1_000_000):
        u_arr = batch.column("user_id").to_numpy(zero_copy_only=False)
        a_arr = batch.column("app_id").to_numpy(zero_copy_only=False)
        rec_arr = batch.column("is_recommended").to_numpy(zero_copy_only=False)

        # Filter to in-universe positives
        pos_mask = rec_arr
        if not np.any(pos_mask):
            continue

        pos_u = u_arr[pos_mask]
        pos_a = a_arr[pos_mask]

        # Check universe membership
        in_univ = np.isin(pos_a, list(universe_app_ids))
        if not np.any(in_univ):
            continue

        valid_u = pos_u[in_univ]
        unique_u, counts = np.unique(valid_u, return_counts=True)
        for u, cnt in zip(unique_u, counts, strict=False):
            u_int = int(u)
            user_pos_counts[u_int] = user_pos_counts.get(u_int, 0) + int(cnt)

    eligible_users = [
        u for u, cnt in user_pos_counts.items() if cnt >= config.min_user_positives
    ]
    eligible_users.sort()
    logger.info("PASS 2: Found %d eligible users with >= %d in-universe positives", len(eligible_users), config.min_user_positives)

    # Deterministic sampling
    rng = np.random.default_rng(config.seed)
    if len(eligible_users) > config.max_users:
        sampled_users = rng.choice(eligible_users, size=config.max_users, replace=False).tolist()
    else:
        sampled_users = eligible_users

    sampled_users_set = set(sampled_users)
    logger.info("Sampled %d users (seed=%d)", len(sampled_users_set), config.seed)

    # ── 4. PASS 3: Extract & Filter Interactions for Sampled Users ───────────
    logger.info("PASS 3: Extracting all interactions for sampled users & universe items...")
    collected_chunks: list[pd.DataFrame] = []

    for batch in dataset.to_batches(
        columns=["app_id", "user_id", "is_recommended", "hours", "date", "review_id"],
        batch_size=1_000_000,
    ):
        u_arr = batch.column("user_id").to_numpy(zero_copy_only=False)
        a_arr = batch.column("app_id").to_numpy(zero_copy_only=False)

        mask = np.isin(u_arr, list(sampled_users_set)) & np.isin(a_arr, list(universe_app_ids))
        if not np.any(mask):
            continue

        batch_table = batch.filter(pa.compute.array_opacity_or_mask(mask) if hasattr(pa.compute, "array_opacity_or_mask") else pa.array(mask))
        collected_chunks.append(batch_table.to_pandas())

    if collected_chunks:
        raw_interactions_df = pd.concat(collected_chunks, ignore_index=True)
    else:
        raw_interactions_df = pd.DataFrame(
            columns=["app_id", "user_id", "is_recommended", "hours", "date", "review_id"]
        )

    logger.info("Collected %d raw interaction rows for sampled users", len(raw_interactions_df))

    # ── 5. Deduplication ─────────────────────────────────────────────────────
    # Deduplicate (user_id, app_id), keeping the latest date (tie-break by larger review_id)
    raw_interactions_df["date_str"] = raw_interactions_df["date"].astype(str)
    raw_interactions_df = raw_interactions_df.sort_values(
        by=["user_id", "app_id", "date_str", "review_id"],
        ascending=[True, True, True, True],
    )
    dedup_df = raw_interactions_df.drop_duplicates(subset=["user_id", "app_id"], keep="last").copy()
    logger.info("After deduplication: %d interactions remaining", len(dedup_df))

    # Convert date to unix epoch timestamp (seconds)
    ts_series = pd.to_datetime(dedup_df["date_str"]).astype("int64") // 10**9
    dedup_df["ts"] = ts_series

    # Re-verify user positive counts after deduplication to ensure >= min_user_positives
    pos_counts_post_dedup = (
        dedup_df[dedup_df["is_recommended"] == True]  # noqa: E712
        .groupby("user_id")
        .size()
    )
    valid_final_users = set(pos_counts_post_dedup[pos_counts_post_dedup >= config.min_user_positives].index)

    final_df = dedup_df[dedup_df["user_id"].isin(valid_final_users)].copy()
    logger.info(
        "Final filtered interactions: %d rows across %d users",
        len(final_df),
        len(valid_final_users),
    )

    # ── 6. Contiguous User Indexing ──────────────────────────────────────────
    sorted_final_users = sorted(valid_final_users)
    user_to_idx: dict[int, int] = {uid: idx for idx, uid in enumerate(sorted_final_users)}
    num_users = len(sorted_final_users)

    final_df["user_idx"] = final_df["user_id"].map(user_to_idx).astype(np.int32)
    final_df["item_idx"] = final_df["app_id"].map(item_to_idx).astype(np.int32)

    # ── 7. Temporal Split (Positives Only) ───────────────────────────────────
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    user_val_cutoff: dict[int, tuple[int, int]] = {}  # user_idx -> (val_ts, val_review_id)
    user_n_train: dict[int, int] = {}

    positives_df = final_df[final_df["is_recommended"] == True].copy()  # noqa: E712
    # Sort strictly by user_idx, ts, review_id
    positives_df = positives_df.sort_values(
        by=["user_idx", "ts", "review_id"],
        ascending=[True, True, True],
    )

    grouped_positives = positives_df.groupby("user_idx", sort=False)
    for u_idx, group in grouped_positives:
        # Last = TEST, Second-to-last = VAL, Everything before = TRAIN
        # Users have >= min_user_positives (>=5) so n_train >= 3
        test_item = group.iloc[-1]
        val_item = group.iloc[-2]
        train_items = group.iloc[:-2]

        test_rows.append({
            "user_idx": int(u_idx),
            "item_idx": int(test_item["item_idx"]),
            "ts": int(test_item["ts"]),
            "hours": float(test_item["hours"]),
        })

        val_rows.append({
            "user_idx": int(u_idx),
            "item_idx": int(val_item["item_idx"]),
            "ts": int(val_item["ts"]),
            "hours": float(val_item["hours"]),
        })

        user_val_cutoff[int(u_idx)] = (int(val_item["ts"]), int(val_item["review_id"]))
        user_n_train[int(u_idx)] = len(train_items)

        for _, t_row in train_items.iterrows():
            train_rows.append({
                "user_idx": int(u_idx),
                "item_idx": int(t_row["item_idx"]),
                "ts": int(t_row["ts"]),
                "hours": float(t_row["hours"]),
            })

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    # ── 8. Negatives Partitioning ────────────────────────────────────────────
    negatives_df = final_df[final_df["is_recommended"] == False].copy()  # noqa: E712
    negatives_rows: list[dict[str, Any]] = []

    for _, n_row in negatives_df.iterrows():
        u_idx = int(n_row["user_idx"])
        n_ts = int(n_row["ts"])
        n_rev = int(n_row["review_id"])

        val_ts, val_rev = user_val_cutoff.get(u_idx, (float("inf"), float("inf")))
        # Negative split = 'train' if (ts, review_id) < (val_ts, val_review_id) else 'holdout'
        split = "train" if (n_ts, n_rev) < (val_ts, val_rev) else "holdout"

        negatives_rows.append({
            "user_idx": u_idx,
            "item_idx": int(n_row["item_idx"]),
            "ts": n_ts,
            "hours": float(n_row["hours"]),
            "split": split,
        })

    neg_df = pd.DataFrame(negatives_rows)
    if neg_df.empty:
        neg_df = pd.DataFrame(columns=["user_idx", "item_idx", "ts", "hours", "split"])

    # ── 9. User Metadata ─────────────────────────────────────────────────────
    logger.info("Reading user activity metadata from %s...", users_csv.name)
    user_metadata_df = pd.read_csv(users_csv, dtype={"user_id": "int32", "products": "int32", "reviews": "int32"})
    user_meta_dict = user_metadata_df.set_index("user_id").to_dict(orient="index")

    users_data: list[dict[str, Any]] = []
    for u_idx, orig_uid in enumerate(sorted_final_users):
        meta = user_meta_dict.get(orig_uid, {})
        users_data.append({
            "user_idx": u_idx,
            "original_user_id": orig_uid,
            "products": int(meta.get("products", 0)),
            "reviews": int(meta.get("reviews", 0)),
            "n_train": int(user_n_train.get(u_idx, 0)),
        })
    users_parquet_df = pd.DataFrame(users_data)

    # ── 10. Item Mappings ────────────────────────────────────────────────────
    mappings_data = [
        {"item_idx": idx, "app_id": idx_to_item[idx]}
        for idx in range(num_universe_items)
    ]
    mappings_df = pd.DataFrame(mappings_data)

    # ── 11. Write Parquets ───────────────────────────────────────────────────
    train_parquet_path = data_path / "train.parquet"
    val_parquet_path = data_path / "val.parquet"
    test_parquet_path = data_path / "test.parquet"
    negatives_parquet_path = data_path / "negatives.parquet"
    users_parquet_path = data_path / "users.parquet"
    mappings_parquet_path = data_path / "mappings.parquet"

    train_df.to_parquet(train_parquet_path, index=False, compression="snappy")
    val_df.to_parquet(val_parquet_path, index=False, compression="snappy")
    test_df.to_parquet(test_parquet_path, index=False, compression="snappy")
    neg_df.to_parquet(negatives_parquet_path, index=False, compression="snappy")
    users_parquet_df.to_parquet(users_parquet_path, index=False, compression="snappy")
    mappings_df.to_parquet(mappings_parquet_path, index=False, compression="snappy")

    logger.info("Saved all interaction artifacts to %s", data_path)

    return {
        "num_users": num_users,
        "num_items": num_universe_items,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "negatives_rows": len(neg_df),
        "train_parquet": train_parquet_path,
        "val_parquet": val_parquet_path,
        "test_parquet": test_parquet_path,
        "negatives_parquet": negatives_parquet_path,
        "users_parquet": users_parquet_path,
        "mappings_parquet": mappings_parquet_path,
    }


def compute_file_sha1(path: Path | str) -> str:
    """Computes SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
