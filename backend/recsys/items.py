"""
Item feature engineering, sparse tag representation, text embeddings, and manifest generation.

Builds model-ready item representations:
  - Top 300 tag sparse multi-hot matrix (scipy.sparse .npz)
  - Standardized numeric features & scalers (pricing, reviews, platform flags)
  - Sentence-transformers all-MiniLM-L6-v2 normalized embeddings [I, 384] with SHA1 text cache
  - Aligned train popularity vector (item_train_pop.npy)
  - Metadata parquet (items.parquet) for database serving
  - Data manifest (data_manifest.json) with verifiable SHA1 hashes and 12-char dataset_version

Rule 4: Zero FastAPI and zero database imports.
Rule 7: CPU only execution.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

from recsys.config import RecsysConfig
from recsys.data_schema import (
    GAMES_COLUMNS,
    METADATA_KEYS,
    validate_header,
    validate_json_lines_keys,
)

logger = logging.getLogger("steamiq.recsys.items")


def _compute_sha1_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def compute_file_sha1(path: Path | str) -> str:
    """Computes SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_item_artifacts(
    raw_dir: Path | str,
    data_dir: Path | str,
    config: RecsysConfig,
    interaction_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Builds item features, embeddings, metadata parquet, and data manifest.
    """
    raw_path = Path(raw_dir)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    games_csv = raw_path / "games.csv"
    metadata_json = raw_path / "games_metadata.json"
    mappings_parquet = data_path / "mappings.parquet"
    train_parquet = data_path / "train.parquet"

    validate_header(games_csv, GAMES_COLUMNS)
    validate_json_lines_keys(metadata_json, METADATA_KEYS)

    if not mappings_parquet.exists():
        raise FileNotFoundError(f"Mappings parquet not found at {mappings_parquet}. Run prepare_interactions first.")

    mappings_df = pd.read_parquet(mappings_parquet)
    # Sort strictly by item_idx to ensure 0..I-1 alignment
    mappings_df = mappings_df.sort_values("item_idx").reset_index(drop=True)
    universe_app_ids = mappings_df["app_id"].tolist()
    app_id_to_idx = {app_id: idx for idx, app_id in enumerate(universe_app_ids)}
    num_items = len(universe_app_ids)

    logger.info("Building item artifacts for %d universe items...", num_items)

    # ── 1. Read games.csv for Universe Items ─────────────────────────────────
    games_df = pd.read_csv(games_csv)
    universe_games_df = games_df[games_df["app_id"].isin(universe_app_ids)].copy()
    games_by_app_id = universe_games_df.set_index("app_id").to_dict(orient="index")

    # Reference date for age calculation: max date_release in entire games.csv
    valid_dates = pd.to_datetime(games_df["date_release"], errors="coerce").dropna()
    max_release_date = valid_dates.max() if not valid_dates.empty else pd.Timestamp("2023-01-01")

    # ── 2. Stream games_metadata.json for Universe Items ─────────────────────
    logger.info("Streaming %s for universe item metadata...", metadata_json.name)
    metadata_by_app_id: dict[int, dict[str, Any]] = {}
    with open(metadata_json, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                app_id = int(record["app_id"])
                if app_id in app_id_to_idx:
                    metadata_by_app_id[app_id] = record
            except Exception:
                continue

    # ── 3. Train Positive Counts (Train Popularity) ──────────────────────────
    train_df = pd.read_parquet(train_parquet, columns=["item_idx"])
    train_pos_counts = train_df["item_idx"].value_counts().to_dict()
    item_train_pop = np.array(
        [train_pos_counts.get(idx, 0) for idx in range(num_items)],
        dtype=np.int32,
    )
    np.save(data_path / "item_train_pop.npy", item_train_pop)

    # ── 4. Tag Vocabulary & Sparse Multi-Hot Matrix ──────────────────────────
    logger.info("Constructing tag vocabulary and multi-hot matrix...")
    item_tags_list: list[list[str]] = []
    all_tag_counts: Counter[str] = Counter()

    for app_id in universe_app_ids:
        meta = metadata_by_app_id.get(app_id, {})
        raw_tags = meta.get("tags")
        if isinstance(raw_tags, list | dict):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags = []
        item_tags_list.append(tags)
        all_tag_counts.update(tags)

    # Top 300 tags across universe items
    top_300_tags = [tag for tag, _ in all_tag_counts.most_common(300)]
    tag_to_vocab_idx = {tag: i for i, tag in enumerate(top_300_tags)}

    # Build sparse matrix [I, 300]
    sparse_rows: list[int] = []
    sparse_cols: list[int] = []
    sparse_data: list[float] = []

    for item_idx, tags in enumerate(item_tags_list):
        for tag in tags:
            if tag in tag_to_vocab_idx:
                sparse_rows.append(item_idx)
                sparse_cols.append(tag_to_vocab_idx[tag])
                sparse_data.append(1.0)

    tag_sparse_matrix = sp.csr_matrix(
        (sparse_data, (sparse_rows, sparse_cols)),
        shape=(num_items, len(top_300_tags)),
        dtype=np.float32,
    )
    sp.save_npz(data_path / "item_tags_sparse.npz", tag_sparse_matrix)

    tag_vocab_data = {
        "tags": top_300_tags,
        "vocab_size": len(top_300_tags),
    }
    with open(data_path / "tag_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tag_vocab_data, f, indent=2)

    # ── 5. Item Descriptions & Text Embeddings ───────────────────────────────
    logger.info("Building text representations & computing MiniLM embeddings...")
    texts_for_embedding: list[str] = []
    has_description_flags: list[bool] = []
    descriptions_short: list[str] = []

    for item_idx, app_id in enumerate(universe_app_ids):
        g_info = games_by_app_id.get(app_id, {})
        meta = metadata_by_app_id.get(app_id, {})

        title = str(g_info.get("title", "")).strip() or f"App {app_id}"
        tags = item_tags_list[item_idx]
        top_10_tags = tags[:10]
        tags_str = ", ".join(top_10_tags)

        raw_desc = meta.get("description", "")
        desc_str = str(raw_desc).strip() if raw_desc else ""

        if desc_str:
            has_description = True
            desc_short = desc_str[:500]
            if tags_str:
                text_repr = f"{title}. Tags: {tags_str}. {desc_short}"
            else:
                text_repr = f"{title}. {desc_short}"
        else:
            has_description = False
            desc_short = ""
            text_repr = f"{title}. Tags: {tags_str}" if tags_str else title

        texts_for_embedding.append(text_repr)
        has_description_flags.append(has_description)
        descriptions_short.append(desc_short)

    desc_coverage = float(np.mean(has_description_flags)) if has_description_flags else 0.0
    logger.info("Item description coverage: %.2f%% (%d/%d)", desc_coverage * 100, int(sum(has_description_flags)), num_items)

    # Load embedding cache if available
    emb_cache_path = data_path / ".embedding_cache.json"
    embedding_cache: dict[str, list[float]] = {}
    if emb_cache_path.exists():
        try:
            with open(emb_cache_path, encoding="utf-8") as f:
                embedding_cache = json.load(f)
        except Exception:
            embedding_cache = {}

    # Check which texts need inference
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []
    text_hashes = [_compute_sha1_hex(t) for t in texts_for_embedding]

    for i, h in enumerate(text_hashes):
        if h not in embedding_cache:
            uncached_indices.append(i)
            uncached_texts.append(texts_for_embedding[i])

    embeddings_matrix = np.zeros((num_items, 384), dtype=np.float32)

    # Fill cached
    for i, h in enumerate(text_hashes):
        if h in embedding_cache:
            embeddings_matrix[i] = np.array(embedding_cache[h], dtype=np.float32)

    # Run sentence-transformers for uncached
    if uncached_texts:
        logger.info("Generating MiniLM embeddings for %d uncached texts...", len(uncached_texts))
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        new_embeddings = model.encode(
            uncached_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        for idx, (original_i, text_hash) in enumerate(zip(uncached_indices, [text_hashes[i] for i in uncached_indices], strict=False)):
            emb_vec = new_embeddings[idx]
            embeddings_matrix[original_i] = emb_vec
            embedding_cache[text_hash] = emb_vec.tolist()

        # Update cache on disk
        with open(emb_cache_path, "w", encoding="utf-8") as f:
            json.dump(embedding_cache, f)

    np.save(data_path / "item_text_emb.npy", embeddings_matrix)

    # ── 6. Numeric Features & Standardization ────────────────────────────────
    logger.info("Computing numeric features and standardizing continuous variables...")
    price_finals: list[float] = []
    discounts: list[float] = []
    positive_ratios: list[float] = []
    user_reviews_list: list[float] = []
    log1p_reviews: list[float] = []
    age_years_list: list[float] = []

    win_flags: list[bool] = []
    mac_flags: list[bool] = []
    linux_flags: list[bool] = []
    deck_flags: list[bool] = []
    release_dates: list[str | None] = []

    for app_id in universe_app_ids:
        g = games_by_app_id.get(app_id, {})
        pf = float(g.get("price_final", 0.0) if pd.notna(g.get("price_final")) else 0.0)
        disc = float(g.get("discount", 0.0) if pd.notna(g.get("discount")) else 0.0)
        pos_r = float(g.get("positive_ratio", 0) if pd.notna(g.get("positive_ratio")) else 0)
        revs = float(g.get("user_reviews", 0) if pd.notna(g.get("user_reviews")) else 0)
        l_revs = float(np.log1p(max(revs, 0.0)))

        rel_date_str = str(g.get("date_release", "")).strip() if pd.notna(g.get("date_release")) else None
        if rel_date_str:
            parsed_d = pd.to_datetime(rel_date_str, errors="coerce")
            if pd.notna(parsed_d):
                days_diff = max((max_release_date - parsed_d).days, 0)
                age_y = float(days_diff / 365.25)
            else:
                age_y = 0.0
        else:
            age_y = 0.0

        w_flag = bool(g.get("win", False)) if pd.notna(g.get("win")) else False
        m_flag = bool(g.get("mac", False)) if pd.notna(g.get("mac")) else False
        l_flag = bool(g.get("linux", False)) if pd.notna(g.get("linux")) else False
        d_flag = bool(g.get("steam_deck", False)) if pd.notna(g.get("steam_deck")) else False

        price_finals.append(pf)
        discounts.append(disc)
        positive_ratios.append(pos_r)
        user_reviews_list.append(revs)
        log1p_reviews.append(l_revs)
        age_years_list.append(age_y)
        win_flags.append(w_flag)
        mac_flags.append(m_flag)
        linux_flags.append(l_flag)
        deck_flags.append(d_flag)
        release_dates.append(rel_date_str if rel_date_str else None)

    # Compute means and stds for continuous features
    continuous_features = {
        "price_final": np.array(price_finals, dtype=np.float32),
        "discount": np.array(discounts, dtype=np.float32),
        "positive_ratio": np.array(positive_ratios, dtype=np.float32),
        "log1p_user_reviews": np.array(log1p_reviews, dtype=np.float32),
        "age_years": np.array(age_years_list, dtype=np.float32),
    }

    scalers: dict[str, dict[str, float]] = {}
    standardized_arrays: dict[str, np.ndarray] = {}

    for name, arr in continuous_features.items():
        m = float(np.mean(arr))
        s = float(np.std(arr))
        scalers[name] = {"mean": m, "std": s}
        standardized_arrays[name] = ((arr - m) / (s + 1e-8)).astype(np.float32)

    with open(data_path / "numeric_scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)

    # ── 7. Metadata items.parquet for Serving ────────────────────────────────
    items_rows: list[dict[str, Any]] = []
    for item_idx, app_id in enumerate(universe_app_ids):
        g = games_by_app_id.get(app_id, {})
        title = str(g.get("title", "")).strip() or f"App {app_id}"

        items_rows.append({
            "app_id": int(app_id),
            "item_idx": int(item_idx),
            "title": title,
            "date_release": release_dates[item_idx],
            "price_final": float(price_finals[item_idx]),
            "positive_ratio": int(positive_ratios[item_idx]),
            "user_reviews": int(user_reviews_list[item_idx]),
            "tags": item_tags_list[item_idx],
            "description_short": descriptions_short[item_idx],
            "has_description": bool(has_description_flags[item_idx]),
            "train_positives": int(item_train_pop[item_idx]),
            "popularity_rank": int(item_idx + 1),  # item_idx 0 is rank 1
            "win": bool(win_flags[item_idx]),
            "mac": bool(mac_flags[item_idx]),
            "linux": bool(linux_flags[item_idx]),
            "steam_deck": bool(deck_flags[item_idx]),
            "header_image_url": f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg",
        })

    items_df = pd.DataFrame(items_rows)
    items_parquet_path = data_path / "items.parquet"
    items_df.to_parquet(items_parquet_path, index=False, compression="snappy")
    logger.info("Saved items.parquet with %d rows", len(items_df))

    # ── 8. Data Manifest ─────────────────────────────────────────────────────
    logger.info("Generating data_manifest.json...")
    manifest_files = [
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "negatives.parquet",
        "users.parquet",
        "mappings.parquet",
        "items.parquet",
        "item_text_emb.npy",
        "item_train_pop.npy",
        "item_tags_sparse.npz",
    ]

    file_sha1s: dict[str, str] = {}
    for filename in manifest_files:
        f_path = data_path / filename
        if f_path.exists():
            file_sha1s[filename] = compute_file_sha1(f_path)

    # Read user / split counts from interaction stats or parquets
    train_count = len(pd.read_parquet(data_path / "train.parquet")) if (data_path / "train.parquet").exists() else 0
    val_count = len(pd.read_parquet(data_path / "val.parquet")) if (data_path / "val.parquet").exists() else 0
    test_count = len(pd.read_parquet(data_path / "test.parquet")) if (data_path / "test.parquet").exists() else 0
    neg_count = len(pd.read_parquet(data_path / "negatives.parquet")) if (data_path / "negatives.parquet").exists() else 0
    users_count = len(pd.read_parquet(data_path / "users.parquet")) if (data_path / "users.parquet").exists() else 0

    counts = {
        "users": users_count,
        "items": num_items,
        "train_rows": train_count,
        "val_rows": val_count,
        "test_rows": test_count,
        "negatives_rows": neg_count,
    }

    # Deterministic dataset_version
    # first 12 hex chars of SHA1 of (profile + seed + counts + file SHA1s)
    version_payload = {
        "profile": config.profile if hasattr(config, "profile") else "dev",
        "seed": config.seed,
        "counts": counts,
        "file_sha1s": dict(sorted(file_sha1s.items())),
    }
    version_str = json.dumps(version_payload, sort_keys=True)
    dataset_version = hashlib.sha1(version_str.encode("utf-8")).hexdigest()[:12]

    manifest = {
        "profile": config.profile if hasattr(config, "profile") else "dev",
        "seed": config.seed,
        "dataset_version": dataset_version,
        "counts": counts,
        "item_description_coverage": round(desc_coverage, 4),
        "file_sha1s": file_sha1s,
    }

    manifest_path = data_path / "data_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved data manifest with dataset_version=%s", dataset_version)

    return {
        "dataset_version": dataset_version,
        "counts": counts,
        "item_description_coverage": desc_coverage,
        "manifest_path": manifest_path,
        "items_parquet": items_parquet_path,
    }
