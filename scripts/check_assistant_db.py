"""
Assistant & Recsys Database Verification Tool.

Purpose:
  Inspects PostgreSQL model_runs, serving_rec_reports, and mart_review_intelligence
  to verify offline-to-online reconciliation without hardcoded mock values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, text
from core.config import settings


def get_sync_engine():
    sync_url = settings.database_url or (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    return create_engine(sync_url)


def print_reports(keys: list[str] | None = None) -> None:
    engine = get_sync_engine()
    with engine.connect() as conn:
        print("=== model_runs (assistant_*) ===")
        runs = conn.execute(
            text(
                "SELECT id, model_name, stage, created_at, metrics "
                "FROM model_runs WHERE model_name LIKE 'assistant_%' "
                "ORDER BY created_at DESC"
            )
        ).fetchall()
        for r in runs:
            print(f"ID: {r[0]} | Name: {r[1]} | Stage: {r[2]} | Created: {r[3]}")
            if r[4]:
                print(f"  Metrics: {r[4]}")

        print("\n=== serving_rec_reports ===")
        if keys:
            query = text(
                "SELECT report_key, updated_at, payload "
                "FROM serving_rec_reports WHERE report_key = ANY(:keys) "
                "ORDER BY report_key"
            )
            reps = conn.execute(query, {"keys": keys}).fetchall()
        else:
            query = text(
                "SELECT report_key, updated_at, payload "
                "FROM serving_rec_reports WHERE report_key LIKE 'assistant_%' "
                "ORDER BY report_key"
            )
            reps = conn.execute(query).fetchall()

        for rep in reps:
            print(f"\n==================== [REPORT: {rep[0]}] ====================")
            print(f"Updated At: {rep[1]}")
            print(json.dumps(rep[2], indent=2))


def print_mart_review_sample(n: int = 5) -> None:
    engine = get_sync_engine()
    with engine.connect() as conn:
        print("=== Table Row Counts ===")
        tables = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")).fetchall()
        for t in sorted([t[0] for t in tables]):
            cnt = conn.execute(text(f"SELECT count(*) FROM {t}")).scalar()
            print(f"  {t}: {cnt}")

        count = conn.execute(text("SELECT count(*) FROM mart_review_intelligence")).scalar()
        print(f"\n=== mart_review_intelligence (Total rows: {count}) ===")
        rows = conn.execute(
            text(
                "SELECT app_id, sentiment_overview, timeline, topics, "
                "loved_features, complaints, summary, materialized_at "
                "FROM mart_review_intelligence LIMIT :limit"
            ),
            {"limit": n},
        ).mappings().fetchall()
        for idx, row in enumerate(rows, 1):
            print(f"\n--- Row {idx} (app_id: {row['app_id']}) ---")
            for k, v in row.items():
                print(f"  {k}: {json.dumps(v, default=str) if isinstance(v, (dict, list)) else v}")


def print_models(models: list[str]) -> None:
    engine = get_sync_engine()
    with engine.connect() as conn:
        print(f"\n=== model_runs for {models} ===")
        rows = conn.execute(
            text(
                "SELECT id, model_name, stage, created_at, hyperparameters, metrics "
                "FROM model_runs WHERE model_name = ANY(:names) "
                "ORDER BY created_at DESC"
            ),
            {"names": models},
        ).fetchall()
        for r in rows:
            print(f"\nModel: {r[1]} | Stage: {r[2]} | ID: {r[0]} | Created: {r[3]}")
            print(f"Hyperparameters: {json.dumps(r[4], indent=2)}")
            print(f"Metrics: {json.dumps(r[5], indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect assistant reports and review intelligence in DB.")
    parser.add_argument("--reports", nargs="*", help="Specific report keys to inspect (e.g. assistant_leaderboard assistant_retrieval)")
    parser.add_argument("--mart-reviews", type=int, default=0, help="Print N sample rows from mart_review_intelligence")
    parser.add_argument("--models", nargs="*", help="Model names to inspect from model_runs")
    args = parser.parse_args()

    if args.models:
        print_models(args.models)
    elif args.mart_reviews > 0:
        print_mart_review_sample(args.mart_reviews)
    else:
        print_reports(args.reports)


if __name__ == "__main__":
    main()
