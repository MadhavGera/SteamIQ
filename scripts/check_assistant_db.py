from sqlalchemy import create_engine, text
from core.config import settings

def main():
    sync_url = settings.database_url or f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    engine = create_engine(sync_url)
    with engine.connect() as conn:
        print("=== model_runs (assistant_*) ===")
        runs = conn.execute(text("SELECT id, model_name, stage, created_at FROM model_runs WHERE model_name LIKE 'assistant_%' ORDER BY created_at DESC")).fetchall()
        for r in runs:
            print(f"ID: {r[0]} | Name: {r[1]} | Stage: {r[2]} | Created: {r[3]}")
        
        print("\n=== serving_rec_reports columns ===")
        cols = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'serving_rec_reports'")).fetchall()
        print([c[0] for c in cols])
        
        print("\n=== serving_rec_reports (assistant_*) ===")
        reps = conn.execute(text("SELECT report_key, updated_at, payload FROM serving_rec_reports WHERE report_key LIKE 'assistant_%' ORDER BY updated_at DESC")).fetchall()
        for rep in reps:
            print(f"Key: {rep[0]} | Updated: {rep[1]}")
            if rep[0] in ('assistant_calibration', 'assistant_retrieval'):
                import json
                print(f"--- Payload for {rep[0]} ---")
                print(json.dumps(rep[2], indent=2)[:400])

if __name__ == "__main__":
    main()
