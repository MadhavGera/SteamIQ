import sys, asyncio
sys.path.insert(0, 'backend')
from sqlalchemy import text
from db.base import AsyncSessionLocal

async def clear_promoted():
    async with AsyncSessionLocal() as session:
        res = await session.execute(text("UPDATE model_runs SET promoted_at = NULL, promoted_by = NULL WHERE stage = 'candidate'"))
        await session.commit()
        print(f"Candidate rows updated: {res.rowcount}")

        # Verify
        check = await session.execute(text("SELECT model_name, stage, promoted_at, promoted_by FROM model_runs WHERE promoted_at IS NOT NULL OR promoted_by IS NOT NULL"))
        rows = check.fetchall()
        print("Runs with promoted_* set:")
        for r in rows:
            print(f"  {r[0]} | {r[1]} | {r[2]} | {r[3]}")

if __name__ == "__main__":
    asyncio.run(clear_promoted())
