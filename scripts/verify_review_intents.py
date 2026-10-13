"""
Verify real review intelligence answers from mart_review_intelligence.
"""
from __future__ import annotations

import asyncio
from assistant.service import AssistantService
from db.base import async_session_factory


async def main() -> None:
    service = AssistantService.get_instance()
    service.initialize()

    async with async_session_factory() as session:
        # 1. REVIEW_SUMMARY
        res1 = await service.ask_async("what do reviews say about hades?", user_mode="player", db_session=session)
        print("=== INTENT 1: REVIEW_SUMMARY ===")
        print("Intent:", res1.intent)
        print("Confidence:", res1.confidence)
        print("Answer:", res1.answer)
        print("Tables read:", res1.data_tables_read)
        print()

        # 2. TOP_COMPLAINTS
        res2 = await service.ask_async("what are the player complaints for hades?", user_mode="player", db_session=session)
        print("=== INTENT 2: TOP_COMPLAINTS ===")
        print("Intent:", res2.intent)
        print("Confidence:", res2.confidence)
        print("Answer:", res2.answer)
        print("Tables read:", res2.data_tables_read)
        print()

        # 3. LOVED_FEATURES
        res3 = await service.ask_async("what do players praise most in hades?", user_mode="player", db_session=session)
        print("=== INTENT 3: LOVED_FEATURES ===")
        print("Intent:", res3.intent)
        print("Confidence:", res3.confidence)
        print("Answer:", res3.answer)
        print("Tables read:", res3.data_tables_read)


if __name__ == "__main__":
    asyncio.run(main())
