"""
Game Match Profile Materialization Pipeline — Phase 5 (Roadmap v2 §3).

Derives the 6-dimension intensity profile for every game:
  - difficulty (0.0 to 10.0)
  - story_weight (0.0 to 10.0)
  - exploration (0.0 to 10.0)
  - combat (0.0 to 10.0)
  - multiplayer (0.0 to 10.0)
  - session_length (0.0 to 10.0)

Derived deterministically from:
  - raw_games (tags, categories, playtime metrics)
  - feature_review_topics, feature_review_complaints, feature_review_features

ADR 0001 Golden Rule Compliance:
  - Reads ONLY from raw_* and feature_* tables.
  - Writes ONLY to mart_game_match_profile.
  - API handlers read directly from mart_game_match_profile.

Usage:
  python -m jobs.materialize_match_profiles --all
  python -m jobs.materialize_match_profiles --app-id 367520
  python -m jobs.materialize_match_profiles --dry-run
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewTopic,
    MartGameMatchProfile,
    RawGame,
)
from jobs.base_job import BaseJob


class MaterializeMatchProfilesJob(BaseJob):
    job_name = "materialize_match_profiles"

    def __init__(
        self,
        *,
        app_id: int | None = None,
        all_games: bool = False,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id
        self.all_games = all_games

    async def _run_async(self, session: AsyncSession) -> None:
        self.logger.info("Starting Game Match profile materialization...")

        # 1. Fetch Target Games
        if self.app_id:
            res = await session.execute(select(RawGame).where(RawGame.app_id == self.app_id))
            target_games = res.scalars().all()
        elif self.all_games:
            res = await session.execute(select(RawGame))
            target_games = res.scalars().all()
        else:
            self.logger.warning("Neither --app-id nor --all specified; defaulting to --all")
            res = await session.execute(select(RawGame))
            target_games = res.scalars().all()

        if not target_games:
            self.logger.warning("No target games found for match profile materialization.")
            return

        target_app_ids = [g.app_id for g in target_games]
        self.logger.info("Materializing match profiles for %d games...", len(target_games))

        # 2. Fetch Supporting NLP Feature Data in Batch
        topics_res = await session.execute(
            select(FeatureReviewTopic).where(FeatureReviewTopic.app_id.in_(target_app_ids))
        )
        topics_by_game: dict[int, list[FeatureReviewTopic]] = {}
        for t in topics_res.scalars().all():
            topics_by_game.setdefault(t.app_id, []).append(t)

        complaints_res = await session.execute(
            select(FeatureReviewComplaint).where(FeatureReviewComplaint.app_id.in_(target_app_ids))
        )
        complaints_by_game: dict[int, list[FeatureReviewComplaint]] = {}
        for c in complaints_res.scalars().all():
            complaints_by_game.setdefault(c.app_id, []).append(c)

        features_res = await session.execute(
            select(FeatureReviewFeature).where(FeatureReviewFeature.app_id.in_(target_app_ids))
        )
        features_by_game: dict[int, list[FeatureReviewFeature]] = {}
        for f in features_res.scalars().all():
            features_by_game.setdefault(f.app_id, []).append(f)

        # 3. Compute 6-Dimension Match Profile for each game
        now = datetime.now(UTC)
        profiles: list[MartGameMatchProfile] = []

        for game in target_games:
            app_id = game.app_id
            game_tags = self._extract_tags_set(game)
            game_categories = self._extract_categories_set(game)
            game_topics = topics_by_game.get(app_id, [])
            game_complaints = complaints_by_game.get(app_id, [])
            game_features = features_by_game.get(app_id, [])

            diff = self._score_difficulty(game_tags, game_complaints, game_topics)
            story = self._score_story(game_tags, game_topics, game_features)
            expl = self._score_exploration(game_tags, game_topics, game_features)
            combat = self._score_combat(game_tags, game_topics, game_features)
            mp = self._score_multiplayer(game_tags, game_categories)
            sess = self._score_session_length(game, game_tags)

            # Confidence score based on data availability
            data_points = len(game_tags) + len(game_topics) * 2 + len(game_complaints)
            conf = min(0.98, max(0.65, 0.65 + (data_points / 30.0) * 0.30))

            summary = self._build_profile_summary(diff, story, expl, combat, mp, sess, game_tags)

            profile = MartGameMatchProfile(
                app_id=app_id,
                difficulty=Decimal(str(round(diff, 2))),
                story_weight=Decimal(str(round(story, 2))),
                exploration=Decimal(str(round(expl, 2))),
                combat=Decimal(str(round(combat, 2))),
                multiplayer=Decimal(str(round(mp, 2))),
                session_length=Decimal(str(round(sess, 2))),
                confidence_score=Decimal(str(round(conf, 2))),
                profile_summary=summary,
                materialized_at=now,
            )
            profiles.append(profile)

        # 4. Save to mart_game_match_profile
        if not self.dry_run:
            # Idempotent delete-insert
            await session.execute(
                delete(MartGameMatchProfile).where(MartGameMatchProfile.app_id.in_(target_app_ids))
            )
            for p in profiles:
                session.add(p)
            await session.commit()
            self.logger.info("Successfully materialized %d game match profiles into mart_game_match_profile.", len(profiles))
        else:
            self.logger.info("[DRY RUN] Would materialize %d match profiles.", len(profiles))

    # ── Tag & Category Extraction Helpers ─────────────────────────────────────

    def _extract_tags_set(self, game: RawGame) -> set[str]:
        tags = set()
        if game.tags and isinstance(game.tags, dict):
            for k in game.tags:
                tags.add(str(k).strip().lower())
        elif game.tags and isinstance(game.tags, list):
            for t in game.tags:
                tags.add(str(t).strip().lower())
        if game.genres and isinstance(game.genres, list):
            for g in game.genres:
                if isinstance(g, dict) and "description" in g:
                    tags.add(g["description"].strip().lower())
        return tags

    def _extract_categories_set(self, game: RawGame) -> set[str]:
        cats = set()
        if game.categories and isinstance(game.categories, list):
            for c in game.categories:
                if isinstance(c, dict) and "description" in c:
                    cats.add(c["description"].strip().lower())
        return cats

    # ── Dimension Scoring Functions (0.0 to 10.0) ─────────────────────────────

    def _score_difficulty(
        self,
        tags: set[str],
        complaints: list[FeatureReviewComplaint],
        topics: list[FeatureReviewTopic],
    ) -> float:
        score = 5.0

        # Positive difficulty drivers
        if "souls-like" in tags or "soulslike" in tags:
            score += 3.5
        if "difficult" in tags or "hard" in tags:
            score += 2.0
        if "precision platformer" in tags or "bullet hell" in tags:
            score += 2.2
        if "permadeath" in tags or "unforgiving" in tags:
            score += 1.8
        if "roguelike" in tags or "rogue-like" in tags or "roguelite" in tags:
            score += 1.2
        if "tactical" in tags or "strategy" in tags:
            score += 0.8

        # Low difficulty dampeners
        if "casual" in tags or "relaxing" in tags or "cozy" in tags:
            score -= 3.0
        if "walking simulator" in tags or "visual novel" in tags or "story rich" in tags and "action" not in tags:
            score -= 2.0
        if "easy" in tags or "family friendly" in tags:
            score -= 2.5

        # Review topic / complaint influence
        diff_complaints = [c for c in complaints if "difficulty" in c.category.lower() or "balance" in c.category.lower()]
        if diff_complaints:
            score += 1.0

        return float(np.clip(score, 1.0, 10.0))

    def _score_story(
        self,
        tags: set[str],
        topics: list[FeatureReviewTopic],
        features: list[FeatureReviewFeature],
    ) -> float:
        score = 3.5

        if "story rich" in tags or "narrative" in tags:
            score += 3.5
        if "visual novel" in tags:
            score += 4.5
        if "lore-rich" in tags or "deep lore" in tags:
            score += 2.5
        if "rpg" in tags or "jrpg" in tags or "crpg" in tags:
            score += 2.0
        if "choices matter" in tags or "multiple endings" in tags:
            score += 2.0
        if "adventure" in tags:
            score += 1.0

        # Minimal story dampeners
        if "battle royale" in tags or "arena shooter" in tags or "sports" in tags:
            score -= 3.0
        if "sandbox" in tags and "rpg" not in tags:
            score -= 1.5

        # Topic review evidence
        lore_topics = [t for t in topics if "lore" in t.topic_label.lower() or "story" in t.topic_label.lower() or "atmosphere" in t.topic_label.lower()]
        if lore_topics:
            score += 1.5

        return float(np.clip(score, 0.5, 10.0))

    def _score_exploration(
        self,
        tags: set[str],
        topics: list[FeatureReviewTopic],
        features: list[FeatureReviewFeature],
    ) -> float:
        score = 4.0

        if "open world" in tags:
            score += 4.0
        if "metroidvania" in tags:
            score += 3.5
        if "exploration" in tags:
            score += 3.0
        if "non-linear" in tags or "secrets" in tags:
            score += 2.0
        if "dungeon crawler" in tags or "sandbox" in tags:
            score += 1.5
        if "adventure" in tags:
            score += 1.0

        # Linear / arcade dampeners
        if "linear" in tags or "on-rails" in tags or "arcade" in tags:
            score -= 2.5
        if "match 3" in tags or "puzzle" in tags and "adventure" not in tags:
            score -= 2.0

        expl_features = [f for f in features if "exploration" in f.feature_name.lower() or "map" in f.feature_name.lower() or "world" in f.feature_name.lower()]
        if expl_features:
            score += 1.2

        return float(np.clip(score, 0.5, 10.0))

    def _score_combat(
        self,
        tags: set[str],
        topics: list[FeatureReviewTopic],
        features: list[FeatureReviewFeature],
    ) -> float:
        score = 4.5

        if "action" in tags:
            score += 2.0
        if "hack and slash" in tags or "beat 'em up" in tags or "souls-like" in tags:
            score += 3.5
        if "fps" in tags or "shooter" in tags or "third-person shooter" in tags:
            score += 3.0
        if "fast-paced" in tags or "combat" in tags:
            score += 2.0
        if "fighting" in tags or "bullet hell" in tags:
            score += 2.5

        # Non-combat dampeners
        if "walking simulator" in tags or "visual novel" in tags:
            score -= 4.0
        if "puzzle" in tags and "action" not in tags:
            score -= 3.5
        if "city builder" in tags or "management" in tags or "farming sim" in tags:
            score -= 3.0
        if "relaxing" in tags or "cozy" in tags:
            score -= 2.0

        combat_topics = [t for t in topics if "combat" in t.topic_label.lower() or "boss" in t.topic_label.lower() or "weapon" in t.topic_label.lower()]
        if combat_topics:
            score += 1.5

        return float(np.clip(score, 0.0, 10.0))

    def _score_multiplayer(self, tags: set[str], categories: set[str]) -> float:
        score = 0.0

        if "multiplayer" in tags or "multi-player" in categories:
            score += 5.0
        if "online pvp" in tags or "online pvp" in categories or "pvp" in tags:
            score += 3.0
        if "co-op" in tags or "online co-op" in categories or "co-op" in categories:
            score += 2.5
        if "mmo" in tags or "mmorpg" in tags:
            score += 4.0
        if "competitive" in tags or "team-based" in tags:
            score += 2.0
        if "local multiplayer" in tags or "local co-op" in tags:
            score += 2.0

        return float(np.clip(score, 0.0, 10.0))

    def _score_session_length(self, game: RawGame, tags: set[str]) -> float:
        score = 5.0

        # Based on reported average / median playtime
        playtime = game.median_playtime_forever or game.average_playtime_forever or 0
        if playtime > 0:
            if playtime < 120:  # < 2 hours
                score = 3.0
            elif playtime < 600:  # 2 - 10 hours
                score = 5.0
            elif playtime < 2400:  # 10 - 40 hours
                score = 7.0
            else:  # 40+ hours
                score = 9.0

        if "short" in tags or "casual" in tags or "micro" in tags:
            score -= 2.0
        if "endless" in tags or "grand strategy" in tags or "4x" in tags or "mmorpg" in tags:
            score += 2.0
        if "roguelite" in tags or "roguelike" in tags:
            score += 1.0

        return float(np.clip(score, 1.0, 10.0))

    # ── Summary Synthesis ─────────────────────────────────────────────────────

    def _build_profile_summary(
        self,
        diff: float,
        story: float,
        expl: float,
        combat: float,
        mp: float,
        sess: float,
        tags: set[str],
    ) -> dict[str, Any]:
        dimensions = [
            ("Difficulty", diff),
            ("Story Weight", story),
            ("Exploration", expl),
            ("Combat Intensity", combat),
            ("Multiplayer Focus", mp),
            ("Session Length", sess),
        ]
        # Top dominant traits (intensity >= 6.5)
        dominant = sorted([d for d in dimensions if d[1] >= 6.0], key=lambda x: -x[1])
        trait_labels = [f"High {d[0]} ({d[1]:.1f}/10)" for d in dominant[:3]]
        if not trait_labels:
            trait_labels = [f"Balanced {dimensions[0][0]} ({diff:.1f}/10)"]

        return {
            "primary_traits": trait_labels,
            "tag_highlights": list(sorted(tags))[:8],
            "intensity_vector": {
                "difficulty": round(diff, 2),
                "story_weight": round(story, 2),
                "exploration": round(expl, 2),
                "combat": round(combat, 2),
                "multiplayer": round(mp, 2),
                "session_length": round(sess, 2),
            },
        }


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Game Match Profiles (Phase 5)")
    parser.add_argument("--app-id", type=int, help="Single Steam App ID to process")
    parser.add_argument("--all", action="store_true", help="Process all games in catalog")
    parser.add_argument("--dry-run", action="store_true", help="Calculate without database commit")

    args = parser.parse_args()
    job = MaterializeMatchProfilesJob(app_id=args.app_id, all_games=args.all, dry_run=args.dry_run)
    job.run()


if __name__ == "__main__":
    main()
