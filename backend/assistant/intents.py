"""
Fixed intent taxonomy and slot definitions for Ask SteamIQ — Phase 6.

Rule 4 & Roadmap §3 Phase 6:
  - Zero FastAPI / SQLAlchemy / DB imports in this module.
  - Pure Python dataclasses and constants.
  - Mode-visibility rules: dev-only intents refused in player mode with an honest explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Intent(StrEnum):
    GAME_OVERVIEW = "game_overview"
    REVIEW_SUMMARY = "review_summary"
    TOP_COMPLAINTS = "top_complaints"
    LOVED_FEATURES = "loved_features"
    SIMILAR_GAMES = "similar_games"
    PRICE_ADVICE = "price_advice"
    PLAYER_MATCH = "player_match"
    SUCCESS_SCORE = "success_score"
    UPDATE_IMPACT = "update_impact"
    RECOMMENDATIONS = "recommendations"
    MARKET_OPPORTUNITY = "market_opportunity"
    RECOMMEND_FOR_ME = "recommend_for_me"
    OUT_OF_SCOPE = "out_of_scope"


ALL_INTENTS: list[str] = [i.value for i in Intent]

# Dev-only intents hidden/refused in player mode
DEV_ONLY_INTENTS: set[str] = {
    Intent.SUCCESS_SCORE.value,
    Intent.UPDATE_IMPACT.value,
    Intent.RECOMMENDATIONS.value,
    Intent.MARKET_OPPORTUNITY.value,
}

# Intents requiring a specific target game (app_id or game title)
GAME_SPECIFIC_INTENTS: set[str] = {
    Intent.GAME_OVERVIEW.value,
    Intent.REVIEW_SUMMARY.value,
    Intent.TOP_COMPLAINTS.value,
    Intent.LOVED_FEATURES.value,
    Intent.SIMILAR_GAMES.value,
    Intent.PRICE_ADVICE.value,
    Intent.PLAYER_MATCH.value,
    Intent.SUCCESS_SCORE.value,
    Intent.UPDATE_IMPACT.value,
    Intent.RECOMMENDATIONS.value,
}

# Intents operating over user library / global market
CATALOG_OR_GLOBAL_INTENTS: set[str] = {
    Intent.MARKET_OPPORTUNITY.value,
    Intent.RECOMMEND_FOR_ME.value,
    Intent.OUT_OF_SCOPE.value,
}


@dataclass
class IntentDefinition:
    intent: str
    display_name: str
    description: str
    source_endpoint: str
    source_table: str
    required_slots: list[str]
    mode_visibility: list[str]  # ["player", "developer"] or ["developer"]
    template_id: str
    keywords: list[str] = field(default_factory=list)


DEV_MODE_REFUSAL_MESSAGE = (
    "This analytical view is available exclusively in Developer Mode. "
    "Switch to Developer Mode using the mode toggle in the top bar to inspect "
    "predictive success scores, SHAP feature drivers, update impact, or market opportunity."
)

MISSING_GAME_SLOT_MESSAGE = (
    "Please specify which game you would like to analyze (e.g., 'What are the main complaints for Hades?')."
)

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions related to Steam game intelligence, player reviews, pricing, "
    "recommendations, and Steam market analytics. Try asking about a game's reception, price, "
    "or similar titles."
)
