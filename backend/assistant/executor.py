"""
Deterministic Executor and Mode-Aware Template Engine — Phase 6.

Key requirements:
  - Mode Parity Invariant: Given (intent, slots), the structured data call arguments
    are mathematically identical under both player and developer modes.
  - Only the template copy and framing differ between modes.
  - Dev-only intents requested in player mode are refused with an honest explanation.
  - Every response carries complete provenance metadata (intent, confidence, model_run_id,
    tables read, materialized_at).
  - When data is missing, renders honest empty states (HONEST-FALLBACK convention).
  - No LLM free-text generation.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from assistant.intents import (
    DEV_MODE_REFUSAL_MESSAGE,
    DEV_ONLY_INTENTS,
    GAME_SPECIFIC_INTENTS,
    MISSING_GAME_SLOT_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    Intent,
)

logger = logging.getLogger("steamiq.assistant.executor")


@dataclass
class StructuredCallArgs:
    """Represents the canonical structured query arguments."""
    intent: str
    app_id: int | None = None
    liked_app_ids: list[int] = field(default_factory=list)
    limit: int = 5


@dataclass
class ExecutionResult:
    """Complete structured response with provenance."""
    answer: str
    intent: str
    confidence: float
    abstained: bool
    user_mode: str
    model_run_id: str | None
    data_tables_read: list[str]
    materialized_at: str | None
    sources: list[dict[str, Any]] = field(default_factory=list)
    structured_args: dict[str, Any] = field(default_factory=dict)
    retrieval_passages: list[dict[str, Any]] = field(default_factory=list)


def build_structured_call_args(
    intent: str,
    slots: dict[str, Any],
) -> StructuredCallArgs:
    """
    Constructs canonical structured call parameters.
    CRITICAL: Does NOT inspect or branch on user_mode.
    Guarantees strict mode parity for all data queries.
    """
    app_id = slots.get("app_id") or slots.get("game_app_id")
    if app_id is not None:
        try:
            app_id = int(app_id)
        except (ValueError, TypeError):
            app_id = None

    liked_raw = slots.get("liked_app_ids") or []
    liked_ids: list[int] = []
    if isinstance(liked_raw, list):
        for lid in liked_raw:
            with contextlib.suppress(ValueError, TypeError):
                liked_ids.append(int(lid))

    return StructuredCallArgs(
        intent=intent,
        app_id=app_id,
        liked_app_ids=liked_ids,
        limit=int(slots.get("limit", 5)),
    )


def execute_template(
    intent: str,
    data: dict[str, Any] | None,
    slots: dict[str, Any],
    user_mode: str,
    confidence: float = 1.0,
    model_run_id: str | None = None,
    retrieval_passages: list[dict[str, Any]] | None = None,
) -> ExecutionResult:
    """
    Fills deterministic response templates according to user_mode and data.
    """
    is_player = user_mode == "player"
    retrieval_passages = retrieval_passages or []

    # 1. Check Dev-Only Intent in Player Mode
    if is_player and intent in DEV_ONLY_INTENTS:
        return ExecutionResult(
            answer=DEV_MODE_REFUSAL_MESSAGE,
            intent=intent,
            confidence=confidence,
            abstained=False,
            user_mode=user_mode,
            model_run_id=model_run_id,
            data_tables_read=[],
            materialized_at=None,
            sources=[],
            structured_args=build_structured_call_args(intent, slots).__dict__,
        )

    # 2. Check Out of Scope
    if intent == Intent.OUT_OF_SCOPE.value:
        answer = OUT_OF_SCOPE_MESSAGE
        if retrieval_passages:
            answer += "\n\nHowever, here are the most relevant catalog excerpts matching your query:"
        return ExecutionResult(
            answer=answer,
            intent=intent,
            confidence=confidence,
            abstained=True,
            user_mode=user_mode,
            model_run_id=model_run_id,
            data_tables_read=[],
            materialized_at=None,
            sources=[],
            structured_args=build_structured_call_args(intent, slots).__dict__,
            retrieval_passages=retrieval_passages,
        )

    # 3. Check Missing Game Slot for Game-Specific Intents
    call_args = build_structured_call_args(intent, slots)
    if intent in GAME_SPECIFIC_INTENTS and call_args.app_id is None:
        answer = MISSING_GAME_SLOT_MESSAGE
        if retrieval_passages:
            answer += "\n\nRelevant review intelligence passages:"
        return ExecutionResult(
            answer=answer,
            intent=intent,
            confidence=confidence,
            abstained=True,
            user_mode=user_mode,
            model_run_id=model_run_id,
            data_tables_read=[],
            materialized_at=None,
            sources=[],
            structured_args=call_args.__dict__,
            retrieval_passages=retrieval_passages,
        )

    # 4. Check Missing Data (Honest Fallback)
    game_title = slots.get("game_title") or (data.get("title") if data else None) or f"Game #{call_args.app_id}"
    if data is None or not data:
        answer = (
            f"No materialized data is currently available for {game_title}. "
            "Please ensure the data ingestion and mart materialization pipeline has run for this title."
        )
        return ExecutionResult(
            answer=answer,
            intent=intent,
            confidence=confidence,
            abstained=False,
            user_mode=user_mode,
            model_run_id=model_run_id,
            data_tables_read=data.get("tables_read", []) if data else [],
            materialized_at=None,
            sources=[],
            structured_args=call_args.__dict__,
            retrieval_passages=retrieval_passages,
        )

    # 5. Intent-Specific Template Rendering
    tables_read = data.get("tables_read", [])
    mat_at = data.get("materialized_at")
    sources = data.get("sources", [])

    if intent == Intent.GAME_OVERVIEW.value:
        title = data.get("title", game_title)
        developer = data.get("developer", "Unknown")
        price = data.get("price_usd", "0.00")
        pos_ratio = data.get("positive_ratio", "--")
        tags = ", ".join(data.get("tags", [])[:5]) or "None"
        release = data.get("release_date", "Unknown")

        if is_player:
            answer = (
                f"**{title}** is developed by {developer} and was released on {release}. "
                f"It is currently priced at ${price} with a {pos_ratio}% positive review rating. "
                f"Key tags: {tags}."
            )
        else:
            owners = data.get("owners_estimate", "--")
            answer = (
                f"**{title}** Overview: Developer: {developer} | Release: {release} | "
                f"Price: ${price} | Positive Ratio: {pos_ratio}% | Owners Estimate: {owners}. "
                f"Core Tags: {tags}."
            )

    elif intent == Intent.REVIEW_SUMMARY.value:
        title = data.get("title", game_title)
        sentiment_label = data.get("review_score_desc", "Positive")
        pos_ratio = data.get("positive_ratio", "--")
        summary_text = (data.get("summary_text") or "").strip()

        if is_player:
            if summary_text:
                answer = (
                    f"Player consensus for **{title}** is **{sentiment_label}** ({pos_ratio}% positive). "
                    f"{summary_text}"
                )
            else:
                answer = (
                    f"Player consensus for **{title}** is **{sentiment_label}** ({pos_ratio}% positive). "
                    "No structured NLP review summary is currently available for this title."
                )
        else:
            rev_count = data.get("user_reviews", "--")
            if summary_text:
                answer = (
                    f"Review Intelligence for **{title}**: Sentiment is **{sentiment_label}** "
                    f"({pos_ratio}% positive over {rev_count} verified reviews). "
                    f"Extracted NLP Summary: {summary_text}"
                )
            else:
                answer = (
                    f"Review Intelligence for **{title}**: Sentiment is **{sentiment_label}** "
                    f"({pos_ratio}% positive over {rev_count} verified reviews). "
                    "Extracted NLP Summary: None materialized."
                )

    elif intent == Intent.TOP_COMPLAINTS.value:
        title = data.get("title", game_title)
        complaints = data.get("complaints", [])
        if not complaints:
            answer = f"No significant recurring player complaints have been recorded for **{title}**."
        else:
            comp_lines = []
            for c in complaints[:4]:
                cat = c.get("category", "Issue")
                vol = c.get("volume_pct", 0.0)
                sev = c.get("severity", "moderate")
                snips = c.get("representative_snippets", [])
                snip_str = f' — e.g. "{snips[0]}"' if snips else ""
                comp_lines.append(f"- **{cat}** ({vol}% of negative reviews, severity: {sev}){snip_str}")
            items_str = "\n".join(comp_lines)
            if is_player:
                answer = f"Here are the top player complaints for **{title}**:\n{items_str}"
            else:
                answer = f"Top NLP complaint clusters for **{title}** (churn drivers):\n{items_str}"

    elif intent == Intent.LOVED_FEATURES.value:
        title = data.get("title", game_title)
        loved = data.get("loved_features", [])
        if not loved:
            answer = f"No specific praised features have been isolated for **{title}**."
        else:
            loved_lines = []
            for feat in loved[:4]:
                name = feat.get("feature_name", feat.get("feature", "Highlight"))
                mentions = feat.get("mention_count", feat.get("mentions", 0))
                intensity = feat.get("praise_intensity", feat.get("intensity", 0))
                loved_lines.append(f"- **{name}** ({intensity}% praise frequency, {mentions} positive mentions)")
            items_str = "\n".join(loved_lines)
            if is_player:
                answer = f"What players love most about **{title}**:\n{items_str}"
            else:
                answer = f"Primary retention drivers and praised features for **{title}**:\n{items_str}"

    elif intent == Intent.SIMILAR_GAMES.value:
        title = data.get("title", game_title)
        similar_list = data.get("similar_games", [])
        if not similar_list:
            answer = f"No similar titles found in catalog for **{title}**."
        else:
            if is_player:
                names = [s.get("title", f"App {s.get('app_id')}") for s in similar_list[:5]]
                answer = f"If you like **{title}**, check out these similar games: {', '.join(names)}."
            else:
                items_str = "\n".join([
                    f"- **{s.get('title', '')}** (Sim: {round(float(s.get('similarity', 0.0)), 3)}, Market Presence: {s.get('market_presence', 'medium')})"
                    for s in similar_list[:5]
                ])
                answer = f"Top Competitors for **{title}** in embedding space:\n{items_str}"

    elif intent == Intent.PRICE_ADVICE.value:
        title = data.get("title", game_title)
        price = data.get("price_usd", "0.00")
        hist_low = data.get("historical_low_usd", price)
        discount = data.get("discount_pct", 0)

        if is_player:
            if discount > 0:
                answer = (
                    f"**{title}** is currently on sale for ${price} ({discount}% off). "
                    f"Its historical low is ${hist_low}. This is a solid time to buy."
                )
            else:
                answer = (
                    f"**{title}** is currently full price at ${price}. "
                    f"Its recorded historical low is ${hist_low}. If you're on a budget, consider waiting for the next Steam sale."
                )
        else:
            median = data.get("genre_median_price_usd", price)
            p25 = data.get("genre_p25_price_usd", "--")
            p75 = data.get("genre_p75_price_usd", "--")
            answer = (
                f"Pricing Intelligence for **{title}**: Current Price: ${price} | "
                f"Genre Median: ${median} (P25: ${p25}, P75: ${p75}) | Historical Low: ${hist_low}. "
                f"Positioning aligns with market median."
            )

    elif intent == Intent.PLAYER_MATCH.value:
        title = data.get("title", game_title)
        diff = data.get("difficulty", 50)
        story = data.get("story_weight", 50)
        session = data.get("session_length", 50)
        combat = data.get("combat", 50)

        if is_player:
            answer = (
                f"Gameplay Intensity Profile for **{title}**:\n"
                f"- Difficulty: {diff}/100\n- Story Weight: {story}/100\n"
                f"- Combat: {combat}/100\n- Session Length: {session}/100."
            )
        else:
            multi = data.get("multiplayer", 0)
            expl = data.get("exploration", 50)
            answer = (
                f"6D Match Profile for **{title}**: Difficulty: {diff} | Story: {story} | "
                f"Combat: {combat} | Exploration: {expl} | Multiplayer: {multi} | Session Length: {session}."
            )

    elif intent == Intent.SUCCESS_SCORE.value:
        title = data.get("title", game_title)
        score = data.get("success_score", 0.0)
        drivers = data.get("shap_drivers", [])
        drivers_str = ", ".join([f"{d.get('feature')}: {d.get('effect')}" for d in drivers[:3]]) or "Balanced features"
        answer = (
            f"Predictive ML Model indicates a **{round(float(score) * 100, 1)}%** success probability for **{title}**. "
            f"Key SHAP drivers: {drivers_str}."
        )

    elif intent == Intent.UPDATE_IMPACT.value:
        title = data.get("title", game_title)
        shift = data.get("sentiment_shift_pct", 0.0)
        ver = data.get("latest_patch", "latest patch")
        answer = (
            f"Update Impact Analysis for **{title}** ({ver}): "
            f"30-day sentiment shifted by {round(float(shift), 2)}% following patch release."
        )

    elif intent == Intent.RECOMMENDATIONS.value:
        title = data.get("title", game_title)
        recs = data.get("strategic_recommendations", [])
        recs_str = "\n".join([f"- {r}" for r in recs[:3]]) if recs else "- Maintain current update cadence."
        answer = f"Strategic Recommendations for **{title}** based on review signals & competitor gaps:\n{recs_str}"

    elif intent == Intent.MARKET_OPPORTUNITY.value:
        top_opps = data.get("opportunities", [])
        if not top_opps:
            answer = "No clear market gaps currently exceed the threshold."
        else:
            items_str = "\n".join([f"- **{o.get('genre_or_tag')}**: Opportunity Score {o.get('score')} ({o.get('rationale', '')})" for o in top_opps[:3]])
            answer = f"Top Underserved Market Niches:\n{items_str}"

    elif intent == Intent.RECOMMEND_FOR_ME.value:
        rec_items = data.get("recommendations", [])
        if not rec_items:
            answer = "Provide a few games from your Steam library to get personalized recommendations."
        else:
            rec_titles = [r.get("title") for r in rec_items[:5] if r.get("title")]
            answer = f"Based on your library, here are your top recommended games: {', '.join(rec_titles)}."

    else:
        answer = f"Information for intent '{intent}' resolved successfully."

    return ExecutionResult(
        answer=answer,
        intent=intent,
        confidence=confidence,
        abstained=False,
        user_mode=user_mode,
        model_run_id=model_run_id,
        data_tables_read=tables_read,
        materialized_at=mat_at,
        sources=sources,
        structured_args=call_args.__dict__,
        retrieval_passages=retrieval_passages,
    )
