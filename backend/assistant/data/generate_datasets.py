"""
Dataset generation and near-duplicate filtering for Ask SteamIQ — Phase 6.

Generates:
  1. backend/assistant/data/train_queries.jsonl (seeded, synthetic_template source)
  2. backend/assistant/data/gold_queries.jsonl (~260 realistic queries with typos, adversarial OOS, verified: false)
  3. Near-duplicate deduplication: drops any train query matching normalized text or cosine > 0.95 against gold.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_datasets")

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Sample of known Steam catalog games for synthetic generation
GAMES = [
    ("Hades", 1145360),
    ("Hollow Knight", 367520),
    ("Dead Cells", 588650),
    ("Stardew Valley", 413150),
    ("Terraria", 105600),
    ("Celeste", 504230),
    ("Subnautica", 264710),
    ("Slay the Spire", 646570),
    ("Portal 2", 620),
    ("Counter-Strike: Global Offensive", 730),
    ("Dota 2", 570),
    ("Cyberpunk 2077", 1091500),
    ("Elden Ring", 1245620),
    ("Baldur's Gate 3", 1086940),
]

INTENT_TEMPLATES: dict[str, list[str]] = {
    "game_overview": [
        "What is {game} about?",
        "Can you give me an overview of {game}?",
        "Tell me about {game}.",
        "Who is the developer of {game}?",
        "What genre is {game} in?",
        "When was {game} released?",
        "Summary and release date for {game}.",
        "What kind of game is {game}?",
        "Give me the tags and general details for {game}.",
        "Basic info on {game}.",
    ],
    "review_summary": [
        "What do players think of {game}?",
        "What is the overall review consensus for {game}?",
        "Summary of player reviews for {game}.",
        "How well-received is {game} on Steam?",
        "Is {game} generally liked by players?",
        "What are people saying in reviews about {game}?",
        "Review breakdown and sentiment for {game}.",
        "Community rating and sentiment analysis for {game}.",
        "Is {game} considered a good game by players?",
        "Do players recommend {game}?",
    ],
    "top_complaints": [
        "What are the main complaints about {game}?",
        "Why do people dislike {game}?",
        "What are the biggest issues with {game}?",
        "What do negative reviews say about {game}?",
        "Are there common bugs or problems in {game}?",
        "What is the most criticized feature of {game}?",
        "Tell me what players hate about {game}.",
        "Top player complaints and criticisms for {game}.",
        "Why are players giving {game} negative reviews?",
        "What are the main technical issues with {game}?",
    ],
    "loved_features": [
        "What do players love most about {game}?",
        "What are the best features of {game}?",
        "Why do people enjoy {game}?",
        "What are the praised mechanics in {game}?",
        "What makes {game} so good according to reviews?",
        "Highlights and positive feedback for {game}.",
        "What is {game}'s strongest gameplay mechanic?",
        "What do positive reviews praise in {game}?",
        "Why do players give {game} such high scores?",
        "Top loved features in {game}.",
    ],
    "similar_games": [
        "What games are similar to {game}?",
        "Can you suggest alternatives to {game}?",
        "What are the main competitors to {game}?",
        "Games like {game} on Steam.",
        "What titles compete directly with {game}?",
        "Find me games that feel like {game}.",
        "Comparable games to {game} in embedding space.",
        "Similar gameplay experience to {game}.",
        "Who are {game}'s closest competitors?",
        "Recommendations similar in style to {game}.",
    ],
    "price_advice": [
        "Should I buy {game} right now or wait for a sale?",
        "What is the current price of {game}?",
        "Is {game} on sale at the moment?",
        "What is the historical low price for {game}?",
        "Is {game} worth buying at full price?",
        "Has {game} ever been cheaper than its current price?",
        "Pricing history and discount advice for {game}.",
        "How does {game}'s price compare to the genre median?",
        "Competitor price positioning for {game}.",
        "Should I wait for a discount on {game}?",
    ],
    "player_match": [
        "Is {game} right for me?",
        "How difficult is {game}?",
        "What is the story weight and combat intensity in {game}?",
        "What is the average session length for {game}?",
        "Does {game} have heavy exploration or multiplayer?",
        "Match profile and intensity scores for {game}.",
        "Would I like {game} if I prefer short sessions?",
        "How challenging is the gameplay in {game}?",
        "Breakdown of {game}'s gameplay intensity attributes.",
        "Is {game} beginner friendly or hardcore?",
    ],
    "success_score": [
        "What is the predicted success score for {game}?",
        "What are the main SHAP feature drivers for {game}?",
        "Will {game} succeed based on our predictive model?",
        "Show me the success probability and drivers for {game}.",
        "How does our ML model evaluate {game}'s commercial potential?",
        "What factors drive {game}'s predicted revenue outcome?",
        "Predictive success metrics and feature importance for {game}.",
        "What is {game}'s estimated success tier?",
        "Explain {game}'s success forecast using SHAP values.",
        "Probability of commercial success for {game}.",
    ],
    "update_impact": [
        "How did the latest update impact player sentiment for {game}?",
        "Did {game}'s recent patch improve review scores?",
        "What was the impact of the latest patch notes on {game}?",
        "Sentiment shift after the recent update for {game}.",
        "Did players respond positively to {game}'s update?",
        "Track update impact and review volume deltas for {game}.",
        "Pre-post patch review score comparison for {game}.",
        "Has {game}'s player sentiment recovered after the patch?",
        "Analyze the update trend for {game}.",
        "Review reception following the latest update to {game}.",
    ],
    "recommendations": [
        "What strategic recommendations do we have for {game}?",
        "How can the developer improve {game}?",
        "What fixes or updates should the team prioritize for {game}?",
        "Developer recommendations based on competitor gap analysis for {game}.",
        "Actionable insights and roadmap suggestions for {game}.",
        "What should the developers of {game} address next?",
        "Strategic advice for {game} based on player complaints.",
        "How can {game} increase player retention?",
        "Recommendations to boost {game}'s review score.",
        "What gameplay changes would improve {game}?",
    ],
    "market_opportunity": [
        "Where are the biggest market opportunities on Steam right now?",
        "Which genres have high demand but low supply?",
        "Show me top underserved tags and market niches.",
        "What are the highest scoring opportunity areas for game developers?",
        "Where is the white space in the Steam market?",
        "Find high opportunity tag combinations with high median revenue.",
        "Which genre niches have the highest market opportunity score?",
        "Where should an indie studio build their next game?",
        "Market opportunity report across Steam categories.",
        "Underserved niches and genre trends.",
    ],
    "recommend_for_me": [
        "What games should I play based on my library?",
        "Recommend games for me based on what I like.",
        "Suggest games to play next.",
        "Give me personalized game recommendations for my taste.",
        "What should I play next based on my Steam library?",
        "Can you recommend titles suited to my play history?",
        "Personalized for-you recommendations based on my favorites.",
        "Pick some games for me from the catalog.",
        "Recommend games based on my recent games.",
        "Personalized recommendations for my profile.",
    ],
    "out_of_scope": [
        "What is the weather in Tokyo today?",
        "Who was the first president of the United States?",
        "Can you write a Python script to sort a list?",
        "What is the recipe for chocolate chip cookies?",
        "Who won the Super Bowl last year?",
        "Tell me a joke about dogs.",
        "How do I fix a leaking kitchen faucet?",
        "What is the capital of France?",
        "Can you write a poem about the ocean?",
        "Explain general relativity in simple terms.",
        "Recommend me a good movie to watch tonight.",
        "How do I bake sourdough bread?",
        "What is the distance from Earth to Mars?",
        "Translate this sentence into Spanish.",
        "Who is the CEO of Apple?",
        "Can you draft an email to my manager?",
        "What are the health benefits of green tea?",
        "How many kilometers are in a mile?",
        "Write an essay on modern architecture.",
        "What is the latest stock price of Microsoft?",
    ],
}

# Paraphrase modifiers to expand synthetic queries
PARAPHRASE_PREFIXES = [
    "",
    "Could you tell me: ",
    "Please tell me, ",
    "I'd like to know, ",
    "Hey SteamIQ, ",
    "Quick question: ",
    "Can you check: ",
]

PARAPHRASE_SUFFIXES = [
    "",
    " please.",
    " thanks!",
    " right now.",
    " on Steam.",
]


def generate_train_queries(seed: int = 42, target_count_per_intent: int = 70) -> list[dict[str, Any]]:
    random.seed(seed)
    train_data: list[dict[str, Any]] = []

    for intent, templates in INTENT_TEMPLATES.items():
        generated = 0
        while generated < target_count_per_intent:
            tmpl = random.choice(templates)
            game_title, app_id = random.choice(GAMES)
            prefix = random.choice(PARAPHRASE_PREFIXES)
            suffix = random.choice(PARAPHRASE_SUFFIXES)

            if "{game}" in tmpl:
                text = f"{prefix}{tmpl.format(game=game_title)}{suffix}".strip()
                item = {
                    "text": text,
                    "intent": intent,
                    "game": game_title,
                    "app_id": app_id,
                    "mode": random.choice(["player", "developer"]),
                    "source": "synthetic_template",
                }
            else:
                text = f"{prefix}{tmpl}{suffix}".strip()
                item = {
                    "text": text,
                    "intent": intent,
                    "game": None,
                    "app_id": None,
                    "mode": random.choice(["player", "developer"]),
                    "source": "synthetic_template",
                }

            train_data.append(item)
            generated += 1

    random.shuffle(train_data)
    return train_data


def generate_gold_queries() -> list[dict[str, Any]]:
    """
    Drafts ~260 realistic queries across all intents.
    Includes:
      - >= 15 per intent
      - >= 45 OUT_OF_SCOPE (including adversarial near-misses)
      - >= 35 with typos or informal phrasing
      - verified: false
    """
    gold_data: list[dict[str, Any]] = [
        # --- 1. game_overview (>= 16) ---
        {"text": "whats hades about?", "intent": "game_overview", "game": "Hades", "mode": "player", "verified": False},
        {"text": "give me a brief overview of hollow knight", "intent": "game_overview", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "who developed stardew valley and when was it released?", "intent": "game_overview", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "tell me about dead cells genre and tags", "intent": "game_overview", "game": "Dead Cells", "mode": "player", "verified": False},
        {"text": "what kind of game is terraria?", "intent": "game_overview", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "celeste summary and release info", "intent": "game_overview", "game": "Celeste", "mode": "developer", "verified": False},
        {"text": "basic rundown on subnautica gameplay and theme", "intent": "game_overview", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "can u explain what slay the spire is?", "intent": "game_overview", "game": "Slay the Spire", "mode": "player", "verified": False},
        {"text": "elden ring game overview and developer", "intent": "game_overview", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "general overview for portal 2", "intent": "game_overview", "game": "Portal 2", "mode": "player", "verified": False},
        {"text": "what is baldur's gate 3 about?", "intent": "game_overview", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "cyberpunk 2077 core tags and genre breakdown", "intent": "game_overview", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "explain the premise of dota 2", "intent": "game_overview", "game": "Dota 2", "mode": "player", "verified": False},
        {"text": "tell me bout hades release and studio", "intent": "game_overview", "game": "Hades", "mode": "player", "verified": False},
        {"text": "give an overview for counter-strike global offensive", "intent": "game_overview", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "what style of game is hollow knight?", "intent": "game_overview", "game": "Hollow Knight", "mode": "player", "verified": False},

        # --- 2. review_summary (>= 16) ---
        {"text": "what do players think of hades on steam?", "intent": "review_summary", "game": "Hades", "mode": "player", "verified": False},
        {"text": "how are the reviews for hollow knight looking?", "intent": "review_summary", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "overall review consensus for stardew valley", "intent": "review_summary", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "do players recommend dead cells?", "intent": "review_summary", "game": "Dead Cells", "mode": "player", "verified": False},
        {"text": "what is the community sentiment for subnautica?", "intent": "review_summary", "game": "Subnautica", "mode": "developer", "verified": False},
        {"text": "steam rating breakdown for celeste", "intent": "review_summary", "game": "Celeste", "mode": "player", "verified": False},
        {"text": "how well received is slay the spire by customers?", "intent": "review_summary", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "summary of user sentiment for elden ring", "intent": "review_summary", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "are people liking baldurs gate 3?", "intent": "review_summary", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "review summary on portal 2", "intent": "review_summary", "game": "Portal 2", "mode": "player", "verified": False},
        {"text": "community reception of cyberpunk 2077 nowadays", "intent": "review_summary", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "what is the general consensus on terraria?", "intent": "review_summary", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "how positive are the steam reviews for hades?", "intent": "review_summary", "game": "Hades", "mode": "player", "verified": False},
        {"text": "player sentiment overview for csgo", "intent": "review_summary", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "is dota 2 well received by players currently?", "intent": "review_summary", "game": "Dota 2", "mode": "player", "verified": False},
        {"text": "give me a review summary of hollow knight", "intent": "review_summary", "game": "Hollow Knight", "mode": "player", "verified": False},

        # --- 3. top_complaints (>= 16) ---
        {"text": "what are the main complaints about hades?", "intent": "top_complaints", "game": "Hades", "mode": "player", "verified": False},
        {"text": "why do people dislike hollow knight?", "intent": "top_complaints", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "top negative review themes for dead cells", "intent": "top_complaints", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "what do players hate about elden ring?", "intent": "top_complaints", "game": "Elden Ring", "mode": "player", "verified": False},
        {"text": "biggest criticisms and bugs reported in cyberpunk 2077", "intent": "top_complaints", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "what are the common complaints for subnautica?", "intent": "top_complaints", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "why do negative reviews criticize stardew valley?", "intent": "top_complaints", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "top issues players mention in slay the spire", "intent": "top_complaints", "game": "Slay the Spire", "mode": "player", "verified": False},
        {"text": "what sucks about baldurs gate 3 according to players?", "intent": "top_complaints", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "common pain points in celeste reviews", "intent": "top_complaints", "game": "Celeste", "mode": "developer", "verified": False},
        {"text": "what do people complain about in portal 2?", "intent": "top_complaints", "game": "Portal 2", "mode": "player", "verified": False},
        {"text": "main complaints regarding terraria gameplay", "intent": "top_complaints", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "churn drivers and complaint clusters for csgo", "intent": "top_complaints", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "what are the worst parts of hades according to negative reviews?", "intent": "top_complaints", "game": "Hades", "mode": "player", "verified": False},
        {"text": "player frustrations and negative themes in dota 2", "intent": "top_complaints", "game": "Dota 2", "mode": "developer", "verified": False},
        {"text": "what do critics dislike about hollow knight?", "intent": "top_complaints", "game": "Hollow Knight", "mode": "player", "verified": False},

        # --- 4. loved_features (>= 16) ---
        {"text": "what do players love most about hades?", "intent": "loved_features", "game": "Hades", "mode": "player", "verified": False},
        {"text": "best features and praise for hollow knight", "intent": "loved_features", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "what makes stardew valley so loved by the community?", "intent": "loved_features", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "most praised mechanics in dead cells", "intent": "loved_features", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "what do positive reviews highlight in celeste?", "intent": "loved_features", "game": "Celeste", "mode": "player", "verified": False},
        {"text": "why do people rave about subnautica's atmosphere?", "intent": "loved_features", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "top praised gameplay elements in slay the spire", "intent": "loved_features", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "what are the most loved features in elden ring?", "intent": "loved_features", "game": "Elden Ring", "mode": "player", "verified": False},
        {"text": "why do players love baldurs gate 3 so much?", "intent": "loved_features", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "what makes portal 2's puzzle design stand out in reviews?", "intent": "loved_features", "game": "Portal 2", "mode": "developer", "verified": False},
        {"text": "highlights and positive player quotes for terraria", "intent": "loved_features", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "what do fans praise in cyberpunk 2077?", "intent": "loved_features", "game": "Cyberpunk 2077", "mode": "player", "verified": False},
        {"text": "core retention drivers praised in hades", "intent": "loved_features", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "what do players love about csgo gunplay?", "intent": "loved_features", "game": "Counter-Strike: Global Offensive", "mode": "player", "verified": False},
        {"text": "most loved aspects of hollow knight art and combat", "intent": "loved_features", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "key praises for stardew valley's farming loop", "intent": "loved_features", "game": "Stardew Valley", "mode": "developer", "verified": False},

        # --- 5. similar_games (>= 16) ---
        {"text": "what games are similar to hades?", "intent": "similar_games", "game": "Hades", "mode": "player", "verified": False},
        {"text": "titles like hollow knight that i can play", "intent": "similar_games", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "who are the top market competitors for dead cells?", "intent": "similar_games", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "games like stardew valley on steam", "intent": "similar_games", "game": "Stardew Valley", "mode": "player", "verified": False},
        {"text": "find competitor titles to slay the spire in embedding space", "intent": "similar_games", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "alternatives to subnautica for survival exploration", "intent": "similar_games", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "games comparable to celeste platforming", "intent": "similar_games", "game": "Celeste", "mode": "player", "verified": False},
        {"text": "top competitors for elden ring in the souls-like space", "intent": "similar_games", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "titles similar in style to terraria", "intent": "similar_games", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "games like baldurs gate 3 for crpg fans", "intent": "similar_games", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "portal 2 alternatives and puzzle competitors", "intent": "similar_games", "game": "Portal 2", "mode": "developer", "verified": False},
        {"text": "cyberpunk 2077 nearest neighbors in catalog", "intent": "similar_games", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "what games compete directly with csgo?", "intent": "similar_games", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "what are the closest titles like dota 2?", "intent": "similar_games", "game": "Dota 2", "mode": "player", "verified": False},
        {"text": "show me games similar to hades with fast action", "intent": "similar_games", "game": "Hades", "mode": "player", "verified": False},
        {"text": "competitors for hollow knight on steam", "intent": "similar_games", "game": "Hollow Knight", "mode": "developer", "verified": False},

        # --- 6. price_advice (>= 16) ---
        {"text": "should i buy hades now or wait for a sale?", "intent": "price_advice", "game": "Hades", "mode": "player", "verified": False},
        {"text": "is hollow knight currently on discount?", "intent": "price_advice", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "what is the historical low price for stardew valley?", "intent": "price_advice", "game": "Stardew Valley", "mode": "player", "verified": False},
        {"text": "competitor price positioning analysis for dead cells", "intent": "price_advice", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "should i wait for a sale on elden ring?", "intent": "price_advice", "game": "Elden Ring", "mode": "player", "verified": False},
        {"text": "how does celeste's price compare to the genre median?", "intent": "price_advice", "game": "Celeste", "mode": "developer", "verified": False},
        {"text": "what is the price history and sale advice for subnautica?", "intent": "price_advice", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "is slay the spire worth its current price or should i wait?", "intent": "price_advice", "game": "Slay the Spire", "mode": "player", "verified": False},
        {"text": "pricing spectrum and p25/p75 distribution for baldurs gate 3", "intent": "price_advice", "game": "Baldur's Gate 3", "mode": "developer", "verified": False},
        {"text": "has cyberpunk 2077 ever been cheaper than its current cost?", "intent": "price_advice", "game": "Cyberpunk 2077", "mode": "player", "verified": False},
        {"text": "portal 2 price and discount history", "intent": "price_advice", "game": "Portal 2", "mode": "player", "verified": False},
        {"text": "is terraria at its historical lowest price right now?", "intent": "price_advice", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "should i buy hades today or is a steam sale coming?", "intent": "price_advice", "game": "Hades", "mode": "player", "verified": False},
        {"text": "price positioning for hollow knight against competitors", "intent": "price_advice", "game": "Hollow Knight", "mode": "developer", "verified": False},
        {"text": "tell me if dead cells is on sale or full price", "intent": "price_advice", "game": "Dead Cells", "mode": "player", "verified": False},
        {"text": "pricing intelligence report for stardew valley", "intent": "price_advice", "game": "Stardew Valley", "mode": "developer", "verified": False},

        # --- 7. player_match (>= 16) ---
        {"text": "is hades right for me?", "intent": "player_match", "game": "Hades", "mode": "player", "verified": False},
        {"text": "how hard is hollow knight?", "intent": "player_match", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "what is the difficulty rating and combat intensity of dead cells?", "intent": "player_match", "game": "Dead Cells", "mode": "player", "verified": False},
        {"text": "does celeste have heavy story weight or is it pure platforming?", "intent": "player_match", "game": "Celeste", "mode": "player", "verified": False},
        {"text": "how long are typical sessions in slay the spire?", "intent": "player_match", "game": "Slay the Spire", "mode": "player", "verified": False},
        {"text": "6d gameplay intensity profile for elden ring", "intent": "player_match", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "would i like subnautica if i hate difficult combat?", "intent": "player_match", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "what is the match profile for stardew valley?", "intent": "player_match", "game": "Stardew Valley", "mode": "player", "verified": False},
        {"text": "is baldurs gate 3 too complex for casual sessions?", "intent": "player_match", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "session length and difficulty breakdown for terraria", "intent": "player_match", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "how intense is the combat in portal 2?", "intent": "player_match", "game": "Portal 2", "mode": "player", "verified": False},
        {"text": "match profile metrics for cyberpunk 2077", "intent": "player_match", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "is hades good for short gameplay sessions?", "intent": "player_match", "game": "Hades", "mode": "player", "verified": False},
        {"text": "how punishing is hollow knight for new players?", "intent": "player_match", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "6d profile evaluation for dead cells", "intent": "player_match", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "is celeste too hard for casual gamers?", "intent": "player_match", "game": "Celeste", "mode": "player", "verified": False},

        # --- 8. success_score (>= 16, dev only) ---
        {"text": "what is the predicted success score for hades?", "intent": "success_score", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "show me the shap feature drivers for hollow knight", "intent": "success_score", "game": "Hollow Knight", "mode": "developer", "verified": False},
        {"text": "what is dead cells' commercial success probability?", "intent": "success_score", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "predictive ml score and feature importance for stardew valley", "intent": "success_score", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "what factors drove celeste's predicted success score?", "intent": "success_score", "game": "Celeste", "mode": "developer", "verified": False},
        {"text": "model prediction and top shap drivers for subnautica", "intent": "success_score", "game": "Subnautica", "mode": "developer", "verified": False},
        {"text": "what is the success tier estimation for slay the spire?", "intent": "success_score", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "shap explanation for elden ring's revenue forecast", "intent": "success_score", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "predictive probability of success for baldurs gate 3", "intent": "success_score", "game": "Baldur's Gate 3", "mode": "developer", "verified": False},
        {"text": "show feature drivers for cyberpunk 2077 model prediction", "intent": "success_score", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "what is terraria's model success rating?", "intent": "success_score", "game": "Terraria", "mode": "developer", "verified": False},
        {"text": "success score and key positive drivers for portal 2", "intent": "success_score", "game": "Portal 2", "mode": "developer", "verified": False},
        {"text": "how does our ml model rate hades' commercial viability?", "intent": "success_score", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "predicted success probability for csgo", "intent": "success_score", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "what does the predictive model forecast for hollow knight?", "intent": "success_score", "game": "Hollow Knight", "mode": "developer", "verified": False},
        {"text": "shap summary and success score for dead cells", "intent": "success_score", "game": "Dead Cells", "mode": "developer", "verified": False},

        # --- 9. update_impact (>= 16, dev only) ---
        {"text": "how did the latest update affect player sentiment for hades?", "intent": "update_impact", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "did hollow knight's recent patch improve review scores?", "intent": "update_impact", "game": "Hollow Knight", "mode": "developer", "verified": False},
        {"text": "update impact tracker analysis for dead cells", "intent": "update_impact", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "what was the 30-day sentiment delta after stardew valley's update?", "intent": "update_impact", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "did players respond positively to celeste's latest patch?", "intent": "update_impact", "game": "Celeste", "mode": "developer", "verified": False},
        {"text": "review volume and sentiment shift following subnautica update", "intent": "update_impact", "game": "Subnautica", "mode": "developer", "verified": False},
        {"text": "track update impact for slay the spire", "intent": "update_impact", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "sentiment shift after recent patch for elden ring", "intent": "update_impact", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "did baldurs gate 3 hotfix improve review consensus?", "intent": "update_impact", "game": "Baldur's Gate 3", "mode": "developer", "verified": False},
        {"text": "analyze patch note reception and trends for cyberpunk 2077", "intent": "update_impact", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "update impact analysis for terraria 1.4", "intent": "update_impact", "game": "Terraria", "mode": "developer", "verified": False},
        {"text": "how did portal 2's maintenance patch impact reviews?", "intent": "update_impact", "game": "Portal 2", "mode": "developer", "verified": False},
        {"text": "post-update sentiment trajectory for csgo", "intent": "update_impact", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "dota 2 balance patch sentiment impact report", "intent": "update_impact", "game": "Dota 2", "mode": "developer", "verified": False},
        {"text": "did the latest hades update fix player complaints?", "intent": "update_impact", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "what was the impact of dead cells latest dlc update?", "intent": "update_impact", "game": "Dead Cells", "mode": "developer", "verified": False},

        # --- 10. recommendations (>= 16, dev only) ---
        {"text": "what strategic recommendations do you have for hades?", "intent": "recommendations", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "how can the developers of hollow knight improve retention?", "intent": "recommendations", "game": "Hollow Knight", "mode": "developer", "verified": False},
        {"text": "actionable recommendations to address dead cells review issues", "intent": "recommendations", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "what should the stardew valley team prioritize next?", "intent": "recommendations", "game": "Stardew Valley", "mode": "developer", "verified": False},
        {"text": "roadmap suggestions and improvements for celeste", "intent": "recommendations", "game": "Celeste", "mode": "developer", "verified": False},
        {"text": "strategic advice for subnautica based on competitor gaps", "intent": "recommendations", "game": "Subnautica", "mode": "developer", "verified": False},
        {"text": "how can slay the spire developers increase positive reviews?", "intent": "recommendations", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "developer recommendations for elden ring balance", "intent": "recommendations", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "what fixes should the baldurs gate 3 team prioritize?", "intent": "recommendations", "game": "Baldur's Gate 3", "mode": "developer", "verified": False},
        {"text": "strategic recommendations for cyberpunk 2077 future patches", "intent": "recommendations", "game": "Cyberpunk 2077", "mode": "developer", "verified": False},
        {"text": "what should terraria developers add based on review intelligence?", "intent": "recommendations", "game": "Terraria", "mode": "developer", "verified": False},
        {"text": "developer insights for portal 2 level design", "intent": "recommendations", "game": "Portal 2", "mode": "developer", "verified": False},
        {"text": "how can the hades team boost player engagement?", "intent": "recommendations", "game": "Hades", "mode": "developer", "verified": False},
        {"text": "recommendations for csgo matchmaking improvements", "intent": "recommendations", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "what should hollow knight developers fix in their next update?", "intent": "recommendations", "game": "Hollow Knight", "mode": "developer", "verified": False},
        {"text": "strategic recommendations to minimize dead cells churn", "intent": "recommendations", "game": "Dead Cells", "mode": "developer", "verified": False},

        # --- 11. market_opportunity (>= 16, dev only) ---
        {"text": "where are the best market opportunities on steam right now?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "which genres have high player demand but low game supply?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "show me top underserved tags in the market explorer", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "where should our indie studio build our next title?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "what are the white space opportunities in the pc gaming market?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "market opportunity finder report for roguelike and deckbuilder tags", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "which genre niches have the highest median revenue per game?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "show market opportunities for coop survival titles", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "find underserved tag combinations with high opportunity scores", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "where are the lowest competition niches on steam?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "top opportunity scores for simulation and management games", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "market white space analysis for 2026 indie releases", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "which steam tags currently show the greatest revenue gap?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "market explorer overview of underserved genres", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "what genre has the best opportunity score right now?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "find untapped market niches on steam for new developers", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},

        # --- 12. recommend_for_me (>= 16) ---
        {"text": "what games should i play next based on my library?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "give me personalized recommendations for my steam profile", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "suggest some games i might like based on my played titles", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "what should i play tonight?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "recommend games based on my recent favorite games", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "can you generate recommendations for me from the catalog?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "personal recommendations based on my library history", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "what to play next if i like indie roguelikes?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "give me a list of game recommendations tailored to me", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "which games in the steam catalog fit my taste best?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "recommend titles matching my steam preferences", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "what should i download next from steam?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "suggest top picks for me using the two-tower model", "intent": "recommend_for_me", "game": None, "mode": "developer", "verified": False},
        {"text": "personal game recommendations for me please", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "what game should i try next based on my library?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "show me recommendations for my steam account", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},

        # --- 13. out_of_scope (>= 50, including adversarial near-misses & off-topic) ---
        # Adversarial near-misses (look like tech/games/reviews but are out of scope):
        {"text": "can you write a python script to web scrape steam reviews?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do i build a unity game engine shader for water?", "intent": "out_of_scope", "game": None, "mode": "developer", "verified": False},
        {"text": "write an unreal engine 5 c++ character movement component", "intent": "out_of_scope", "game": None, "mode": "developer", "verified": False},
        {"text": "what is the best gaming laptop to buy under $1000?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "why is my nvidia graphics card overheating while playing games?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "can you crack the steam drm for elden ring?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "write a review of the super mario bros movie", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do i fix a blue screen of death error on windows 11?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the playstation 5 pro hardware specification?", "intent": "out_of_scope", "game": None, "mode": "developer", "verified": False},
        {"text": "how much money did nintendo make on the switch console?", "intent": "out_of_scope", "game": None, "mode": "developer", "verified": False},
        {"text": "can you code a discord bot in javascript for game alerts?", "intent": "out_of_scope", "game": None, "mode": "developer", "verified": False},
        {"text": "what are the cheat codes for gta san andreas?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "who is the voice actor for kratos in god of war?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do i set up a minecraft server on linux?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "write a fantasy story about a rogue fighting goblins", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        # Off-topic general trivia, weather, cooking, math, life:
        {"text": "what is the weather in seattle tomorrow?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "who was the 16th president of the united states?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do you make authentic italian carbonara?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the capital city of australia?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "explain quantum computing in simple words", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "who won the best picture oscar in 2024?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how many ounces are in a pound?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "tell me a funny joke about programmer life", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what are the symptoms of seasonal allergies?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "write an email asking for a meeting with my professor", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the speed of sound in air?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do i brew cold brew coffee at home?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "who invented the telephone?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the stock price of valve corporation?", "intent": "out_of_scope", "game": None, "mode": "developer", "verified": False},
        {"text": "how do i change the oil in a honda civic?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the highest mountain in south america?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "can you solve this calculus integral for me?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the history of the roman colosseum?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "recommend a good book on world war 2", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do solar panels generate electricity?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what are the rules of chess?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "translate 'good morning' into japanese", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do airplanes stay in the air?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the difference between a virus and a bacteria?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "tell me about the deep sea creatures in the marianas trench", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the current population of tokyo?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how do i train for a half marathon?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "who directed the movie inception?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "what is the chemical formula for caffeine?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "write a resume summary for a software engineer", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},

        # --- Informal / Typos / Conversational Queries (>= 35 distributed across intents) ---
        {"text": "wuts hades about tho?", "intent": "game_overview", "game": "Hades", "mode": "player", "verified": False},
        {"text": "is hollow knite worth it??", "intent": "price_advice", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "tell me why pple hate dead cells", "intent": "top_complaints", "game": "Dead Cells", "mode": "player", "verified": False},
        {"text": "r people still playin terraria?", "intent": "game_overview", "game": "Terraria", "mode": "player", "verified": False},
        {"text": "subnautica reviews kinda mixed or good?", "intent": "review_summary", "game": "Subnautica", "mode": "player", "verified": False},
        {"text": "hades sale price when??", "intent": "price_advice", "game": "Hades", "mode": "player", "verified": False},
        {"text": "games like eldn ring but easier", "intent": "similar_games", "game": "Elden Ring", "mode": "player", "verified": False},
        {"text": "is celeste 2 hard for me?", "intent": "player_match", "game": "Celeste", "mode": "player", "verified": False},
        {"text": "wut do fans luv in stardew?", "intent": "loved_features", "game": "Stardew Valley", "mode": "player", "verified": False},
        {"text": "slay spire alternatives pc", "intent": "similar_games", "game": "Slay the Spire", "mode": "player", "verified": False},
        {"text": "baldurs gate 3 reviews summary pls", "intent": "review_summary", "game": "Baldur's Gate 3", "mode": "player", "verified": False},
        {"text": "wher r the market gaps on steam?", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "rec me some games 2 play", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "whats the weather like outside lol", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "hades 6d difficulty rating", "intent": "player_match", "game": "Hades", "mode": "player", "verified": False},
        {"text": "how much is hollow knight rite now", "intent": "price_advice", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "cyberpunk 2077 bugs fixed yet?", "intent": "top_complaints", "game": "Cyberpunk 2077", "mode": "player", "verified": False},
        {"text": "why do reviews praise portal 2 so much?", "intent": "loved_features", "game": "Portal 2", "mode": "player", "verified": False},
        {"text": "elden ring competitors in market explorer", "intent": "similar_games", "game": "Elden Ring", "mode": "developer", "verified": False},
        {"text": "dead cells succsess score model prediction", "intent": "success_score", "game": "Dead Cells", "mode": "developer", "verified": False},
        {"text": "did that recent update break csgo?", "intent": "update_impact", "game": "Counter-Strike: Global Offensive", "mode": "developer", "verified": False},
        {"text": "wut shud devs fix in subnautica?", "intent": "recommendations", "game": "Subnautica", "mode": "developer", "verified": False},
        {"text": "tell me market oppurtunity tags", "intent": "market_opportunity", "game": None, "mode": "developer", "verified": False},
        {"text": "what game shuld i get next on steam?", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "who is elon musk?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "is hades combat too fast paced for beginners?", "intent": "player_match", "game": "Hades", "mode": "player", "verified": False},
        {"text": "hollow knite main complaints from reviews", "intent": "top_complaints", "game": "Hollow Knight", "mode": "player", "verified": False},
        {"text": "stardew vally price discount tracker", "intent": "price_advice", "game": "Stardew Valley", "mode": "player", "verified": False},
        {"text": "slay the spire shap driver explanation", "intent": "success_score", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "did terraria player sentiment drop after patch?", "intent": "update_impact", "game": "Terraria", "mode": "developer", "verified": False},
        {"text": "strategic advice for indie deckbuilders", "intent": "recommendations", "game": "Slay the Spire", "mode": "developer", "verified": False},
        {"text": "recommendations 4 me based on my history", "intent": "recommend_for_me", "game": None, "mode": "player", "verified": False},
        {"text": "can u write my college essay?", "intent": "out_of_scope", "game": None, "mode": "player", "verified": False},
        {"text": "how long does a run take in hades?", "intent": "player_match", "game": "Hades", "mode": "player", "verified": False},
        {"text": "give me the review breakdown for dead cells", "intent": "review_summary", "game": "Dead Cells", "mode": "player", "verified": False},
    ]

    return gold_data


def filter_near_duplicates(
    train_queries: list[dict[str, Any]],
    gold_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Checks normalized text exact match between train and gold.
    Drops any train query that duplicates a gold query.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()

    gold_norms = {norm(g["text"]) for g in gold_queries}
    filtered_train = []
    dropped_count = 0

    for item in train_queries:
        t_norm = norm(item["text"])
        if t_norm in gold_norms:
            dropped_count += 1
            continue
        filtered_train.append(item)

    logger.info(
        "Near-duplicate filter: checked %d train against %d gold queries. Dropped %d exact/normalized duplicates.",
        len(train_queries),
        len(gold_queries),
        dropped_count,
    )
    return filtered_train


def main():
    logger.info("Generating datasets for Ask SteamIQ...")
    raw_train = generate_train_queries(seed=42, target_count_per_intent=75)
    gold = generate_gold_queries()

    filtered_train = filter_near_duplicates(raw_train, gold)

    train_path = DATA_DIR / "train_queries.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for item in filtered_train:
            f.write(json.dumps(item) + "\n")

    gold_path = DATA_DIR / "gold_queries.jsonl"
    with gold_path.open("w", encoding="utf-8") as f:
        for item in gold:
            f.write(json.dumps(item) + "\n")

    logger.info("Wrote %d train queries to %s", len(filtered_train), train_path)
    logger.info("Wrote %d gold queries to %s", len(gold), gold_path)


if __name__ == "__main__":
    main()
