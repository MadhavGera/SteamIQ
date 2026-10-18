# Human Holdout Query Set Specification

## Purpose
This directory contains the human-authored holdout evaluation dataset for Ask SteamIQ intent routing.
Unlike synthetic or agent-paraphrased queries, this evaluation set must be authored by human project owners or annotators to measure true out-of-distribution real-world generalization without synthetic templating artifacts.

## File Location
- File: `backend/assistant/data/holdout_queries_human.jsonl`
- Status: Scaffold awaiting human annotation ($\ge 100$ queries required; evaluated automatically once $\ge 50$ rows are present).

## Format Specification
Each row must be a valid single-line JSON object containing:
- `text`: (str, required) The exact user query verbatim as entered.
- `intent`: (str, required) The ground-truth intent class name. Must be one of the 13 canonical intents below.
- `game`: (str, optional) Name of the target game if referenced.
- `mode`: (str, optional) "player" or "developer".
- `source`: "human_holdout"
- `written_by`: "human"

### Canonical Intent Classes
1. `game_overview`
2. `review_summary`
3. `sentiment_breakdown`
4. `loved_features`
5. `similar_games`
6. `price_check`
7. `historical_low`
8. `specs_check`
9. `player_count`
10. `release_date`
11. `refund_policy`
12. `developer_info`
13. `out_of_scope`

## Evaluation Behavior in Pipeline
The training and evaluation pipeline (`backend/jobs/train_assistant.py`) automatically monitors `holdout_queries_human.jsonl`:
- If `len(rows) >= 50`: Evaluates router variants and temperature calibrator on human holdout, reporting accuracy, macro-F1, selective coverage at $\tau$, and confidence intervals as an independent third benchmark column alongside Gold Test and Synthetic Paraphrase Holdout.
- If `len(rows) < 50`: Explicitly records `"human holdout not yet provided (awaiting human authoring)"` in all serialized report artifacts and model cards, preventing any ungrounded claims.
