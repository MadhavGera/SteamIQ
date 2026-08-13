# SteamIQ — Final Project Roadmap v2 (Player + Developer)

**Status:** Proposed amendment to `docs/steamiq-final-roadmap.md` (v1) and `docs/steamiq-0002-ui-ux-blueprint.md`
**Supersedes:** v1 roadmap phase sequence stays intact (Phase 0–7 unchanged in order). This document only changes *what* Phases 5 and 6 deliver, and amends the locked IA. Phases 0–4 are untouched — do not re-open them.

---

## 0. What changed and why

The player-facing expansion is a good addition **in principle** — one shared Game Intelligence Engine with two audiences on top of it, not two products. That's consistent with the architecture you already committed to in ADR 0001 (single Postgres, single embedding space, zone-prefixed tables everyone reads through `serving_*`/`mart_*`).

But the original expansion proposal designs it as if starting from scratch, which produces a second top-level IA (`Explore` vs `Developer Studio`) and several "new" pages that duplicate tabs your `0002-ui-ux-blueprint.md` already locked (pricing folded into Market, competitors already a tab, no separate Similar Games page). Implementing it as written would mean silently violating your own locked-IA golden rule instead of amending it.

**Decision:** integrate the player experience as a **mode toggle on the existing IA**, reusing existing tables wherever possible, and add exactly one new table for the one genuinely new feature (game-profile matching). No new phase is inserted; the work lands in Phase 5 (materialization/decision layer) and lightly touches Phase 6 (assistant).

---

## 1. Revised architecture (shared engine, two decision outputs)

```
                         DATA SOURCES
                              │
              ┌───────────────┼───────────────┐
           Steam API       Reviews          SteamSpy
              │               │                 │
          raw_games      raw_reviews    raw_player_snapshots
                                          raw_price_history
                              │
                              ▼
                    FEATURE PIPELINES (Phase 2–4)
                              │
       ┌──────────────────────┼───────────────────────┐
    feature_review_*      model_game_embeddings   feature_game_features
    (sentiment/topics/     serving_similar_games    feature_market_features
     complaints/loved)                               model_runs / serving_predictions
                              │
                              ▼
                   DECISION / MART LAYER (Phase 5)
                              │
     ┌───────────┬────────────┼────────────┬───────────────┐
mart_game_    mart_trends  mart_opportunity  mart_game_match  serving_recommendations
overview                    _scores          _profile  ← NEW
                              │
                              ▼
                   API (reads serving_*/mart_* only)
                              │
                ┌─────────────┴─────────────┐
           PLAYER MODE                  DEVELOPER MODE
        (same Game Intelligence      (same Game Intelligence page
         page, framing toggled)       + dev-only global pages)
                              │
                        Ask SteamIQ (Phase 6, user_mode-aware)
```

No second vector store, no second ingestion path, no second nav shell. The mode toggle changes *labels, framing, and which tabs render* — not which system answers the question.

---

## 2. IA amendment (proposed replacement for §1 of `0002-ui-ux-blueprint.md`)

This must be committed as an explicit amendment to that doc before frontend work starts — its own golden rule requires that, and it's currently marked "locked."

```
SteamIQ
│
├── Search / Landing
├── Dashboard
│
├── Game Intelligence  (per game, tabbed — shared by both modes)
│   ├── Overview
│   ├── Reviews              — Review Intelligence
│   ├── Player Activity      — Player Behaviour / activity
│   ├── Market               — Price + Success/Revenue Intelligence
│   │                           Player mode: "buy now or wait" framing
│   │                           Dev mode: competitor price positioning
│   ├── Competitors          — Similarity Engine
│   │                           Player mode label: "Similar Games"
│   │                           Dev mode label: "Competitors" (+ market presence column)
│   ├── Match                — NEW. Player mode only. "Is this for me?"
│   ├── Updates               — Dev mode only (Update Impact Tracker)
│   └── Recommendations       — Dev mode only (Recommendation Engine)
│
├── Compare Games
├── Market Explorer           — Dev mode only (Opportunity Finder + Trends)
├── Ask SteamIQ               — shared, responses adapt to mode
└── Settings                  (mode toggle lives here + as a persistent header control)
```

**New locked decisions this round:**
- Mode toggle (Player / Developer) is global and session-persisted. It never forks routes — `/game/[app_id]` is one page for both modes.
- `Match` tab is the only net-new tab. Everything else is an existing tab with conditional visibility and/or relabeled framing.
- `Updates`, `Recommendations`, and `Market Explorer` are hidden (not deleted, not separately routed) in Player mode.

**Rejected from the original proposal:** a second top-level nav (`Explore`/`Developer Studio` as separate trees), a standalone Price Intelligence page, a standalone Similar Games page. All folded into the structure above.

---

## 3. Phase deltas

Phases 0–4 are unchanged — do not reopen. Only Phase 5 and Phase 6 gain new deliverables.

### Phase 5 — Decision Intelligence (revised)

**Goal (unchanged):** turn stored intelligence into materialized, dashboard-ready insight — now audience-aware.

**Deliverables — existing (unchanged):**
- `mart_game_overview`, `mart_trends`, `mart_opportunity_scores`
- Opportunity Finder (weighted scoring, analytics-only)
- Pricing Intelligence: comparable-range output, no causal price-elasticity claims
- Update Impact Tracker
- Recommendation Engine (hybrid rules + model output)

**Deliverables — new:**
- **`mart_game_match_profile`** (new `mart_*` table): per-game intensity scores on a small fixed dimension set — difficulty, story weight, exploration, combat, multiplayer, session length. Derived from `feature_review_topics` + `raw_game_tags` via a weighted-scoring formula identical in spirit to Opportunity Finder — **not a new ML model, no new phase**. Populated by the same scheduled materialization job pattern as the other `mart_*` tables.
- **Game Match scoring**: given a user-supplied preference vector (a handful of checkbox values, never stored server-side unless the user opts to save it), compute a percentage match against `mart_game_match_profile`. This is a documented, narrow exception to the Golden Rule: the handler performs a small fixed-size comparison (≤6 numbers) against precomputed data, not an aggregate or a table scan. Flag it as such in `core/README.md` next to the Golden Rule so it isn't mistaken for a crack in the rule.
- **`serving_similar_games`**: add one additive column, `market_presence` (derived from `raw_player_snapshots` volume / review count), so the existing row serves both "Similar Games" (player) and "Competitors" (dev) without a second table.
- Pricing Intelligence gains player-facing copy ("current price is X% above historical low") reading the *same* `mart_*` output the dev-side comparable-range view uses — presentation only, one new ingestion requirement to verify: **confirm the price-history job runs on a recurring schedule**, not just the Phase 1 one-off snapshot, or "historical lowest" has nothing to compare against.

**Exit criteria (added):** Game Match returns a percentage score for any seeded game given a sample preference vector, computed entirely from `mart_game_match_profile`; the dashboard and game page still load entirely from `mart_*`/`serving_*` with zero live aggregation elsewhere on the page.

### Phase 6 — AI Assistant (revised)

**Deliverables — new:**
- Intent router gains a `user_mode` parameter (`player` / `developer`). Same deterministic rule set and `schema_metadata.json`, different answer templates — e.g. "should I buy this now" and "why is my sentiment declining" both resolve through the same structured path, just routed to different phrasing.
- No new retrieval path, no new RAG index. This is a templating change on an already-planned component.

**Exit criteria (added):** the example query set resolves correctly under both `user_mode` values without any change to the deterministic router's structured-call logic.

### Phases 0–4, 7 — unchanged
Everything already specified stands as written. In particular: no new vector store, no new job runner, no new ingestion source, no change to the MLflow/model_runs/registry work in Phase 4.

---

## 4. Feature ownership (merged)

| Feature | Player | Developer | Zone / Table | Phase |
|---|:---:|:---:|---|---|
| Game Search | ✅ | ✅ | `raw_games` → `mart_game_overview` | 1 / 5 |
| Game Overview | ✅ | ✅ | `mart_game_overview` | 5 |
| Community Sentiment / Topics / Loved / Complaints / Summary | ✅ | ✅ | `feature_review_*` | 2 |
| Similar Games / Competitors | ✅ | ✅ | `serving_similar_games` (+`market_presence`) | 3 |
| Price History / Historical Low | ✅ | ✅ | `raw_price_history` → `mart_*` | 1 / 5 |
| Player Activity | ✅ | ✅ | `raw_player_snapshots` | 1 |
| **Game Match ("is this for me")** | ✅ | — | **`mart_game_match_profile`** (new) | **5** |
| Success / Retention / Playtime / Churn Prediction | — | ✅ | `feature_game_features`, `model_runs`, `serving_predictions` | 4 |
| Market Opportunity Finder | — | ✅ | `mart_opportunity_scores` | 5 |
| Update Impact Tracker | — | ✅ | `mart_trends` / job output | 5 |
| Recommendation Engine | — | ✅ | `serving_recommendations` | 5 |
| Ask SteamIQ | ✅ | ✅ | `schema_metadata.json` + `user_mode` routing | 6 |

---

## 5. Revised scope map

| Phase | MVP | Strong Final Version | Stretch |
|---|---|---|---|
| 0 — Architecture & ADR | ✅ (+ this IA amendment recorded) | | |
| 1 — Foundation | ✅ | | |
| 2 — NLP Core | ✅ | | |
| 3 — Similarity + Competitors | ✅ | `market_presence` column added | |
| 4 — Predictive ML | ✅ | Revenue category, activity forecast | |
| 5 — Decision Intelligence | | ✅ Opportunity Finder, Update Impact, Recommendation Engine, SHAP, **Game Match** | Pricing simulator refinements |
| 6 — AI Assistant | | ✅ RAG assistant, **mode-aware templating** | |
| 7 — Productionization | | ✅ Redis, MLflow, registry, CI/CD, golden-output tests | Cloud deployment, portfolio dashboard, auto-retraining, What-If Simulator |

Nothing moves from Stretch into core because of this expansion — Game Match is cheap enough to sit in the existing Final scope for Phase 5 without displacing anything.

---

## 6. One rule to keep visible (unchanged, one addendum)

> No API handler computes an aggregate, runs a model, or scans `raw_*`/`feature_*` tables. It reads from `serving_*` or `mart_*` only. If the data isn't there yet, add or fix a job — never add a bigger query to the handler.
>
> **Addendum:** Game Match's request-time comparison against a user-supplied preference vector is the one explicit, documented exception — a fixed-size (≤6 value) comparison against precomputed `mart_*` data, not an aggregate. Any other "just this once" handler computation is still a violation.