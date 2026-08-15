# SteamIQ — Core Shared Library

## The Golden Rule (ADR 0001, Decision 2)

> **No API handler computes an aggregate, runs a model, or scans `raw_*` or `feature_*` tables.**
> **It reads from `serving_*` or `mart_*` only.**
> **If the data isn't there yet, add or fix a job — never add a bigger query to the handler.**

### Golden Rule Addendum (Game Match Scoring Exception — Phase 5 / Roadmap v2 §3)

> **Game Match scoring (`POST /api/v1/games/{app_id}/match`):**
> Given a user-supplied preference vector ($\le 6$ intensity numbers, never persisted server-side), the endpoint computes an in-memory percentage match directly against the pre-materialized row in `mart_game_match_profile`.
> This is a documented, narrow exception to the Golden Rule: the handler performs a small fixed-size comparison ($\le 6$ numbers) against precomputed data in a single indexed mart row, not an aggregate, model run, or table scan.

Pin this on your monitor. It's the single rule that, if followed from Phase 1, prevents SteamIQ from repeating the mistake nearly every past project made.

---

## What lives in core/

| Module | Purpose |
|---|---|
| `app.py` | `create_app()` — FastAPI factory, CORS, middleware, router registration |
| `config.py` | `Settings` (pydantic-settings) — single source of truth for all env vars |
| `response.py` | `ApiResponse[T]` envelope — wraps every response, success or error |
| `errors.py` | Exception hierarchy + global handlers — all errors become `ApiResponse` |
| `health.py` | `/health` router — DB connectivity + version |

## CORS policy (ADR 0001, Decision 4)

CORS is configured with `CORS_ALLOWED_ORIGINS` from env — a comma-separated list of explicit origins.

**Never use `["*"]`.** Even in development.

```
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

For staging/production, set this to the real origin(s).

## ApiResponse envelope (ADR 0001, Decision 5)

Every endpoint returns `ApiResponse[T]`. Never return a raw Pydantic model.

```python
# ✅ Correct
return ApiResponse.ok(data=game_schema, took_ms=timer.elapsed_ms)

# ❌ Wrong — raw model, no envelope
return game_schema
```

Error responses use `SteamIQException` subclasses:
```python
raise GameNotFoundError(f"No game found with app_id {app_id}")
# → {"success": false, "error": {"code": "game_not_found", "message": "..."}}
```

## Phase 5 mart transition (complete)

`api/games.py` now reads from `mart_game_overview` directly. The Phase 1 stopgap (`raw_games` read) has been removed in Phase 5 per ADR 0001, Decision 2. All handlers read exclusively from `mart_*` and `serving_*` tables.

