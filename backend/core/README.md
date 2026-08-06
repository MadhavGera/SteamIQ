# SteamIQ — Core Shared Library

## The Golden Rule (ADR 0001, Decision 2)

> **No API handler computes an aggregate, runs a model, or scans `raw_*` or `feature_*` tables.**
> **It reads from `serving_*` or `mart_*` only.**
> **If the data isn't there yet, add or fix a job — never add a bigger query to the handler.**

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

## Phase 1 stopgap (track and remove)

`api/games.py` reads from `raw_games` directly. Every read is marked:
```python
# TODO(Phase5): repoint to mart_game_overview once it exists
```

This stopgap is **bounded** — it goes away in Phase 5. Do not add new raw_* reads in handlers.
