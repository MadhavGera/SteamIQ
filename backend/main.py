"""
SteamIQ FastAPI application entrypoint.

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or via Makefile:
    make dev
"""
from __future__ import annotations

import logging
from pathlib import Path

# ─── Load .env before anything else reads env vars ────────────────────────────
# Uses absolute path — never os.getcwd() (ADR 0001, Decision 8)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env_file, override=False)
    except ImportError:
        pass  # pydantic-settings will still read .env directly

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ─── App ──────────────────────────────────────────────────────────────────────
from core.app import create_app  # noqa: E402

app = create_app()
