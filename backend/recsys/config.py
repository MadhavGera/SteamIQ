"""
Recommender System Configuration & Execution Profiles.

Defines:
  - RecsysConfig: Frozen dataclass parameterizing dataset sampling, filtering thresholds,
    and runtime device settings.
  - get_recsys_config(): Factory method constructing profile configuration from application settings.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecsysConfig:
    profile: str
    max_users: int
    max_items: int
    eval_users: int
    min_user_positives: int = 5
    min_item_positives: int = 20
    max_history: int = 50
    seed: int = 42
    num_threads: int = 4
    device: str = "cpu"
    raw_dir: Path | None = None
    data_dir: Path | None = None


def get_recsys_config(
    profile: str | None = None,
    settings_obj: Any = None,
) -> RecsysConfig:
    """
    Constructs a frozen RecsysConfig based on requested profile ("dev" or "full")
    and core application settings.

    Profiles:
      - dev: max_users=20000, max_items=3000, eval_users=5000
      - full: max_users=200000, max_items=10000, eval_users=50000
      - shared: min_user_positives=5, min_item_positives=20, max_history=50
    """
    if settings_obj is None:
        from core.config import settings
        settings_obj = settings

    target_profile = (profile or getattr(settings_obj, "recsys_profile", "dev")).strip().lower()

    if target_profile == "full":
        max_users = 200000
        max_items = 10000
        eval_users = 50000
        active_profile = "full"
    else:
        max_users = 20000
        max_items = 3000
        eval_users = 5000
        active_profile = "dev"

    raw_dir = getattr(settings_obj, "resolved_recsys_raw_dir", None)
    data_dir = getattr(settings_obj, "resolved_recsys_data_dir", None)
    seed = getattr(settings_obj, "recsys_seed", 42)
    num_threads = getattr(settings_obj, "recsys_num_threads", 4)
    device = getattr(settings_obj, "recsys_device", "cpu")

    return RecsysConfig(
        profile=active_profile,
        max_users=max_users,
        max_items=max_items,
        eval_users=eval_users,
        min_user_positives=5,
        min_item_positives=20,
        max_history=50,
        seed=seed,
        num_threads=num_threads,
        device=device,
        raw_dir=raw_dir,
        data_dir=data_dir,
    )
