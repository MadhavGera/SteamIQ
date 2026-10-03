"""
SteamIQ Recommender System Package.

Pure ML/Recsys modeling, configuration, schema validation, and utilities.
Rule 4: Zero FastAPI and zero database imports.
"""
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data_schema import (
    GAMES_COLUMNS,
    METADATA_KEYS,
    RECOMMENDATIONS_COLUMNS,
    USERS_COLUMNS,
    SchemaError,
    validate_header,
    validate_json_lines_keys,
)

__all__ = [
    "RecsysConfig",
    "get_recsys_config",
    "GAMES_COLUMNS",
    "METADATA_KEYS",
    "RECOMMENDATIONS_COLUMNS",
    "USERS_COLUMNS",
    "SchemaError",
    "validate_header",
    "validate_json_lines_keys",
]
