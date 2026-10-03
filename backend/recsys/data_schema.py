"""
Data Schemas and Header Validation for Kaggle Steam Datasets.

Defines:
  - Exact expected schema columns for games.csv, games_metadata.json, users.csv, recommendations.csv
  - SchemaError exception
  - validate_header() function to ensure exact schema compliance without auto-renaming.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

GAMES_COLUMNS: list[str] = [
    "app_id",
    "title",
    "date_release",
    "win",
    "mac",
    "linux",
    "rating",
    "positive_ratio",
    "user_reviews",
    "price_final",
    "price_original",
    "discount",
    "steam_deck",
]

METADATA_KEYS: list[str] = [
    "app_id",
    "description",
    "tags",
]

USERS_COLUMNS: list[str] = [
    "user_id",
    "products",
    "reviews",
]

RECOMMENDATIONS_COLUMNS: list[str] = [
    "app_id",
    "helpful",
    "funny",
    "date",
    "is_recommended",
    "hours",
    "user_id",
    "review_id",
]


class SchemaError(ValueError):
    """Raised when CSV or JSON file schema does not strictly match expected column layout."""
    pass


def validate_header(path: str | Path, expected: Sequence[str]) -> list[str]:
    """
    Validates that a CSV file's header strictly contains all expected columns and no unexpected columns.

    Args:
        path: Path to CSV file.
        expected: Expected column names in exact expected order or set.

    Returns:
        List of found column names.

    Raises:
        SchemaError: If missing or unexpected columns are found.
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header_row = next(reader, None)

    if header_row is None:
        raise SchemaError(f"File '{file_path.name}' is empty or missing a header row.")

    actual_cols = [c.strip() for c in header_row]

    expected_set = set(expected)
    actual_set = set(actual_cols)

    missing = [c for c in expected if c not in actual_set]
    unexpected = [c for c in actual_cols if c not in expected_set]

    if missing or unexpected:
        error_parts = [f"Schema validation failed for '{file_path.name}'."]
        if missing:
            error_parts.append(f"Missing columns: {missing}")
        if unexpected:
            error_parts.append(f"Unexpected columns: {unexpected}")
        error_parts.append(f"Expected: {list(expected)}, Found: {actual_cols}")
        raise SchemaError(" | ".join(error_parts))

    return actual_cols


def validate_json_lines_keys(path: str | Path, expected_keys: Sequence[str], sample_lines: int = 10) -> list[str]:
    """
    Validates that lines in a JSON-Lines file contain expected keys.

    Args:
        path: Path to JSONL file.
        expected_keys: Expected JSON keys.
        sample_lines: Number of lines to validate from start of file.

    Returns:
        List of found keys on first line.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    found_keys: list[str] = []

    with open(file_path, encoding="utf-8-sig", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"Invalid JSON Line at {file_path.name}:{line_num}: {exc}") from exc

            if not isinstance(record, dict):
                raise SchemaError(f"Expected JSON object at {file_path.name}:{line_num}, got {type(record).__name__}")

            actual_set = set(record.keys())
            missing = [k for k in expected_keys if k not in actual_set]
            if missing:
                raise SchemaError(
                    f"Schema validation failed for JSONL '{file_path.name}' line {line_num}. Missing keys: {missing}"
                )

            if not found_keys:
                found_keys = list(record.keys())

            if line_num >= sample_lines:
                break

    if not found_keys:
        raise SchemaError(f"File '{file_path.name}' is empty.")

    return found_keys
