"""
Unit tests for recsys schema validation and memory-safe audit job.

Rule 9: Unit tests pass OFFLINE using tiny synthetic data (never needs real Kaggle files).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.audit_recsys_data import AuditRecsysDataJob
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


def test_validate_header_success(tmp_path: Path):
    """Test header validation succeeds on matching schema."""
    csv_file = tmp_path / "valid_games.csv"
    csv_file.write_text(",".join(GAMES_COLUMNS) + "\n10,Game One,2022-01-01,True,False,False,Positive,95,100,9.99,9.99,0,True\n")

    found_cols = validate_header(csv_file, GAMES_COLUMNS)
    assert found_cols == GAMES_COLUMNS


def test_validate_header_pp_id_failure(tmp_path: Path):
    """Test header validation fails with clear message on 'pp_id' typo."""
    bad_columns = ["pp_id"] + GAMES_COLUMNS[1:]
    csv_file = tmp_path / "bad_games.csv"
    csv_file.write_text(",".join(bad_columns) + "\n10,Game One,2022-01-01\n")

    with pytest.raises(SchemaError) as exc_info:
        validate_header(csv_file, GAMES_COLUMNS)

    err_msg = str(exc_info.value)
    assert "pp_id" in err_msg
    assert "app_id" in err_msg
    assert "Missing columns" in err_msg
    assert "Unexpected columns" in err_msg


def test_validate_header_missing_file():
    """Test FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        validate_header("non_existent_file.csv", GAMES_COLUMNS)


def test_validate_header_empty_file(tmp_path: Path):
    """Test SchemaError on empty file."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(SchemaError, match="empty or missing a header row"):
        validate_header(empty_file, GAMES_COLUMNS)


def test_validate_json_lines_keys(tmp_path: Path):
    """Test JSON Lines schema validation."""
    jsonl_file = tmp_path / "metadata.json"
    lines = [
        json.dumps({"app_id": 10, "description": "Fun game", "tags": ["RPG", "Action"]}),
        json.dumps({"app_id": 20, "description": "Another game", "tags": ["Strategy"]}),
    ]
    jsonl_file.write_text("\n".join(lines) + "\n")

    found = validate_json_lines_keys(jsonl_file, METADATA_KEYS)
    assert "app_id" in found
    assert "description" in found

    # Missing key test
    bad_jsonl = tmp_path / "bad_metadata.json"
    bad_jsonl.write_text(json.dumps({"app_id": 10, "description": "No tags key"}) + "\n")
    with pytest.raises(SchemaError, match="Missing keys"):
        validate_json_lines_keys(bad_jsonl, METADATA_KEYS)


def test_recsys_config_profiles():
    """Test dev and full profile configuration values."""
    dev_cfg = get_recsys_config("dev")
    assert isinstance(dev_cfg, RecsysConfig)
    assert dev_cfg.max_users == 20_000
    assert dev_cfg.max_items == 3_000
    assert dev_cfg.eval_users == 5_000
    assert dev_cfg.min_user_positives == 5
    assert dev_cfg.min_item_positives == 20
    assert dev_cfg.max_history == 50
    assert dev_cfg.seed == 42

    full_cfg = get_recsys_config("full")
    assert full_cfg.max_users == 200_000
    assert full_cfg.max_items == 10_000
    assert full_cfg.eval_users == 50_000


def test_audit_recsys_data_synthetic_20_rows(tmp_path: Path):
    """
    Test memory-safe audit job with a 20-row synthetic dataset.
    Verifies metric computations, deduplication, quantile bounds, and join coverage.
    """
    raw_dir = tmp_path / "raw_steam"
    data_dir = tmp_path / "recsys"
    raw_dir.mkdir()
    data_dir.mkdir()

    # 1. games.csv: 4 catalog games (10, 20, 30, 40)
    games_content = [
        ",".join(GAMES_COLUMNS),
        "10,Hollow Knight,2017-02-24,True,True,True,Overwhelmingly Positive,97,100000,14.99,14.99,0,True",
        "20,Celeste,2018-01-25,True,True,True,Overwhelmingly Positive,97,50000,19.99,19.99,0,True",
        "30,Dead Cells,2018-08-07,True,True,True,Overwhelmingly Positive,96,80000,24.99,24.99,0,True",
        "40,Hades,2020-09-17,True,True,False,Overwhelmingly Positive,98,120000,24.99,24.99,0,True",
    ]
    (raw_dir / "games.csv").write_text("\n".join(games_content) + "\n", encoding="utf-8")

    # 2. games_metadata.json (JSON Lines)
    metadata_lines = [
        json.dumps({"app_id": 10, "description": "Challenging 2D metroidvania", "tags": ["Metroidvania", "Souls-like", "2D"]}),
        json.dumps({"app_id": 20, "description": "Precision platformer", "tags": ["Platformer", "Pixel Graphics"]}),
        json.dumps({"app_id": 30, "description": "", "tags": ["Rogue-lite", "Action"]}),  # Empty description
        json.dumps({"app_id": 40, "description": "God-like rogue-like", "tags": ["Action Roguelike", "Mythology", "Indie", "Great Soundtrack"]}),
    ]
    (raw_dir / "games_metadata.json").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    # 3. users.csv: users 1001, 1002, 1003
    users_content = [
        ",".join(USERS_COLUMNS),
        "1001,15,4",
        "1002,30,8",
        "1003,10,2",
    ]
    (raw_dir / "users.csv").write_text("\n".join(users_content) + "\n", encoding="utf-8")

    # 4. recommendations.csv: 20 rows
    # Users: 1001, 1002, 1003, 1004 (1004 not in users.csv)
    # App IDs: 10, 20, 30, 40, 999 (999 not in games.csv)
    # User 1001: 6 positive reviews (all in catalog) -> has >= 5 positives
    # User 1002: 5 positive reviews (4 in catalog, 1 out of catalog app_id=999) -> 5 total pos, 4 catalog pos
    # User 1003: 4 positive reviews -> < 5 positives
    # User 1004: 1 negative review
    # Duplicate (1001, 10) occurs twice -> 1 duplicate pair
    # Duplicate (1002, 20) occurs twice -> 1 duplicate pair (Total duplicates = 2)
    # Total rows = 20
    # Positive rows = 16, Negative rows = 4
    # Playtime hours range: min = 1.5, max = 250.0
    # Years: 2021 (5 rows), 2022 (10 rows), 2023 (5 rows)
    recs_rows = [
        ",".join(RECOMMENDATIONS_COLUMNS),
        # User 1001 interactions (positives = 6, 1 duplicate pair on app 10)
        "10,5,1,2021-01-10,True,25.0,1001,1",
        "10,2,0,2021-02-15,True,30.0,1001,2",  # Duplicate (1001, 10)
        "20,10,0,2021-03-01,True,15.5,1001,3",
        "30,0,0,2021-04-12,True,50.0,1001,4",
        "40,1,0,2021-05-20,True,100.0,1001,5",
        "20,0,0,2022-01-05,True,12.0,1001,6",
        # User 1002 interactions (positives = 5: 4 in catalog, 1 out of catalog, 1 negative, 1 duplicate on app 20)
        "10,0,0,2022-02-10,True,10.0,1002,7",
        "20,0,0,2022-03-15,True,8.0,1002,8",
        "20,0,0,2022-04-20,True,8.5,1002,9",  # Duplicate (1002, 20)
        "30,4,1,2022-05-01,True,45.0,1002,10",
        "999,0,0,2022-06-11,True,1.5,1002,11",  # Out of catalog
        "40,0,0,2022-07-22,False,2.0,1002,12",  # Negative
        # User 1003 interactions (positives = 4, negatives = 2)
        "10,0,0,2022-08-30,True,80.0,1003,13",
        "20,0,0,2022-09-14,True,60.0,1003,14",
        "30,0,0,2022-10-01,True,250.0,1003,15",
        "40,0,0,2023-01-15,True,15.0,1003,16",
        "999,0,0,2023-02-20,False,3.0,1003,17",
        "999,0,0,2023-03-25,False,4.0,1003,18",
        # User 1004 interactions (negatives = 1, positives = 1)
        "10,0,0,2023-04-05,True,5.0,1004,19",
        "20,0,0,2023-05-18,False,6.0,1004,20",
    ]
    (raw_dir / "recommendations.csv").write_text("\n".join(recs_rows) + "\n", encoding="utf-8")

    # Run audit job with small chunk size to exercise chunk iteration
    job = AuditRecsysDataJob(
        raw_dir=raw_dir,
        data_dir=data_dir,
        chunk_size=7,
        dry_run=False,
    )
    results = job.run()

    # Assertions on returned metrics
    assert results["total_rows"] == 20
    assert results["unique_users"] == 4  # 1001, 1002, 1003, 1004
    assert results["unique_items"] == 5  # 10, 20, 30, 40, 999
    # Duplicate pairs:
    # (1001, 10), (1001, 20), (1002, 20), (1003, 999) -> 4 duplicates total
    assert results["duplicate_user_app_pairs"] == 4
    assert results["users_with_ge_5_positives"] == 2  # 1001 (6) and 1002 (5)
    assert results["users_with_ge_5_positives_in_catalog"] == 1  # 1001 (6), while 1002 only has 4 in catalog
    assert results["date_min"] == "2021-01-10"
    assert results["date_max"] == "2023-05-18"
    assert results["rows_per_year"] == {"2021": 5, "2022": 10, "2023": 5}
    assert results["hours_quantiles"]["p0"] == 1.5
    assert results["hours_quantiles"]["p100"] == 250.0

    # Join coverage assertions
    coverage = results["join_coverage"]
    # 17 interactions out of 20 have app_id in games.csv (3 rows have app_id=999)
    assert coverage["interaction_app_ids_in_games_share"] == 17 / 20
    # 3 out of 4 games have non-empty description
    assert coverage["games_with_non_empty_description_share"] == 3 / 4
    # 3 out of 4 unique users are found in users.csv (1001, 1002, 1003 in users.csv; 1004 is not)
    assert coverage["interaction_users_in_users_csv_share"] == 3 / 4

    # Verify audit.json output file exists and is valid JSON
    audit_file = data_dir / "audit.json"
    assert audit_file.exists()
    saved_data = json.loads(audit_file.read_text(encoding="utf-8"))
    assert saved_data["total_rows"] == 20
    assert saved_data["users_with_ge_5_positives"] == 2
