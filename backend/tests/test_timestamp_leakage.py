"""
Unit and Integrity Test: Timestamp-Aware Masking & Future Leakage Prevention (Issue I7).

Verifies:
  1. Uninformative Validation Check: By construction, train interactions occur prior
     to the validation target (t_train < t_val). Therefore, before/after checks on
     the validation split produce identical fold-in histories.
  2. Test Sequence Future Leakage Prevention: When interactions occur after target
     timestamp t_target:
       (a) Validation positive is included in fold-in history IF AND ONLY IF t_val < t_test.
           If t_val >= t_test, it is strictly excluded (preventing future information leakage).
       (b) Negative interactions with t_neg < t_target are masked in the seen mask.
       (c) Negative interactions with t_neg >= t_target are NOT masked (preserving un-seen test state).
       (d) Target item is strictly never masked in seen mask.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from recsys.evaluate import build_user_evaluation_inputs


def create_synthetic_leakage_dataset():
    """
    Creates a synthetic dataset with controlled timestamps:
      User 1:
        - Train positive: item 10 at t=100
        - Validation positive: item 20 at t=200
        - Test target positive: item 30 at t=300
        - Train negative: item 40 at t=50
        - Future negative (held-out): item 50 at t=350 (AFTER test target!)
        - Past negative (held-out): item 60 at t=250 (BEFORE test target!)

      User 2 (Anomalous future validation sequence):
        - Train positive: item 11 at t=100
        - Validation positive: item 21 at t=400 (AFTER test target at t=300!)
        - Test target positive: item 31 at t=300
    """
    train_df = pd.DataFrame({
        "user_idx": [1, 2],
        "item_idx": [10, 11],
        "ts": [100.0, 100.0],
    })

    val_df = pd.DataFrame({
        "user_idx": [1, 2],
        "item_idx": [20, 21],
        "ts": [200.0, 400.0],  # User 2 val is in the future relative to test!
    })

    test_df = pd.DataFrame({
        "user_idx": [1, 2],
        "item_idx": [30, 31],
        "ts": [300.0, 300.0],
    })

    negatives_df = pd.DataFrame({
        "user_idx": [1, 1, 1],
        "item_idx": [40, 50, 60],
        "ts": [50.0, 350.0, 250.0],
        "split": ["train", "test", "test"],
    })

    data = SimpleNamespace(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=negatives_df,
        n_items=100,
    )
    return data


def test_validation_split_uninformative_by_construction():
    """
    Demonstrates Issue I7: on the validation split, all training items occurred
    strictly before the validation target timestamp.
    """
    data = create_synthetic_leakage_dataset()

    histories, targets, seen_masks, hist_lens = build_user_evaluation_inputs(
        data=data,
        eval_users=[1],
        split="val",
        max_history=50,
    )

    # For user 1 on val split:
    # History contains only train positives (item 10)
    assert list(histories[0]) == [10]
    assert targets[0] == 20
    # Seen mask contains train positive 10 and train negative 40
    assert 10 in seen_masks[0]
    assert 40 in seen_masks[0]
    # Val target 20 must NOT be in seen mask
    assert 20 not in seen_masks[0]


def test_test_split_prevents_future_validation_leakage():
    """
    Demonstrates leak prevention on test split:
      - User 1: val positive (t=200) < test target (t=300) -> INCLUDED in fold-in history.
      - User 2: val positive (t=400) > test target (t=300) -> EXCLUDED from fold-in history.
    """
    data = create_synthetic_leakage_dataset()

    histories, targets, seen_masks, hist_lens = build_user_evaluation_inputs(
        data=data,
        eval_users=[1, 2],
        split="test",
        max_history=50,
    )

    # User 1: t_val (200) < t_test (300) -> history has [10, 20]
    assert list(histories[0]) == [10, 20], "User 1 should fold in past validation positive"

    # User 2: t_val (400) > t_test (300) -> history must ONLY have [11], excluding future item 21!
    assert list(histories[1]) == [11], "User 2 must NOT leak future validation positive occurring after test target"
    assert 21 not in seen_masks[1], "Future item 21 must not be in seen mask"


def test_timestamp_aware_negative_masking():
    """
    Demonstrates timestamp-aware negative masking on test split:
      - Item 40: train negative (t=50 < 300) -> MASKED in seen
      - Item 60: held-out negative with t=250 < 300 -> MASKED in seen
      - Item 50: future negative with t=350 > 300 -> NOT MASKED in seen
      - Item 30: test target -> NEVER MASKED
    """
    data = create_synthetic_leakage_dataset()

    _, targets, seen_masks, _ = build_user_evaluation_inputs(
        data=data,
        eval_users=[1],
        split="test",
        max_history=50,
    )

    seen = seen_masks[0]

    assert 40 in seen, "Train negative must be masked"
    assert 60 in seen, "Past negative (t=250 < t_target=300) must be masked"
    assert 50 not in seen, "Future negative (t=350 > t_target=300) must NOT be masked"
    assert targets[0] == 30
    assert 30 not in seen, "Test target must NEVER be masked"
