"""Tests for token decoding constraints."""

import math

import pytest

from src.decoding import apply_token_constraints, select_next_token


def test_select_next_token_returns_highest_logit() -> None:
    """Select the token with the highest score."""
    assert select_next_token([0.1, 0.9, 0.2]) == 1


def test_apply_token_constraints_blocks_disallowed_tokens() -> None:
    """Set disallowed token scores to negative infinity."""
    result = apply_token_constraints(
        [0.1, 0.9, 0.2],
        {0, 2},
    )

    assert result[0] == 0.1
    assert result[1] == -math.inf
    assert result[2] == 0.2


def test_apply_token_constraints_rejects_empty_allowlist() -> None:
    """Reject an empty set of allowed tokens."""
    with pytest.raises(ValueError):
        apply_token_constraints([0.1, 0.9], set())


def test_apply_token_constraints_rejects_invalid_token_ids() -> None:
    """Reject token IDs outside the logits range."""
    with pytest.raises(ValueError):
        apply_token_constraints([0.1, 0.9], {2})