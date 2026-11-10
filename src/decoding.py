"""Utilities for selecting and constraining generated tokens."""

import math
import numpy as np


def select_next_token(logits: list[float]) -> int:
    """Return the index of the highest logit."""
    if not logits:
        raise ValueError("Cannot select a token from empty logits.")

    return logits.index(max(logits))

def apply_token_constraints(
    logits: list[float],
    allowed_token_ids: set[int],
) -> list[float]:
    """Set the logits of disallowed tokens to negative infinity."""
    if not allowed_token_ids:
        raise ValueError("At least one token must be allowed.")

    if any(
        token_id < 0 or token_id >= len(logits)
        for token_id in allowed_token_ids
    ):
        raise ValueError("An allowed token ID is outside the logits range.")

    return [
        score if token_id in allowed_token_ids else -math.inf
        for token_id, score in enumerate(logits)
    ]


def select_constrained_token(
    logits: list[float],
    allowed_token_ids: set[int],
) -> int:
    """Select the highest-scoring token from the allowed tokens."""
    constrained_logits = apply_token_constraints(
        logits,
        allowed_token_ids,
    )
    return select_next_token(constrained_logits)
