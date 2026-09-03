#!/usr/bin/env python3
"""Decide whether a pull request may receive another automatic Pi review."""

from __future__ import annotations

import json
import sys
from typing import Any

AUTOMATIC_REVIEW_AUTHOR = "github-actions[bot]"
MAX_AUTOMATIC_REVIEWS = 2


def should_run_automatic_review(reviews: list[dict[str, Any]]) -> bool:
    """Return whether fewer than two automatic Pi reviews already exist.

    :param reviews: GitHub pull-request review objects.
    :returns: Whether the workflow may start another automatic review.
    :raises ValueError: If a review lacks the expected author login.
    """
    automatic_review_count = 0
    for review in reviews:
        try:
            author = review["user"]["login"]
        except (KeyError, TypeError) as error:
            raise ValueError("review payload lacks user.login") from error
        if author == AUTOMATIC_REVIEW_AUTHOR:
            automatic_review_count += 1
    return automatic_review_count < MAX_AUTOMATIC_REVIEWS


def main() -> None:
    """Read GitHub reviews from stdin and print a workflow-compatible boolean.

    :raises ValueError: If the API payload is not a review list.
    """
    reviews = json.load(sys.stdin)
    if not isinstance(reviews, list):
        raise ValueError("review payload must be a JSON list")
    sys.stdout.write(f"{str(should_run_automatic_review(reviews)).lower()}\n")


if __name__ == "__main__":
    main()
