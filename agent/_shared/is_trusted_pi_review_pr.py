#!/usr/bin/env python3
"""Validate the pull-request trust boundary for manual Pi reviews."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

ALLOWED_PR_AUTHOR = "ktinubu"


def is_trusted_pull_request(pull_request: dict[str, Any], repository: str) -> bool:
    """Return whether a pull request is owner-authored and from the current repository.

    :param pull_request: GitHub pull-request API object.
    :param repository: Expected full repository name.
    :returns: Whether the pull request may reach review secrets.
    :raises ValueError: If required pull-request metadata is absent.
    """
    try:
        author = pull_request["user"]["login"]
        head_repository = pull_request["head"]["repo"]["full_name"]
    except (KeyError, TypeError) as error:
        raise ValueError("pull-request payload lacks author or head repository") from error
    return author == ALLOWED_PR_AUTHOR and head_repository == repository


def main() -> int:
    """Read pull-request metadata and return a silent trust decision.

    :returns: Zero for a trusted pull request, otherwise one.
    :raises ValueError: If the API payload is not a pull-request object.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()

    pull_request = json.load(sys.stdin)
    if not isinstance(pull_request, dict):
        raise ValueError("pull-request payload must be a JSON object")
    trusted = is_trusted_pull_request(pull_request, args.repository)
    return 0 if trusted else 1


if __name__ == "__main__":
    sys.exit(main())
