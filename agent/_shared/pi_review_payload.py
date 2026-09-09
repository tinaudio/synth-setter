#!/usr/bin/env python3
"""Build deterministic review delivery payloads from final adjudications."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

if __package__:
    from agent._shared.pi_review_render import build_adjudicated_review
    from agent._shared.pi_review_routing import ReviewAdjudication
else:
    from pi_review_render import build_adjudicated_review
    from pi_review_routing import ReviewAdjudication


def _build_parser() -> argparse.ArgumentParser:
    """Build the payload CLI parser.

    :returns: Parser for adjudication and review metadata inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--review-body", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follow-up", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate adjudications and write the canonical delivery payload.

    :param argv: Optional CLI arguments.
    :returns: Process exit status.
    """
    args = _build_parser().parse_args(argv)
    adjudications = TypeAdapter(tuple[ReviewAdjudication, ...]).validate_json(
        args.adjudications.read_text()
    )
    for adjudication in adjudications:
        adjudication.verify_fingerprint()
    payload = build_adjudicated_review(
        pr_number=args.pr_number,
        repo=args.repo,
        review_body=args.review_body.read_text(),
        adjudications=adjudications,
    )
    if args.follow_up:
        payload = payload.model_copy(update={"event": "COMMENT"})
    args.output.write_text(payload.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
