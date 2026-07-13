#!/usr/bin/env python
"""Shard the ``requires_vst`` test files into a GitHub Actions matrix.

Reads ``pytest --collect-only -q`` output (one node ID per line) on stdin,
reduces it to the unique set of test files, and round-robins them into
``--splits`` shards. Emits a matrix object (``{"include": [{"shard", "files"}]}``)
as a ``matrix=<json>`` line for the ``nightly-vst-sweep.yml`` discover job to
hand to ``fromJSON``. Exits non-zero on an empty selection — a 0-test collection
must fail the workflow, never ship a silently-green net.

Stdlib-only (argparse, no click) so the discover job runs it with the runner's
bare ``python3``, no venv or dependency install.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import TypeAlias

#: A GitHub Actions matrix object: ``{"include": [{"shard": n, "files": str}]}``.
Matrix: TypeAlias = dict[str, list[dict[str, object]]]


def collect_test_files(collected: str) -> list[str]:
    """Return the sorted unique test files in ``pytest --collect-only`` output.

    :param collected: raw ``--collect-only -q`` stdout (one node ID per line);
        non-node lines (summaries, warnings) are ignored.
    :returns: sorted unique ``tests/...`` file paths, node-ID suffix stripped.
    """
    return sorted(
        {
            line.split("::", 1)[0]
            for line in collected.splitlines()
            if line.startswith("tests/") and "::" in line
        }
    )


def build_matrix(files: Sequence[str], splits: int) -> Matrix:
    """Round-robin ``files`` into ``splits`` shards as a GHA matrix object.

    Shard files are space-joined and split back with ``read -ra`` in the
    workflow, so a path containing a space would corrupt the shard; test paths
    never contain spaces.

    :param files: test files to distribute (order preserved within a shard).
    :param splits: shard count; must be >= 1.
    :returns: ``{"include": [{"shard": n, "files": "a.py b.py"}, ...]}``; shards
        left empty (fewer files than splits) are omitted.
    :raises ValueError: if ``splits`` < 1; if ``files`` is empty (an empty matrix
        would ship a silently-green net); or if any path contains whitespace
        (the space-join / ``read -ra`` round-trip would misroute it).
    """
    if splits < 1:
        raise ValueError(f"splits must be >= 1, got {splits}")
    if not files:
        raise ValueError("no requires_vst tests collected — refusing an empty matrix")
    paths_with_whitespace = [path for path in files if any(char.isspace() for char in path)]
    if paths_with_whitespace:
        raise ValueError(
            f"test paths must not contain whitespace (breaks read -ra): {paths_with_whitespace}"
        )
    buckets: list[list[str]] = [[] for _ in range(splits)]
    for index, path in enumerate(files):
        buckets[index % splits].append(path)
    include: list[dict[str, object]] = [
        {"shard": i + 1, "files": " ".join(bucket)} for i, bucket in enumerate(buckets) if bucket
    ]
    return {"include": include}


def main(argv: Sequence[str] | None = None) -> int:
    """Build the shard matrix from stdin node IDs and emit ``matrix=<json>``.

    Writes the matrix line to ``$GITHUB_OUTPUT`` when set, else to stdout.

    :param argv: command-line arguments (defaults to ``sys.argv[1:]``).
    :returns: process exit code — 0 on success, 1 on an empty/invalid selection.
    """
    parser = argparse.ArgumentParser(description="Shard requires_vst tests into a GHA matrix.")
    parser.add_argument(
        "--splits", type=int, required=True, help="Number of shards to spread files across."
    )
    args = parser.parse_args(argv)

    files = collect_test_files(sys.stdin.read())
    try:
        matrix = build_matrix(files, args.splits)
    except ValueError as exc:
        sys.stderr.write(f"::error::{exc}\n")
        return 1

    payload = "matrix=" + json.dumps(matrix)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")
    sys.stderr.write(f"discovered {len(files)} files across {len(matrix['include'])} shards\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
