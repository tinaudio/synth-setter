"""Tests for scripts/ci/shard_vst_tests.py — the nightly VST shard matrix builder.

The module turns ``pytest --collect-only`` output into a GitHub Actions matrix.
The contract under test: which unique files are extracted, how they round-robin
into shards, and that a 0-test collection fails instead of emitting an empty
(silently-green) matrix.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.ci.shard_vst_tests import build_matrix, collect_test_files, main

_COLLECTED = "\n".join(
    [
        "tests/data/vst/test_a.py::test_one",
        "tests/data/vst/test_a.py::test_two",
        "tests/test_b.py::test_three",
        "tests/tools/test_c.py::test_four[case with spaces]",
        "63/3132 tests collected (3069 deselected) in 6.47s",
        "",
    ]
)


def test_collect_test_files_dedupes_node_ids_to_sorted_files() -> None:
    """Multiple node IDs in one file collapse to a single sorted path entry."""
    assert collect_test_files(_COLLECTED) == [
        "tests/data/vst/test_a.py",
        "tests/test_b.py",
        "tests/tools/test_c.py",
    ]


def test_collect_test_files_ignores_non_node_summary_lines() -> None:
    """Summary / blank lines without a ``tests/...::`` shape are dropped."""
    assert collect_test_files("63/3132 tests collected\n\nwarnings summary\n") == []


def test_build_matrix_round_robins_files_across_shards() -> None:
    """Files distribute round-robin so each shard gets an interleaved subset."""
    files = ["tests/a.py", "tests/b.py", "tests/c.py", "tests/d.py", "tests/e.py"]
    matrix = build_matrix(files, splits=2)
    assert matrix == {
        "include": [
            {"shard": 1, "files": "tests/a.py tests/c.py tests/e.py"},
            {"shard": 2, "files": "tests/b.py tests/d.py"},
        ]
    }


def test_build_matrix_omits_empty_shards_when_fewer_files_than_splits() -> None:
    """With fewer files than splits, trailing empty shards are dropped."""
    matrix = build_matrix(["tests/a.py"], splits=4)
    assert matrix == {"include": [{"shard": 1, "files": "tests/a.py"}]}


def test_build_matrix_raises_on_empty_file_list() -> None:
    """An empty selection raises rather than emitting a silently-green matrix."""
    with pytest.raises(ValueError, match="refusing an empty matrix"):
        build_matrix([], splits=4)


def test_build_matrix_raises_on_non_positive_splits() -> None:
    """``splits`` below 1 is rejected."""
    with pytest.raises(ValueError, match="splits must be >= 1"):
        build_matrix(["tests/a.py"], splits=0)


def test_main_appends_matrix_line_to_github_output_file() -> None:
    """With ``$GITHUB_OUTPUT`` set, ``main`` appends the ``matrix=`` line and returns 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "gh_output")
        with (
            patch.dict(os.environ, {"GITHUB_OUTPUT": output_path}),
            patch("sys.stdin", io.StringIO(_COLLECTED)),
        ):
            assert main(["--splits", "2"]) == 0
        line = Path(output_path).read_text(encoding="utf-8").strip()
    assert line.startswith("matrix=")
    matrix = json.loads(line[len("matrix=") :])
    files_across_shards = " ".join(cell["files"] for cell in matrix["include"])
    assert "tests/data/vst/test_a.py" in files_across_shards
    assert "tests/tools/test_c.py" in files_across_shards


def test_main_prints_matrix_to_stdout_when_github_output_unset() -> None:
    """Without ``$GITHUB_OUTPUT``, ``main`` prints the ``matrix=`` line to stdout."""
    env_without_output = {k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"}
    captured = io.StringIO()
    with (
        patch.dict(os.environ, env_without_output, clear=True),
        patch("sys.stdin", io.StringIO(_COLLECTED)),
        redirect_stdout(captured),
    ):
        assert main(["--splits", "2"]) == 0
    assert captured.getvalue().strip().startswith("matrix=")


def test_main_returns_one_when_no_tests_collected() -> None:
    """A collection with no ``requires_vst`` node IDs makes ``main`` return 1."""
    with patch("sys.stdin", io.StringIO("0 tests collected\n")):
        assert main(["--splits", "4"]) == 1
