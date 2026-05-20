"""Unit tests for ``agent/_shared/review_sentinel.py``.

The helper is shared between the ``/repo-review-full-no-comments`` skill
(which writes the file) and ``agent/hooks/pre-pr-review-gate.sh`` (which
parses the path supplied via ``REVIEW_FULL=<path>`` on ``gh pr create``).
Both sides must agree on the filename format, so the tests assert the
round-trip and the rejection behavior for malformed inputs.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = _REPO_ROOT / "agent" / "_shared" / "review_sentinel.py"


def _load_helper() -> ModuleType:
    """Import ``review_sentinel`` directly by path so the test doesn't need agent/ on sys.path.

    :returns: The imported module.
    :raises RuntimeError: If the helper file can't be located by ``importlib``.
    """
    spec = importlib.util.spec_from_file_location("review_sentinel", _HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not locate {_HELPER_PATH} via importlib")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper() -> ModuleType:
    """Load ``review_sentinel`` once per module via path-based import.

    :returns: The imported helper module.
    """
    return _load_helper()


@pytest.mark.parametrize(
    "sha",
    [
        "0" * 40,
        "a" * 40,
        "da0a8209b632757713d322797b51c640f6742cc1",
    ],
)
def test_make_review_filename_accepts_40_char_lowercase_hex(helper: ModuleType, sha: str) -> None:
    """Filename builder accepts any valid 40-char lowercase hex SHA.

    :param helper: The loaded helper module.
    :param sha: Candidate SHA value.
    """
    filename = helper.make_review_filename(sha)
    assert filename == f"repo-review-full-no-comments.{sha}.md"


@pytest.mark.parametrize(
    "bad_sha",
    [
        "",
        "abc",
        "Z" * 40,
        "A" * 40,
        "a" * 39,
        "a" * 41,
        "g" * 40,
        " " * 40,
    ],
)
def test_make_review_filename_rejects_invalid_sha(helper: ModuleType, bad_sha: str) -> None:
    """Filename builder raises ValueError for anything that isn't 40-char lowercase hex.

    :param helper: The loaded helper module.
    :param bad_sha: Invalid SHA value (wrong length, uppercase, non-hex chars, etc.).
    """
    with pytest.raises(ValueError, match="expected 40-char lowercase hex SHA"):
        helper.make_review_filename(bad_sha)


def test_parse_review_filename_round_trips_basename(helper: ModuleType) -> None:
    """``parse(make(sha))`` returns the original SHA.

    :param helper: The loaded helper module.
    """
    sha = "da0a8209b632757713d322797b51c640f6742cc1"
    assert helper.parse_review_filename(helper.make_review_filename(sha)) == sha


def test_parse_review_filename_accepts_full_path(helper: ModuleType) -> None:
    """Parsing strips directory components before matching.

    :param helper: The loaded helper module.
    """
    sha = "f" * 40
    full = f".agent-reviews/repo-review-full-no-comments.{sha}.md"
    assert helper.parse_review_filename(full) == sha


@pytest.mark.parametrize(
    "bad_filename",
    [
        "some-other-file.md",
        "repo-review-full-no-comments.short.md",
        "repo-review-full-no-comments.txt",
        f"repo-review-full-no-comments.{'Z' * 40}.md",
        f"repo-review-full-no-comments.{'A' * 40}.md",
        # Mid-string uppercase hex char — pins per-char hex-only matching
        # against an all-uppercase fixture that would survive a `re.IGNORECASE`
        # regression by themselves.
        f"repo-review-full-no-comments.{'a' * 20}A{'a' * 19}.md",
        f"repo-review-full.{'a' * 40}.md",
        f"repo-review-full-no-comments.{'a' * 40}",
        f"prefix-repo-review-full-no-comments.{'a' * 40}.md",
    ],
)
def test_parse_review_filename_rejects_non_matching_names(
    helper: ModuleType, bad_filename: str
) -> None:
    """Parsing returns None for anything that doesn't match the sentinel pattern.

    :param helper: The loaded helper module.
    :param bad_filename: A name that should NOT decode to a SHA.
    """
    assert helper.parse_review_filename(bad_filename) is None


def test_make_review_path_uses_default_directory(helper: ModuleType) -> None:
    """``make_review_path`` joins the helper's REVIEW_DIR by default.

    :param helper: The loaded helper module.
    """
    sha = "a" * 40
    assert helper.make_review_path(sha) == f".agent-reviews/repo-review-full-no-comments.{sha}.md"


def test_make_review_path_honors_custom_base_dir(helper: ModuleType) -> None:
    """Caller can override the base directory.

    :param helper: The loaded helper module.
    """
    sha = "a" * 40
    assert (
        helper.make_review_path(sha, base_dir="other/dir")
        == f"other/dir/repo-review-full-no-comments.{sha}.md"
    )


def test_cli_make_then_parse_round_trips() -> None:
    """The CLI surface used by the bash gate round-trips a SHA through make + parse."""
    sha = "1" * 40
    made = subprocess.run(  # noqa: S603
        ["python3", str(_HELPER_PATH), "make", sha],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert made == f"repo-review-full-no-comments.{sha}.md"

    parsed = subprocess.run(  # noqa: S603
        ["python3", str(_HELPER_PATH), "parse", made],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert parsed == sha


def test_cli_parse_exits_nonzero_on_bad_filename() -> None:
    """``parse`` exits non-zero when the input doesn't match the sentinel pattern.

    This is the failure path the bash gate trips on to emit BLOCKED.
    """
    result = subprocess.run(  # noqa: S603
        ["python3", str(_HELPER_PATH), "parse", "not-a-sentinel.md"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_cli_path_subcommand_emits_review_dir_filename() -> None:
    """``path <sha>`` prints ``<REVIEW_DIR>/<filename>`` — the form SKILL.md uses."""
    sha = "b" * 40
    result = subprocess.run(  # noqa: S603
        ["python3", str(_HELPER_PATH), "path", sha],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f".agent-reviews/repo-review-full-no-comments.{sha}.md"


@pytest.mark.parametrize(
    "argv_tail",
    [
        [],
        ["make"],
        ["bogus", "x"],
    ],
)
def test_cli_argv_validation_exits_2_with_usage(argv_tail: list[str]) -> None:
    """Missing-arg / unknown-subcommand cases exit 2 and print ``usage:`` on stderr.

    Pins behavior so a regression flipping ``< 3`` to ``<= 3`` or widening
    the subcommand set silently can't survive.

    :param argv_tail: Argv after the script path.
    """
    result = subprocess.run(  # noqa: S603
        ["python3", str(_HELPER_PATH), *argv_tail],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "usage:" in result.stderr


@pytest.mark.parametrize("subcommand", ["make", "path"])
def test_cli_make_and_path_exit_2_on_bad_sha(subcommand: str) -> None:
    """``make``/``path`` translate ValueError into exit 2 with the SHA message on stderr.

    :param subcommand: Which CLI verb to exercise (both wrap ValueError the same way).
    """
    result = subprocess.run(  # noqa: S603
        ["python3", str(_HELPER_PATH), subcommand, ""],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "expected 40-char lowercase hex SHA" in result.stderr
