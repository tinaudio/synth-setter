"""Regression tests for the BLOCK-severity sub-gate in ``pre-pr-review-gate.sh``.

The gate already rejects a sentinel that still lists ``[comment-hygiene:*]``
findings; ``REVIEW_BLOCK_GATE`` widens that to ANY ``[<skill>:block]`` tag
(``synth-setter``, ``code-health``, ``ml-test``, ...). These tests invoke the
hook script directly with a HEAD-SHA sentinel (lag 0, passes ancestry) and
assert the exit contract.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_PATH = _REPO_ROOT / "agent" / "hooks" / "pre-pr-review-gate.sh"
_SENTINEL_PY = _REPO_ROOT / "agent" / "_shared" / "review_sentinel.py"


def _head_sha() -> str:
    """Return the repo's HEAD SHA (lag 0, so the gate's ancestry check passes).

    :returns: The 40-char HEAD commit SHA.
    """
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _head_sentinel(tmp_path: Path, body: str) -> Path:
    """Write ``body`` to a HEAD-SHA sentinel file (≥200 bytes) and return its path.

    :param tmp_path: Directory the synthetic review file lives in.
    :param body: Sentinel content; padded to clear the 200-byte stub guard.
    :returns: Path to the written sentinel.
    """
    filename = subprocess.run(  # noqa: S603
        ["python3", str(_SENTINEL_PY), "make", _head_sha()],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    review = tmp_path / filename
    review.write_text(body + ("padding\n" * 40))
    return review


def _run_gate(review: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the hook with a ``gh pr create`` command carrying ``REVIEW_FULL=<review>``.

    cwd is the repo root and ``REVIEW_MAX_LAG`` is widened so only the
    block-gate logic under test can change the outcome.

    :param review: Sentinel path passed as the trailing ``REVIEW_FULL=`` comment.
    :param env: Extra environment variables overlaid on the inherited environment.
    :returns: Completed subprocess result with captured stdout/stderr and exit code.
    """
    payload: dict[str, Any] = {
        "tool_input": {
            "command": f"gh pr create --title x --body y  # REVIEW_FULL={review}",
        },
    }
    overlay = {"REVIEW_MAX_LAG": "1000"}
    if env:
        overlay.update(env)
    return subprocess.run(  # noqa: S603
        ["bash", str(_HOOK_PATH)],  # noqa: S607
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
        env={**os.environ, **overlay},
    )


def test_gate_blocks_when_sentinel_lists_synth_setter_block_finding(tmp_path: Path) -> None:
    """A ``[synth-setter:block]`` finding trips the default block-gate (exit 2).

    :param tmp_path: pytest tmp dir for the synthetic sentinel.
    """
    review = _head_sentinel(
        tmp_path,
        "# repo-review-full-no-comments\n\n"
        "- **L42** — **[synth-setter:block]** mutable default argument.\n",
    )
    result = _run_gate(review, env={"REVIEW_COMMENT_GATE": "off"})
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "unresolved BLOCK finding" in result.stderr


def test_gate_off_mode_allows_sentinel_with_block_finding(tmp_path: Path) -> None:
    """``REVIEW_BLOCK_GATE=off`` is the documented escape hatch (exit 0).

    :param tmp_path: pytest tmp dir for the synthetic sentinel.
    """
    review = _head_sentinel(
        tmp_path,
        "# repo-review-full-no-comments\n\n"
        "- **L42** — **[synth-setter:block]** mutable default argument.\n",
    )
    result = _run_gate(review, env={"REVIEW_COMMENT_GATE": "off", "REVIEW_BLOCK_GATE": "off"})
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_gate_does_not_fire_on_warn_only_sentinel(tmp_path: Path) -> None:
    """A sentinel with only ``[python-style:warn]`` (no block tag) passes (exit 0).

    :param tmp_path: pytest tmp dir for the synthetic sentinel.
    """
    review = _head_sentinel(
        tmp_path,
        "# repo-review-full-no-comments\n\n"
        "- **L7** — **[python-style:warn]** prefer a comprehension here.\n",
    )
    result = _run_gate(review, env={"REVIEW_COMMENT_GATE": "off"})
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_block_gate_excludes_comment_hygiene_blocks(tmp_path: Path) -> None:
    """A ``[comment-hygiene:block]`` finding is the comment-gate's domain, not the block-gate's.

    With ``REVIEW_COMMENT_GATE=off`` and the default block-gate, the comment-hygiene
    block passes (exit 0) — proving the two gates don't overlap.

    :param tmp_path: pytest tmp dir for the synthetic sentinel.
    """
    review = _head_sentinel(
        tmp_path,
        "# repo-review-full-no-comments\n\n"
        "- **L9** — **[comment-hygiene:block]** comment inside a run: block-scalar.\n",
    )
    result = _run_gate(review, env={"REVIEW_COMMENT_GATE": "off"})
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_gate_allows_clean_pass_sentinel(tmp_path: Path) -> None:
    """A clean PASS sentinel (no bracketed findings) passes (exit 0).

    :param tmp_path: pytest tmp dir for the synthetic sentinel.
    """
    review = _head_sentinel(
        tmp_path,
        "# repo-review-full-no-comments\n\n## Summary\n\n0 BLOCK, 0 WARN. PASS.\n",
    )
    result = _run_gate(review)
    assert result.returncode == 0, (result.returncode, result.stderr)
