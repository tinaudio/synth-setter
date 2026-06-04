"""Tests for the PR-title sub-gate in ``pre-pr-review-gate.sh``.

The gate extracts the inline ``--title`` from a ``gh pr create`` command and
lints it with ``uvx --from gitlint-core ... gitlint``. These tests stub ``uvx``
with a PATH shim whose exit code and output are fixed per-case, so they exercise
the gate's reaction contract (block / warn / fail-open / skip) without invoking
real gitlint or the network.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_PATH = _REPO_ROOT / "agent" / "hooks" / "pre-pr-review-gate.sh"
_SENTINEL_PY = _REPO_ROOT / "agent" / "_shared" / "review_sentinel.py"

# A gitlint verdict line (`<lineno>: <RULEID> ...`); the gate keys on this shape
# to tell a rejection from a uvx launch failure (which shares exit code 2).
_VIOLATION = "1: CT1 Title does not start with one of feat, fix, ...\n"
_TWO_VIOLATIONS = '1: T1 Title exceeds max length (101>72): "x"\n1: CT1 Title does not start ...\n'
# uv's own failure text — no verdict line, so the gate must fail open.
_UV_LAUNCH_ERROR = "error: failed to create directory `/x/uv/tools`: Permission denied\n"


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


def _clean_sentinel(tmp_path: Path) -> Path:
    """Write a finding-free HEAD-SHA sentinel (≥200 bytes) so the review gate passes.

    :param tmp_path: Directory the synthetic review file lives in.
    :returns: Path to the written sentinel.
    """
    filename = subprocess.run(  # noqa: S603
        ["python3", str(_SENTINEL_PY), "make", _head_sha()],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    review = tmp_path / filename
    review.write_text(
        "# repo-review-full-no-comments\n\n0 BLOCK, 0 WARN. PASS.\n" + ("pad\n" * 40)
    )
    return review


def _fake_uvx(bin_dir: Path, exit_code: int, stdin_log: Path, output: str) -> None:
    """Install a stub ``uvx`` on PATH that records stdin, emits ``output``, and exits.

    The recorded stdin lets a test assert the extracted title was piped through
    verbatim; ``output`` stands in for gitlint's verdict lines (or uv's launch
    error) and ``exit_code`` for its status.

    :param bin_dir: Directory prepended to PATH; the stub is written here.
    :param exit_code: Exit status the stub returns.
    :param stdin_log: File the stub writes its stdin to.
    :param output: Text the stub writes to stderr (gitlint's verdict or a uv error).
    """
    uvx = bin_dir / "uvx"
    uvx.write_text(
        "#!/usr/bin/env bash\n"
        f"cat > {shlex.quote(str(stdin_log))}\n"
        f"printf '%s' {shlex.quote(output)} >&2\n"
        f"exit {exit_code}\n"
    )
    uvx.chmod(0o755)


def _run_title_gate(
    tmp_path: Path,
    command: str,
    *,
    uvx_exit: int = 0,
    output: str = "",
    gate: str = "block",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the hook against ``command`` with a stubbed ``uvx`` and a clean sentinel.

    :param tmp_path: pytest tmp dir for the stub, its stdin log, and the sentinel.
    :param command: The ``gh pr create`` command the hook receives (no sentinel
        suffix — one is appended here so the review gate passes on a good title).
    :param uvx_exit: Exit code the stubbed ``uvx`` returns.
    :param output: Stderr text the stubbed ``uvx`` emits.
    :param gate: Value for ``PR_TITLE_GATE`` (block/warn/off).
    :returns: The completed process and the path the stub logs its stdin to.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stdin_log = tmp_path / "uvx_stdin.txt"
    _fake_uvx(bin_dir, uvx_exit, stdin_log, output)
    review = _clean_sentinel(tmp_path)
    payload: dict[str, Any] = {
        "tool_input": {"command": f"{command}  # REVIEW_FULL={review}"},
    }
    overlay = {
        "REVIEW_MAX_LAG": "1000",
        "PR_TITLE_GATE": gate,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(  # noqa: S603
        ["bash", str(_HOOK_PATH)],  # noqa: S607
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
        env={**os.environ, **overlay},
    )
    return result, stdin_log


def test_title_gate_blocks_non_conventional_title_exit_2(tmp_path: Path) -> None:
    """A gitlint verdict on the title (one violation) blocks PR creation with exit 2.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path, 'gh pr create --title "made it better" --body b', uvx_exit=1, output=_VIOLATION
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "PR title is not a conventional commit" in result.stderr
    assert stdin_log.read_text().strip() == "made it better"


def test_title_gate_blocks_multi_violation_title_exit_2(tmp_path: Path) -> None:
    """A title with 2+ violations (gitlint exit 2/3) still blocks, not fails open.

    Regression guard: gitlint's exit code is the violation count, which collides
    with uvx's own exit 2 — so the gate must key on the verdict text, not the code.

    :param tmp_path: pytest tmp dir.
    """
    result, _ = _run_title_gate(
        tmp_path, 'gh pr create --title "bad" --body b', uvx_exit=3, output=_TWO_VIOLATIONS
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "PR title is not a conventional commit" in result.stderr


def test_title_gate_allows_conventional_title_exit_0(tmp_path: Path) -> None:
    """Gitlint accepting the title (stub exit 0, no verdict) falls through to exit 0.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path, 'gh pr create --title "feat(x): valid" --body b', uvx_exit=0
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert stdin_log.read_text().strip() == "feat(x): valid"


def test_title_gate_warn_mode_does_not_block(tmp_path: Path) -> None:
    """``PR_TITLE_GATE=warn`` downgrades a rejection to a warning (exit 0).

    :param tmp_path: pytest tmp dir.
    """
    result, _ = _run_title_gate(
        tmp_path,
        'gh pr create --title "nope" --body b',
        uvx_exit=1,
        output=_VIOLATION,
        gate="warn",
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "WARNING: PR title is not a conventional commit" in result.stderr


def test_title_gate_off_skips_check(tmp_path: Path) -> None:
    """``PR_TITLE_GATE=off`` skips the gate entirely — uvx is never invoked.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path,
        'gh pr create --title "made it better" --body b',
        uvx_exit=1,
        output=_VIOLATION,
        gate="off",
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert not stdin_log.exists()


def test_title_gate_rejects_invalid_gate_value(tmp_path: Path) -> None:
    """An out-of-enum ``PR_TITLE_GATE`` is rejected up front (exit 2), uvx unused.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path, 'gh pr create --title "feat: x" --body b', gate="bogus"
    )
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "PR_TITLE_GATE must be one of" in result.stderr
    assert not stdin_log.exists()


def test_title_gate_fail_open_when_uvx_errors(tmp_path: Path) -> None:
    """A uvx launch failure (exit 2, no verdict line) fails open: warn but allow.

    :param tmp_path: pytest tmp dir.
    """
    result, _ = _run_title_gate(
        tmp_path, 'gh pr create --title "anything" --body b', uvx_exit=2, output=_UV_LAUNCH_ERROR
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "could not lint the PR title" in result.stderr


def test_title_gate_skips_when_no_title_flag(tmp_path: Path) -> None:
    """With no ``--title`` to inspect (e.g. ``--fill``), the gate skips — uvx unused.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path, "gh pr create --fill", uvx_exit=1, output=_VIOLATION
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert not stdin_log.exists()


def test_title_gate_skips_dangling_title_flag(tmp_path: Path) -> None:
    """A ``--title`` with no following value yields no title, so the gate skips.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path, "gh pr create --title", uvx_exit=1, output=_VIOLATION
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert not stdin_log.exists()


def test_title_gate_skips_unparseable_command(tmp_path: Path) -> None:
    """A command shlex cannot tokenize (unbalanced quote) fails safe: skip, defer to CI.

    :param tmp_path: pytest tmp dir.
    """
    result, stdin_log = _run_title_gate(
        tmp_path, 'gh pr create --title "unterminated', uvx_exit=1, output=_VIOLATION
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert not stdin_log.exists()


def test_title_gate_pipes_title_with_shell_metacharacters_literally(tmp_path: Path) -> None:
    """A title with shell metacharacters is piped verbatim, never executed.

    :param tmp_path: pytest tmp dir.
    """
    canary = tmp_path / "pwned"
    title = f"feat: x; touch {canary}"
    _, stdin_log = _run_title_gate(
        tmp_path, f"gh pr create --title {shlex.quote(title)} --body b", uvx_exit=0
    )
    assert stdin_log.read_text().strip() == title
    assert not canary.exists()


@pytest.mark.parametrize(
    "command",
    [
        'gh pr create --title "feat: spaced and: punctuated" --body b',
        "gh pr create --title='feat: spaced and: punctuated' --body b",
        'gh pr create -t "feat: spaced and: punctuated" --body b',
    ],
)
def test_title_gate_extracts_title_across_flag_forms(tmp_path: Path, command: str) -> None:
    """``--title X``, ``--title=X`` and ``-t X`` all yield the same piped title.

    :param tmp_path: pytest tmp dir.
    :param command: A ``gh pr create`` variant carrying the title differently.
    """
    _, stdin_log = _run_title_gate(tmp_path, command, uvx_exit=0)
    assert stdin_log.read_text().strip() == "feat: spaced and: punctuated"
