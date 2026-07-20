"""Make targets resolve Python tools from the checkout-local virtualenv."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _write_tool(path: Path, origin: str) -> None:
    """Write a command stub that records which environment supplied it.

    :param path: Executable path to create.
    :param origin: Value written to ``TOOL_MARKER`` when invoked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "{origin}" > "$TOOL_MARKER"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("target", "tool"),
    [("format", "pre-commit"), ("test-fast", "pytest")],
)
def test_make_target_with_foreign_environment_uses_checkout_venv(
    tmp_path: Path, target: str, tool: str
) -> None:
    """An inherited environment cannot redirect developer Make targets.

    :param tmp_path: Pytest fixture providing a throwaway checkout.
    :param target: Make target under test.
    :param tool: Python environment executable used by the target.
    """
    checkout = tmp_path / "checkout $HOME with spaces"
    checkout.mkdir()
    shutil.copy(PROJECT_ROOT / "Makefile", checkout / "Makefile")
    marker = tmp_path / "tool-origin.txt"
    global_bin = tmp_path / "global-bin"
    _write_tool(global_bin / tool, "global")
    _write_tool(checkout / ".venv" / "bin" / tool, "worktree")
    make = shutil.which("make")
    assert make is not None

    env = {
        **os.environ,
        "PATH": f"{global_bin}:{os.environ['PATH']}",
        "TOOL_MARKER": str(marker),
        "VIRTUAL_ENV": "/foreign/checkout/.venv",
    }
    subprocess.run(  # noqa: S603 — resolved make binary and allowlisted target
        [make, target], cwd=checkout, env=env, check=True
    )

    assert marker.read_text(encoding="utf-8") == "worktree\n"


def _full_cpu_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """Create a checkout with the real target and a recording worktree pytest.

    :param tmp_path: Pytest fixture providing an isolated checkout.
    :returns: Checkout and invocation-log paths.
    """
    checkout = tmp_path / "isolated-worktree"
    wrapper = checkout / "src" / "synth_setter" / "scripts" / "run-linux-vst-headless.sh"
    wrapper.parent.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "Makefile", checkout / "Makefile")
    shutil.copy(PROJECT_ROOT / wrapper.relative_to(checkout), wrapper)

    log_path = tmp_path / "pytest-invocations.txt"
    pytest_path = checkout / ".venv" / "bin" / "pytest"
    pytest_path.parent.mkdir(parents=True)
    pytest_path.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$PYTEST_INVOCATION_LOG"\n',
        encoding="utf-8",
    )
    pytest_path.chmod(0o755)
    return checkout, log_path


def _run_full_cpu(checkout: Path, log_path: Path, uname: str) -> subprocess.CompletedProcess[str]:
    """Run the real full CPU target without an inherited Python environment.

    :param checkout: Isolated checkout containing the Makefile under test.
    :param log_path: Path where the worktree pytest harness records invocations.
    :param uname: Platform lane selected by the Makefile.
    :returns: Completed Make invocation.
    """
    make = shutil.which("make")
    assert make is not None
    env = {
        "HOME": str(checkout),
        "PATH": SYSTEM_PATH,
        "PYTEST_INVOCATION_LOG": str(log_path),
    }
    return subprocess.run(  # noqa: S603 — resolved make binary and allowlisted target
        [make, "--no-print-directory", f"UNAME_S={uname}", "test-full-cpu"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.infra
@pytest.mark.skipif(os.uname().sysname != "Linux", reason="Linux wrapper requires X11 tools")
def test_full_cpu_linux_stripped_environment_uses_worktree_pytest_through_wrapper(
    tmp_path: Path,
) -> None:
    """The Linux headless lane executes the checkout-local pytest.

    :param tmp_path: Pytest fixture providing an isolated checkout.
    """
    checkout, log_path = _full_cpu_checkout(tmp_path)

    result = _run_full_cpu(checkout, log_path, "Linux")

    assert result.returncode == 0, result.stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == ["-n auto -m not gpu and not mps"]


@pytest.mark.infra
def test_full_cpu_darwin_stripped_environment_uses_worktree_pytest_harness(
    tmp_path: Path,
) -> None:
    """The faithfully selected Darwin lanes execute the checkout-local harness.

    This selects the Makefile's Darwin branch without claiming to emulate macOS; the executable
    harness verifies both platform-specific pytest invocations.

    :param tmp_path: Pytest fixture providing an isolated checkout.
    """
    checkout, log_path = _full_cpu_checkout(tmp_path)

    result = _run_full_cpu(checkout, log_path, "Darwin")

    assert result.returncode == 0, result.stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "-n auto -m not gpu and not mps and not requires_vst",
        "-m requires_vst and not gpu and not mps",
    ]
