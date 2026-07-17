"""CI marker filters live only in the `make test-ci-*` targets — see #1353.

Finding 1 of the testing audit: every CI workflow re-spelled its pytest marker
expression inline, so they drifted (test.yml dropped `not requires_vst`,
nightly.yml silently ran `slow`). The marker strings now live in three Makefile
targets and the workflows call them. These tests pin both halves: the targets
carry the canonical expressions, and the workflows invoke the targets instead
of an inline `pytest -m`, so a future edit can't reintroduce drift unnoticed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# make target -> the canonical marker expression its recipe must carry.
TARGET_MARKERS: dict[str, str] = {
    "test-ci-unit": "not slow and not gpu and not mps",
    "test-ci-slow": "slow and not gpu and not mps and not requires_vst",
    "test-ci-nightly": "not gpu and not mps and not requires_vst",
}

# workflow file -> the make target it must invoke instead of inline pytest.
WORKFLOW_TARGETS: dict[str, str] = {
    "test.yml": "test-ci-unit",
    "cpu-slow.yml": "test-ci-slow",
    "nightly.yml": "test-ci-nightly",
}


def _recipe(makefile: str, target: str) -> str:
    """Return the tab-indented recipe body for ``target`` in the Makefile text.

    :param makefile: full Makefile contents.
    :param target: target name whose recipe to extract.
    :returns: the recipe lines joined by newlines (empty if the target has none).
    """
    lines = makefile.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"{target}:"):
            recipe = []
            for body in lines[i + 1 :]:
                if body.startswith("\t"):
                    recipe.append(body)
                elif body.strip() == "":
                    continue
                else:
                    break
            return "\n".join(recipe)
    pytest.fail(f"Makefile missing target {target!r}")


@pytest.mark.infra
@pytest.mark.parametrize(("target", "marker_expr"), sorted(TARGET_MARKERS.items()))
def test_make_target_carries_canonical_marker(
    project_root: Path, target: str, marker_expr: str
) -> None:
    """Each `test-ci-*` recipe runs pytest with its canonical `-m` expression.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param target: the make target under test.
    :param marker_expr: the marker expression its recipe must pass to pytest.
    """
    recipe = _recipe((project_root / "Makefile").read_text(), target)
    assert f'-m "{marker_expr}"' in recipe, (
        f'`make {target}` recipe must run `pytest -m "{marker_expr}"`; got:\n{recipe}'
    )


RECORDED_PARALLEL_FAILURE_EXIT_CODE = 7
RECORDED_SERIAL_FAILURE_EXIT_CODE = 8


def _write_command_recorder(
    path: Path, *, parallel_exit_code: int = 0, serial_exit_code: int = 0
) -> None:
    """Write an executable that records arguments and returns configured lane results.

    :param path: Script path to create.
    :param parallel_exit_code: Exit code for invocations containing ``-n auto``.
    :param serial_exit_code: Exit code for other invocations.
    """
    path.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$MAKE_TEST_LOG"\n'
        f'case "$*" in *"-n auto"*) exit {parallel_exit_code};; *) exit {serial_exit_code};; esac\n'
    )
    path.chmod(0o755)


def _recording_environment(tmp_path: Path, log_path: Path) -> dict[str, str]:
    """Return an environment that puts recording executables before system commands.

    :param tmp_path: Directory containing the recording executables.
    :param log_path: File where recording executables write their arguments.
    :returns: Environment for the sandboxed make invocation.
    """
    return os.environ | {
        "MAKE_TEST_LOG": str(log_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }


def _run_full_cpu_target(
    project_root: Path,
    tmp_path: Path,
    *,
    uname: str,
    parallel_exit_code: int = 0,
    serial_exit_code: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run ``test-full-cpu`` with recording stand-ins for external commands.

    :param project_root: Repository whose Makefile target to run.
    :param tmp_path: Directory for stand-in executables and their log.
    :param uname: Platform value passed to the Makefile.
    :param parallel_exit_code: Exit code for parallel pytest invocations.
    :param serial_exit_code: Exit code for serial pytest invocations.
    :returns: Completed make process and recorded command arguments.
    """
    pytest_recorder = tmp_path / "pytest"
    wrapper_recorder = tmp_path / "headless-wrapper"
    log_path = tmp_path / "commands.log"
    _write_command_recorder(
        pytest_recorder,
        parallel_exit_code=parallel_exit_code,
        serial_exit_code=serial_exit_code,
    )
    _write_command_recorder(wrapper_recorder)

    result = subprocess.run(  # noqa: S603 — resolved make binary and allowlisted target
        [
            shutil.which("make") or "make",
            "--no-print-directory",
            f"UNAME_S={uname}",
            f"HEADLESS_WRAPPER={wrapper_recorder}",
            "test-full-cpu",
        ],
        cwd=project_root,
        env=_recording_environment(tmp_path, log_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, log_path.read_text().splitlines()


@pytest.mark.infra
def test_full_cpu_darwin_non_vst_failure_still_runs_serial_vst_lane(
    project_root: Path, tmp_path: Path
) -> None:
    """Darwin reports both lane results while retaining a non-VST failure.

    :param project_root: Locates the repository Makefile.
    :param tmp_path: Provides isolated stand-in executables and logs.
    """
    result, commands = _run_full_cpu_target(
        project_root,
        tmp_path,
        uname="Darwin",
        parallel_exit_code=RECORDED_PARALLEL_FAILURE_EXIT_CODE,
    )

    assert result.returncode != 0
    assert "Error 1" in result.stderr
    assert commands == [
        "-n auto -m not gpu and not mps and not requires_vst",
        "-m requires_vst and not gpu and not mps",
    ]


@pytest.mark.infra
@pytest.mark.parametrize(
    ("parallel_exit_code", "serial_exit_code"),
    [
        (0, RECORDED_SERIAL_FAILURE_EXIT_CODE),
        (RECORDED_PARALLEL_FAILURE_EXIT_CODE, RECORDED_SERIAL_FAILURE_EXIT_CODE),
    ],
)
def test_full_cpu_darwin_lane_failure_returns_aggregate_failure(
    project_root: Path,
    tmp_path: Path,
    parallel_exit_code: int,
    serial_exit_code: int,
) -> None:
    """Darwin returns failure when the serial lane or both lanes fail.

    :param project_root: Locates the repository Makefile.
    :param tmp_path: Provides isolated stand-in executables and logs.
    :param parallel_exit_code: Recorded parallel-lane result.
    :param serial_exit_code: Recorded serial-lane result.
    """
    result, commands = _run_full_cpu_target(
        project_root,
        tmp_path,
        uname="Darwin",
        parallel_exit_code=parallel_exit_code,
        serial_exit_code=serial_exit_code,
    )

    assert result.returncode != 0
    assert "Error 1" in result.stderr
    assert commands == [
        "-n auto -m not gpu and not mps and not requires_vst",
        "-m requires_vst and not gpu and not mps",
    ]


@pytest.mark.infra
def test_full_cpu_linux_runs_all_cpu_tests_through_headless_wrapper(
    project_root: Path, tmp_path: Path
) -> None:
    """Linux keeps the complete CPU lane parallel behind its display wrapper.

    :param project_root: Locates the repository Makefile.
    :param tmp_path: Provides isolated stand-in executables and logs.
    """
    result, commands = _run_full_cpu_target(project_root, tmp_path, uname="Linux")

    assert result.returncode == 0
    assert commands == ["pytest -n auto -m not gpu and not mps"]


@pytest.mark.infra
@pytest.mark.parametrize(("workflow", "target"), sorted(WORKFLOW_TARGETS.items()))
def test_workflow_invokes_make_target(project_root: Path, workflow: str, target: str) -> None:
    """The workflow calls `make <target>` rather than re-spelling pytest markers.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param workflow: workflow filename whose test step must call the target.
    :param target: the make target it must invoke.
    """
    text = (project_root / ".github" / "workflows" / workflow).read_text()
    invokes = re.search(rf"make {re.escape(target)}(?=\s|$)", text, re.MULTILINE)
    assert invokes is not None, (
        f"{workflow} must run `make {target}` so its marker filter stays in the "
        f"Makefile (see #1353)"
    )


@pytest.mark.infra
@pytest.mark.parametrize("workflow", sorted(WORKFLOW_TARGETS))
def test_workflow_has_no_inline_pytest_marker(project_root: Path, workflow: str) -> None:
    """No `pytest -m` survives inline — the make target is the only source.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param workflow: workflow filename that must not re-spell a marker filter.
    """
    text = (project_root / ".github" / "workflows" / workflow).read_text()
    # Collapse `\`-continuations so a `pytest \<newline>-m gpu` invocation split
    # across lines (the test-gpu.yml / test-mps.yml style) can't evade the guard.
    collapsed = text.replace("\\\n", " ")
    inline = re.search(r"pytest\b[^\n]*\s-m\s", collapsed)
    assert inline is None, (
        f"{workflow} re-spells a pytest marker filter inline ({inline.group(0)!r}); "
        f"move it into a `make test-ci-*` target to keep one source of truth (#1353)"
    )
