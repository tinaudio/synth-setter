"""Pin :func:`synth_setter.workspace.operator_workspace` resolution order.

Covers env override, checkout detection, CWD fallback, and the
``PROJECT_ROOT`` side effect. See #1261 for the import-time crash these
tests pin against.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from synth_setter import workspace

# Enables the ``pytester`` fixture used by the cache-teardown regression test
# below, which needs a second, order-pinned pytest session.
pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _reset_workspace_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the ``@cache`` and any leaked ``$PROJECT_ROOT`` between cases.

    :param monkeypatch: Pytest fixture for env-var isolation.
    """
    workspace.operator_workspace.cache_clear()
    monkeypatch.delenv("PROJECT_ROOT", raising=False)


def _stage_synthetic_package(tmp_path: Path) -> Path:
    """Copy ``src/synth_setter`` into ``tmp_path/site-packages`` for subprocess use.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :returns: Path to the synthetic ``site-packages`` directory; passed
        as ``PYTHONPATH`` to launcher subprocesses that need to import
        ``synth_setter`` without a ``.project-root`` reachable.
    """
    src_pkg = Path(__file__).resolve().parents[1] / "src" / "synth_setter"
    dest = tmp_path / "site-packages"
    dest.mkdir()
    shutil.copytree(src_pkg, dest / "synth_setter", symlinks=True)
    for parent in (dest / "synth_setter").parents:
        assert not (parent / ".project-root").is_file(), (
            f"unexpected .project-root under {parent} — test setup leaked"
        )
    return dest


def _import_probe(module: str) -> str:
    """Build the subprocess source that imports ``module`` and reports ``$PROJECT_ROOT``.

    The probe leaves via :func:`os._exit` once the value is flushed, so the
    result reflects the import only. Running CPython finalization over the
    launchers' native stack instead would fold an unrelated teardown failure
    into this contract — the macOS ``recursive_mutex`` abort in #3429/#3506
    scored a ``SIGABRT`` on a process that had already printed the right path.

    :param module: Module the probe imports.
    :returns: Python source for ``python -c``.
    """
    return (
        "import os, sys\n"
        f"import {module}\n"
        "sys.stdout.write(os.environ.get('PROJECT_ROOT', '<unset>') + '\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )


def _run_probe(source: str, dest: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter with no ``.project-root`` reachable.

    :param source: Python source to execute.
    :param dest: Synthetic ``site-packages`` directory to put on ``PYTHONPATH``.
    :param tmp_path: Working directory, so the cwd fallback resolves here.
    :returns: The completed subprocess.
    """
    return subprocess.run(  # noqa: S603 — invoking python with controlled argv
        [sys.executable, "-s", "-c", source],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(dest),
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``$SYNTH_SETTER_WORKSPACE`` takes precedence over the checkout.

    :param monkeypatch: Pytest fixture for env-var setup.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    resolved = workspace.operator_workspace()
    assert resolved == tmp_path.resolve()
    assert resolved.is_absolute()


def test_blank_env_override_uses_checkout_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only ``$SYNTH_SETTER_WORKSPACE`` falls back to checkout discovery.

    :param monkeypatch: Pytest fixture for env-var setup.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", "   ")
    expected = Path(workspace.__file__).resolve().parents[2]
    resolved = workspace.operator_workspace()
    assert resolved == expected


def test_env_override_strips_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Surrounding whitespace on ``$SYNTH_SETTER_WORKSPACE`` is ignored.

    :param monkeypatch: Pytest fixture for env-var setup.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", f"  {tmp_path}  ")
    resolved = workspace.operator_workspace()
    assert resolved == tmp_path.resolve()


def test_checkout_root_detected_from_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env override, the parents-walk lands on the test file's checkout root.

    :param monkeypatch: Pytest fixture for env-var isolation.
    """
    monkeypatch.delenv("SYNTH_SETTER_WORKSPACE", raising=False)
    expected = Path(workspace.__file__).resolve().parents[2]
    resolved = workspace.operator_workspace()
    assert resolved == expected
    assert (resolved / ".project-root").is_file()


def test_project_root_env_set_as_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The helper publishes the resolved path as ``$PROJECT_ROOT`` for Hydra interpolation.

    :param monkeypatch: Pytest fixture for env-var setup.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    workspace.operator_workspace()
    assert os.environ["PROJECT_ROOT"] == str(tmp_path.resolve())


def test_project_root_env_not_overwritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pre-set ``$PROJECT_ROOT`` is preserved — operator wins.

    :param monkeypatch: Pytest fixture for env-var setup.
    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    monkeypatch.setenv("PROJECT_ROOT", "/already/set")
    monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
    workspace.operator_workspace()
    assert os.environ["PROJECT_ROOT"] == "/already/set"


def test_cwd_fallback_when_no_checkout_reachable(tmp_path: Path) -> None:
    """Spawn a subprocess from a layout with no ``.project-root`` reachable.

    Mirrors the wheel-install crash repro in #1261: copy ``synth_setter``
    into a clean tree, point ``PYTHONPATH`` at it, leave both
    workspace-resolution envs unset so the cwd branch fires.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    dest = _stage_synthetic_package(tmp_path)

    proc = _run_probe(
        "from synth_setter.workspace import operator_workspace; print(operator_workspace())",
        dest,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path.resolve())


@pytest.mark.parametrize(
    "module",
    [
        "synth_setter.cli.train",
        "synth_setter.cli.eval",
        "synth_setter.cli.generate_dataset",
        "synth_setter.cli.finalize_dataset",
    ],
)
def test_launcher_imports_without_project_root(tmp_path: Path, module: str) -> None:
    """Each console-script module imports cleanly and publishes ``$PROJECT_ROOT``.

    Failure here means #1261's import-time crash regressed: a launcher
    walked up from ``__file__`` looking for ``.project-root`` and raised
    ``FileNotFoundError`` before publishing its ``main`` callable. No
    workspace env is set so the cwd branch fires — the exact wheel-install
    failure mode the issue described. The post-import ``$PROJECT_ROOT``
    assertion confirms the side effect ``configs/paths/default.yaml``'s
    ``${oc.env:PROJECT_ROOT}`` interpolation depends on actually fires.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param module: Parametrized console-script module path under test.
    """
    dest = _stage_synthetic_package(tmp_path)

    proc = _run_probe(_import_probe(module), dest, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path.resolve()), (
        f"launcher import did not publish $PROJECT_ROOT (got {proc.stdout.strip()!r})"
    )


def test_launcher_probe_when_interpreter_teardown_aborts_still_reports_the_import(
    tmp_path: Path,
) -> None:
    """A module whose teardown aborts still scores its import as the success it was.

    Pins #3429/#3506: the launchers' native stack aborted during CPython
    finalization on macOS (``recursive_mutex lock failed``) long after the
    import had published the right ``$PROJECT_ROOT``, and the probe reported
    that as an import regression. Here an ``atexit`` abort stands in for that
    teardown on any platform.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    dest = _stage_synthetic_package(tmp_path)
    (dest / "aborts_at_teardown.py").write_text(
        "import atexit, os\n"
        "from synth_setter.workspace import operator_workspace\n"
        "operator_workspace()\n"
        "atexit.register(os.abort)\n",
        encoding="utf-8",
    )

    proc = _run_probe(_import_probe("aborts_at_teardown"), dest, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tmp_path.resolve())


def test_launcher_probe_when_the_import_raises_reports_failure(tmp_path: Path) -> None:
    """The probe still fails when the import itself raises — the #1261 contract.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    dest = _stage_synthetic_package(tmp_path)
    (dest / "raises_on_import.py").write_text(
        "raise FileNotFoundError('.project-root')\n", encoding="utf-8"
    )

    proc = _run_probe(_import_probe("raises_on_import"), dest, tmp_path)

    assert proc.returncode != 0
    assert "FileNotFoundError" in proc.stderr
    assert proc.stdout.strip() == ""


def test_project_autouse_teardown_clears_a_leaked_workspace_for_the_following_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project's real autouse teardown drops a leaked workspace between tests.

    ``operator_workspace`` is ``@cache``d, so a test that points
    ``$SYNTH_SETTER_WORKSPACE`` at its ``tmp_path`` leaves that directory
    cached process-wide; ``monkeypatch`` restores the env var but cannot
    restore the cache. Later tests in the same xdist worker then resolve
    workspace-relative paths under a deleted temp directory (#3188).

    Runs a controlled two-test session (deterministic order, ``-p
    no:randomly``) that loads the project's real ``tests/conftest.py`` as a
    plugin, so the production autouse teardown — not a stand-in — is what
    governs cleanup. The first test leaks its ``tmp_path``; the second
    asserts the resolved workspace is a real checkout. It passes only
    because the real fixture cleared the cache in between, so removing or
    breaking that fixture fails this check.

    :param pytester: Pytest fixture that runs the order-pinned sub-session.
    :param monkeypatch: Puts the repo root on ``PYTHONPATH`` so the
        subprocess can import ``tests.conftest`` as a plugin.
    """
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("PYTHONPATH", str(repo_root))
    pytester.makepyfile("""
        from synth_setter import workspace

        def test_1_leak_a_temp_workspace(monkeypatch, tmp_path):
            monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
            assert workspace.operator_workspace() == tmp_path.resolve()

        def test_2_resolution_falls_back_to_the_checkout():
            resolved = workspace.operator_workspace()
            assert (resolved / ".project-root").is_file(), resolved
    """)

    result = pytester.runpytest_subprocess(
        "-p", "tests.conftest", "-p", "no:randomly", "-p", "no:cacheprovider"
    )

    result.assert_outcomes(passed=2)
