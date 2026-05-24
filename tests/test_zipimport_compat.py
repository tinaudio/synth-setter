"""Verify package resources work under zipimport.

Without this lane, every refactor toward the ``Traversable`` API is unverified
— the unpacked test suite cannot tell the difference between code that
correctly stays on the protocol and code that has accidentally drifted back
to ``__fspath__`` / ``.glob()`` / ``str(traversable)``-as-real-path. Both
shapes pass when the install layout happens to be a ``PosixPath``.

The fixture zips the ``synth_setter`` source tree once per session and spawns
a fresh Python with ``PYTHONPATH=<zip>`` and a wiped ``sys.path`` so the
subprocess can only find the package via ``zipimport``. Each test injects a
small Python program over ``-c`` and asserts on its stdout — failure modes
(missing ``__fspath__``, broken ``open``, ``glob`` returning ``[]``) surface
as a non-zero exit with a readable traceback.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synth_setter_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Zip the in-tree ``src/synth_setter`` for zipimport.

    Uses :func:`shutil.make_archive` with ``base_dir="synth_setter"`` so the
    archive root contains ``synth_setter/__init__.py`` — the exact layout
    Python's zipimporter expects on ``sys.path``. Built once per pytest
    session; the resulting path is reused across the smoke tests below.

    :param tmp_path_factory: pytest-provided factory for session-scoped temp
        directories; the zip lives under one of its mktemp dirs.
    :returns: Absolute path to the built ``.zip``.
    """
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    assert (src_root / "synth_setter" / "__init__.py").is_file(), (
        f"synth_setter package not found under {src_root}; the smoke test expects a src/ layout"
    )
    target_dir = tmp_path_factory.mktemp("zipimport")
    archive_stem = target_dir / "synth_setter"
    archive_path = Path(
        shutil.make_archive(
            base_name=str(archive_stem),
            format="zip",
            root_dir=str(src_root),
            base_dir="synth_setter",
        )
    )
    return archive_path


def _run_in_zipped_python(zip_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Spawn a subprocess that can only ``import synth_setter`` via the zip.

    ``PYTHONNOUSERSITE`` disables ``~/.local/lib/...`` injection; ``PYTHONPATH``
    is set to the zip alone; ``PYTHONDONTWRITEBYTECODE`` keeps the test from
    littering ``__pycache__`` next to the source tree.

    :param zip_path: Archive built by :func:`synth_setter_zip`.
    :param code: Python source to execute under ``python -c``.
    :returns: The completed process; caller asserts on ``returncode`` /
        ``stdout`` / ``stderr``.
    """
    return subprocess.run(  # noqa: S603 — controlled argv for the layout smoke test
        [sys.executable, "-s", "-c", code],
        env={
            "PYTHONPATH": str(zip_path),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_synth_setter_resources_importable_from_zip(synth_setter_zip: Path) -> None:
    """``synth_setter.resources`` resolves under zipimport with no on-disk install.

    :param synth_setter_zip: Session-scoped fixture path to the in-tree zip.
    """
    result = _run_in_zipped_python(
        synth_setter_zip,
        "import synth_setter.resources; print(type(synth_setter.resources).__name__)",
    )
    assert result.returncode == 0, f"zipimport of synth_setter.resources failed: {result.stderr}"
    assert "module" in result.stdout


def test_configs_dir_iteration_works_under_zipimport(synth_setter_zip: Path) -> None:
    """``configs_dir().iterdir()`` returns Traversables that work without ``__fspath__``.

    Exercises the schema-test pattern: list YAML names under a configs
    subdirectory without ever materializing a real ``Path``. Catches regressions
    that swap ``iterdir()``/``name`` for ``.glob()``/``.stem``.

    :param synth_setter_zip: Session-scoped fixture path to the in-tree zip.
    """
    code = textwrap.dedent("""
        from synth_setter.resources import configs_dir

        trainer_dir = configs_dir() / "trainer"
        assert trainer_dir.is_dir(), f"trainer/ not found in zipped configs"

        yaml_names = sorted(
            p.name
            for p in trainer_dir.iterdir()
            if p.is_file() and p.name.endswith(".yaml")
        )
        assert yaml_names, "no YAMLs under configs/trainer/ via zipimport"
        print("\\n".join(yaml_names))
    """)
    result = _run_in_zipped_python(synth_setter_zip, code)
    assert result.returncode == 0, f"iterdir failed: {result.stderr}"
    assert "default.yaml" in result.stdout


def test_configs_dir_read_text_works_under_zipimport(synth_setter_zip: Path) -> None:
    """``(configs_dir() / "X.yaml").read_text()`` returns expected file content.

    Exercises the ``OmegaConf.create(traversable.read_text())`` pattern used in
    place of ``OmegaConf.load(str(traversable))`` — the latter calls ``open()``
    on a synthetic path that doesn't exist under zip.

    :param synth_setter_zip: Session-scoped fixture path to the in-tree zip.
    """
    code = textwrap.dedent("""
        from synth_setter.resources import configs_dir

        text = (configs_dir() / "train.yaml").read_text()
        assert text, "train.yaml read_text() returned empty"
        assert "defaults" in text, "train.yaml missing Hydra defaults key"
        print("OK")
    """)
    result = _run_in_zipped_python(synth_setter_zip, code)
    assert result.returncode == 0, f"read_text failed: {result.stderr}"
    assert "OK" in result.stdout


def test_as_file_materializes_wrapper_under_zipimport(synth_setter_zip: Path) -> None:
    """``as_file(vst_headless_wrapper())`` yields a real ``Path`` even under zip.

    Production subprocess sites depend on this — ``as_file`` must extract the
    zip entry to a tempfile and yield its filesystem path, and the path must
    survive long enough for ``subprocess.check_call`` to ``exec()`` it.

    :param synth_setter_zip: Session-scoped fixture path to the in-tree zip.
    """
    code = textwrap.dedent("""
        from pathlib import Path
        from synth_setter.resources import as_file, vst_headless_wrapper

        with as_file(vst_headless_wrapper()) as wrapper_path:
            assert isinstance(wrapper_path, Path), type(wrapper_path).__name__
            assert wrapper_path.is_file(), f"{wrapper_path} not on disk inside as_file"
            assert wrapper_path.name.endswith("run-linux-vst-headless.sh"), wrapper_path.name
            assert "Xvfb" in wrapper_path.read_text(), "wrapper missing Xvfb invocation"
        print("OK")
    """)
    result = _run_in_zipped_python(synth_setter_zip, code)
    assert result.returncode == 0, f"as_file(vst_headless_wrapper) failed: {result.stderr}"
    assert "OK" in result.stdout


def test_as_file_materializes_generate_script_under_zipimport(
    synth_setter_zip: Path,
) -> None:
    """``as_file(generate_vst_dataset_script())`` yields a runnable ``Path``.

    The subprocess launcher passes this path to ``python <script>`` once per
    shard; if ``as_file`` doesn't extract the ``.py`` under zip, every shard
    render is a ``FileNotFoundError``.

    :param synth_setter_zip: Session-scoped fixture path to the in-tree zip.
    """
    code = textwrap.dedent("""
        from pathlib import Path
        from synth_setter.resources import as_file, generate_vst_dataset_script

        with as_file(generate_vst_dataset_script()) as script_path:
            assert isinstance(script_path, Path), type(script_path).__name__
            assert script_path.is_file(), f"{script_path} not on disk inside as_file"
            assert script_path.name.endswith("generate_vst_dataset.py"), script_path.name
            assert "__main__" in script_path.read_text(), "script missing __main__ guard"
        print("OK")
    """)
    result = _run_in_zipped_python(synth_setter_zip, code)
    assert result.returncode == 0, f"as_file(generate_vst_dataset_script) failed: {result.stderr}"
    assert "OK" in result.stdout


def test_generate_script_actually_executes_under_zipimport(synth_setter_zip: Path) -> None:
    """Spawn the materialized renderer as ``python <script> --help`` and assert exit 0.

    Stronger than the materialization check above: the script's own module-top
    imports (``rootutils.setup_root``, then ``from synth_setter…``) have to
    survive being invoked from a tempfile location. Catches the failure mode
    where ``rootutils`` can't walk up from a temp path to find ``.project-root``
    and crashes the renderer before the launcher's argv even reaches the CLI.

    :param synth_setter_zip: Session-scoped fixture path to the in-tree zip.
    """
    code = textwrap.dedent("""
        import subprocess, sys
        from synth_setter.resources import as_file, generate_vst_dataset_script

        with as_file(generate_vst_dataset_script()) as script_path:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        assert result.returncode == 0, (
            f"renderer --help exited {result.returncode}: {result.stderr[:600]}"
        )
        assert "--plugin_path" in result.stdout, "expected CLI flag missing from --help"
        print("OK")
    """)
    result = _run_in_zipped_python(synth_setter_zip, code)
    assert result.returncode == 0, f"renderer execution under zip failed: {result.stderr}"
    assert "OK" in result.stdout
