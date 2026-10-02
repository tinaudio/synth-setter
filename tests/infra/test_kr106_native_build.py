"""Exercise KR106-native wheel builds with isolated compiler configuration."""

import asyncio
import os
import shutil
import sys
import sysconfig
import textwrap
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "packages/kr106-native"
MISSING_COMPILER = "kr106-native-missing-cxx"


async def _run_command(
    command: list[str], working_directory: Path, environment: dict[str, str]
) -> CompletedProcess[str]:
    """Run a command while retaining its output for assertion failures.

    :param command: Executable and arguments, without shell interpretation.
    :param working_directory: Subprocess working directory.
    :param environment: Isolated subprocess environment.
    :returns: Exit status and captured text output.
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=working_directory,
        env=environment,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return CompletedProcess(command, await process.wait(), stdout.decode(), stderr.decode())


def _build_environment(sysconfig_directory: Path) -> dict[str, str]:
    """Return an environment whose Python build compiler does not exist.

    :param sysconfig_directory: Directory for the overridden sysconfig module.
    :returns: Environment directing Python to the altered build configuration.
    """
    module_name = "_sysconfigdata_kr106_native_missing_cxx"
    build_time_vars = dict(sysconfig.get_config_vars())
    build_time_vars["CXX"] = MISSING_COMPILER
    (sysconfig_directory / f"{module_name}.py").write_text(
        f"build_time_vars = {build_time_vars!r}\n"
    )

    environment = os.environ.copy()
    environment.pop("CXX", None)
    environment["_PYTHON_SYSCONFIGDATA_NAME"] = module_name
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(sysconfig_directory), environment.get("PYTHONPATH")])
    )
    return environment


def _build_wheel(output_directory: Path, environment: dict[str, str]) -> CompletedProcess[str]:
    """Build the package wheel in an isolated output directory.

    :param output_directory: Destination for the built wheel.
    :param environment: Compiler selection and Python configuration.
    :returns: Build exit status and captured output.
    """
    uv_executable = shutil.which("uv")
    assert uv_executable is not None
    return asyncio.run(
        _run_command(
            [
                uv_executable,
                "build",
                "--wheel",
                "--out-dir",
                str(output_directory),
                str(PACKAGE_ROOT),
            ],
            output_directory.parent,
            environment,
        )
    )


@pytest.mark.slow
def test_native_wheel_build_ignores_sysconfig_compiler_and_renders(
    tmp_path: Path,
) -> None:
    """Build and consume a wheel when Python's default CXX cannot be found.

    :param tmp_path: Isolated build and import directories.
    """
    sysconfig_directory = tmp_path / "sysconfig"
    sysconfig_directory.mkdir()
    wheel_directory = tmp_path / "wheels"
    wheel_directory.mkdir()

    build = _build_wheel(wheel_directory, _build_environment(sysconfig_directory))

    assert build.returncode == 0, build.stdout + build.stderr
    wheel_path = next(wheel_directory.glob("kr106_native-*.whl"))
    import_prefix = tmp_path / "wheel-import"
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(import_prefix)

    render = asyncio.run(
        _run_command(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import kr106_native
                    import numpy as np

                    audio = kr106_native.render_note(
                        parameters={44: 0.35},
                        midi_note=60,
                        velocity=100,
                        start_sample=0,
                        end_sample=2048,
                        num_samples=4096,
                        sample_rate=44_100.0,
                    )
                    assert audio.shape == (2, 4096)
                    assert np.isfinite(audio).all()
                    assert np.max(np.abs(audio)) > 1e-5
                    """
                ),
            ],
            tmp_path,
            {**os.environ, "PYTHONPATH": str(import_prefix)},
        )
    )

    assert render.returncode == 0, render.stdout + render.stderr


@pytest.mark.slow
def test_native_wheel_build_honors_explicit_cxx_override(tmp_path: Path) -> None:
    """Fail the build before source retrieval when a user selects a bad CXX.

    :param tmp_path: Isolated build destination.
    """
    wheel_directory = tmp_path / "wheels"
    wheel_directory.mkdir()
    environment = os.environ.copy()
    environment["CXX"] = MISSING_COMPILER

    build = _build_wheel(wheel_directory, environment)

    assert build.returncode != 0
    assert MISSING_COMPILER in build.stdout + build.stderr


def _sync_native_consumer(
    project_directory: Path, environment: dict[str, str]
) -> CompletedProcess[str]:
    """Synchronize the isolated consumer project with uv.

    :param project_directory: Temporary project with the local native dependency.
    :param environment: Isolated uv cache and installation paths.
    :returns: Sync exit status and captured output.
    """
    uv_executable = shutil.which("uv")
    assert uv_executable is not None
    return asyncio.run(_run_command([uv_executable, "sync"], project_directory, environment))


def _installed_native_version(
    environment_directory: Path, working_directory: Path, environment: dict[str, str]
) -> CompletedProcess[str]:
    """Return the version exposed by the installed compiled extension.

    :param environment_directory: Temporary virtual environment containing the extension.
    :param working_directory: Import probe's current directory.
    :param environment: Subprocess environment.
    :returns: Import probe exit status and printed version.
    """
    return asyncio.run(
        _run_command(
            [
                str(environment_directory / "bin/python"),
                "-c",
                "import kr106_native; print(kr106_native.get_version())",
            ],
            working_directory,
            environment,
        )
    )


@pytest.mark.slow
def test_uv_sync_rebuilds_native_extension_after_cpp_source_change(tmp_path: Path) -> None:
    """Rebuild a copied native dependency when only its C++ source changes.

    :param tmp_path: Isolated consumer, package copy, cache, and virtual environment.
    """
    project_directory = tmp_path / "consumer"
    package_directory = project_directory / "kr106-native"
    project_directory.mkdir()
    shutil.copytree(PACKAGE_ROOT, package_directory)
    (project_directory / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "kr106-native-cache-consumer"
            version = "0.0.0"
            requires-python = ">=3.12"
            dependencies = ["kr106-native"]

            [tool.uv]
            package = false

            [tool.uv.sources]
            kr106-native = { path = "kr106-native" }
            """
        ).lstrip()
    )
    environment_directory = tmp_path / "environment"
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    environment["UV_PROJECT_ENVIRONMENT"] = str(environment_directory)

    initial_sync = _sync_native_consumer(project_directory, environment)
    assert initial_sync.returncode == 0, initial_sync.stdout + initial_sync.stderr
    initial_version = _installed_native_version(
        environment_directory, project_directory, environment
    )
    assert initial_version.returncode == 0, initial_version.stdout + initial_version.stderr
    assert initial_version.stdout.strip() == "2.5.13"

    source_path = package_directory / "src/kr106_native.cpp"
    source = source_path.read_text()
    source_path.write_text(
        source.replace(
            'constexpr char kVersion[] = "2.5.13";',
            'constexpr char kVersion[] = "2.5.13-cache-key";',
        )
    )

    updated_sync = _sync_native_consumer(project_directory, environment)
    assert updated_sync.returncode == 0, updated_sync.stdout + updated_sync.stderr
    updated_version = _installed_native_version(
        environment_directory, project_directory, environment
    )
    assert updated_version.returncode == 0, updated_version.stdout + updated_version.stderr
    assert updated_version.stdout.strip() == "2.5.13-cache-key"
