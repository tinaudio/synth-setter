"""Invariants for the Grafana Alloy / Pyroscope eBPF profiling support baked into the image.

The launcher is exercised as a real subprocess: `pyroscope.ebpf` fails *silently* (it collects
nothing) when a precondition is unmet, so the preflight's refusals are the behavior worth pinning.
Setup and the RunPod limitation live in docs/reference/profiling.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Resolved once: ruff flags a bare "bash" as a partial executable path (S607).
_BASH = shutil.which("bash") or "/bin/bash"

REQUIRED_CREDENTIAL_VARS = (
    "GRAFANA_CLOUD_PYROSCOPE_ENDPOINT",
    "GRAFANA_CLOUD_PYROSCOPE_USER",
    "GRAFANA_CLOUD_PYROSCOPE_API_KEY",
)

# The kernel's PROC_PID_INIT_INO: /proc/self/ns/pid carries this inode only in the initial
# (host) PID namespace, which is the one pyroscope.ebpf needs.
INIT_PID_NAMESPACE_INODE = "4026531836"

# Every launcher diagnostic carries this prefix, so its presence proves the launcher itself ran.
LAUNCHER_LOG_PREFIX = "start-alloy-profiling:"

# eBPF unwinding needs these beyond the SYS_ADMIN the devcontainers already grant for FUSE.
_EBPF_RUN_ARGS = (
    "--cap-add=BPF",
    "--cap-add=PERFMON",
    "--cap-add=SYS_PTRACE",
    "--cap-add=SYS_RESOURCE",
    "--cap-add=DAC_READ_SEARCH",
)


@pytest.fixture(scope="session")
def launcher_script(project_root: Path) -> Path:
    """Absolute path to the opt-in Alloy profiling launcher.

    :param project_root: Repo checkout holding the script.
    :returns: Path to `scripts/docker/start_alloy_profiling.sh`.
    """
    return project_root / "scripts" / "docker" / "start_alloy_profiling.sh"


@pytest.fixture(scope="session")
def alloy_config(project_root: Path) -> Path:
    """Absolute path to the Alloy profiling config baked into the image.

    :param project_root: Repo checkout holding the config.
    :returns: Path to `docker/ubuntu22_04/alloy/profiling.alloy`.
    """
    return project_root / "docker" / "ubuntu22_04" / "alloy" / "profiling.alloy"


@pytest.fixture(scope="session")
def runtime_dockerfile(project_root: Path) -> Path:
    """Absolute path to the runtime image Dockerfile.

    :param project_root: Repo checkout holding the Dockerfile.
    :returns: Path to `docker/ubuntu22_04/Dockerfile`.
    """
    return project_root / "docker" / "ubuntu22_04" / "Dockerfile"


@pytest.fixture
def stub_alloy_bin(tmp_path: Path) -> tuple[Path, Path]:
    """Build a directory holding an `alloy` stub that records the fact it ran.

    :param tmp_path: Per-test temporary directory.
    :returns: The directory to prepend to PATH, and the marker file the stub writes.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "alloy-was-launched"
    stub = bin_dir / "alloy"
    stub.write_text(f'#!/usr/bin/env bash\necho "$@" > "{marker}"\n')
    stub.chmod(0o755)
    return bin_dir, marker


def _run_launcher(
    script: Path, env_overrides: dict[str, str], path_prefix: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real launcher with a controlled environment and capture its outcome.

    :param script: The launcher to execute.
    :param env_overrides: Variables layered onto a credential-stripped copy of the environment.
    :param path_prefix: Directory prepended to PATH, so a stub `alloy` shadows any real one.
    :returns: The completed process, with stdout and stderr captured as text.
    """
    env = {k: v for k, v in os.environ.items() if k not in REQUIRED_CREDENTIAL_VARS}
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}{os.pathsep}{env['PATH']}"
    env.update(env_overrides)
    return subprocess.run(  # noqa: S603 — resolved bash over a repo-owned script
        [_BASH, str(script)], env=env, capture_output=True, text=True, check=False
    )


def _run_preflight(
    script: Path, args: str, env_overrides: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Source the launcher and invoke its preflight directly with explicit host state.

    :param script: The launcher to source.
    :param args: Preflight arguments as a shell word list: euid, PID-namespace inode, tracefs dir.
    :param env_overrides: Variables layered onto a credential-stripped copy of the environment.
    :returns: The completed process, with stdout and stderr captured as text.
    """
    env = {k: v for k, v in os.environ.items() if k not in REQUIRED_CREDENTIAL_VARS}
    env.update(env_overrides)
    return subprocess.run(  # noqa: S603 — resolved bash over a repo-owned script
        [_BASH, "-c", f'source "{script}"; alloy_profiling_preflight {args}'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.infra
def test_launcher_without_enable_flag_exits_cleanly_without_starting_alloy(
    launcher_script: Path, stub_alloy_bin: tuple[Path, Path]
) -> None:
    """Profiling is off by default: the launcher is inert when the enable flag is unset.

    :param launcher_script: The opt-in launcher under test.
    :param stub_alloy_bin: PATH directory holding the `alloy` stub, and its marker file.
    """
    bin_dir, marker = stub_alloy_bin
    result = _run_launcher(launcher_script, {}, path_prefix=bin_dir)

    assert result.returncode == 0, f"launcher must no-op when disabled, got: {result.stderr}"
    assert not marker.exists(), "launcher started Alloy despite profiling being disabled"


@pytest.mark.infra
def test_launcher_enabled_without_credentials_fails_naming_every_missing_variable(
    launcher_script: Path, stub_alloy_bin: tuple[Path, Path]
) -> None:
    """Enabling profiling without Grafana Cloud credentials refuses and names what is missing.

    :param launcher_script: The opt-in launcher under test.
    :param stub_alloy_bin: PATH directory holding the `alloy` stub, and its marker file.
    """
    bin_dir, marker = stub_alloy_bin
    result = _run_launcher(
        launcher_script, {"SYNTH_SETTER_PROFILING_ENABLED": "1"}, path_prefix=bin_dir
    )

    assert result.returncode != 0, "launcher must refuse to start without credentials"
    assert not marker.exists(), "launcher started Alloy without credentials"
    for var in REQUIRED_CREDENTIAL_VARS:
        assert var in result.stderr, f"diagnostic must name the missing {var}"


@pytest.mark.infra
@pytest.mark.skipif(os.geteuid() == 0, reason="asserts the non-root refusal; this session is root")
def test_launcher_enabled_as_non_root_fails_naming_the_root_requirement(
    launcher_script: Path, stub_alloy_bin: tuple[Path, Path]
) -> None:
    """With credentials present but no root, the launcher refuses instead of collecting nothing.

    :param launcher_script: The opt-in launcher under test.
    :param stub_alloy_bin: PATH directory holding the `alloy` stub, and its marker file.
    """
    bin_dir, marker = stub_alloy_bin
    result = _run_launcher(
        launcher_script,
        {
            "SYNTH_SETTER_PROFILING_ENABLED": "1",
            "GRAFANA_CLOUD_PYROSCOPE_ENDPOINT": "https://profiles-prod-001.grafana.net",
            "GRAFANA_CLOUD_PYROSCOPE_USER": "123456",
            "GRAFANA_CLOUD_PYROSCOPE_API_KEY": "glc_example",
        },
        path_prefix=bin_dir,
    )

    assert result.returncode != 0, "launcher must refuse to start as non-root"
    assert not marker.exists(), "launcher started Alloy as non-root"
    assert "root" in result.stderr.lower(), (
        f"diagnostic must name the root requirement: {result.stderr}"
    )


@pytest.mark.infra
def test_preflight_in_nested_pid_namespace_reports_the_pid_host_requirement(
    launcher_script: Path,
) -> None:
    """A nested PID namespace is the silent-failure case, so preflight must name `--pid=host`.

    :param launcher_script: The launcher whose preflight is invoked directly.
    """
    credentials: dict[str, str] = dict.fromkeys(REQUIRED_CREDENTIAL_VARS, "set")
    # euid 0, a PID-namespace inode that is not the host's, tracefs present.
    result = _run_preflight(launcher_script, "0 4026534147 /sys/kernel/tracing", credentials)

    assert result.returncode != 0, "nested PID namespace must fail preflight"
    assert "--pid=host" in result.stderr, f"diagnostic must name the remedy: {result.stderr}"


@pytest.mark.infra
@pytest.mark.skipif(
    shutil.which("unshare") is None, reason="needs unshare to build a real PID namespace"
)
def test_launcher_inside_a_real_nested_pid_namespace_refuses_to_start(
    launcher_script: Path, stub_alloy_bin: tuple[Path, Path]
) -> None:
    """Run in an actual nested PID namespace, the launcher refuses rather than collecting nothing.

    The unshared process is uid 0 with credentials set, so only the namespace check can stop it.

    :param launcher_script: The opt-in launcher under test.
    :param stub_alloy_bin: PATH directory holding the `alloy` stub, and its marker file.
    """
    bin_dir, marker = stub_alloy_bin
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        SYNTH_SETTER_PROFILING_ENABLED="1",
        GRAFANA_CLOUD_PYROSCOPE_ENDPOINT="https://profiles-prod-001.grafana.net",
        GRAFANA_CLOUD_PYROSCOPE_USER="123456",
        GRAFANA_CLOUD_PYROSCOPE_API_KEY="glc_example",
    )
    unshare = shutil.which("unshare")
    assert unshare is not None
    result = subprocess.run(  # noqa: S603 — resolved unshare over a repo-owned script
        [unshare, "-Urpf", "--mount-proc", _BASH, str(launcher_script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # unshare fails in several ways where namespaces are restricted (GitHub runners refuse the
    # uid_map write). Any of them leaves the launcher un-run, which its log prefix reveals.
    if LAUNCHER_LOG_PREFIX not in result.stderr:
        pytest.skip(f"could not build a nested PID namespace here: {result.stderr.strip()}")

    assert result.returncode != 0, "launcher must refuse inside a nested PID namespace"
    assert not marker.exists(), "launcher started Alloy inside a nested PID namespace"
    assert "--pid=host" in result.stderr, f"diagnostic must name the remedy: {result.stderr}"


@pytest.mark.infra
def test_preflight_without_tracefs_reports_the_missing_mount(launcher_script: Path) -> None:
    """Without tracefs the profiler cannot attach tracepoints, so preflight names the mount.

    :param launcher_script: The launcher whose preflight is invoked directly.
    """
    credentials: dict[str, str] = dict.fromkeys(REQUIRED_CREDENTIAL_VARS, "set")
    result = _run_preflight(
        launcher_script, f"0 {INIT_PID_NAMESPACE_INODE} /nonexistent/tracefs", credentials
    )

    assert result.returncode != 0, "missing tracefs must fail preflight"
    assert "/nonexistent/tracefs" in result.stderr, "diagnostic must name the absent mount point"


@pytest.mark.infra
def test_preflight_with_every_precondition_met_reports_success(launcher_script: Path) -> None:
    """The happy path passes: root, host PID namespace, tracefs mounted, credentials present.

    :param launcher_script: The launcher whose preflight is invoked directly.
    """
    credentials: dict[str, str] = dict.fromkeys(REQUIRED_CREDENTIAL_VARS, "set")
    result = _run_preflight(launcher_script, f"0 {INIT_PID_NAMESPACE_INODE} /tmp", credentials)

    assert result.returncode == 0, (
        f"preflight must pass when every precondition holds: {result.stderr}"
    )


@pytest.mark.infra
def test_start_runs_alloy_against_the_supplied_config_outside_the_working_tree(
    launcher_script: Path, stub_alloy_bin: tuple[Path, Path], tmp_path: Path
) -> None:
    """Once preflight passes, Alloy runs on the config and stores its data outside the checkout.

    Alloy's `--storage.path` defaults to `data-alloy/` relative to the working directory, which in
    a devcontainer is the bind-mounted repo.

    :param launcher_script: The launcher whose start step is invoked directly.
    :param stub_alloy_bin: PATH directory holding the `alloy` stub, and its marker file.
    :param tmp_path: Per-test temporary directory holding the config stand-in.
    """
    bin_dir, marker = stub_alloy_bin
    config = tmp_path / "profiling.alloy"
    config.write_text("// config under test\n")
    env = dict(os.environ, SYNTH_SETTER_ALLOY_BIN=str(bin_dir / "alloy"))

    subprocess.run(  # noqa: S603 — resolved bash over a repo-owned script
        [_BASH, "-c", f'source "{launcher_script}"; alloy_profiling_start "{config}"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    invocation = marker.read_text().strip()
    assert invocation == f"run --disable-reporting --storage.path=/tmp/alloy-data {config}"


@pytest.mark.infra
def test_runtime_image_installs_alloy_from_a_checksum_pinned_release(
    runtime_dockerfile: Path,
) -> None:
    """Alloy is installed from a pinned release with a SHA256 per supported arch.

    :param runtime_dockerfile: The runtime image Dockerfile.
    """
    text = runtime_dockerfile.read_text()
    version = re.search(r"^ARG ALLOY_VERSION=(v[\d.]+)$", text, re.MULTILINE)
    assert version, "Dockerfile must pin ALLOY_VERSION to an exact release tag"
    for arch in ("AMD64", "ARM64"):
        digest = re.search(rf"^ARG ALLOY_SHA256_{arch}=[0-9a-f]{{64}}$", text, re.MULTILINE)
        assert digest, f"Dockerfile must pin a 64-hex ALLOY_SHA256_{arch}"
    assert "sha256sum -c -" in text, "the downloaded Alloy archive must be checksum-verified"


@pytest.mark.infra
def test_runtime_image_ships_the_profiling_config_and_launcher(runtime_dockerfile: Path) -> None:
    """The image bakes both the Alloy config and the launcher so pods need no file mounts.

    :param runtime_dockerfile: The runtime image Dockerfile.
    """
    text = runtime_dockerfile.read_text()
    assert "docker/ubuntu22_04/alloy/profiling.alloy" in text, (
        "Dockerfile must COPY the Alloy profiling config into the image"
    )
    assert "start_alloy_profiling.sh" in text, (
        "Dockerfile must COPY the launcher onto PATH so pods can opt in"
    )


@pytest.mark.infra
def test_alloy_config_forwards_ebpf_profiles_to_the_grafana_cloud_endpoint(
    alloy_config: Path,
) -> None:
    """The config wires pyroscope.ebpf into a pyroscope.write pointed at Grafana Cloud.

    :param alloy_config: The Alloy profiling config baked into the image.
    """
    text = alloy_config.read_text()
    assert "pyroscope.ebpf" in text, "config must declare a pyroscope.ebpf collector"
    write_block = re.search(r'pyroscope\.write\s+"(\w+)"', text)
    assert write_block, "config must declare a pyroscope.write sink"
    assert f"pyroscope.write.{write_block.group(1)}.receiver" in text, (
        "the eBPF collector must forward to the declared pyroscope.write receiver"
    )


@pytest.mark.infra
def test_alloy_config_reads_every_grafana_credential_from_the_environment(
    alloy_config: Path,
) -> None:
    """No credential is baked into the config; each is resolved via `sys.env` at run time.

    :param alloy_config: The Alloy profiling config baked into the image.
    """
    text = alloy_config.read_text()
    for var in REQUIRED_CREDENTIAL_VARS:
        assert f'sys.env("{var}")' in text, f"{var} must be read from the environment, not baked"


@pytest.mark.infra
@pytest.mark.parametrize("architecture", ["aarch64", "x86_64"])
def test_alloy_process_filter_matches_resolved_uv_runtime_interpreter(
    alloy_config: Path, runtime_dockerfile: Path, architecture: str
) -> None:
    """The process filter selects the canonical executable exposed through ``/proc``.

    :param alloy_config: The Alloy profiling config baked into the image.
    :param runtime_dockerfile: The runtime Dockerfile declaring the uv Python installation.
    :param architecture: uv platform architecture used in its managed interpreter path.
    """
    dockerfile_text = runtime_dockerfile.read_text()
    install_dir = re.search(r"^ENV UV_PYTHON_INSTALL_DIR=(\S+)$", dockerfile_text, re.MULTILINE)
    python_version = re.search(
        r'^RUN uv venv --python ([\d.]+) "\$VIRTUAL_ENV"$', dockerfile_text, re.MULTILINE
    )
    assert install_dir and python_version, "Dockerfile must pin the uv-managed runtime Python"

    config_text = alloy_config.read_text()
    keep_rule = re.search(
        r'source_labels = \["__meta_process_exe"\]\s+regex\s+=\s+"([^"]+)"\s+action\s+=\s+"keep"',
        config_text,
    )
    assert keep_rule, "Alloy config must keep selected process executables"

    major_minor = python_version.group(1).rsplit(".", 1)[0]
    resolved_interpreter = (
        f"{install_dir.group(1)}/cpython-{python_version.group(1)}-linux-{architecture}-gnu/"
        f"bin/python{major_minor}"
    )
    assert re.fullmatch(keep_rule.group(1), resolved_interpreter), (
        f"the process filter drops the resolved runtime interpreter {resolved_interpreter}"
    )


@pytest.mark.infra
def test_every_devcontainer_grants_the_ebpf_profiling_capabilities(
    devcontainer_json_paths: list[Path],
) -> None:
    """Each devcontainer grants the capabilities the eBPF unwinder needs beyond SYS_ADMIN.

    :param devcontainer_json_paths: All .devcontainer/*/devcontainer.json files.
    """
    for path in devcontainer_json_paths:
        run_args = json.loads(path.read_text()).get("runArgs", [])
        missing = [arg for arg in _EBPF_RUN_ARGS if arg not in run_args]
        assert not missing, f"{path}: runArgs missing {missing} needed by pyroscope.ebpf"


@pytest.mark.infra
def test_no_devcontainer_repeats_a_run_arg(devcontainer_json_paths: list[Path]) -> None:
    """RunArgs carry no duplicates — repeats are copy-paste artifacts, not intent.

    :param devcontainer_json_paths: All .devcontainer/*/devcontainer.json files.
    """
    for path in devcontainer_json_paths:
        run_args = json.loads(path.read_text()).get("runArgs", [])
        duplicated = sorted({arg for arg in run_args if run_args.count(arg) > 1})
        assert not duplicated, f"{path}: runArgs repeats {duplicated}"


@pytest.mark.infra
def test_env_example_documents_every_grafana_cloud_credential(project_root: Path) -> None:
    """`.env.example` lists each Grafana Cloud variable so operators know the full surface.

    :param project_root: Repo checkout holding `.env.example`.
    """
    text = (project_root / ".env.example").read_text()
    for var in REQUIRED_CREDENTIAL_VARS:
        assert var in text, f".env.example must document {var}"
