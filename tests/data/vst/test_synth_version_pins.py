"""Pin each synth group's ``synth_version`` against its runtime artifact.

The version belongs to synth identity and is composed from the root
``configs/synth/<synth>.yaml`` group (#2565). Workers still inspect the
installed artifact through ``extract_renderer_version`` before rendering.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from importlib.resources import as_file
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.core import extract_renderer_version
from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME
from synth_setter.resources import vst_headless_wrapper
from synth_setter.synth_spec import SYNTHS, SynthName
from tests._vst import TEST_SYNTH, TEST_SYNTH_VERSION

_VST_SYNTHS = sorted(
    str(name) for name, synth in SYNTHS.items() if Path(synth.plugin_path).suffix == ".vst3"
)
_TORCHSYNTH_SYNTHS = sorted(
    str(name) for name, synth in SYNTHS.items() if synth.plugin_path == TORCHSYNTH_PLUGIN_NAME
)
_PROBE_RESULT_PREFIX = "SYNTH_SETTER_VST_VERSION="
_PROBE_TIMEOUT_SECONDS = 60


def _probe_installed_plugin_version(plugin_path: Path, *, headless: bool = False) -> str:
    """Read one VST version in a fresh process.

    :param plugin_path: Installed VST3 bundle to inspect.
    :param headless: Run the child behind the shipped Linux display wrapper.
    :returns: Version reported by the child probe.
    """
    probe_script = Path(__file__).with_name("_version_probe.py")
    command = [sys.executable, str(probe_script), str(plugin_path)]
    if not headless or not sys.platform.startswith("linux"):
        return _run_version_probe(command, plugin_path)

    with as_file(vst_headless_wrapper()) as wrapper:
        return _run_version_probe(["bash", str(wrapper), *command], plugin_path)


def _run_version_probe(command: list[str], plugin_path: Path) -> str:
    """Run a child probe and validate its result contract.

    :param command: Child process command.
    :param plugin_path: Plugin path used in diagnostics.
    :returns: Version reported by the child probe.
    :raises AssertionError: The child fails or emits an invalid result contract.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - test-owned argv
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"VST version probe timed out after {_PROBE_TIMEOUT_SECONDS}s for {plugin_path}\n"
            f"stdout:\n{exc.stdout or '<empty>'}\nstderr:\n{exc.stderr or '<empty>'}"
        ) from exc

    diagnostics = (
        f"stdout:\n{completed.stdout or '<empty>'}\n"
        f"stderr:\n{completed.stderr or '<empty>'}"
    )
    if completed.returncode != 0:
        if completed.returncode < 0:
            signal_number = -completed.returncode
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                exit_status = f"signal {signal_number}"
            else:
                exit_status = f"signal {signal_name} ({signal_number})"
        else:
            exit_status = f"exit code {completed.returncode}"
        raise AssertionError(
            f"VST version probe failed with {exit_status} for {plugin_path}\n{diagnostics}"
        )

    result_lines = [
        line.removeprefix(_PROBE_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(_PROBE_RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise AssertionError(
            f"VST version probe emitted {len(result_lines)} result lines for {plugin_path}\n"
            f"{diagnostics}"
        )
    try:
        version = json.loads(result_lines[0])
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"VST version probe emitted invalid JSON for {plugin_path}: {result_lines[0]!r}"
        ) from exc
    if not isinstance(version, str):
        raise AssertionError(
            f"VST version probe emitted a non-string version for {plugin_path}: {version!r}"
        )
    return version


def _composed_synth(group: str) -> tuple[str, str]:
    """Compose one root synth group and read its artifact identity.

    :param group: Synth group name below ``configs/synth``.
    :returns: ``(plugin_path, synth_version)`` declared by the synth group.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        synth = compose(config_name=f"synth/{group}").synth
    return synth.plugin_path, synth.synth_version


def test_test_synth_version_matches_the_selected_synth_group() -> None:
    """The shared VST fixture reports the selected synth group's version."""
    _, synth_version = _composed_synth(TEST_SYNTH)

    assert TEST_SYNTH_VERSION == synth_version


@pytest.mark.parametrize("group", sorted(SYNTHS))
def test_every_synth_group_resolves_a_synth_version(group: str) -> None:
    """Every registered synth composes the non-blank registry version.

    :param group: Synth group name under test.
    """
    _, synth_version = _composed_synth(group)

    assert synth_version == SYNTHS[SynthName(group)].synth_version
    assert synth_version.strip()


def test_vst_version_probe_uses_fresh_process_returns_child_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe ignores parent loader state and returns its child's version.

    :param tmp_path: Temporary plugin bundle root.
    :param monkeypatch: Replaces the parent process loader with a failure.
    """
    monkeypatch.setattr(
        "tests.data.vst.test_synth_version_pins.extract_renderer_version",
        lambda _: pytest.fail("the parent process must not load the VST"),
    )
    plugin = tmp_path / "Fixture.vst3"
    metadata = plugin / "Contents/moduleinfo.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"Version": "9.8.7"}', encoding="utf-8")

    assert _probe_installed_plugin_version(plugin) == "9.8.7"


def test_vst_version_probe_headless_linux_uses_shipped_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A headless Linux probe starts the child behind the shipped display wrapper.

    :param tmp_path: Temporary plugin path root used in the command.
    :param monkeypatch: Captures the child command without launching it.
    """
    plugin = tmp_path / "Cardinal.vst3"
    commands: list[list[str]] = []

    def _capture(command: list[str], _plugin_path: Path) -> str:
        commands.append(command)
        return "1.0"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "tests.data.vst.test_synth_version_pins._run_version_probe",
        _capture,
    )

    assert _probe_installed_plugin_version(plugin, headless=True) == "1.0"
    assert commands[0][:2] == ["bash", str(vst_headless_wrapper())]


def test_vst_version_probe_child_error_includes_diagnostics(tmp_path: Path) -> None:
    """A failed probe identifies the plugin and preserves its child traceback.

    :param tmp_path: Temporary missing plugin path root.
    """
    missing_plugin = tmp_path / "Missing.vst3"

    with pytest.raises(AssertionError) as exc_info:
        _probe_installed_plugin_version(missing_plugin)

    diagnostic = str(exc_info.value)
    assert str(missing_plugin) in diagnostic
    assert "FileNotFoundError" in diagnostic
    assert "stderr:" in diagnostic


def test_vst_version_probe_child_sigsegv_becomes_test_failure(tmp_path: Path) -> None:
    """A child segfault reports its signal without aborting the pytest process.

    :param tmp_path: Temporary plugin path root used in diagnostics.
    """
    plugin = tmp_path / "Crashing.vst3"
    command = [
        sys.executable,
        "-c",
        (
            "import os, resource, signal; "
            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
            "os.kill(os.getpid(), signal.SIGSEGV)"
        ),
    ]

    with pytest.raises(AssertionError, match=r"signal SIGSEGV \(11\)"):
        _run_version_probe(command, plugin)


@pytest.mark.skipif(not hasattr(signal, "SIGRTMIN"), reason="POSIX real-time signals required")
def test_vst_version_probe_child_unknown_signal_becomes_test_failure(tmp_path: Path) -> None:
    """An unnamed child signal reports its number through the assertion contract.

    :param tmp_path: Temporary plugin path root used in diagnostics.
    """
    plugin = tmp_path / "Crashing.vst3"
    signal_number = signal.SIGRTMIN + 1
    command = [
        sys.executable,
        "-c",
        f"import os; os.kill(os.getpid(), {signal_number})",
    ]

    with pytest.raises(AssertionError, match=rf"signal {signal_number}"):
        _run_version_probe(command, plugin)


@pytest.mark.requires_vst
@pytest.mark.parametrize("group", _VST_SYNTHS)
def test_vst_synth_group_pins_the_installed_plugin_version(group: str) -> None:
    """Each VST synth's pin matches the version read from its plugin bundle.

    :param group: Synth group name under test.
    """
    plugin_path, synth_version = _composed_synth(group)

    installed_version = _probe_installed_plugin_version(
        Path(plugin_path), headless=group == "cardinal"
    )

    assert installed_version == synth_version


@pytest.mark.parametrize("group", _TORCHSYNTH_SYNTHS)
def test_torchsynth_group_pins_the_installed_package_version(group: str) -> None:
    """Each torchsynth pin matches the installed package version.

    :param group: Synth group name under test.
    """
    plugin_path, synth_version = _composed_synth(group)

    assert plugin_path == TORCHSYNTH_PLUGIN_NAME
    assert extract_renderer_version(Path(plugin_path)) == synth_version
