"""CLI tests for ``synth-setter-introspect-plugin`` (issue #1596).

Plugin loading is patched at the ``cli.introspect_plugin.load_plugin`` boundary
— a real load needs a ``.vst3`` binary and X11; the real-plugin path is covered
by the ``requires_vst`` e2e in ``test_introspect_real_plugin.py``. Everything
downstream of the boundary (drafting, emission, file writes) runs for real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pytest
from click.testing import CliRunner

from synth_setter.cli.introspect_plugin import main
from synth_setter.data.vst.param_spec import ParamSpec
from tests.data.vst._introspect_fakes import (
    IntrospectFakeParameter,
    IntrospectFakePlugin,
    exec_module,
)


@dataclass(frozen=True)
class CliRun:
    """Outcome of one CLI invocation plus the isolated cwd it wrote into.

    .. attribute :: exit_code

       Process exit code reported by click.

    .. attribute :: output

       Captured stdout/stderr text.

    .. attribute :: cwd

       The isolated working directory the run wrote its outputs into.
    """

    exit_code: int
    output: str
    cwd: Path


InvokeCli: TypeAlias = Callable[..., CliRun]


@pytest.fixture
def fake_plugin() -> IntrospectFakePlugin:
    """Build a two-parameter fake plugin standing in for the loaded VST3.

    :returns: Fake with one continuous and one categorical parameter.
    """
    return IntrospectFakePlugin(
        {
            "cutoff": IntrospectFakeParameter(float, [0.0, 0.5, 1.0]),
            "filter_type": IntrospectFakeParameter(str, ["LP", "HP"], raw_values=[0.0, 1.0]),
        },
        preset_data=b"VST3\x01\x00fake-state",
        name="Fake Synth",
    )


@pytest.fixture
def invoke_cli(
    fake_plugin: IntrospectFakePlugin, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> InvokeCli:
    """Invoke the CLI in an isolated cwd with plugin loading patched to the fake.

    :param fake_plugin: The plugin double ``load_plugin`` is patched to return.
    :param monkeypatch: Used to patch the ``load_plugin`` boundary.
    :param tmp_path: Parent for the isolated working directory.
    :returns: ``_invoke(*args)`` callable returning a ``CliRun``.
    """
    monkeypatch.setattr(
        "synth_setter.cli.introspect_plugin.load_plugin", lambda _path: fake_plugin
    )

    def _invoke(*args: str) -> CliRun:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            # --plugin-path validates existence; the patched loader never reads it.
            Path(cwd, "fake.vst3").touch()
            result = runner.invoke(main, args, catch_exceptions=False)
        return CliRun(exit_code=result.exit_code, output=result.output, cwd=Path(cwd))

    return _invoke


def test_cli_writes_importable_spec_module_and_preset(invoke_cli: InvokeCli) -> None:
    """The CLI writes a re-executable spec module and the captured preset bytes.

    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    """
    run = invoke_cli("--plugin-path", "fake.vst3", "--spec-name", "fake_synth")

    assert run.exit_code == 0
    spec_path = run.cwd / "fake_synth_param_spec.py"
    preset_path = run.cwd / "fake_synth-base.vstpreset"
    assert preset_path.read_bytes() == b"VST3\x01\x00fake-state"

    spec_text = spec_path.read_text()
    assert "plugin: fake.vst3" in spec_text
    spec = exec_module(spec_text)["FAKE_SYNTH_PARAM_SPEC"]
    assert isinstance(spec, ParamSpec)
    assert spec.synth_param_names == ["cutoff", "filter_type"]


def test_cli_reports_draft_summary_and_next_steps(invoke_cli: InvokeCli) -> None:
    """The CLI reports counts, output paths, and the registration next step.

    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    """
    run = invoke_cli("--plugin-path", "fake.vst3", "--spec-name", "fake_synth")

    assert "Drafted 2 parameter(s), skipped 0." in run.output
    assert "fake_synth_param_spec.py" in run.output
    assert "fake_synth-base.vstpreset" in run.output
    assert "param_spec_registry" in run.output


def test_cli_reports_skipped_count_for_degenerate_parameter(
    fake_plugin: IntrospectFakePlugin, invoke_cli: InvokeCli
) -> None:
    """A degenerate parameter shows up in the CLI's skipped count.

    :param fake_plugin: The plugin double; gains a single-valued parameter here.
    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    """
    fake_plugin.parameters["m1"] = IntrospectFakeParameter(float, [0.0])

    run = invoke_cli("--plugin-path", "fake.vst3", "--spec-name", "fake_synth")

    assert "Drafted 2 parameter(s), skipped 1." in run.output


def test_cli_provenance_version_falls_back_to_unknown(invoke_cli: InvokeCli) -> None:
    """A plugin file with no readable bundle metadata records ``version unknown``.

    The placeholder ``fake.vst3`` is an empty file, so version extraction
    fails and the provenance line must carry the fallback.

    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    """
    run = invoke_cli("--plugin-path", "fake.vst3", "--spec-name", "fake_synth")

    assert "(version unknown)" in (run.cwd / "fake_synth_param_spec.py").read_text()


def test_cli_honors_explicit_output_paths(invoke_cli: InvokeCli) -> None:
    """``--out-spec`` / ``--out-preset`` override the spec-name-derived defaults.

    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    """
    run = invoke_cli(
        "--plugin-path",
        "fake.vst3",
        "--spec-name",
        "fake_synth",
        "--out-spec",
        "out/custom_spec.py",
        "--out-preset",
        "out/custom.vstpreset",
    )

    assert run.exit_code == 0
    assert (run.cwd / "out" / "custom_spec.py").exists()
    assert (run.cwd / "out" / "custom.vstpreset").read_bytes() == b"VST3\x01\x00fake-state"


def test_cli_rejects_spec_name_that_is_not_an_identifier(invoke_cli: InvokeCli) -> None:
    """A non-identifier ``--spec-name`` fails with a message naming the constraint.

    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    """
    run = invoke_cli("--plugin-path", "fake.vst3", "--spec-name", "my-synth!")

    assert run.exit_code != 0
    assert "identifier" in run.output


def test_cli_refuses_to_overwrite_existing_spec_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An existing output file aborts the run before plugin load unless ``--force``.

    No ``load_plugin`` patch is installed: the guard must fire before any load.

    :param monkeypatch: Used to run the CLI inside ``tmp_path``.
    :param tmp_path: Working directory holding the pre-existing spec file.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake.vst3").touch()
    existing = tmp_path / "fake_synth_param_spec.py"
    existing.write_text("# hand-tuned, do not clobber\n")

    result = CliRunner().invoke(
        main, ["--plugin-path", "fake.vst3", "--spec-name", "fake_synth"], catch_exceptions=False
    )

    assert result.exit_code != 0
    assert "--force" in result.output
    assert existing.read_text() == "# hand-tuned, do not clobber\n"


def test_cli_refuses_to_overwrite_existing_preset_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-existing baseline preset also trips the overwrite guard.

    :param monkeypatch: Used to run the CLI inside ``tmp_path``.
    :param tmp_path: Working directory holding the pre-existing preset file.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake.vst3").touch()
    existing = tmp_path / "fake_synth-base.vstpreset"
    existing.write_bytes(b"VST3-hand-captured")

    result = CliRunner().invoke(
        main, ["--plugin-path", "fake.vst3", "--spec-name", "fake_synth"], catch_exceptions=False
    )

    assert result.exit_code != 0
    assert "--force" in result.output
    assert existing.read_bytes() == b"VST3-hand-captured"


def test_cli_force_overwrites_existing_outputs(
    fake_plugin: IntrospectFakePlugin, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--force`` replaces existing outputs with the fresh draft.

    :param fake_plugin: The plugin double ``load_plugin`` is patched to return.
    :param monkeypatch: Used to patch ``load_plugin`` and run inside ``tmp_path``.
    :param tmp_path: Working directory holding the pre-existing spec file.
    """
    monkeypatch.setattr(
        "synth_setter.cli.introspect_plugin.load_plugin", lambda _path: fake_plugin
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake.vst3").touch()
    (tmp_path / "fake_synth_param_spec.py").write_text("# stale draft\n")

    result = CliRunner().invoke(
        main,
        ["--plugin-path", "fake.vst3", "--spec-name", "fake_synth", "--force"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "FAKE_SYNTH_PARAM_SPEC" in (tmp_path / "fake_synth_param_spec.py").read_text()


def test_cli_force_keeps_existing_spec_when_capture_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under ``--force``, a failed preset capture leaves the hand-tuned spec untouched.

    :param monkeypatch: Used to patch ``load_plugin`` and run inside ``tmp_path``.
    :param tmp_path: Working directory holding the pre-existing spec file.
    """

    class _CaptureFailsPlugin(IntrospectFakePlugin):
        @property
        def preset_data(self) -> bytes:
            """Fail on capture, simulating a plugin whose state read crashes.

            :returns: Never returns.
            :raises RuntimeError: Always.
            """
            raise RuntimeError("state read failed")

        @preset_data.setter
        def preset_data(self, value: bytes) -> None:
            """Accept the constructor's initial assignment.

            :param value: Ignored.
            """

    plugin = _CaptureFailsPlugin({"cutoff": IntrospectFakeParameter(float, [0.0, 1.0])})
    monkeypatch.setattr("synth_setter.cli.introspect_plugin.load_plugin", lambda _path: plugin)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake.vst3").touch()
    existing = tmp_path / "fake_synth_param_spec.py"
    existing.write_text("# hand-tuned, do not clobber\n")

    result = CliRunner().invoke(
        main,
        ["--plugin-path", "fake.vst3", "--spec-name", "fake_synth", "--force"],
        catch_exceptions=True,
    )

    assert result.exit_code != 0
    assert existing.read_text() == "# hand-tuned, do not clobber\n"


def test_cli_loads_starting_preset_before_capture(invoke_cli: InvokeCli, tmp_path: Path) -> None:
    """``--preset-path`` state is applied before the baseline is captured.

    The fake's ``load_preset`` adopts the file's bytes as ``preset_data``, so
    the captured file carries the loaded bytes only if loading happened first.

    :param invoke_cli: Fixture invoking the CLI with plugin loading patched.
    :param tmp_path: Holds the starting preset passed via ``--preset-path``.
    """
    start = tmp_path / "start.vstpreset"
    start.write_bytes(b"VST3-loaded-state")

    run = invoke_cli(
        "--plugin-path", "fake.vst3", "--spec-name", "fake_synth", "--preset-path", str(start)
    )

    assert run.exit_code == 0
    assert (run.cwd / "fake_synth-base.vstpreset").read_bytes() == b"VST3-loaded-state"
