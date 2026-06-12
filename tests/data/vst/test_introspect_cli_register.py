"""``--register`` mode tests for ``synth-setter-introspect-plugin`` (issue #1596).

Register mode wires a drafted spec into a checkout, so these tests run the CLI
against checkout copies built in tmp dirs — a skeleton (the real registry +
render config files) for the focused behaviors, and a full copy of the
installed ``synth_setter`` source tree for the end-to-end test, which then
imports the modified registry in a clean subprocess and composes the generated
render config exactly the way ``generate_dataset`` would.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import synth_setter
from synth_setter.cli.introspect_plugin import main
from synth_setter.pipeline.schemas.spec import RenderConfig
from tests.data.vst._introspect_fakes import (
    IntrospectFakeParameter,
    IntrospectFakePlugin,
    assert_ruff_format_clean,
    exec_module,
)

_REAL_PKG_DIR = Path(synth_setter.__file__).parent


@pytest.fixture
def fake_plugin(
    fake_plugin: IntrospectFakePlugin, monkeypatch: pytest.MonkeyPatch
) -> IntrospectFakePlugin:
    """Patch ``load_plugin`` to return the shared two-parameter fake.

    Overrides the conftest fixture of the same name, adding the patching every
    register-mode test needs.

    :param fake_plugin: The shared fake from ``conftest.py``.
    :param monkeypatch: Used to patch the ``load_plugin`` boundary.
    :returns: The shared fake, now wired behind ``load_plugin``.
    """
    monkeypatch.setattr(
        "synth_setter.cli.introspect_plugin.load_plugin", lambda _path: fake_plugin
    )
    return fake_plugin


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Build a skeleton checkout: real registry + render configs, fake plugin bundle.

    :param tmp_path: Parent for the skeleton.
    :returns: The checkout root.
    """
    root = tmp_path / "checkout"
    vst_dir = root / "src/synth_setter/data/vst"
    vst_dir.mkdir(parents=True)
    shutil.copy(_REAL_PKG_DIR / "data/vst/param_spec_registry.py", vst_dir)
    render_dir = root / "src/synth_setter/configs/render"
    render_dir.mkdir(parents=True)
    shutil.copy(_REAL_PKG_DIR / "configs/render/surge_xt.yaml", render_dir)
    bundle = root / "plugins/fake.vst3/Contents"
    bundle.mkdir(parents=True)
    (bundle / "moduleinfo.json").write_text('{"Version": "9.9.9"}')
    return root


def _register(checkout: Path, *extra: str, spec_name: str = "fake_synth") -> Result:
    """Invoke the CLI in register mode against ``checkout``.

    :param checkout: Checkout root passed as ``--repo-root``.
    :param *extra: Additional CLI arguments.
    :param spec_name: Registry key passed as ``--spec-name``.
    :returns: The click invocation result.
    """
    args = [
        "--plugin-path",
        str(checkout / "plugins/fake.vst3"),
        "--spec-name",
        spec_name,
        "--register",
        "--repo-root",
        str(checkout),
        *extra,
    ]
    return CliRunner().invoke(main, args, catch_exceptions=False)


def test_register_writes_all_artifacts_into_the_checkout_layout(
    checkout: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """Register mode lands spec, preset, csv, and render config at the conventional paths.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    result = _register(checkout)

    assert result.exit_code == 0, result.output
    spec_source = (checkout / "src/synth_setter/data/vst/fake_synth_param_spec.py").read_text()
    spec = exec_module(spec_source)["FAKE_SYNTH_PARAM_SPEC"]
    assert spec.synth_param_names == ["cutoff", "filter_type"]
    # The committed module must not instruct a registration step that already happened.
    assert "Then register the" not in spec_source
    assert "Registered in" in spec_source
    assert (
        checkout / "presets/fake_synth-base.vstpreset"
    ).read_bytes() == b"VST3\x01\x00fake-state"
    assert (checkout / "fake_synth_params.csv").exists()
    assert (checkout / "src/synth_setter/configs/render/fake_synth.yaml").exists()


def test_register_adds_spec_to_the_registry_module(
    checkout: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """The checkout's registry gains the import and both dict entries, format-clean.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    _register(checkout)

    registry = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()
    assert "from synth_setter.data.vst.fake_synth_param_spec import FAKE_SYNTH_PARAM_SPEC" in (
        registry
    )
    assert '"fake_synth": FAKE_SYNTH_PARAM_SPEC,' in registry
    assert '"fake_synth": "presets/fake_synth-base.vstpreset",' in registry
    assert_ruff_format_clean(registry)


def test_register_render_config_pins_relative_plugin_path_and_version(
    checkout: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """The render config records the checkout-relative plugin path and bundle version.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    _register(checkout)

    cfg = OmegaConf.load(checkout / "src/synth_setter/configs/render/fake_synth.yaml")
    assert cfg.plugin_path == "plugins/fake.vst3"
    assert cfg.param_spec_name == "fake_synth"
    assert cfg.preset_path == "presets/fake_synth-base.vstpreset"
    assert cfg.renderer_version == "9.9.9"


def test_register_reports_the_generate_dataset_next_step(
    checkout: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """The CLI tells the user how to render with the newly wired synth.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    result = _register(checkout)

    assert "render=fake_synth" in result.output


def test_register_warns_when_renderer_version_is_unknown(
    checkout: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """A bundle without version metadata still registers, but warns about the pin.

    ``generate_dataset`` cross-checks ``renderer_version`` against the loaded
    plugin, so an ``"unknown"`` pin must be surfaced for hand-editing.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    bare = checkout / "plugins/bare.vst3"
    bare.touch()

    result = CliRunner().invoke(
        main,
        [
            "--plugin-path",
            str(bare),
            "--spec-name",
            "fake_synth",
            "--register",
            "--repo-root",
            str(checkout),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "WARNING: renderer_version" in result.output
    cfg = OmegaConf.load(checkout / "src/synth_setter/configs/render/fake_synth.yaml")
    assert cfg.renderer_version == "unknown"


def test_register_refuses_existing_spec_module_without_force(checkout: Path) -> None:
    """An existing hand-tuned spec module aborts before plugin load, registry untouched.

    No ``load_plugin`` patch is installed: the guard must fire before any load.

    :param checkout: Skeleton checkout fixture.
    """
    spec_path = checkout / "src/synth_setter/data/vst/fake_synth_param_spec.py"
    spec_path.write_text("# hand-tuned, do not clobber\n")
    registry_before = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()

    result = _register(checkout)

    assert result.exit_code != 0
    assert "--force" in result.output
    assert spec_path.read_text() == "# hand-tuned, do not clobber\n"
    registry_after = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()
    assert registry_after == registry_before


def test_register_force_rerun_converges_on_the_same_registry(
    checkout: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """A ``--force`` re-run overwrites the artifacts and leaves the registry stable.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    first = _register(checkout)
    registry_once = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()

    second = _register(checkout, "--force")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    registry_twice = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()
    assert registry_twice == registry_once


def test_register_conflicting_spec_name_fails_before_plugin_load(checkout: Path) -> None:
    """A spec name already in the registry (surge_xt) aborts without loading the plugin.

    No ``load_plugin`` patch is installed: the conflict must surface first.

    :param checkout: Skeleton checkout fixture.
    """
    result = _register(checkout, spec_name="surge_xt")

    assert result.exit_code != 0
    assert "surge_xt" in result.output


def test_register_rejects_explicit_out_paths(checkout: Path) -> None:
    """``--register`` owns the destinations; combining it with ``--out-*`` is an error.

    :param checkout: Skeleton checkout fixture.
    """
    result = _register(checkout, "--out-spec", "elsewhere.py")

    assert result.exit_code != 0
    assert "--out-spec" in result.output


def test_register_autodetects_repo_root_from_cwd(
    checkout: Path, fake_plugin: IntrospectFakePlugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--repo-root``, the enclosing checkout is found from the cwd.

    :param checkout: Skeleton checkout fixture.
    :param fake_plugin: Patches the plugin-load boundary.
    :param monkeypatch: Used to chdir into a nested checkout directory.
    """
    nested = checkout / "presets"
    nested.mkdir(exist_ok=True)
    monkeypatch.chdir(nested)

    result = CliRunner().invoke(
        main,
        [
            "--plugin-path",
            str(checkout / "plugins/fake.vst3"),
            "--spec-name",
            "fake_synth",
            "--register",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert (checkout / "src/synth_setter/configs/render/fake_synth.yaml").exists()


def test_register_outside_a_checkout_fails_with_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Register mode outside a checkout (and without ``--repo-root``) names the fix.

    :param tmp_path: A bare directory that is not a checkout.
    :param monkeypatch: Used to chdir into the bare directory.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake.vst3").touch()

    result = CliRunner().invoke(
        main,
        ["--plugin-path", "fake.vst3", "--spec-name", "fake_synth", "--register"],
        catch_exceptions=False,
    )

    assert result.exit_code != 0
    assert "--repo-root" in result.output


def test_register_capture_failure_leaves_registry_and_render_config_unwritten(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed preset capture must not leave a half-wired checkout behind.

    :param checkout: Skeleton checkout fixture.
    :param monkeypatch: Used to patch the plugin-load boundary.
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
    registry_before = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()

    result = CliRunner().invoke(
        main,
        [
            "--plugin-path",
            str(checkout / "plugins/fake.vst3"),
            "--spec-name",
            "fake_synth",
            "--register",
            "--repo-root",
            str(checkout),
        ],
        catch_exceptions=True,
    )

    assert result.exit_code != 0
    registry_after = (checkout / "src/synth_setter/data/vst/param_spec_registry.py").read_text()
    assert registry_after == registry_before
    assert not (checkout / "src/synth_setter/configs/render/fake_synth.yaml").exists()


def test_register_end_to_end_wires_a_runnable_synth_into_a_full_checkout_copy(
    tmp_path: Path, fake_plugin: IntrospectFakePlugin
) -> None:
    """The committed register-mode changes make the synth resolvable by the pipeline.

    Copies the installed ``synth_setter`` source tree into a tmp checkout, runs
    the CLI, then proves the wiring the way ``generate_dataset`` consumes it:

    - a clean subprocess (``PYTHONPATH`` pointed at the copy) imports the
      modified ``param_spec_registry`` and samples the new spec;
    - the parent composes ``render=fake_synth`` from the copy's Hydra configs
      and validates it into a strict ``RenderConfig``.

    :param tmp_path: Parent for the checkout copy.
    :param fake_plugin: Patches the plugin-load boundary.
    """
    root = tmp_path / "checkout"
    shutil.copytree(
        _REAL_PKG_DIR,
        root / "src/synth_setter",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    bundle = root / "plugins/fake.vst3/Contents"
    bundle.mkdir(parents=True)
    (bundle / "moduleinfo.json").write_text('{"Version": "9.9.9"}')

    result = _register(root)
    assert result.exit_code == 0, result.output

    probe = textwrap.dedent(
        """
        import json
        from synth_setter.data.vst.param_spec_registry import param_specs, preset_paths

        spec = param_specs["fake_synth"]
        sampled = spec.sample()
        print(
            json.dumps(
                {
                    "preset_path": preset_paths["fake_synth"],
                    "encoded_width": len(spec),
                    "sampled_params": len(sampled),
                }
            )
        )
        """
    )
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["preset_path"] == "presets/fake_synth-base.vstpreset"

    spec_source = (root / "src/synth_setter/data/vst/fake_synth_param_spec.py").read_text()
    expected_width = len(exec_module(spec_source)["FAKE_SYNTH_PARAM_SPEC"])
    assert report["encoded_width"] == expected_width
    assert report["sampled_params"] > 0

    with initialize_config_dir(
        config_dir=str(root / "src/synth_setter/configs"), version_base="1.3"
    ):
        cfg = compose(overrides=["+render=fake_synth"])
    raw = OmegaConf.to_container(cfg.render, resolve=True)
    assert isinstance(raw, dict)
    render = RenderConfig(**{k: v for k, v in raw.items() if isinstance(k, str)})
    assert render.param_spec_name == "fake_synth"
    assert render.plugin_path == "plugins/fake.vst3"
    assert render.preset_path == "presets/fake_synth-base.vstpreset"
    assert render.renderer_version == "9.9.9"
