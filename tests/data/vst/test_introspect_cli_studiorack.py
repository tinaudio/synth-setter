from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

import synth_setter
from synth_setter.cli.introspect_plugin import main
from tests.data.vst._introspect_fakes import IntrospectFakePlugin

_REAL_PKG_DIR = Path(synth_setter.__file__).parent


def _checkout(tmp_path: Path) -> Path:
    """Create the minimum checkout layout required by registration.

    :param tmp_path: Scratch root receiving the checkout.
    :returns: Skeleton checkout root.
    """
    root = tmp_path / "checkout"
    vst_dir = root / "src/synth_setter/data/vst"
    vst_dir.mkdir(parents=True)
    shutil.copy(_REAL_PKG_DIR / "data/vst/param_spec_registry.py", vst_dir)
    shutil.copy(_REAL_PKG_DIR / "synth_spec.py", vst_dir.parents[1])
    render_dir = root / "src/synth_setter/configs/render"
    render_dir.mkdir(parents=True)
    shutil.copy(_REAL_PKG_DIR / "configs/render/vst.yaml", render_dir)
    return root


def test_register_studiorack_plugin_links_managed_bundle_and_records_alias(
    tmp_path: Path,
    fake_plugin: IntrospectFakePlugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed registration records a stable alias instead of a host path.

    :param tmp_path: Scratch root for checkout and managed plugin storage.
    :param fake_plugin: Real introspection surface behind the patched loader boundary.
    :param monkeypatch: Replaces the native plugin loader with the test plugin.
    """
    checkout = _checkout(tmp_path)
    manifest = checkout / "studiorack.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "test-project",
                "type": "project",
                "plugins": {"example/synth": "9.9.9"},
                "vst3Bundles": {"example/synth": "Example Synth.vst3"},
            }
        )
    )
    managed = tmp_path / "managed"
    bundle = managed / "VST3/example/synth/9.9.9/Example Synth.vst3/Contents"
    bundle.mkdir(parents=True)
    (bundle / "moduleinfo.json").write_text('{"Version": "9.9.9"}')
    monkeypatch.setattr(
        "synth_setter.cli.introspect_plugin.load_plugin",
        lambda _path, _name=None: fake_plugin,
    )

    result = CliRunner().invoke(
        main,
        [
            "--studiorack-plugin",
            "example/synth",
            "--studiorack-manifest",
            str(manifest),
            "--studiorack-plugins-dir",
            str(managed),
            "--spec-name",
            "fake_synth",
            "--register",
            "--repo-root",
            str(checkout),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    alias = checkout / "plugins/Example Synth.vst3"
    assert alias.is_symlink()
    assert alias.resolve() == bundle.parent.resolve()
    identity = OmegaConf.load(checkout / "src/synth_setter/configs/synth/fake_synth.yaml")
    assert identity.plugin_path == "plugins/Example Synth.vst3"
    assert identity.synth_version == "9.9.9"


def test_register_studiorack_conflict_preserves_existing_alias(tmp_path: Path) -> None:
    """Registration validation runs before changing a managed alias.

    :param tmp_path: Scratch root for checkout and managed plugin storage.
    """
    checkout = _checkout(tmp_path)
    old_bundle = tmp_path / "old/Surge XT.vst3"
    old_bundle.mkdir(parents=True)
    alias = checkout / "plugins/Surge XT.vst3"
    alias.parent.mkdir()
    alias.symlink_to(old_bundle, target_is_directory=True)
    manifest = checkout / "studiorack.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "test-project",
                "type": "project",
                "plugins": {"example/synth": "9.9.9"},
                "vst3Bundles": {"example/synth": "Surge XT.vst3"},
            }
        )
    )
    new_bundle = tmp_path / "managed/VST3/example/synth/9.9.9/Surge XT.vst3"
    new_bundle.mkdir(parents=True)

    result = CliRunner().invoke(
        main,
        [
            "--studiorack-plugin",
            "example/synth",
            "--studiorack-manifest",
            str(manifest),
            "--studiorack-plugins-dir",
            str(tmp_path / "managed"),
            "--spec-name",
            "surge_xt",
            "--register",
            "--repo-root",
            str(checkout),
        ],
    )

    assert result.exit_code != 0
    assert alias.resolve() == old_bundle.resolve()


def test_register_rejects_plugin_path_with_studiorack_plugin(
    tmp_path: Path,
) -> None:
    """Managed and explicit plugin selectors cannot be combined.

    :param tmp_path: Scratch root for the checkout and explicit bundle.
    """
    checkout = _checkout(tmp_path)
    explicit = tmp_path / "explicit.vst3"
    explicit.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "--plugin-path",
            str(explicit),
            "--studiorack-plugin",
            "example/synth",
            "--spec-name",
            "fake_synth",
            "--register",
            "--repo-root",
            str(checkout),
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
