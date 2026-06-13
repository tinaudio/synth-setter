"""Tests for ``synth_setter.data.vst.registration`` (issue #1596).

The registry transform is exercised against the repo's real
``param_spec_registry.py`` source so the anchors are pinned to the file the
CLI actually modifies; tool cleanliness (ruff format / isort) is checked with
the real ruff binary, mirroring what pre-commit runs on a user's commit.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synth_setter.data.vst import param_spec_registry
from synth_setter.data.vst.registration import (
    checkout_relative_path,
    find_repo_root,
    registration_paths,
    registry_with_spec,
    render_config_yaml,
)
from tests.data.vst._introspect_fakes import assert_ruff_format_clean

REGISTRY_SOURCE = Path(param_spec_registry.__file__).read_text(encoding="utf-8")


def _dict_keys(source: str, name: str) -> list[str]:
    """Extract the literal string keys of module-level dict ``name`` from ``source``.

    :param source: Python module source.
    :param name: Name of the module-level dict assignment to read.
    :returns: The dict's string keys, in order.
    :raises AssertionError: ``source`` has no module-level dict named ``name``.
    """
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            assert isinstance(node.value, ast.Dict)
            return [ast.literal_eval(k) for k in node.value.keys if k is not None]
    raise AssertionError(f"no module-level dict named {name} in source")


def test_registry_with_spec_adds_key_to_both_dicts() -> None:
    """The transform registers the spec in ``param_specs`` and ``preset_paths``."""
    result = registry_with_spec(REGISTRY_SOURCE, "fake_synth")

    assert "fake_synth" in _dict_keys(result, "param_specs")
    assert "fake_synth" in _dict_keys(result, "preset_paths")


def test_registry_with_spec_maps_preset_to_conventional_path() -> None:
    """The ``preset_paths`` entry points at ``presets/<name>-base.vstpreset``."""
    result = registry_with_spec(REGISTRY_SOURCE, "fake_synth")

    assert '"fake_synth": "presets/fake_synth-base.vstpreset",' in result


def test_registry_with_spec_imports_the_generated_module() -> None:
    """The transform imports ``<NAME>_PARAM_SPEC`` from the generated module."""
    result = registry_with_spec(REGISTRY_SOURCE, "fake_synth")

    imported = {
        (node.module, alias.name)
        for node in ast.parse(result).body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("synth_setter.data.vst.fake_synth_param_spec", "FAKE_SYNTH_PARAM_SPEC") in imported


def test_registry_with_spec_output_is_ruff_format_clean() -> None:
    """The modified registry survives ``ruff format --check`` unchanged."""
    assert_ruff_format_clean(registry_with_spec(REGISTRY_SOURCE, "fake_synth"))


@pytest.mark.parametrize("spec_name", ["aaa_synth", "zzz_synth"])
def test_registry_with_spec_output_is_isort_clean(spec_name: str) -> None:
    """The inserted import lands in sorted position for ruff's I001 check.

    Both ends of the alphabet are exercised so the insertion point is derived, not fixed.

    :param spec_name: Spec name sorting before/after the existing imports.
    """
    result = registry_with_spec(REGISTRY_SOURCE, spec_name)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "I001",
            "--stdin-filename",
            "src/synth_setter/data/vst/param_spec_registry.py",
            "-",
        ],
        input=result,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"I001 unsorted imports:\n{proc.stdout}\n{proc.stderr}"


def test_registry_with_spec_already_registered_identically_is_a_noop() -> None:
    """Re-applying the same registration returns the source unchanged."""
    once = registry_with_spec(REGISTRY_SOURCE, "fake_synth")

    assert registry_with_spec(once, "fake_synth") == once


def test_registry_with_spec_conflicting_existing_key_raises() -> None:
    """A spec name already registered by hand (surge_xt) is rejected."""
    with pytest.raises(ValueError, match="surge_xt"):
        registry_with_spec(REGISTRY_SOURCE, "surge_xt")


def test_registry_with_spec_unrecognized_source_raises() -> None:
    """A source without the registry's dict anchors fails loudly, not silently."""
    with pytest.raises(ValueError, match="param_spec_registry"):
        registry_with_spec("print('not the registry')\n", "fake_synth")


def test_render_config_yaml_pins_synth_identity_over_surge_xt_defaults() -> None:
    """The emitted config inherits surge_xt knobs and overrides the identity fields."""
    text = render_config_yaml(
        "fake_synth", plugin_path="plugins/fake.vst3", renderer_version="9.9.9"
    )

    cfg = yaml.safe_load(text)
    assert cfg["defaults"] == ["surge_xt"]
    assert cfg["plugin_path"] == "plugins/fake.vst3"
    assert cfg["preset_path"] == "presets/fake_synth-base.vstpreset"
    assert cfg["param_spec_name"] == "fake_synth"
    assert cfg["renderer_version"] == "9.9.9"


def test_render_config_yaml_quotes_arbitrary_plugin_path() -> None:
    """A plugin path with YAML-hostile characters round-trips through safe_load."""
    hostile = '/tmp/odd: "name" #1.vst3'  # noqa: S108 — literal fixture path, never opened

    text = render_config_yaml("fake_synth", plugin_path=hostile, renderer_version="1.0")

    assert yaml.safe_load(text)["plugin_path"] == hostile


def test_render_config_yaml_preserves_reserved_word_spec_name_as_string() -> None:
    """A spec name that is a YAML 1.1 boolean literal stays a string after parsing."""
    text = render_config_yaml("on", plugin_path="plugins/on.vst3", renderer_version="1.0")

    assert yaml.safe_load(text)["param_spec_name"] == "on"


def test_checkout_relative_path_inside_checkout_is_relative(tmp_path: Path) -> None:
    """A plugin inside the checkout is recorded checkout-relative (POSIX form).

    :param tmp_path: Stands in for the checkout root.
    """
    plugin = tmp_path / "plugins" / "fake.vst3"
    plugin.mkdir(parents=True)

    assert checkout_relative_path(str(plugin), tmp_path) == "plugins/fake.vst3"


def test_checkout_relative_path_outside_checkout_is_absolute(tmp_path: Path) -> None:
    """A plugin outside the checkout is pinned to its resolved absolute path.

    :param tmp_path: Holds both the plugin dir and the disjoint checkout.
    """
    outside = tmp_path / "elsewhere" / "fake.vst3"
    outside.mkdir(parents=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    assert checkout_relative_path(str(outside), checkout) == str(outside.resolve())


def test_checkout_relative_path_through_plugins_symlink_stays_relative(tmp_path: Path) -> None:
    """An in-checkout ``plugins/`` symlink to a system path records relative.

    ``make link-plugins`` symlinks ``plugins/<name>.vst3`` at the system VST3
    dir outside the checkout, so dereferencing the symlink escapes the tree.
    The recorded path must stay checkout-relative regardless.

    :param tmp_path: Holds the disjoint checkout and system plugin dirs.
    """
    checkout = tmp_path / "checkout"
    (checkout / "plugins").mkdir(parents=True)
    system_plugin = tmp_path / "system" / "Dexed.vst3"
    system_plugin.mkdir(parents=True)
    link = checkout / "plugins" / "Dexed.vst3"
    link.symlink_to(system_plugin)

    assert checkout_relative_path(str(link), checkout) == "plugins/Dexed.vst3"


def test_checkout_relative_path_relative_input_resolves_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative plugin path (the CLI's own form) records checkout-relative.

    The CLI passes ``--plugin-path plugins/<name>.vst3`` verbatim from a cwd at
    the checkout root, so the bare relative path must resolve against the cwd.

    :param tmp_path: Stands in for the checkout root.
    :param monkeypatch: Chdirs into the checkout for the relative-path read.
    """
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "Dexed.vst3").mkdir()
    monkeypatch.chdir(tmp_path)

    assert checkout_relative_path("plugins/Dexed.vst3", tmp_path) == "plugins/Dexed.vst3"


def test_registration_paths_lay_out_the_repo_convention(tmp_path: Path) -> None:
    """Destinations follow the spec-module / preset / csv / render-config convention.

    :param tmp_path: Stands in for the checkout root.
    """
    paths = registration_paths(tmp_path, "fake_synth")

    assert paths.spec_module == tmp_path / "src/synth_setter/data/vst/fake_synth_param_spec.py"
    assert paths.preset == tmp_path / "presets/fake_synth-base.vstpreset"
    assert paths.csv == tmp_path / "fake_synth_params.csv"
    assert paths.render_config == tmp_path / "src/synth_setter/configs/render/fake_synth.yaml"
    assert paths.registry == tmp_path / "src/synth_setter/data/vst/param_spec_registry.py"


def test_find_repo_root_walks_up_to_the_checkout_root(tmp_path: Path) -> None:
    """The checkout root is found from a nested cwd via the registry marker file.

    :param tmp_path: Stands in for the checkout root.
    """
    registry = tmp_path / "src/synth_setter/data/vst/param_spec_registry.py"
    registry.parent.mkdir(parents=True)
    registry.touch()
    nested = tmp_path / "docs" / "deep"
    nested.mkdir(parents=True)

    assert find_repo_root(nested) == tmp_path


def test_find_repo_root_outside_a_checkout_returns_none(tmp_path: Path) -> None:
    """A directory tree without the registry marker yields ``None``.

    :param tmp_path: A bare directory that is not a checkout.
    """
    assert find_repo_root(tmp_path) is None
