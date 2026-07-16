"""Tests that ``RenderConfig`` fields are surfaced in ``surge_xt.yaml`` and overridable.

Verifies that:
- Plain ``render.<field>=...`` Hydra CLI overrides (no ``+``) are accepted when
  ``surge_xt.yaml`` surfaces the field — Hydra struct mode rejects unknown keys,
  so a passing compose proves the field is present in the composed tree (#489).
- A no-override compose yields the values hard-coded in ``surge_xt.yaml`` on
  every platform (not the ``RenderConfig`` model's platform-dependent defaults).
- ``render=obxf`` composes into a valid ``RenderConfig`` pinning OB-Xf's identity.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module

from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig

# Off-default values for each field surfaced in ``render/surge_xt.yaml``.  Each
# value must differ from the YAML default so the assertion distinguishes "override
# landed" from "coincidentally matched the default".
_SURFACED_RENDER_DEFAULTS: dict[str, object] = {
    "samples_per_render_batch": 16,
    "max_retries": 3,
    "parallel": True,
    "plugin_reload_cadence": "render",
    "gui_toggle_cadence": "never",
    "param_sample_cadence": "shard",
}

# An experiment that sets none of ``_SURFACED_RENDER_DEFAULTS``, so a successful
# plain override proves the key comes from the base render config, not the experiment.
_NO_CADENCE_EXPERIMENT = "experiment=generate_dataset/ci-materialize-test"


def test_render_config_names_plugin_state_path_as_the_pedalboard_state_input() -> None:
    """Render configuration exposes the pedalboard state file as plugin_state_path."""
    config = RenderConfig(
        plugin_path="plugin.vst3",
        plugin_state_path="state.vstpreset",
        param_spec_name=ParamSpecName("surge_xt"),
        renderer_version="1.0.0",
        sample_rate=44100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        samples_per_shard=1,
    )

    assert config.plugin_state_path == "state.vstpreset"


def _spec_from_dataset_overrides(overrides: list[str]) -> DatasetSpec:
    """Compose ``dataset.yaml`` with extra overrides and round-trip through ``DatasetSpec``.

    :param overrides: Hydra override strings appended after the experiment selector.
    :returns: The validated spec built from the composed cfg.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=[_NO_CADENCE_EXPERIMENT, *overrides])
    return DatasetSpec.from_hydra_cfg(cfg)


@pytest.mark.parametrize(("field", "override_value"), list(_SURFACED_RENDER_DEFAULTS.items()))
def test_base_render_config_accepts_plain_override_for_surfaced_default(
    field: str, override_value: object
) -> None:
    """A plain ``render.<field>=`` override composes against a no-cadence experiment.

    Struct mode rejects ``render.<field>=`` (without ``+``) when the key is absent
    from the composed tree, so this passing proves ``render/surge_xt.yaml`` surfaces
    the field's default; the override value round-trips onto the spec.

    :param field: RenderConfig field surfaced in the base render config.
    :param override_value: Off-default value passed on the Hydra CLI for that field.
    """
    spec = _spec_from_dataset_overrides([f"render.{field}={override_value}"])
    assert getattr(spec.render, field) == override_value


def test_base_render_config_surfaced_defaults_compose_correctly() -> None:
    """A no-override compose yields the values surfaced in ``surge_xt.yaml`` on all platforms.

    Pins the values written to the YAML (not the ``RenderConfig`` model's
    ``default_factory`` values, which are platform-dependent). ``gui_toggle_cadence``
    is ``"once"`` — safe on Darwin where ``"render"`` is rejected (#714).
    """
    spec = _spec_from_dataset_overrides([])
    assert spec.render.samples_per_render_batch == 32
    assert spec.render.max_retries == 0
    assert spec.render.parallel is False
    assert spec.render.plugin_reload_cadence == "once"
    assert spec.render.gui_toggle_cadence == "once"
    assert spec.render.param_sample_cadence == "sample"


@pytest.mark.parametrize(
    ("name", "num_params"),
    [("torchsynth_adsr", 8), ("torchsynth_simple", 19), ("torchsynth_full", 79)],
)
def test_render_torchsynth_composes_into_valid_render_config(name: str, num_params: int) -> None:
    """Each ``render=torchsynth_*`` group composes into a valid in-process ``RenderConfig``.

    :param name: Render group / param-spec registry key under test.
    :param num_params: Expected encoded parameter width.
    """
    spec = _spec_from_dataset_overrides([f"render={name}"])

    assert spec.render.param_spec_name == name
    assert spec.render.renderer_backend == "torchsynth"
    assert spec.render.plugin_path == "torchsynth"
    assert spec.render.plugin_state_path == ""
    assert spec.render.renderer_version == "1.0.2"
    assert spec.render.gui_toggle_cadence == "never"
    # One shared voice per shard: rebuilding it per render would dominate render time.
    assert spec.render.plugin_reload_cadence == "once"
    assert spec.num_params == num_params


def test_render_obxf_composes_into_valid_render_config() -> None:
    """``render=obxf`` composes into a valid ``RenderConfig``; plugin_path stays repo-relative and num_params resolves without ``KeyError``."""
    spec = _spec_from_dataset_overrides(["render=obxf"])

    assert spec.render.param_spec_name == "obxf"
    assert spec.render.renderer_version == "1.0.3"
    assert spec.render.plugin_path == "plugins/OB-Xf.vst3"
    assert spec.render.plugin_state_path == "presets/obxf-base.vstpreset"
    assert spec.num_params == 187
    # Inherited from the surge_xt base group, proving defaults: [surge_xt] is live.
    assert spec.render.plugin_reload_cadence == "once"
