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

from synth_setter.pipeline.schemas.spec import DatasetSpec

# Off-default values for each field surfaced in ``render/surge_xt.yaml``.  Each
# value must differ from the YAML default so the assertion distinguishes "override
# landed" from "coincidentally matched the default".
_SURFACED_RENDER_DEFAULTS: dict[str, object] = {
    "samples_per_render_batch": 16,
    "max_retries": 3,
    "parallel": True,
    "plugin_reload_cadence": "once",
    "gui_toggle_cadence": "never",
    "param_sample_cadence": "shard",
}

# An experiment that sets none of ``_SURFACED_RENDER_DEFAULTS``, so a successful
# plain override proves the key comes from the base render config, not the experiment.
_NO_CADENCE_EXPERIMENT = "experiment=generate_dataset/ci-materialize-test"


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
    assert spec.render.plugin_reload_cadence == "render"
    assert spec.render.gui_toggle_cadence == "once"
    assert spec.render.param_sample_cadence == "sample"


def test_render_obxf_composes_into_valid_render_config() -> None:
    """``render=obxf`` composes into a ``RenderConfig`` pinning OB-Xf's identity.

    ``plugin_path`` must stay repo-relative so renders resolve against the checkout.
    """
    spec = _spec_from_dataset_overrides(["render=obxf"])

    assert spec.render.param_spec_name == "obxf"
    assert spec.render.renderer_version == "1.0.3"
    assert spec.render.plugin_path == "plugins/OB-Xf.vst3"
    assert spec.render.preset_path == "presets/obxf-base.vstpreset"
    # Inherited from the surge_xt base group, proving defaults: [surge_xt] is live.
    assert spec.render.plugin_reload_cadence == "render"
