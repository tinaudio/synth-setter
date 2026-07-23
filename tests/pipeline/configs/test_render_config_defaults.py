"""Tests for the generic VST render base and concrete synth render groups."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig

_GENERIC_RENDER_FIELDS = {
    "audio_dtype",
    "channels",
    "gui_toggle_cadence",
    "max_retries",
    "mel_spec_dtype",
    "min_loudness",
    "parallel",
    "param_sample_cadence",
    "plugin_reload_cadence",
    "renderer_backend",
    "sample_rate",
    "samples_per_render_batch",
    "samples_per_shard",
    "signal_duration_seconds",
    "velocity",
}

# Each value differs from the VST base default so the assertion distinguishes an
# applied override from a value that merely matches the default.
_SURFACED_RENDER_DEFAULTS: dict[str, object] = {
    "audio_dtype": "float32",
    "mel_spec_dtype": "float16",
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


def _compose_render_group(group: str) -> DictConfig:
    """Compose one render group through Hydra.

    :param group: Render group name below ``configs/render``.
    :returns: The composed ``render`` node.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(config_name=f"render/{group}").render


def test_vst_render_group_contains_only_generic_render_fields() -> None:
    """``render=vst`` provides generic knobs without selecting a synth identity."""
    cfg = _compose_render_group("vst")

    assert set(cfg) == _GENERIC_RENDER_FIELDS


def test_vst_render_group_accepts_appended_synth_identity() -> None:
    """A generic VST eval scaffold composes with caller-supplied synth identity."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval",
            overrides=[
                "experiment=surge/fake_oracle",
                "render=vst",
                "+render.param_spec_name=obxf",
                "+render.plugin_state_path=presets/obxf-base.vstpreset",
                "+render.plugin_path=plugins/OB-Xf.vst3",
                "+render.renderer_version=1.0.3",
            ],
        )

    assert cfg.render.param_spec_name == "obxf"
    assert cfg.render.plugin_state_path == "presets/obxf-base.vstpreset"
    assert cfg.render.plugin_path == "plugins/OB-Xf.vst3"
    assert cfg.render.renderer_version == "1.0.3"
    assert cfg.render.plugin_reload_cadence == "once"


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
    from the composed tree, so this passing proves ``render/vst.yaml`` surfaces the
    field's default; the override value round-trips onto the spec.

    :param field: RenderConfig field surfaced in the base render config.
    :param override_value: Off-default value passed on the Hydra CLI for that field.
    """
    spec = _spec_from_dataset_overrides([f"render.{field}={override_value}"])
    assert getattr(spec.render, field) == override_value


def test_base_render_config_surfaced_defaults_compose_correctly() -> None:
    """A no-override compose yields the values inherited from ``vst.yaml`` on all platforms.

    Pins the values written to the YAML (not the ``RenderConfig`` model's
    ``default_factory`` values, which are platform-dependent). ``gui_toggle_cadence``
    is ``"once"`` — safe on Darwin where ``"render"`` is rejected (#714).
    """
    spec = _spec_from_dataset_overrides([])
    assert spec.render.audio_dtype == "float16"
    assert spec.render.mel_spec_dtype == "float32"
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
    assert spec.render.audio_dtype == "float16"
    assert spec.render.mel_spec_dtype == "float32"
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
    assert spec.render.plugin_reload_cadence == "once"


@pytest.mark.parametrize(
    ("group", "param_spec_name", "plugin_state_path"),
    [
        ("surge_4", "surge_4", "presets/surge-mini.vstpreset"),
        ("surge_simple", "surge_simple", "presets/surge-simple.vstpreset"),
    ],
)
def test_surge_subset_render_groups_keep_surge_xt_identity(
    group: str, param_spec_name: str, plugin_state_path: str
) -> None:
    """Surge subset groups override only their spec and preset identity.

    :param group: Surge subset render group.
    :param param_spec_name: Expected subset ParamSpec registry key.
    :param plugin_state_path: Expected subset preset path.
    """
    cfg = _compose_render_group(group)
    surge_xt = _compose_render_group("surge_xt")

    assert cfg.param_spec_name == param_spec_name
    assert cfg.plugin_state_path == plugin_state_path
    assert cfg.plugin_path == surge_xt.plugin_path
    assert cfg.renderer_version == surge_xt.renderer_version
    assert cfg.plugin_reload_cadence == surge_xt.plugin_reload_cadence
