"""Contracts for the full Ultramaster KR-106 parameter surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import CategoricalParameter
from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN_PATH = _REPO_ROOT / "plugins" / "Ultramaster KR-106.vst3"
_EXPECTED_SYNTH_PARAMS = (
    "program",
    "adsr_mode",
    "bender_dco",
    "bender_vcf",
    "bender_lfo",
    "arp_rate",
    "lfo_rate",
    "lfo_delay",
    "dco_lfo",
    "dco_pwm",
    "dco_sub",
    "dco_noise",
    "hpf",
    "vcf_freq",
    "vcf_res",
    "vcf_env",
    "vcf_lfo",
    "vcf_kbd",
    "volume",
    "attack",
    "decay",
    "sustain",
    "release",
    "transpose",
    "hold",
    "arpeggio",
    "pulse",
    "saw",
    "sub_sw",
    "chorus_off",
    "chorus_i",
    "chorus_ii",
    "octave",
    "arp_mode",
    "arp_range",
    "lfo_mode",
    "pwm_mode",
    "vcf_env_inv",
    "vca_mode",
    "bender",
    "tuning",
    "power",
    "porta_mode",
    "porta_rate",
    "transpose_offset",
    "master_volume",
    "voices",
    "vcf_oversample",
    "ignore_velocity",
    "arp_limit_kbd",
    "arp_sync_host",
    "lfo_sync_host",
    "mono_retrigger",
    "send_midi_sysex",
    "arp_quantize",
    "lfo_quantize",
    "oscillator_mode",
    "bypass",
)


def test_ultramaster_kr106_spec_covers_every_host_parameter() -> None:
    """The registered spec exposes the complete v2.5.13 automatable surface."""
    assert tuple(param_specs["ultramaster_kr106"].synth_param_names) == _EXPECTED_SYNTH_PARAMS


def test_ultramaster_kr106_spec_has_expected_full_width() -> None:
    """All controls plus note conditioning occupy the expected vector width."""
    spec = param_specs["ultramaster_kr106"]

    assert len(spec.synth_params) == 58
    assert spec.synth_param_length == 116
    assert spec.note_param_length == 3
    assert spec.encoded_width == 119


def test_ultramaster_kr106_spec_round_trip_preserves_values() -> None:
    """A deterministic sample survives encoding and decoding."""
    spec = param_specs["ultramaster_kr106"]
    synth, note = spec.sample(np.random.default_rng(106))

    encoded = spec.encode(synth, note)
    decoded_synth, decoded_note = spec.decode(encoded)

    assert encoded.dtype == np.float32
    assert encoded.shape == (spec.encoded_width,)
    assert np.isfinite(encoded).all()
    assert ((0.0 <= encoded) & (encoded <= 1.0)).all()
    assert decoded_synth == pytest.approx(synth, abs=1e-6)
    assert decoded_note["pitch"] == note["pitch"]
    assert decoded_note["note_start_and_end"] == pytest.approx(
        note["note_start_and_end"], abs=1e-6
    )


def test_ultramaster_kr106_every_categorical_setting_can_be_sampled() -> None:
    """No registered switch or selector value has zero sampling probability."""
    categorical_params = [
        param
        for param in param_specs["ultramaster_kr106"].synth_params
        if isinstance(param, CategoricalParameter)
    ]
    weights = [weight for param in categorical_params for weight in param.weights]

    assert min(weights) > 0.0


def test_ultramaster_kr106_preset_is_committed() -> None:
    """The registered baseline resolves to a captured plugin state."""
    preset = _REPO_ROOT / plugin_state_paths["ultramaster_kr106"]
    assert preset.is_file()


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_live_plugin_matches_registered_surface() -> None:
    """The real bundled VST and committed specification expose identical names."""
    if not _PLUGIN_PATH.is_dir():
        pytest.skip(f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}")

    from synth_setter.data.vst.core import load_plugin, load_preset

    plugin = load_plugin(str(_PLUGIN_PATH))
    load_preset(plugin, str(_REPO_ROOT / plugin_state_paths["ultramaster_kr106"]))

    assert len(plugin.parameters) == len(_EXPECTED_SYNTH_PARAMS)  # type: ignore[attr-defined]
    assert set(plugin.parameters) == set(_EXPECTED_SYNTH_PARAMS)  # type: ignore[attr-defined]


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_categorical_params_cover_every_host_setting() -> None:
    """Every discrete host value has an exact renderer-native value in the spec."""
    if not _PLUGIN_PATH.is_dir():
        pytest.skip(f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}")

    from synth_setter.data.vst.core import load_plugin

    plugin = load_plugin(str(_PLUGIN_PATH))
    spec = param_specs["ultramaster_kr106"]
    categorical_params = {
        param.name: param
        for param in spec.synth_params
        if isinstance(param, CategoricalParameter)
    }

    for name, param in categorical_params.items():
        host_param = plugin.parameters[name]  # type: ignore[attr-defined]
        expected_labels = set()
        for value in host_param.valid_values:
            host_param.raw_value = host_param.get_raw_value_for(value)
            expected_labels.add(host_param.string_value)

        reached_labels = set()
        for raw_value in param.raw_values:
            host_param.raw_value = raw_value
            reached_labels.add(host_param.string_value)

        assert reached_labels == expected_labels, name


@pytest.mark.slow
@pytest.mark.requires_vst
def test_ultramaster_kr106_generate_dataset_stages_consumable_lance_shard(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production entrypoint renders and stages a real KR-106 Lance row.

    :param fake_r2_remote: Local filesystem backing the real rclone transport.
    :param monkeypatch: Pins the single-worker process contract.
    :param tmp_path: Local output directory for the generated shard.
    """
    if not _PLUGIN_PATH.is_dir():
        pytest.skip(f"Ultramaster KR-106 is not installed at {_PLUGIN_PATH}")

    from hydra import compose, initialize_config_module
    from omegaconf import open_dict

    from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
    from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2
    from synth_setter.pipeline.data.lance_staging import shard_has_complete_attempt

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/ultramaster-kr106-lance-smoke"],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(_REPO_ROOT)
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.work_dir = str(_REPO_ROOT)
        cfg.train_val_test_sizes = [1, 0, 0]
        cfg.finalize_inline = False
        cfg.synth.plugin_path = str(_PLUGIN_PATH)
        cfg.synth.plugin_state_path = str(
            _REPO_ROOT / plugin_state_paths["ultramaster_kr106"]
        )
        cfg.render.samples_per_shard = 1
        cfg.r2.prefix = "fake-r2/ultramaster-kr106-e2e/"
        cfg.logger = None

    spec = spec_from_cfg(cfg)
    from_hydra(cfg)

    assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)
    assert validate_all_shards_from_r2(spec) == []
