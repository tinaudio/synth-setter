"""Real DawDreamer dataset generation and host-to-host audio comparison."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from hydra import compose, initialize_config_module
import lance
import numpy as np
import pytest

from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.renderers import DawDreamerRenderer
from synth_setter.data.vst.shapes import AUDIO_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.tools import build_param_map
from tests._vst import (
    PLUGIN_PATH,
    TEST_PARAM_SPEC_NAME,
    TEST_PRESET_PATH,
    TEST_SYNTH,
    TEST_SYNTH_VERSION,
)
from tests.data.vst.test_generate_vst_dataset import (
    _HARDCODED_NOTE_PARAMS,
    _HARDCODED_SYNTH_PARAMS,
)

_EXPERIMENT_BY_SYNTH = {
    "surge_xt": "surge-xt-dawdreamer-smoke",
    "ultramaster_kr106": "ultramaster-kr106-lance-smoke",
}
_PARAMETER_MAP_BY_SYNTH = {
    "surge_xt": "src/synth_setter/data/vst/surge_xt_param_map.json",
    "ultramaster_kr106": "src/synth_setter/data/vst/ultramaster_kr106_param_map.json",
}
_PARITY_LIMITS_BY_SYNTH = {
    "surge_xt": {"mss": 22.0, "wmfcc": 25.0, "sot": 0.35, "rms": 0.8},
    "ultramaster_kr106": {"mss": 3.0, "wmfcc": 4.0, "sot": 0.01, "rms": 0.95},
}


def _read_lance_column(path: Path, field: str) -> np.ndarray:
    """Materialize one fixed-shape tensor column from a Lance shard.

    :param path: Rendered ``.lance`` shard directory.
    :param field: Column name to read.
    :returns: The column stacked into a ``(num_rows, *shape)`` array.
    """
    chunk = lance.dataset(str(path)).to_table(columns=[field]).column(field).combine_chunks()
    return chunk.to_numpy_ndarray()


def _dawdreamer_experiment_config() -> RenderConfig:
    """Compose the selected synth's DawDreamer smoke experiment.

    :returns: Validated render config with test-only seed and attempt overrides applied.
    """
    experiment = _EXPERIMENT_BY_SYNTH.get(TEST_SYNTH)
    if experiment is None:
        pytest.skip(f"No DawDreamer parity fixture is registered for {TEST_SYNTH}")
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                f"experiment=generate_dataset/{experiment}",
                f"synth.plugin_path={PLUGIN_PATH}",
                f"synth.plugin_state_path={TEST_PRESET_PATH}",
                f"synth.param_spec_name={TEST_PARAM_SPEC_NAME}",
                f"synth.synth_version={TEST_SYNTH_VERSION}",
            ],
        )
    config = RenderConfig.from_cfg_nodes(cfg.render, cfg.synth)
    # base_seed / attempts_per_sample are RenderConfig fields, not render-group keys,
    # so pin them post-validation the way the launcher injects the per-shard seed.
    return config.model_copy(update={"base_seed": 1808, "attempts_per_sample": 1})


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.parametrize(
    ("synth", "parameter_map_path", "preset_path"),
    [
        (
            "surge_xt",
            "src/synth_setter/data/vst/surge_4_param_map.json",
            "presets/surge-mini.vstpreset",
        ),
        (
            "surge_xt",
            "src/synth_setter/data/vst/surge_simple_param_map.json",
            "presets/surge-simple.vstpreset",
        ),
        (
            "surge_xt",
            "src/synth_setter/data/vst/surge_xt_param_map.json",
            "presets/surge-base.vstpreset",
        ),
        (
            "ultramaster_kr106",
            "src/synth_setter/data/vst/ultramaster_kr106_param_map.json",
            "presets/ultramaster_kr106-base.vstpreset",
        ),
    ],
    ids=("surge-4", "surge-simple", "surge-xt", "ultramaster-kr106"),
)
def test_dawdreamer_parameter_map_matches_live_plugin(
    synth: str,
    parameter_map_path: str,
    preset_path: str,
) -> None:
    """Each committed DawDreamer map matches its settled preset identities.

    :param synth: Registered synth required by this map.
    :param parameter_map_path: Joint parameter map under test.
    :param preset_path: VST preset paired with the map.
    """
    if TEST_SYNTH != synth:
        pytest.skip(f"Parameter map fixture requires {synth}")

    config = _dawdreamer_experiment_config()
    DawDreamerRenderer(
        plugin_path=str(Path(PLUGIN_PATH).resolve()),
        sample_rate=config.sample_rate,
        channels=config.channels,
        signal_duration_seconds=config.signal_duration_seconds,
        plugin_state_path=str(Path(preset_path).resolve()),
        parameter_map=load_param_map(Path(parameter_map_path)),
    )


@pytest.mark.slow
@pytest.mark.requires_vst
def test_dump_dawdreamer_cli_writes_settled_live_identities(tmp_path: Path) -> None:
    """The real introspection CLI emits identities consumable by the runtime map.

    :param tmp_path: Temporary output directory.
    """
    parameter_map_path = _PARAMETER_MAP_BY_SYNTH.get(TEST_SYNTH)
    if parameter_map_path is None:
        pytest.skip(f"No DawDreamer parameter map fixture is registered for {TEST_SYNTH}")
    parameter_map = load_param_map(Path(parameter_map_path))
    output_path = tmp_path / "dawdreamer.json"

    result = CliRunner().invoke(
        build_param_map.main,
        [
            "dump-dawdreamer",
            "--plugin",
            str(Path(PLUGIN_PATH).resolve()),
            "--plugin-name",
            parameter_map.plugin,
            "--plugin-version",
            parameter_map.dawdreamer.plugin_version,
            "--preset",
            str(Path(TEST_PRESET_PATH).resolve()),
            "--preset-resource",
            parameter_map.preset_resource,
            "--out",
            str(output_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    dump = build_param_map.HostDump.model_validate_json(output_path.read_text(encoding="utf-8"))
    live_names = {parameter.index: parameter.name for parameter in dump.params}
    for identity in parameter_map.params.values():
        assert live_names[identity.dawdreamer.index] == identity.dawdreamer.name


@pytest.mark.slow
@pytest.mark.requires_vst
def test_dawdreamer_dataset_audio_is_similar_to_pedalboard(tmp_path: Path) -> None:
    """Both hosts generate a real dataset row with perceptually similar audio.

    :param tmp_path: Temporary directory for generated Lance shards.
    """
    dawdreamer_config = _dawdreamer_experiment_config().model_copy(update={"samples_per_shard": 2})
    pedalboard_config = dawdreamer_config.model_copy(update={"renderer_backend": "pedalboard"})
    pedalboard_path = tmp_path / "pedalboard.lance"
    dawdreamer_path = tmp_path / "dawdreamer.lance"
    synth_params = _HARDCODED_SYNTH_PARAMS
    note_params = _HARDCODED_NOTE_PARAMS
    if TEST_SYNTH == "ultramaster_kr106":
        synth_params, _ = resolve_param_spec(ParamSpecName(TEST_PARAM_SPEC_NAME)).sample(
            np.random.default_rng(106)
        )
        synth_params = {**synth_params, "master_volume": 1.0}
        note_params = {"pitch": 60, "note_start_and_end": (0.1, 1.5)}
    fixed_synth = [synth_params, synth_params]
    fixed_note = [note_params, note_params]
    dawdreamer_renderer = make_audio_renderer(dawdreamer_config)
    assert isinstance(dawdreamer_renderer, DawDreamerRenderer)
    factory_audio = dawdreamer_renderer.render(
        synth_params,
        note_params["pitch"],
        dawdreamer_config.velocity,
        note_params["note_start_and_end"],
    )
    assert np.isfinite(factory_audio).all()
    assert np.max(np.abs(factory_audio)) > 1e-4
    missing_keys = synth_params.keys() - dawdreamer_renderer._parameter_indices.keys()
    assert not missing_keys
    host_indices = [dawdreamer_renderer._parameter_indices[key] for key in synth_params]
    assert len(host_indices) == len(set(host_indices))
    make_lance_dataset(
        pedalboard_path,
        pedalboard_config,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )
    make_lance_dataset(
        dawdreamer_path,
        dawdreamer_config,
        fixed_synth_params_list=fixed_synth,
        fixed_note_params_list=fixed_note,
    )

    pedalboard_rows = _read_lance_column(pedalboard_path, AUDIO_FIELD).astype(np.float32)
    dawdreamer_rows = _read_lance_column(dawdreamer_path, AUDIO_FIELD).astype(np.float32)
    pedalboard_params = _read_lance_column(pedalboard_path, PARAM_ARRAY_FIELD)[0]
    dawdreamer_params = _read_lance_column(dawdreamer_path, PARAM_ARRAY_FIELD)[0]

    assert np.array_equal(pedalboard_params, dawdreamer_params)
    assert np.isfinite(pedalboard_rows).all()
    assert np.isfinite(dawdreamer_rows).all()
    assert np.max(np.abs(pedalboard_rows)) <= 1.0
    assert np.max(np.abs(dawdreamer_rows)) <= 1.0
    pedalboard_audio = pedalboard_rows[0]
    dawdreamer_audio = dawdreamer_rows[0]
    assert np.max(np.abs(pedalboard_audio)) > 1e-4
    assert np.max(np.abs(dawdreamer_audio)) > 1e-4

    metrics = {
        "mss": compute_mss(pedalboard_audio, dawdreamer_audio),
        "rms": compute_rms(pedalboard_audio, dawdreamer_audio),
        "sot": compute_sot(pedalboard_audio, dawdreamer_audio),
        "wmfcc": compute_wmfcc(pedalboard_audio, dawdreamer_audio),
    }
    limits = _PARITY_LIMITS_BY_SYNTH[TEST_SYNTH]
    assert metrics["mss"] < limits["mss"], metrics
    assert metrics["wmfcc"] < limits["wmfcc"], metrics
    assert metrics["sot"] < limits["sot"], metrics
    assert metrics["rms"] > limits["rms"], metrics
