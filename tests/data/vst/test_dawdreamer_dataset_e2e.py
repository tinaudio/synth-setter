"""Real DawDreamer dataset generation and host-to-host audio comparison."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest
from click.testing import CliRunner
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.renderers import DawDreamerRenderer
from synth_setter.data.vst.shapes import AUDIO_FIELD, PARAM_ARRAY_FIELD
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)
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


def _read_lance_column(path: Path, field: str) -> np.ndarray:
    """Materialize one fixed-shape tensor column from a Lance shard.

    :param path: Rendered ``.lance`` shard directory.
    :param field: Column name to read.
    :returns: The column stacked into a ``(num_rows, *shape)`` array.
    """
    chunk = lance.dataset(str(path)).to_table(columns=[field]).column(field).combine_chunks()
    return chunk.to_numpy_ndarray()


def _dawdreamer_experiment_config() -> RenderConfig:
    """Compose the DawDreamer smoke experiment with the test plugin paths.

    :returns: Validated render config with test-only seed and attempt overrides applied.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/surge-xt-dawdreamer-smoke",
                f"render.synth.plugin_path={PLUGIN_PATH}",
                f"render.synth.plugin_state_path={TEST_PRESET_PATH}",
                f"render.synth.param_spec_name={TEST_PARAM_SPEC_NAME}",
                f"render.synth.synth_version={TEST_SYNTH_VERSION}",
            ],
        )
    config = RenderConfig.model_validate(OmegaConf.to_container(cfg.render, resolve=True))
    # base_seed / attempts_per_sample are RenderConfig fields, not render-group keys,
    # so pin them post-validation the way the launcher injects the per-shard seed.
    return config.model_copy(update={"base_seed": 1808, "attempts_per_sample": 1})


@pytest.mark.slow
@pytest.mark.requires_vst
@pytest.mark.parametrize(
    ("parameter_map_path", "preset_path"),
    [
        ("src/synth_setter/data/vst/surge_4_param_map.json", "presets/surge-mini.vstpreset"),
        (
            "src/synth_setter/data/vst/surge_simple_param_map.json",
            "presets/surge-simple.vstpreset",
        ),
        ("src/synth_setter/data/vst/surge_xt_param_map.json", "presets/surge-base.vstpreset"),
    ],
    ids=("surge-4", "surge-simple", "surge-xt"),
)
def test_dawdreamer_parameter_map_matches_live_plugin(
    parameter_map_path: str,
    preset_path: str,
) -> None:
    """Each committed DawDreamer map matches its settled preset identities.

    :param parameter_map_path: Joint parameter map under test.
    :param preset_path: VST preset paired with the map.
    """
    if TEST_SYNTH != "surge_xt":
        pytest.skip("DawDreamer parameter map fixtures use the Surge XT plugin")

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
@pytest.mark.requires_surgepy
def test_dawdreamer_dump_build_roundtrip_loads_real_settled_map(tmp_path: Path) -> None:
    """Real host dumps build a map accepted by the settled DawDreamer renderer.

    :param tmp_path: Temporary host-dump and map destinations.
    """
    if TEST_SYNTH != "surge_xt":
        pytest.skip("DawDreamer parameter map fixtures use the Surge XT plugin")
    clap_path = Path("/usr/lib/clap/Surge XT.clap")
    if not clap_path.exists():
        pytest.skip(f"Surge XT CLAP fixture is unavailable: {clap_path}")

    pedalboard_dump = tmp_path / "pedalboard.json"
    clap_dump = tmp_path / "clap.json"
    dawdreamer_dump = tmp_path / "dawdreamer.json"
    surgepy_dump = tmp_path / "surgepy.json"
    output_map = tmp_path / "surge_xt_param_map.json"
    runner = CliRunner()

    def invoke(*args: str) -> None:
        result = runner.invoke(build_param_map.main, list(args), catch_exceptions=False)
        assert result.exit_code == 0, result.output

    invoke(
        "dump-pedalboard",
        "--plugin",
        str(PLUGIN_PATH),
        "--preset",
        str(TEST_PRESET_PATH),
        "--preset-resource",
        "presets/surge-base.vstpreset",
        "--out",
        str(pedalboard_dump),
    )
    invoke("dump-clap", "--plugin", str(clap_path), "--out", str(clap_dump))
    invoke(
        "dump-dawdreamer",
        "--plugin",
        str(PLUGIN_PATH),
        "--plugin-name",
        "Surge XT",
        "--plugin-version",
        str(TEST_SYNTH_VERSION),
        "--preset",
        str(TEST_PRESET_PATH),
        "--preset-resource",
        "presets/surge-base.vstpreset",
        "--out",
        str(dawdreamer_dump),
    )
    invoke(
        "dump-surgepy",
        "--preset",
        "presets/surge-base.fxp",
        "--preset-resource",
        "presets/surge-base.fxp",
        "--out",
        str(surgepy_dump),
    )
    invoke(
        "build",
        "--pedalboard-dump",
        str(pedalboard_dump),
        "--clap-dump",
        str(clap_dump),
        "--dawdreamer-dump",
        str(dawdreamer_dump),
        "--surgepy-dump",
        str(surgepy_dump),
        "--param-spec-name",
        "surge_xt",
        "--out",
        str(output_map),
    )

    config = _dawdreamer_experiment_config()
    generated_map = load_param_map(output_map)
    DawDreamerRenderer(
        plugin_path=str(Path(PLUGIN_PATH).resolve()),
        sample_rate=config.sample_rate,
        channels=config.channels,
        signal_duration_seconds=config.signal_duration_seconds,
        plugin_state_path=str(Path(TEST_PRESET_PATH).resolve()),
        parameter_map=generated_map,
    )
    assert all(
        identity.dawdreamer.name == identity.pedalboard.name
        for identity in generated_map.params.values()
    )


@pytest.mark.slow
@pytest.mark.requires_vst
def test_dawdreamer_dataset_audio_is_similar_to_pedalboard(tmp_path: Path) -> None:
    """Both hosts generate a real dataset row with perceptually similar audio.

    :param tmp_path: Temporary directory for generated Lance shards.
    """
    if TEST_SYNTH != "surge_xt":
        pytest.skip("DawDreamer comparison fixture uses the Surge XT parameter map")

    dawdreamer_config = _dawdreamer_experiment_config().model_copy(update={"samples_per_shard": 2})
    pedalboard_config = dawdreamer_config.model_copy(update={"renderer_backend": "pedalboard"})
    pedalboard_path = tmp_path / "pedalboard.lance"
    dawdreamer_path = tmp_path / "dawdreamer.lance"
    fixed_synth = [_HARDCODED_SYNTH_PARAMS, _HARDCODED_SYNTH_PARAMS]
    fixed_note = [_HARDCODED_NOTE_PARAMS, _HARDCODED_NOTE_PARAMS]
    dawdreamer_renderer = make_audio_renderer(dawdreamer_config)
    assert isinstance(dawdreamer_renderer, DawDreamerRenderer)
    factory_audio = dawdreamer_renderer.render(
        _HARDCODED_SYNTH_PARAMS,
        _HARDCODED_NOTE_PARAMS["pitch"],
        dawdreamer_config.velocity,
        _HARDCODED_NOTE_PARAMS["note_start_and_end"],
    )
    assert np.isfinite(factory_audio).all()
    assert np.max(np.abs(factory_audio)) > 1e-4
    missing_keys = _HARDCODED_SYNTH_PARAMS.keys() - dawdreamer_renderer._parameter_indices.keys()
    assert not missing_keys
    host_indices = [dawdreamer_renderer._parameter_indices[key] for key in _HARDCODED_SYNTH_PARAMS]
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
    assert metrics["mss"] < 22.0, metrics
    assert metrics["wmfcc"] < 25.0, metrics
    assert metrics["sot"] < 0.35, metrics
    assert metrics["rms"] > 0.8, metrics
