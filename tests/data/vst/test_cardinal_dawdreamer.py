"""Cardinal's DawDreamer contract: registered identity and live mapped-slot control."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.cli.generate_dataset import spec_from_cfg
from synth_setter.data.vst.cardinal_param_spec import CARDINAL_HOST_PARAMETER_TARGETS
from synth_setter.data.vst.param_map import load_param_map
from synth_setter.data.vst.param_spec_registry import param_specs, plugin_state_paths
from synth_setter.data.vst.renderers import DawDreamerRenderer
from synth_setter.data.vst.writers import make_lance_dataset
from synth_setter.resources import as_file, param_map
from synth_setter.synth_spec import SYNTHS, SynthName

_PLUGIN_PATH = "plugins/CardinalSynth.vst3"
_SAMPLE_RATE = 44_100
_NOTE_WINDOW = (0.1, 1.5)
_BASE_PARAMS = {
    "parameter_1_v": 0.5,
    "parameter_2_v": 0.5,
    "parameter_3_v": 0.1,
    "parameter_4_v": 0.25,
    "parameter_5_v": 0.75,
    "parameter_6_v": 0.3,
    "parameter_7_v": 0.8,
    "parameter_8_v": 0.78,
    "parameter_9_v": 1.0,
}


def _cardinal_param_map():
    """Load the committed Cardinal cross-host map.

    :returns: Validated joint map for the ``cardinal`` spec.
    """
    with as_file(param_map("cardinal")) as path:
        return load_param_map(path)


def test_cardinal_param_map_covers_every_curated_slot() -> None:
    """The committed map resolves each curated slot to a distinct DawDreamer index."""
    joint_map = _cardinal_param_map()
    indices = joint_map.dawdreamer_indices()

    assert set(indices) == set(param_specs["cardinal"].synth_param_names)
    assert set(indices) == set(CARDINAL_HOST_PARAMETER_TARGETS)
    assert len(set(indices.values())) == len(indices)


def test_cardinal_param_map_has_no_clap_provenance() -> None:
    """Cardinal ships no CLAP build, so the CLAP projection is unavailable rather than wrong."""
    joint_map = _cardinal_param_map()

    assert joint_map.clap is None
    with pytest.raises(ValueError, match="no CLAP provenance"):
        joint_map.clap_projection()


def test_cardinal_identity_pins_the_mapped_rack_patch() -> None:
    """Cardinal's registered identity carries the patch its host slots depend on."""
    synth = SYNTHS[SynthName("cardinal")]

    assert synth.param_spec_name == "cardinal"
    assert synth.plugin_path == _PLUGIN_PATH
    assert synth.plugin_state_path == "presets/cardinal-base.vstpreset"
    assert plugin_state_paths["cardinal"] == synth.plugin_state_path


@pytest.mark.requires_vst
@pytest.mark.slow
def test_cardinal_mapped_slot_changes_audio_through_dawdreamer() -> None:
    """A curated host slot drives the routed signal once the preset has settled (#2543)."""
    if not Path(_PLUGIN_PATH).exists():
        pytest.skip(f"Cardinal bundle not found at {_PLUGIN_PATH}")
    renderer = DawDreamerRenderer(
        plugin_path=_PLUGIN_PATH,
        sample_rate=_SAMPLE_RATE,
        channels=2,
        signal_duration_seconds=2.0,
        plugin_state_path=plugin_state_paths["cardinal"],
        parameter_map=_cardinal_param_map(),
        reload_plugin_each_render=True,
    )

    quiet = renderer.render({**_BASE_PARAMS, "parameter_7_v": 0.2}, 60, 100, _NOTE_WINDOW)
    loud = renderer.render({**_BASE_PARAMS, "parameter_7_v": 1.0}, 60, 100, _NOTE_WINDOW)

    assert np.isfinite(quiet).all() and np.isfinite(loud).all()
    # Without the post-preset settle render the VCA slot is inert and these are equal.
    assert _rms(loud) > _rms(quiet) * 2


def _generate_cardinal_shard(output_dir: str) -> None:
    """Generate two real rows in an isolated process.

    :param output_dir: Directory for the generated Lance shard.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/cardinal-dawdreamer-smoke",
                "render.samples_per_shard=2",
            ],
        )
    root = str(Path.cwd())
    cfg.paths.root_dir = root
    cfg.paths.output_dir = output_dir
    cfg.paths.work_dir = output_dir
    spec = spec_from_cfg(cfg)
    make_lance_dataset(Path(output_dir) / "shard.lance", spec.render)


@pytest.mark.requires_vst
@pytest.mark.slow
def test_cardinal_worker_survives_analysis_between_plugin_reloads(tmp_path: Path) -> None:
    """A real shard exits cleanly after analysis runs between Cardinal reloads.

    :param tmp_path: Isolates the generated Lance dataset.
    """
    if not Path(_PLUGIN_PATH).exists():
        pytest.skip(f"Cardinal bundle not found at {_PLUGIN_PATH}")
    process = multiprocessing.get_context("spawn").Process(
        target=_generate_cardinal_shard,
        args=(str(tmp_path),),
    )

    process.start()
    process.join(timeout=90)

    if process.is_alive():
        process.kill()
        pytest.fail("Cardinal generation subprocess did not finish within 90 seconds")
    assert process.exitcode == 0


@pytest.mark.requires_vst
@pytest.mark.slow
def test_cardinal_repeated_parameters_render_identical_audio() -> None:
    """Per-render reload makes Cardinal reproducible despite its free-running Rack engine."""
    if not Path(_PLUGIN_PATH).exists():
        pytest.skip(f"Cardinal bundle not found at {_PLUGIN_PATH}")
    renderer = DawDreamerRenderer(
        plugin_path=_PLUGIN_PATH,
        sample_rate=_SAMPLE_RATE,
        channels=2,
        signal_duration_seconds=2.0,
        plugin_state_path=plugin_state_paths["cardinal"],
        parameter_map=_cardinal_param_map(),
        reload_plugin_each_render=True,
    )

    first = renderer.render(_BASE_PARAMS, 60, 100, _NOTE_WINDOW)
    renderer.render({**_BASE_PARAMS, "parameter_7_v": 0.35}, 60, 100, _NOTE_WINDOW)
    repeated = renderer.render(_BASE_PARAMS, 60, 100, _NOTE_WINDOW)

    np.testing.assert_array_equal(first, repeated)


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Cardinal applies restored preset state on its audio thread, so host writes "
    "issued before the first processBlock are dropped (#2543)",
)
def test_cardinal_accepts_parameters_written_before_its_preset_settles() -> None:
    """Host writes land without a settle render — xpass means the settle can be removed."""
    if not Path(_PLUGIN_PATH).exists():
        pytest.skip(f"Cardinal bundle not found at {_PLUGIN_PATH}")
    daw = pytest.importorskip("dawdreamer")
    indices = _cardinal_param_map().dawdreamer_indices()

    def render(vca_level: float) -> np.ndarray:
        engine = daw.RenderEngine(_SAMPLE_RATE, 512)
        plugin = engine.make_plugin_processor("synth", _PLUGIN_PATH)
        plugin.load_vst3_preset(plugin_state_paths["cardinal"])
        engine.load_graph([(plugin, [])])
        for name, value in {**_BASE_PARAMS, "parameter_7_v": vca_level}.items():
            plugin.set_parameter(indices[name], value)
        plugin.add_midi_note(60, 100, _NOTE_WINDOW[0], _NOTE_WINDOW[1] - _NOTE_WINDOW[0])
        engine.render(2.0)
        return np.asarray(engine.get_audio())

    assert _rms(render(1.0)) > _rms(render(0.2)) * 2


def _rms(audio: np.ndarray) -> float:
    """Reduce finite stereo audio to one amplitude value.

    :param audio: Channel-first rendered audio.
    :returns: Root-mean-square amplitude across channels and samples.
    """
    return float(np.sqrt(np.mean(np.square(audio))))
