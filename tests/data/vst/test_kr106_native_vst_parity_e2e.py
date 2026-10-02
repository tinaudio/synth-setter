"""Real KR106 native/VST compatibility through the production Lance writer."""

import json
from collections.abc import Callable
from pathlib import Path

import lance
import numpy as np
import pytest
from hydra import compose, initialize_config_module

from synth_setter.data.vst.param_spec_registry import resolve_param_spec
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

pytestmark = [pytest.mark.slow, pytest.mark.requires_vst]


def _config(backend: str) -> RenderConfig:
    """Compose a real native or hosted configuration for the same KR106 labels.

    :param backend: Renderer under comparison.
    :returns: Validated four-second, three-row rendering configuration.
    """
    experiment = (
        "ultramaster-kr106-native-lance-smoke"
        if backend == "kr106_native"
        else "ultramaster-kr106-lance-smoke"
    )
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                f"experiment=generate_dataset/{experiment}",
                f"render.renderer_backend={backend}",
                "render.samples_per_shard=3",
                "render.samples_per_render_batch=1",
                "render.gui_toggle_cadence=never",
            ],
        )
    return RenderConfig.from_cfg_nodes(cfg.render, cfg.synth)


def _column(path: Path, field: str) -> np.ndarray:
    """Read a persisted tensor column through the actual Lance reader.

    :param path: Dataset path.
    :param field: Tensor field name.
    :returns: Stacked column values.
    """
    return (
        lance.dataset(str(path))
        .to_table(columns=[field])
        .column(field)
        .combine_chunks()
        .to_numpy_ndarray()
    )


@pytest.fixture(scope="module")
def rendered_rows(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Render identical seeded A/B/A parameter rows through all three backends.

    :param tmp_path_factory: Temporary dataset directory factory.
    :returns: Persisted audio and parameter columns indexed by backend.
    """
    root = tmp_path_factory.mktemp("kr106-parity")
    spec = resolve_param_spec(ParamSpecName("ultramaster_kr106"))
    sampled, _ = spec.sample(np.random.default_rng(106))
    first = {**sampled, "master_volume": 1.0}
    # Pair the existing seeded J60 parity fixture with an unmodulated J106 saw.
    second = {
        **first,
        "adsr_mode": 1.0,
        "attack": 0.0,
        "bender": 0.5,
        "chorus_i": 0.0,
        "chorus_ii": 0.0,
        "chorus_off": 1.0,
        "decay": 0.0,
        "dco_lfo": 0.0,
        "dco_noise": 0.0,
        "dco_sub": 0.0,
        "hpf": 1.0 / 3.0,
        "master_volume": 0.35,
        "octave": 0.5,
        "pulse": 0.0,
        "release": 0.2,
        "saw": 1.0,
        "sub_sw": 0.0,
        "sustain": 1.0,
        "transpose_offset": 0.4,
        "tuning": 0.5,
        "vca_mode": 0.0,
        "vcf_env": 0.0,
        "vcf_freq": 0.7,
        "vcf_kbd": 0.0,
        "vcf_lfo": 0.0,
        "vcf_res": 0.0,
    }
    patches = [first, second, first]
    notes = [{"pitch": 60, "note_start_and_end": (0.1, 1.5)}] * 3
    result = {}
    for backend in ("kr106_native", "dawdreamer", "pedalboard"):
        path = root / f"{backend}.lance"
        make_lance_dataset(
            path,
            _config(backend),
            fixed_synth_params_list=patches,
            fixed_note_params_list=notes,
        )
        result[backend] = (
            _column(path, AUDIO_FIELD).astype(np.float32),
            _column(path, PARAM_ARRAY_FIELD),
        )
    return result


@pytest.mark.parametrize("backend", ["kr106_native", "dawdreamer", "pedalboard"])
def test_kr106_backend_persists_valid_rows(
    rendered_rows: dict[str, tuple[np.ndarray, np.ndarray]], backend: str
) -> None:
    """Actual Lance rows are finite, audible, bounded, and preserve repeated labels.

    :param rendered_rows: Real persisted outputs from all backends.
    :param backend: Backend whose A/B/A outputs are checked.
    """
    audio, params = rendered_rows[backend]
    assert audio.shape == (3, 2, 176400)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(np.abs(audio[0])) > 1e-4
    assert not np.array_equal(audio[0], audio[1])
    np.testing.assert_array_equal(params[0], params[2])


def test_kr106_native_persisted_audio_is_state_isolated(
    rendered_rows: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Native A/B/A renders produce identical persisted samples despite the intervening patch.

    :param rendered_rows: Real persisted outputs from all backends.
    """
    audio, _ = rendered_rows["kr106_native"]
    np.testing.assert_array_equal(audio[0], audio[2])


@pytest.mark.parametrize("host", ["dawdreamer", "pedalboard"])
@pytest.mark.parametrize("row", [0, 1])
def test_kr106_native_matches_host_audio_and_labels(
    rendered_rows: dict[str, tuple[np.ndarray, np.ndarray]],
    host: str,
    row: int,
    record_property: Callable[[str, object], None],
) -> None:
    """Native rendering meets the existing KR106 cross-host perceptual limits per row.

    :param rendered_rows: Real persisted outputs from all backends.
    :param host: Independent VST host reference.
    :param row: Distinct patch index.
    :param record_property: Persists measured values in the CI JUnit artifact.
    """
    native_audio, native_params = rendered_rows["kr106_native"]
    host_audio, host_params = rendered_rows[host]
    np.testing.assert_array_equal(native_params, host_params)
    metrics = {
        "mss": compute_mss(native_audio[row], host_audio[row]),
        "rms": compute_rms(native_audio[row], host_audio[row]),
        # SOT normalizes each bin over time; idle noise otherwise dominates quiet tails.
        "sot": compute_sot(native_audio[row, :, 4410:66150], host_audio[row, :, 4410:66150]),
        "full_clip_sot": compute_sot(native_audio[row], host_audio[row]),
        "wmfcc": compute_wmfcc(native_audio[row], host_audio[row]),
    }
    record_property("audio_metrics", json.dumps({key: float(value) for key, value in metrics.items()}))
    assert metrics["mss"] < 3.0, metrics
    assert metrics["rms"] > 0.95, metrics
    assert metrics["sot"] < 0.01, metrics
    assert metrics["wmfcc"] < 4.0, metrics
