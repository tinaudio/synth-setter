"""Native KR106 identity, dispatch, isolation, and audio contracts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.kr106_native_renderer import KR106NativeRenderer
from synth_setter.data.vst.kr106_native_runtime import import_kr106_native
from synth_setter.data.vst.param_map import load_param_map


@pytest.fixture
def renderer() -> KR106NativeRenderer:
    """Construct the real native backend with the committed baseline.

    :returns: An isolated renderer for four-second stereo clips.
    """
    root = Path(__file__).resolve().parents[3]
    return KR106NativeRenderer(
        plugin_path="kr106_native",
        sample_rate=44100,
        channels=2,
        signal_duration_seconds=4.0,
        plugin_state_path=str(root / "presets/ultramaster_kr106-native.json"),
        parameter_map=load_param_map(
            root / "src/synth_setter/data/vst/ultramaster_kr106_param_map.json"
        ),
        block_size=512,
    )


def test_relative_baseline_survives_worker_directory_change(
    renderer: KR106NativeRenderer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker output directories do not change which pinned preset gets loaded.

    :param renderer: Real native renderer with an absolute baseline.
    :param tmp_path: Worker output directory.
    :param monkeypatch: Changes the current directory for this test.
    """
    expected = renderer.render({}, 60, 100, (0.1, 1.0))
    monkeypatch.chdir(tmp_path)
    worker = replace(renderer, plugin_state_path="presets/ultramaster_kr106-native.json")
    np.testing.assert_array_equal(worker.render({}, 60, 100, (0.1, 1.0)), expected)


def test_native_render_returns_owned_stereo_audio(renderer: KR106NativeRenderer) -> None:
    """A native clip is finite, audible, channel-first float32 audio.

    :param renderer: Real native renderer.
    """
    audio = renderer.render({}, 60, 100, (0.1, 1.0))
    assert audio.shape == (2, 176400)
    assert audio.dtype == np.float32
    assert audio.flags.c_contiguous
    assert audio.flags.owndata
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.001


def test_native_mono_is_the_stereo_downmix(renderer: KR106NativeRenderer) -> None:
    """Requesting mono changes channel layout without changing synthesis.

    :param renderer: Real stereo native renderer.
    """
    stereo = renderer.render({}, 60, 100, (0.1, 1.0))
    mono = replace(renderer, channels=1).render({}, 60, 100, (0.1, 1.0))
    np.testing.assert_array_equal(mono, stereo.mean(axis=0, keepdims=True))


@pytest.mark.parametrize(
    "overrides",
    [
        {"plugin_path": "wrong_backend"},
        {"plugin_state_path": ""},
        {"channels": 3},
        {"sample_rate": 0},
        {"signal_duration_seconds": float("nan")},
        {"block_size": 0},
    ],
)
def test_native_invalid_configuration_is_rejected(
    renderer: KR106NativeRenderer, overrides: dict[str, object]
) -> None:
    """Invalid renderer configuration fails before synthesis.

    :param renderer: Valid native renderer.
    :param overrides: One invalid configuration field.
    """
    with pytest.raises(ValueError):
        replace(renderer, **overrides)


def test_native_modified_baseline_is_rejected(
    renderer: KR106NativeRenderer, tmp_path: Path
) -> None:
    """Even a valid JSON reserialization must match the pinned baseline digest.

    :param renderer: Native renderer with verified baseline provenance.
    :param tmp_path: Directory for the modified artifact.
    """
    altered = tmp_path / "altered.json"
    altered.write_bytes(Path(renderer.plugin_state_path).read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(renderer, plugin_state_path=str(altered))


def test_native_render_isolates_rows_and_preserves_output(renderer: KR106NativeRenderer) -> None:
    """Rendering another patch cannot change earlier audio or the repeated patch.

    :param renderer: Real native renderer.
    """
    first = renderer.render({"vcf_freq": 0.7}, 60, 100, (0.1, 1.0))
    retained = first.copy()
    other = renderer.render({"vcf_freq": 0.2}, 67, 90, (0.0, 0.5))
    repeated = renderer.render({"vcf_freq": 0.7}, 60, 100, (0.1, 1.0))
    np.testing.assert_array_equal(first, retained)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_native_render_rejects_invalid_normalized_value(
    renderer: KR106NativeRenderer, value: float
) -> None:
    """Invalid normalized controls fail before native rendering.

    :param renderer: Real native renderer.
    :param value: Out-of-contract normalized control.
    """
    with pytest.raises(ValueError, match="finite.*\\[0, 1\\]"):
        renderer.render({"vcf_freq": value}, 60, 100, (0.0, 1.0))


@pytest.mark.parametrize(
    ("start", "end", "start_sample", "end_sample"),
    [
        (13 / 44100, 0.05, 13, 2205),
        (0.0, 26 / 44100, 0, 26),
        (np.nextafter(17 / 44100, np.inf), 0.05, 18, 2205),
        (0.0, np.nextafter(17 / 44100, np.inf), 0, 18),
    ],
)
def test_native_float_event_boundaries_match_integer_sample_render(
    renderer: KR106NativeRenderer, start: float, end: float, start_sample: int, end_sample: int
) -> None:
    """Floating-point endpoints retain exact sample-boundary semantics.

    :param renderer: Real native renderer with the committed baseline.
    :param start: Requested note-on time.
    :param end: Requested note-off time.
    :param start_sample: Correct note-on index independent of float multiplication.
    :param end_sample: Correct note-off index independent of float multiplication.
    """
    short = replace(renderer, signal_duration_seconds=0.1)
    baseline = json.loads(Path(short.plugin_state_path).read_text())["parameters"]
    expected = import_kr106_native().render_note(
        {},
        60,
        100,
        start_sample,
        end_sample,
        4410,
        44100.0,
        512,
        baseline_parameters={int(key): value for key, value in baseline.items()},
    )
    np.testing.assert_array_equal(short.render({}, 60, 100, (start, end)), expected)


def test_native_render_rejects_unmapped_control(renderer: KR106NativeRenderer) -> None:
    """Unknown names never silently disappear from a sampled patch.

    :param renderer: Real native renderer.
    """
    with pytest.raises(KeyError, match="unknown"):
        renderer.render({"not_a_control": 0.5}, 60, 100, (0.0, 1.0))


def test_native_parameter_conversion_preserves_plugin_ranges(
    renderer: KR106NativeRenderer,
) -> None:
    """Normalized categorical and signed controls reach their actual native ranges.

    :param renderer: Real native renderer.
    """
    values = renderer.native_parameter_values(
        {"octave": 0.5, "tuning": 0.75, "voices": 0.5, "master_volume": 0.35}
    )
    assert values == {"octave": 1.0, "tuning": 0.5, "voices": 8.0, "master_volume": 0.35}
