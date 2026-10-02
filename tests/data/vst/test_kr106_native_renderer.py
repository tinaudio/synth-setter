"""Native KR106 identity, dispatch, isolation, and audio contracts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synth_setter.data.vst.kr106_native_renderer import KR106NativeRenderer
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
