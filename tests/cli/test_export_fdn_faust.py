"""Production-path tests for fixed-value BasicFDN Faust export."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from dawdreamer.dawdreamer import RenderEngine
from pyFDN import FDNBuild, save_fdn_build

from synth_setter.cli.export_fdn_faust import main
from synth_setter.data.basic_fdn import BasicFDN

_SAMPLE_COUNT = 512
_BLOCK_SIZE = 128


def _mimo_build() -> FDNBuild:
    """Return a stable MIMO build with distinct filters at every hook.

    :returns: Complete two-input, three-output FDN build.
    """
    return FDNBuild(
        A=np.array([[0.2, 0.1], [-0.1, 0.2]]),
        B=np.array([[1.0, 0.3], [0.2, 1.0]]),
        C=np.array([[1.0, 0.4], [0.1, 1.0], [0.5, -0.2]]),
        D=np.array([[0.1, 0.0], [0.0, 0.2], [0.1, -0.1]]),
        delays=np.array([17, 29]),
        fs=48_000.0,
        post_delay=np.array(
            [
                [
                    [0.8, 0.7],
                    [0.1, 0.05],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [-0.1, -0.2],
                    [0.0, 0.0],
                ]
            ]
        ),
        post_matrix=np.array(
            [
                [
                    [0.6, 0.9],
                    [0.05, 0.02],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [-0.15, -0.1],
                    [0.0, 0.0],
                ]
            ]
        ),
        post_output=np.array(
            [
                [
                    [1.1, 0.9, 0.8],
                    [0.05, 0.1, 0.02],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [-0.1, -0.2, -0.15],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ),
    )


def _render_faust_transfer(source: str, build: FDNBuild, input_index: int) -> np.ndarray:
    """Render one real DawDreamer impulse through a selected input channel.

    :param source: Generated fixed-value Faust source.
    :param build: FDN geometry and sample rate for the render.
    :param input_index: Input channel receiving the impulse.
    :returns: Response shaped ``(samples, outputs)``.
    """
    impulse = np.zeros((build.B.shape[1], _SAMPLE_COUNT), dtype=np.float32)
    impulse[input_index, 0] = 1.0
    engine = RenderEngine(build.fs, _BLOCK_SIZE)
    playback = engine.make_playback_processor("impulse", impulse)
    faust = engine.make_faust_processor("fdn")
    faust.num_voices = 0
    assert faust.set_dsp_string(source)
    assert faust.compile()
    assert faust.get_parameters_description() == []
    assert engine.load_graph([(playback, []), (faust, [playback.get_name()])])
    assert engine.render(_SAMPLE_COUNT / build.fs)
    return engine.get_audio().T


@pytest.mark.parametrize("input_index", [0, 1])
def test_export_fdn_faust_mimo_build_matches_each_reference_transfer(
    tmp_path: Path, input_index: int
) -> None:
    """The public CLI artifact preserves direct, feedback, delay, and SOS paths.

    :param tmp_path: Per-test directory for the build and generated Faust source.
    :param input_index: MIMO input whose complete output transfer is rendered.
    """
    build = _mimo_build()
    input_path = tmp_path / "fdn.json"
    output_path = tmp_path / "fdn.dsp"
    save_fdn_build(input_path, build)

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code == 0, result.output
    actual = _render_faust_transfer(output_path.read_text(), build, input_index)
    expected = BasicFDN(build).impulse_response(_SAMPLE_COUNT)[:, :, input_index]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_export_fdn_faust_string_sample_rate_rejected_without_output(tmp_path: Path) -> None:
    """Strict input validation rejects coercion before publishing an artifact.

    :param tmp_path: Per-test directory for malformed input and absent output.
    """
    input_path = tmp_path / "fdn.json"
    output_path = tmp_path / "fdn.dsp"
    save_fdn_build(input_path, _mimo_build())
    malformed = input_path.read_text().replace('"sample_rate": 48000.0', '"sample_rate": "48000"')
    input_path.write_text(malformed)

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code != 0
    assert "sample_rate" in result.output
    assert not output_path.exists()


def test_export_fdn_faust_existing_output_refuses_overwrite(tmp_path: Path) -> None:
    """The safe default preserves an existing destination byte for byte.

    :param tmp_path: Per-test directory for input and occupied output.
    """
    input_path = tmp_path / "fdn.json"
    output_path = tmp_path / "fdn.dsp"
    save_fdn_build(input_path, _mimo_build())
    output_path.write_bytes(b"existing")

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code != 0
    assert "overwrite" in result.output
    assert output_path.read_bytes() == b"existing"
