"""Production-process Faust dataset generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import lance
import numpy as np
import pytest
from pydantic_settings import CliApp

from synth_setter.data.vst import generate_vst_dataset
from synth_setter.data.vst.shapes import AUDIO_FIELD, PARAM_ARRAY_FIELD
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.synth_spec import SynthName, SynthSpec
from tests._vst import VST_SUBPROCESS_TIMEOUT_SECONDS


@pytest.mark.slow
def test_faust_generate_cli_writes_real_lance_row(tmp_path: Path) -> None:
    """The production renderer subprocess writes audible audio and exact-width params.

    :param tmp_path: Isolated Lance shard destination.
    """
    shard = tmp_path / "faust.lance"
    config = RenderConfig(
        synth=SynthSpec(
            name=SynthName("faust_bright_organ"),
            param_spec_name=ParamSpecName("faust_bright_organ"),
            plugin_path="faust",
            plugin_state_path="",
            synth_version="0.8.3",
        ),
        renderer_backend="dawdreamer_faust",
        sample_rate=44100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        attempts_per_sample=5,
        base_seed=1808,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    script = Path(generate_vst_dataset.__file__)

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script), str(shard), *CliApp.serialize(config)],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=VST_SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr
    table = lance.dataset(str(shard)).to_table(columns=[AUDIO_FIELD, PARAM_ARRAY_FIELD])
    audio = table.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()[0]
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()[0]
    assert table.num_rows == 1
    assert audio.shape == (2, int(config.sample_rate * config.signal_duration_seconds))
    assert audio.dtype == np.float16
    assert params.shape == (13,)
    assert params.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.isfinite(params).all()
    assert np.all((params >= 0.0) & (params <= 1.0))
    assert float(np.max(np.abs(audio))) > 1e-4
    assert float(np.max(np.abs(audio))) <= 1.0
