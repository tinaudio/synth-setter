"""Production CLI to Lance coverage for the Faust super-shimmer FDN."""

from __future__ import annotations

import subprocess
from pathlib import Path

import lance
import numpy as np
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat, RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName
from tests._vst import VST_SUBPROCESS_TIMEOUT_SECONDS


@pytest.mark.slow
def test_faust_super_shimmer_fdn_generate_cli_writes_consumable_lance_row(
    tmp_path: Path,
) -> None:
    """The real CLI compiles the impulse source and persists its rendered row.

    :param tmp_path: Isolated Lance shard destination.
    """
    config = RenderConfig(
        synth=SYNTHS[SynthName("faust_super_shimmer_fdn")],
        renderer_backend="dawdreamer",
        backend_version="0.8.3",
        sample_rate=44_100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-70.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        attempts_per_sample=5,
        base_seed=1816,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    spec = DatasetSpec(
        task_name="faust-super-shimmer-fdn-e2e",
        output_format=OutputFormat.LANCE,
        train_val_test_sizes=(1, 0, 0),
        base_seed=config.base_seed,
        r2={"bucket": "unused"},  # type: ignore[arg-type]
        render=config,
    )
    args = build_generate_args(spec, spec.shards[0], tmp_path)
    shard = Path(args[2])

    result = subprocess.run(  # noqa: S603
        args,
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=VST_SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr
    table = lance.dataset(str(shard)).to_table(
        columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD]
    )
    audio = table.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()[0]
    mel_spec = table.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()[0]
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()[0]
    assert table.num_rows == 1
    assert audio.shape == (2, 176_400)
    assert audio.dtype == np.float16
    assert mel_spec.shape == (2, 128, 401)
    assert mel_spec.dtype == np.float32
    assert params.shape == (32,)
    assert params.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.isfinite(mel_spec).all()
    assert np.isfinite(params).all()
    assert np.all((params >= 0.0) & (params <= 1.0))
    assert float(np.max(np.abs(audio[:, 6_000:]))) > 1e-5
    assert float(np.max(np.abs(audio))) <= 1.0
