"""Production-process Faust dataset generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat, RenderConfig
from synth_setter.synth_spec import SYNTHS, SynthName
from tests._vst import VST_SUBPROCESS_TIMEOUT_SECONDS


@pytest.mark.slow
def test_faust_generate_cli_writes_real_lance_row(tmp_path: Path) -> None:
    """The production renderer subprocess writes audible audio and exact-width params.

    :param tmp_path: Isolated Lance shard destination.
    """
    config = RenderConfig(
        synth=SYNTHS[SynthName("faust_bright_organ")],
        renderer_backend="dawdreamer",
        backend_version="0.8.3",
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
    spec = DatasetSpec(
        task_name="faust-e2e",
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
    dataset = lance.dataset(str(shard))
    assert dataset.schema.names == [
        "audio",
        "mel_spec",
        "param_array",
        "debug",
        "audio_mp3",
        "audio_uuid",
    ]
    table = dataset.to_table(columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD])
    audio = table.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()[0]
    mel_spec = table.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()[0]
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()[0]
    assert table.num_rows == 1
    assert audio.shape == (2, int(config.sample_rate * config.signal_duration_seconds))
    assert audio.dtype == np.float16
    assert mel_spec.shape == (2, 128, 401)
    assert mel_spec.dtype == np.float32
    assert params.shape == (13,)
    assert params.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.isfinite(mel_spec).all()
    assert np.isfinite(params).all()
    assert np.all((params >= 0.0) & (params <= 1.0))
    assert float(np.max(np.abs(audio))) > 1e-4
    assert float(np.max(np.abs(audio))) <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("identity", "channels", "param_width", "base_seed"),
    [
        ("faust_bright_organ", 2, 13, 1808),
        ("faust_syrinx_bird", 1, 59, 1808),
        # Syrinx2 onset is sparse, so this draws a production seed whose row passes.
        ("faust_syrinx2_bird", 1, 77, 3),
        ("faust_tract3_bird", 1, 103, 1808),
    ],
)
def test_faustwasm_generate_cli_writes_real_lance_row(
    tmp_path: Path,
    identity: str,
    channels: int,
    param_width: int,
    base_seed: int,
) -> None:
    """The production CLI drives Python through Node into a consumable Lance row.

    :param tmp_path: Isolated Lance shard destination.
    :param identity: Registered Faust source identity.
    :param channels: Native source output channels.
    :param param_width: Encoded synth-and-note parameter width.
    :param base_seed: Master seed for the sampled production-path row.
    """
    config = RenderConfig(
        synth=SYNTHS[SynthName(identity)],
        renderer_backend="faustwasm",
        backend_version="0.18.3",
        block_size=64,
        render_contract_version=2,
        sample_rate=44100,
        channels=channels,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-100.0,
        samples_per_render_batch=1,
        samples_per_shard=1,
        attempts_per_sample=5,
        base_seed=base_seed,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )
    spec = DatasetSpec(
        task_name="faustwasm-e2e",
        output_format=OutputFormat.LANCE,
        train_val_test_sizes=(1, 0, 0),
        base_seed=config.base_seed,
        r2={"bucket": "unused"},  # type: ignore[arg-type]
        render=config,
    )
    args = build_generate_args(spec, spec.shards[0], tmp_path)
    shard = Path(args[2])
    assert args[args.index("--block_size") + 1] == "64"

    result = subprocess.run(  # noqa: S603
        args,
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=VST_SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr
    dataset = lance.dataset(str(shard))
    expected_schema = pa.schema(
        [
            pa.field(
                "audio",
                pa.fixed_shape_tensor(pa.float16(), (channels, 176_400)),
                nullable=False,
            ),
            pa.field(
                "mel_spec",
                pa.fixed_shape_tensor(pa.float32(), (channels, 128, 401)),
                nullable=False,
            ),
            pa.field(
                "param_array",
                pa.fixed_shape_tensor(pa.float32(), (param_width,)),
                nullable=False,
            ),
            pa.field("debug", pa.json_(), nullable=False),
            pa.field(
                "audio_mp3",
                pa.binary(),
                nullable=False,
                metadata={b"mime_type": b"audio/mpeg"},
            ),
            pa.field("audio_uuid", pa.string(), nullable=False),
        ]
    )
    assert dataset.schema.remove_metadata() == expected_schema

    table = dataset.to_table(columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD])
    audio = table.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()[0]
    mel_spec = table.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()[0]
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()[0]
    assert table.num_rows == 1
    assert audio.shape == (channels, 176_400)
    assert audio.dtype == np.float16
    assert mel_spec.shape == (channels, 128, 401)
    assert mel_spec.dtype == np.float32
    assert params.shape == (param_width,)
    assert params.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.isfinite(mel_spec).all()
    assert np.isfinite(params).all()
    assert np.all((params >= 0.0) & (params <= 1.0))
    assert float(np.max(np.abs(audio))) > 1e-4
    assert float(np.max(np.abs(audio))) <= 1.0
