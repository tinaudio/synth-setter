"""Replay an h5 shard into a ``compute_audio_metrics.py``-compatible audio pair dir.

The output directory contains:

- ``replayed.h5`` — a fresh shard rendered from the input h5's params via
  ``make_dataset``, with the same ``audio``/``mel_spec``/``param_array``
  schema and attrs as the input.
- ``sample_NNNNNN/{target.wav, pred.wav}`` per row, where ``target.wav`` is
  the input h5's audio (cast float16 → float32) and ``pred.wav`` is the
  replayed audio. The dir feeds straight into
  ``scripts/compute_audio_metrics.py`` for MSS / wMFCC / SOT / RMS metrics.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import click
import h5py
import hdf5plugin  # noqa: F401  side-effect: registers Blosc2 filter for h5py reads
import numpy as np
import rootutils
from loguru import logger
from pedalboard.io import AudioFile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import param_specs, preset_paths  # noqa: E402
from src.data.vst.generate_vst_dataset import (  # noqa: E402
    load_fixed_params_from_h5,
    make_dataset,
)
from src.data.vst.param_spec import ParamSpec  # noqa: E402

_SAMPLE_DIR_FORMAT = "sample_{idx:06d}"
_REPLAYED_H5_FILENAME = "replayed.h5"

_T = TypeVar("_T")


def _read_audio_attr(dataset: h5py.Dataset, name: str, cast_to: Callable[[Any], _T]) -> _T:
    """Read a scalar attr off the audio dataset and cast it to a Python scalar.

    Centralizes the ``# pyright: ignore[reportArgumentType]`` workaround for
    h5py's loosely-typed ``attrs`` mapping in one place.
    """
    return cast_to(dataset.attrs[name])


def _write_pair(
    out_dir: Path,
    idx: int,
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
    channels: int,
) -> Path:
    """Write ``target.wav`` and ``pred.wav`` into ``<out_dir>/sample_<idx>/``.

    ``target`` and ``pred`` are ``(channels, frames)`` float32; transposed to
    ``(frames, channels)`` for ``AudioFile.write``. Returns the sample subdir.
    """
    sample_dir = out_dir / _SAMPLE_DIR_FORMAT.format(idx=idx)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(sample_dir / "target.wav"), "w", sample_rate, channels) as f:
        f.write(target.T)
    with AudioFile(str(sample_dir / "pred.wav"), "w", sample_rate, channels) as f:
        f.write(pred.T)
    return sample_dir


def replay_h5_to_audio_pairs(
    h5_path: Path,
    output_dir: Path,
    plugin_path: str,
    preset_path: str,
    param_spec: ParamSpec,
    num_samples: int | None = None,
) -> int:
    """Replay ``h5_path`` into ``<output_dir>/replayed.h5`` plus per-row WAV pairs.

    Reads ``sample_rate``, ``channels``, ``signal_duration_seconds``,
    ``velocity``, and ``min_loudness`` from the input h5's ``audio`` dataset
    attrs. When ``num_samples`` is given, only the first ``num_samples`` rows
    are replayed; raises ``ValueError`` if it exceeds the h5's row count.
    Returns the number of pairs written.
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)

    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    fixed_synth_list, fixed_note_list = load_fixed_params_from_h5(str(h5_path), param_spec)

    with h5py.File(h5_path, "r") as f:
        audio_dataset = cast(h5py.Dataset, f["audio"])
        sample_rate = _read_audio_attr(audio_dataset, "sample_rate", float)
        channels = _read_audio_attr(audio_dataset, "channels", int)
        signal_duration_seconds = _read_audio_attr(audio_dataset, "signal_duration_seconds", float)
        velocity = _read_audio_attr(audio_dataset, "velocity", int)
        min_loudness = _read_audio_attr(audio_dataset, "min_loudness", float)
        h5_row_count = audio_dataset.shape[0]
        if num_samples is not None and num_samples > h5_row_count:
            raise ValueError(f"num_samples={num_samples} exceeds h5 row count={h5_row_count}")
        rows_to_render = h5_row_count if num_samples is None else num_samples
        target_audio = audio_dataset[:rows_to_render].astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    replayed_h5_path = output_dir / _REPLAYED_H5_FILENAME
    logger.info(f"Replaying {rows_to_render} rows from {h5_path} into {replayed_h5_path}")

    with h5py.File(replayed_h5_path, "w") as f:
        make_dataset(
            hdf5_file=f,
            num_samples=rows_to_render,
            plugin_path=plugin_path,
            preset_path=preset_path,
            sample_rate=sample_rate,
            channels=channels,
            velocity=velocity,
            signal_duration_seconds=signal_duration_seconds,
            min_loudness=min_loudness,
            param_spec=param_spec,
            sample_batch_size=rows_to_render,
            fixed_synth_params_list=fixed_synth_list[:rows_to_render],
            fixed_note_params_list=fixed_note_list[:rows_to_render],
        )

    with h5py.File(replayed_h5_path, "r") as f:
        replayed_audio_dataset = cast(h5py.Dataset, f["audio"])
        pred_audio = replayed_audio_dataset[...].astype(np.float32)

    logger.info(f"Writing {rows_to_render} WAV pairs into {output_dir}")
    for i in range(rows_to_render):
        _write_pair(output_dir, i, target_audio[i], pred_audio[i], sample_rate, channels)

    logger.info(f"Wrote {rows_to_render} pairs to {output_dir}")
    return rows_to_render


@click.command()
@click.argument("h5_path", type=str)
@click.argument("output_dir", type=str)
@click.option("--plugin_path", "-p", type=str, default="plugins/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default=None)
@click.option(
    "--param_spec",
    "-t",
    type=click.Choice(list(param_specs.keys())),
    required=True,
)
@click.option(
    "--num_samples",
    "-n",
    type=int,
    default=None,
    help="Replay only the first N rows of the h5 (default: all rows).",
)
def main(
    h5_path: str,
    output_dir: str,
    plugin_path: str,
    preset_path: str | None,
    param_spec: str,
    num_samples: int | None,
) -> None:
    """Replay H5_PATH into OUTPUT_DIR/{replayed.h5, sample_NNNNNN/{target,pred}.wav}."""
    spec = param_specs[param_spec]
    resolved_preset_path = preset_path if preset_path is not None else preset_paths[param_spec]
    replay_h5_to_audio_pairs(
        h5_path=Path(h5_path),
        output_dir=Path(output_dir),
        plugin_path=plugin_path,
        preset_path=resolved_preset_path,
        param_spec=spec,
        num_samples=num_samples,
    )


if __name__ == "__main__":
    main()
