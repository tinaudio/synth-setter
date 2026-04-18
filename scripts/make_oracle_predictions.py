"""Produce eval.py predict-mode output directly from an h5 shard.

Writes pred-0.pt, target-audio-0.pt, and target-params-0.pt into out_dir,
using target params as the "prediction" (identity oracle). Useful as a floor
for the audio-metric pipeline: any non-zero distance between pred.wav and
target.wav after scripts/predict_vst_audio.py reflects renderer
non-determinism and the encode/decode round-trip, not model error.

Example:
    python scripts/make_oracle_predictions.py shard-201.h5 oracle-predictions/
"""

from pathlib import Path
from typing import cast

import click
import h5py
import hdf5plugin  # noqa: F401  # registers the Blosc2 codec used by the shards
import torch


def write_oracle_predictions(h5_path: Path, out_dir: Path) -> int:
    """Write single-batch pred / target-audio / target-params .pt files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        # Single batch containing the whole shard. predict_vst_audio.py numbers
        # outputs as sample_{k} via a cumulative offset over pred-*.pt in
        # Path.glob order — which is alphabetical, so pred-10.pt sorts before
        # pred-2.pt. One file is the only layout that keeps sample_{k} aligned
        # with h5 row k so metrics.csv rows can be joined back to the shard.
        audio_ds = cast(h5py.Dataset, f["audio"])
        params_ds = cast(h5py.Dataset, f["param_array"])
        audio = torch.from_numpy(audio_ds[:]).to(torch.float32)
        params = torch.from_numpy(params_ds[:]).to(torch.float32)

    # h5 stores params in [0, 1]; SurgeXTDataset rescales to [-1, 1] before the
    # model sees them, and PredictionWriter dumps both pred-*.pt and
    # target-params-*.pt in the rescaled space. predict_vst_audio.py inverts
    # this with (x + 1) / 2 before param_spec.decode, so we must match here.
    params = params * 2 - 1

    torch.save(audio, out_dir / "target-audio-0.pt")
    torch.save(params, out_dir / "target-params-0.pt")
    # Oracle: pred is literally the target params. Replaces the model output.
    torch.save(params, out_dir / "pred-0.pt")

    return audio.shape[0]


@click.command()
@click.argument("h5_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
def main(h5_path: Path, out_dir: Path) -> None:
    """Write oracle pred/target tensors for an h5 shard into ``out_dir``."""
    num_rows = write_oracle_predictions(h5_path, out_dir)
    click.echo(f"Wrote {num_rows} rows to {out_dir}")


if __name__ == "__main__":
    main()
