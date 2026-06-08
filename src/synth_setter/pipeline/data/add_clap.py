"""Inject LAION CLAP audio embeddings into finalized dataset split files.

Post-finalization producer: reads each split file's ``audio`` dataset, downmixes
to mono, runs a CLAP audio encoder, and writes a real ``clap`` dataset ``(N, D)``
into the same file. Mirrors the music2latent injection but targets the split
files (``train/val/test.h5``) the datamodule opens directly — the resharder's
virtual layout only carries the core fields, so a side embedding must live in
the split file itself.

The HDF5 plumbing (:func:`add_clap_embeddings`) takes an injected ``encode``
callable so it is testable without loading CLAP; :func:`load_clap_audio_encoder`
is the thin shell that builds the real encoder (lazy ``transformers`` import).

Run as ``python -m synth_setter.pipeline.data.add_clap <data_dir>``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import click
import h5py
import hdf5plugin
import numpy as np
import structlog

from synth_setter.data.vst.shapes import AUDIO_FIELD, CLAP_EMBEDDING_DIM, CLAP_FIELD

logger = structlog.get_logger(__name__)

DEFAULT_CLAP_CHECKPOINT = "laion/clap-htsat-unfused"
# Default project render rate, used when a shard lacks a sample_rate attr.
DEFAULT_SAMPLE_RATE = 44100
# CLAP's feature extractor rejects any other input rate, so audio is resampled to
# this before encoding.
CLAP_SAMPLE_RATE = 48000
# Boolean attr set on the output dataset only after every row is written, so a
# crash mid-write leaves the field present-but-incomplete and a rerun recomputes
# it instead of trusting a partially-zero dataset.
COMPLETE_ATTR = "clap_complete"
_SPLIT_FILENAMES = ("train.h5", "val.h5", "test.h5")

# Maps a mono ``(B, T)`` batch and its sample rate to a ``(B, D)`` embedding batch.
EncodeFn = Callable[[np.ndarray, int], np.ndarray]


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average the channel axis of an ``(B, C, T)`` batch into ``(B, T)`` mono float32.

    :param audio: Audio batch shaped ``(B, C, T)``.
    :returns: Mono batch shaped ``(B, T)`` as float32.
    """
    return audio.mean(axis=1, dtype=np.float32)


def add_clap_embeddings(
    h5_path: str | Path,
    encode: EncodeFn,
    *,
    batch_size: int = 256,
    field: str = CLAP_FIELD,
    expected_dim: int | None = None,
    overwrite: bool = False,
) -> int:
    """Compute and store CLAP embeddings for every row of one HDF5 split file.

    Reads ``audio`` in batches, downmixes to mono, calls ``encode``, and writes a
    real ``field`` dataset ``(N, D)`` whose width ``D`` is taken from the encoder
    output. The ``audio`` dataset may be virtual; it resolves against sibling
    shards. A ``field`` that already carries the completion marker is skipped
    unless ``overwrite`` is set; a present-but-incomplete field (crashed run) is
    recomputed.

    :param h5_path: Split/shard file, opened read-write.
    :param encode: Maps a mono ``(B, T)`` batch and sample rate to a ``(B, D)``
        float32 embedding batch.
    :param batch_size: Rows per encode call.
    :param field: Output dataset name.
    :param expected_dim: When set, the encoder's output width ``D`` is asserted to
        equal it (guards the default checkpoint against a silent width change);
        ``None`` accepts whatever width the encoder produces.
    :param overwrite: Recompute even a completed ``field`` instead of skipping it.
    :returns: Rows encoded; 0 when a completed field is skipped.
    :raises ValueError: If ``expected_dim`` is set and the encoder width differs.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r+") as f:
        if field in f:
            if not overwrite and f[field].attrs.get(COMPLETE_ATTR, False):
                logger.info("clap_field_complete_skipping", path=str(h5_path), field=field)
                return 0
            del f[field]

        audio_ds = cast(h5py.Dataset, f[AUDIO_FIELD])
        num_rows = audio_ds.shape[0]
        sample_rate = int(audio_ds.attrs.get("sample_rate", DEFAULT_SAMPLE_RATE))

        out_ds: h5py.Dataset | None = None
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            # _downmix_to_mono's mean(dtype=float32) promotes the fp16 slice, so no
            # separate full-batch upcast; np.asarray coerces a torch tensor to ndarray.
            mono = _downmix_to_mono(audio_ds[start:end])
            embeddings = np.asarray(encode(mono, sample_rate))
            if out_ds is None:
                width = embeddings.shape[1]
                if expected_dim is not None and width != expected_dim:
                    raise ValueError(
                        f"CLAP encoder produced width {width}, expected {expected_dim}"
                    )
                # Blosc2 to match the core datasets' on-disk compression. Blosc2 is
                # hdf5plugin's documented public API but absent from its stub __all__.
                out_ds = f.create_dataset(
                    field,
                    shape=(num_rows, width),
                    dtype=np.float32,
                    compression=hdf5plugin.Blosc2(),  # pyright: ignore[reportPrivateImportUsage]
                )
            out_ds[start:end] = embeddings
        if out_ds is not None:
            out_ds.attrs[COMPLETE_ATTR] = True
        f.flush()

    logger.info("clap_embeddings_written", path=str(h5_path), rows=num_rows, field=field)
    return num_rows


def load_clap_audio_encoder(
    checkpoint: str = DEFAULT_CLAP_CHECKPOINT,
    device: str | None = None,
) -> EncodeFn:
    """Load a CLAP checkpoint and return a mono-batch encode callable.

    The ``transformers`` and ``torch`` imports are deferred so the HDF5 core stays
    importable (and testable with a fake encoder) without the heavy dependency or
    a model download.

    :param checkpoint: HuggingFace CLAP model id.
    :param device: Torch device string; defaults to cuda when available, else cpu.
    :returns: Encode callable mapping a mono ``(B, T)`` batch and sample rate to a
        ``(B, D)`` float32 embedding batch.
    """
    import torch
    import torchaudio.functional as audio_fn
    from transformers import ClapModel, ClapProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # transformers' processor/model call surface is dynamically typed; Any at this
    # external boundary keeps the type checker honest about what it cannot verify.
    model: Any = ClapModel.from_pretrained(checkpoint)
    model = model.to(device).eval()
    processor: Any = ClapProcessor.from_pretrained(checkpoint)

    @torch.no_grad()
    def encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(mono))
        if sample_rate != CLAP_SAMPLE_RATE:
            wav = audio_fn.resample(wav, sample_rate, CLAP_SAMPLE_RATE)
        inputs = processor(
            audio=list(wav.numpy()), sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # get_audio_features returns the audio-tower output whose pooler_output
        # holds the projected, L2-normalized joint-space embedding (B, D).
        features = model.get_audio_features(**inputs)
        return features.pooler_output.cpu().numpy()

    return encode


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--checkpoint", default=DEFAULT_CLAP_CHECKPOINT, show_default=True, help="HuggingFace CLAP id."
)
@click.option(
    "--batch-size", "-b", type=int, default=256, show_default=True, help="Rows per batch."
)
@click.option("--device", default=None, help="Torch device (default: cuda if available else cpu).")
@click.option("--field", default=CLAP_FIELD, show_default=True, help="Output dataset name.")
@click.option(
    "--overwrite", is_flag=True, help="Replace an existing CLAP field instead of skipping."
)
def main(
    data_dir: Path,
    checkpoint: str,
    batch_size: int,
    device: str | None,
    field: str,
    overwrite: bool,
) -> None:
    """Inject CLAP embeddings into each ``train/val/test.h5`` under DATA_DIR.

    :param data_dir: Finalized dataset root holding the split files.
    :param checkpoint: HuggingFace CLAP model id.
    :param batch_size: Rows per encode call.
    :param device: Torch device string, or None to auto-select.
    :param field: Output dataset name.
    :param overwrite: Replace an existing CLAP field instead of skipping.
    :raises click.ClickException: When no split files are found under DATA_DIR.
    """
    split_paths = [data_dir / name for name in _SPLIT_FILENAMES if (data_dir / name).exists()]
    if not split_paths:
        raise click.ClickException(f"No split files {_SPLIT_FILENAMES} found under {data_dir}")

    # Only pin the width for the default checkpoint; alternates may project to a
    # different dimension and must be free to write their own width.
    expected_dim = CLAP_EMBEDDING_DIM if checkpoint == DEFAULT_CLAP_CHECKPOINT else None
    encode = load_clap_audio_encoder(checkpoint, device)
    for path in split_paths:
        add_clap_embeddings(
            path,
            encode,
            batch_size=batch_size,
            field=field,
            expected_dim=expected_dim,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    main()
