"""Post-processing for distributed shard generation.

After distributed workers upload shards to R2, this script downloads them,
reshards into train/val/test virtual datasets, computes normalization stats,
and uploads the finalized dataset back to R2.

Usage:
    python scripts/finalize_shards.py \\
        --r2-prefix runs/surge_simple/batch42 \\
        --r2-bucket my-bucket \\
        --output-dir data/surge_simple

    # Skip re-upload (just download + reshard + stats locally)
    python scripts/finalize_shards.py \\
        --r2-prefix runs/surge_simple/batch42 \\
        --r2-bucket my-bucket \\
        --output-dir data/surge_simple \\
        --skip-upload

    # Use dynamic resharding (reads per-shard sizes from HDF5 metadata)
    python scripts/finalize_shards.py \\
        --r2-prefix runs/surge_simple/batch42 \\
        --r2-bucket my-bucket \\
        --output-dir data/surge_simple \\
        --dynamic-reshard
"""

import subprocess  # nosec B404
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import click
import h5py
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.reshard_data_dynamic_shard import reshard_split as reshard_split_dynamic
from src.data.uploader import DatasetUploader, RcloneUploader

_STATS_SCRIPT = "scripts/get_dataset_stats.py"


def reshard_split_fixed(
    shard_files: list[Path],
    output_file: Path,
    shard_size: int = 10_000,
) -> None:
    """Create a virtual HDF5 dataset assuming fixed-size shards.

    Unlike the dynamic variant, this does NOT open each shard to read its size.
    It assumes every shard has exactly ``shard_size`` samples, which avoids
    the O(N) memory growth from HDF5 metadata accumulation when opening many
    shards (see #235).

    Args:
        shard_files: Ordered list of shard HDF5 file paths. Each must contain
            datasets named 'audio', 'mel_spec', and 'param_array'.
        output_file: Path for the output virtual HDF5 file.
        shard_size: Number of samples per shard (default 10 000).

    Raises:
        ValueError: If shard_files is empty.
    """
    if not shard_files:
        raise ValueError("No shard files provided")

    output_dir = output_file.parent
    split_len = len(shard_files) * shard_size

    # Read shapes from the first shard only.
    with h5py.File(shard_files[0], "r") as f:
        audio_shape = f["audio"].shape[1:]
        mel_shape = f["mel_spec"].shape[1:]
        param_shape = f["param_array"].shape[1:]

    vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
    vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
    vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

    for i, file in enumerate(shard_files):
        # Use path relative to the output file's directory for portability.
        rel_path = file.resolve().relative_to(output_dir.resolve())
        rel_str = str(rel_path)

        vs_audio = h5py.VirtualSource(
            rel_str, "audio", dtype=np.float32, shape=(shard_size, *audio_shape)
        )
        vs_mel = h5py.VirtualSource(
            rel_str, "mel_spec", dtype=np.float32, shape=(shard_size, *mel_shape)
        )
        vs_param = h5py.VirtualSource(
            rel_str, "param_array", dtype=np.float32, shape=(shard_size, *param_shape)
        )

        range_start = i * shard_size
        range_end = (i + 1) * shard_size

        vl_audio[range_start:range_end] = vs_audio
        vl_mel[range_start:range_end] = vs_mel
        vl_param[range_start:range_end] = vs_param

    with h5py.File(output_file, "w") as f:
        f.create_virtual_dataset("audio", vl_audio)
        f.create_virtual_dataset("mel_spec", vl_mel)
        f.create_virtual_dataset("param_array", vl_param)


def _rclone_download(
    remote_path: str,
    local_dir: Path,
    bucket: str,
    rclone_remote: str = "r2",
) -> None:
    """Download files from an rclone remote to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    source = f"{rclone_remote}:{bucket}/{remote_path}"
    cmd = [
        "rclone",
        "copy",
        source,
        str(local_dir),
        "--progress",
        "--checksum",
        "--transfers",
        "200",
        "--checkers",
        "200",
    ]
    subprocess.run(cmd, check=True)  # nosec B603


def finalize_shards(
    output_dir: Path,
    download_fn: Callable[[str, Path], None],
    r2_prefix: str,
    val_shards: int = 1,
    test_shards: int = 1,
    uploader: DatasetUploader | None = None,
    dynamic_reshard: bool = False,
    shard_size: int = 10_000,
) -> Path:
    """Download shards from R2, reshard into splits, compute stats, and upload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download shards
    shard_dir = output_dir / "shards"
    print(f"[download] {r2_prefix}/shards -> {shard_dir}", flush=True)
    download_fn(f"{r2_prefix}/shards", shard_dir)

    # 2. Validate
    files = sorted(shard_dir.glob("shard-*.h5"))
    if not files:
        print(f"ERROR: No shard-*.h5 files found in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    train_shards = len(files) - val_shards - test_shards
    if train_shards < 0:
        print(
            f"ERROR: not enough shards for val({val_shards}) + test({test_shards}) "
            f"— only {len(files)} shards available.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Reshard
    if dynamic_reshard:
        reshard_fn = reshard_split_dynamic
        mode_label = "dynamic"
    else:
        reshard_fn = partial(reshard_split_fixed, shard_size=shard_size)
        mode_label = f"fixed (shard_size={shard_size})"
    print(f"[reshard] mode: {mode_label}", flush=True)

    splits = {
        "train": files[:train_shards],
        "val": files[train_shards : train_shards + val_shards],
        "test": files[train_shards + val_shards :],
    }
    for split_name, split_files in splits.items():
        if not split_files:
            print(f"[reshard] skipping {split_name} (0 shards)")
            continue
        out_path = output_dir / f"{split_name}.h5"
        print(f"[reshard] {split_name}: {len(split_files)} shards -> {out_path}", flush=True)
        reshard_fn(split_files, out_path)

    # 4. Compute stats
    train_h5 = output_dir / "train.h5"
    print(f"[stats] computing normalization stats from {train_h5}", flush=True)
    subprocess.run(["python", _STATS_SCRIPT, str(train_h5)], check=True)  # nosec B603 B607

    # 5. Upload
    if uploader is not None:
        print(f"[upload] uploading {output_dir} -> {r2_prefix}", flush=True)
        uploader.upload(output_dir, r2_prefix)
        print("[upload] done", flush=True)
    else:
        print("[upload] skipped (no uploader configured)", flush=True)

    return output_dir


@click.command()
@click.option("--r2-prefix", required=True, help="R2 prefix (e.g. 'runs/surge_simple/batch42').")
@click.option(
    "--r2-bucket", default=None, envvar="R2_BUCKET", help="R2 bucket name (or R2_BUCKET env)."
)
@click.option("--output-dir", "-o", type=str, required=True, help="Local output directory.")
@click.option(
    "--val-shards", "-v", type=int, default=1, show_default=True, help="Validation shard count."
)
@click.option(
    "--test-shards", "-e", type=int, default=1, show_default=True, help="Test shard count."
)
@click.option(
    "--dry-run-upload", is_flag=True, default=False, help="Pass --dry-run to rclone upload."
)
@click.option("--skip-upload", is_flag=True, default=False, help="Skip uploading results to R2.")
@click.option(
    "--dynamic-reshard",
    is_flag=True,
    default=False,
    help="Use dynamic resharding (reads per-shard HDF5 sizes). Default: fixed-size.",
)
@click.option(
    "--shard-size",
    type=int,
    default=10_000,
    show_default=True,
    help="Samples per shard for fixed-size resharding (ignored with --dynamic-reshard).",
)
def main(
    r2_prefix: str,
    r2_bucket: str | None,
    output_dir: str,
    val_shards: int,
    test_shards: int,
    dry_run_upload: bool,
    skip_upload: bool,
    dynamic_reshard: bool,
    shard_size: int,
) -> None:
    """Download shards from R2, reshard, compute stats, and upload results."""
    if not r2_bucket:
        print(
            "ERROR: No R2 bucket configured. Pass --r2-bucket or set R2_BUCKET env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    download_fn = partial(_rclone_download, bucket=r2_bucket)

    uploader: DatasetUploader | None = None
    if not skip_upload:
        uploader = RcloneUploader(bucket=r2_bucket, dry_run=dry_run_upload)

    finalize_shards(
        output_dir=Path(output_dir),
        download_fn=download_fn,
        r2_prefix=r2_prefix,
        val_shards=val_shards,
        test_shards=test_shards,
        uploader=uploader,
        dynamic_reshard=dynamic_reshard,
        shard_size=shard_size,
    )


if __name__ == "__main__":
    main()
