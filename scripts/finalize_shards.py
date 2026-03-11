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
"""

import subprocess  # nosec B404
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import click
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.reshard_data_dynamic_shard import reshard_split
from src.data.uploader import DatasetUploader, RcloneUploader

_STATS_SCRIPT = "scripts/get_dataset_stats.py"


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
        reshard_split(split_files, out_path)

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
def main(
    r2_prefix: str,
    r2_bucket: str | None,
    output_dir: str,
    val_shards: int,
    test_shards: int,
    dry_run_upload: bool,
    skip_upload: bool,
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
    )


if __name__ == "__main__":
    main()
