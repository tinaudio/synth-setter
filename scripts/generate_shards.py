"""Generate HDF5 dataset shards for a VST synthesizer.

Produces N shard files in shard_dir/, each with shard_size
samples. A later resharding step (reshard_data_dynamic_shard.py) assigns
shards to train/val/test splits.

Each shard is named shard-{instance_id}-{seq}.h5 where instance_id
identifies the worker that produced it (auto-generated UUID if omitted).

Usage:
    python scripts/generate_shards.py \\
        --num-shards 12 --shard-size 10000 \\
        --output-dir data/surge_simple --param-spec surge_simple

    # With explicit instance ID (for distributed workers)
    python scripts/generate_shards.py \\
        --num-shards 10 --shard-size 10000 \\
        --output-dir data/surge_simple --param-spec surge_simple \\
        --instance-id worker01
"""

import json
import subprocess  # nosec B404
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

from src.data.uploader import DatasetUploader, RcloneUploader

_GENERATE_SCRIPT = "src/data/vst/generate_vst_dataset.py"
_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"
_DEFAULT_PLUGIN_PATH = "plugins/Surge XT.vst3"
_DEFAULT_PRESET_PATH = "presets/surge-base.vstpreset"


def generate_shards(
    shard_dir: Path,
    num_shards: int,
    shard_size: int,
    param_spec: str,
    instance_id: str | None = None,
    plugin_path: str = _DEFAULT_PLUGIN_PATH,
    preset_path: str = _DEFAULT_PRESET_PATH,
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -55.0,
    sample_batch_size: int = 32,
    headless: bool = False,
    uploader: DatasetUploader | None = None,
    r2_prefix: str | None = None,
) -> Path:
    """Generate num_shards HDF5 shard files into shard_dir.

    Args:
        shard_dir: Directory to write shard files into. Created if needed.
        num_shards: Number of shard files to generate.
        shard_size: Number of samples per shard.
        param_spec: Parameter specification name (e.g. 'surge_simple').
        instance_id: Worker identifier baked into filenames. Auto-generated
            from uuid4 if not provided.
        plugin_path: Path to the VST3 plugin binary.
        preset_path: Path to the VST preset file.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        velocity: MIDI note velocity.
        signal_duration_seconds: Duration of each audio sample.
        min_loudness: Minimum loudness threshold in dB.
        sample_batch_size: Batch size for HDF5 writes.
        headless: Wrap subprocess with Xvfb virtual display for headless Linux.
        uploader: Optional uploader for pushing shards to R2.
        r2_prefix: R2 path prefix (e.g. 'runs/batch42').

    Returns:
        Path to the shard_dir.
    """
    if instance_id is None:
        instance_id = uuid.uuid4().hex[:8]

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    for seq in range(num_shards):
        shard_name = f"shard-{instance_id}-{seq:04d}.h5"  # noqa: E231
        shard_path = shard_dir / shard_name

        cmd = [
            "python",
            _GENERATE_SCRIPT,
            str(shard_path),
            str(shard_size),
            "--plugin_path",
            plugin_path,
            "--preset_path",
            preset_path,
            "--sample_rate",
            str(sample_rate),
            "--channels",
            str(channels),
            "--velocity",
            str(velocity),
            "--signal_duration_seconds",
            str(signal_duration_seconds),
            "--min_loudness",
            str(min_loudness),
            "--param_spec",
            param_spec,
            "--sample_batch_size",
            str(sample_batch_size),
        ]
        if headless:
            cmd = [_HEADLESS_WRAPPER] + cmd
        print(
            f"[generate_shards] shard {seq + 1}/{num_shards}: "
            f"{shard_size} samples -> {shard_path}",
            flush=True,
        )
        subprocess.run(cmd, check=True)  # nosec B603

    # Write worker metadata to the parent directory
    output_dir = shard_dir.parent
    meta = {
        "instance_id": instance_id,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "param_spec": param_spec,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "generation": {
            "plugin_path": plugin_path,
            "preset_path": preset_path,
            "sample_rate": sample_rate,
            "channels": channels,
            "velocity": velocity,
            "signal_duration_seconds": signal_duration_seconds,
            "min_loudness": min_loudness,
            "sample_batch_size": sample_batch_size,
        },
    }
    meta_path = output_dir / f"{instance_id}-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[generate_shards] worker metadata -> {meta_path}", flush=True)

    # Upload shards + metadata to R2 if configured
    if uploader is not None and r2_prefix is not None:
        uploader.upload(shard_dir, f"{r2_prefix}/shards")
        uploader.upload(output_dir, r2_prefix)
        print(f"[generate_shards] uploaded to {r2_prefix}", flush=True)

    return shard_dir


@click.command()
@click.option("--num-shards", "-n", type=int, required=True, help="Number of shard files.")
@click.option("--shard-size", "-s", type=int, required=True, help="Samples per shard.")
@click.option(
    "--output-dir",
    "-o",
    type=str,
    required=True,
    help="Output directory. Shards go into output_dir/shards/.",
)
@click.option(
    "--param-spec",
    "-p",
    type=str,
    default="surge_simple",
    show_default=True,
    help="Param spec name: 'surge_xt' or 'surge_simple'.",
)
@click.option(
    "--instance-id",
    type=str,
    default=None,
    help="Worker instance ID (default: auto-generated 8-char UUID).",
)
@click.option("--plugin-path", default=_DEFAULT_PLUGIN_PATH, show_default=True)
@click.option("--preset-path", default=_DEFAULT_PRESET_PATH, show_default=True)
@click.option("--sample-rate", default=44100.0, show_default=True)
@click.option("--channels", default=2, show_default=True)
@click.option("--velocity", default=100, show_default=True)
@click.option("--signal-duration-seconds", default=4.0, show_default=True)
@click.option("--min-loudness", default=-55.0, show_default=True)
@click.option("--sample-batch-size", default=32, show_default=True)
@click.option(
    "--headless",
    is_flag=True,
    default=False,
    help="Run under Xvfb virtual display for headless Linux environments without a GUI.",
)
@click.option(
    "--r2-bucket", default=None, envvar="R2_BUCKET", help="R2 bucket name. Omit to skip upload."
)
@click.option(
    "--r2-prefix", default=None, help="R2 path prefix (e.g. 'runs/batch42'). Omit to skip upload."
)
@click.option("--dry-run-upload", is_flag=True, default=False, help="Pass --dry-run to rclone.")
def main(
    num_shards: int,
    shard_size: int,
    output_dir: str,
    param_spec: str,
    instance_id: str | None,
    plugin_path: str,
    preset_path: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    sample_batch_size: int,
    headless: bool,
    r2_bucket: str | None,
    r2_prefix: str | None,
    dry_run_upload: bool,
) -> None:
    """Generate HDF5 dataset shards (split-agnostic)."""
    uploader = None
    if r2_bucket and r2_prefix:
        uploader = RcloneUploader(bucket=r2_bucket, dry_run=dry_run_upload)

    shard_dir = Path(output_dir) / "shards"
    generate_shards(
        shard_dir=shard_dir,
        num_shards=num_shards,
        shard_size=shard_size,
        param_spec=param_spec,
        instance_id=instance_id,
        plugin_path=plugin_path,
        preset_path=preset_path,
        sample_rate=sample_rate,
        channels=channels,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        min_loudness=min_loudness,
        sample_batch_size=sample_batch_size,
        headless=headless,
        uploader=uploader,
        r2_prefix=r2_prefix,
    )


if __name__ == "__main__":
    main()
