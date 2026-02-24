"""VST dataset pipeline — generate splits, compute stats, upload to R2.

Orchestrates the full dataset generation pipeline for any VST synthesizer:
  1. Calls generate_vst_dataset.py (via headless X11 wrapper) for each split
  2. Computes mel-spectrogram normalization stats from the train split
  3. Writes a metadata.json recording generation parameters and git provenance
  4. Uploads everything to Cloudflare R2 (optional; omit --r2-prefix to skip)

This script is VST-agnostic. The choice of synthesizer is controlled by
--plugin-path, --preset-path, and --param-spec. It was originally written for
Surge XT but is designed to work with any compatible VST3 plugin.

Usage (inside Docker, via docker_entrypoint.sh):
    python scripts/run_dataset_pipeline.py \\
        --param-spec surge_simple \\
        --train-samples 10000 \\
        --val-samples 1000 \\
        --test-samples 1000 \\
        --output-dir data/surge_simple \\
        --r2-prefix runs/surge_simple/abc1234

Skip the upload by omitting --r2-prefix (useful for local runs or CI without R2).
See docs/pipeline.md for full documentation and plug-and-play command examples.
"""

import json
import os
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.uploader import DatasetUploader, LocalFakeUploader, RcloneUploader

# Default paths matching SurgeDataModule expectations and Docker image layout
_DEFAULT_PLUGIN_PATH = "/usr/lib/vst3/Surge XT.vst3"
_DEFAULT_PRESET_PATH = "presets/surge-base.vstpreset"
_GENERATE_SCRIPT = "src/data/vst/generate_vst_dataset.py"
_STATS_SCRIPT = "scripts/get_dataset_stats.py"
_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"

# Maps param_spec name -> Hydra data config name (configs/data/<name>.yaml).
# Add a new entry here when adding support for a new VST or param set.
PARAM_SPEC_TO_DATA_CONFIG = {
    "surge_simple": "surge_simple",  # configs/data/surge_simple.yaml  (92 params)
    "surge_xt": "surge",  # configs/data/surge.yaml         (189 params)
}

# Number of parameters per spec (recorded in metadata.json for traceability)
_PARAM_SPEC_NUM_PARAMS = {
    "surge_simple": 92,
    "surge_xt": 189,
}


def _validate_param_spec(param_spec: str) -> None:
    """Validate param_spec against known values, printing an error and exiting if invalid."""
    if param_spec not in PARAM_SPEC_TO_DATA_CONFIG:
        valid = ", ".join(f"'{k}'" for k in PARAM_SPEC_TO_DATA_CONFIG)
        print(
            f"ERROR: Unknown PARAM_SPEC '{param_spec}'. Valid values: {valid}.",
            file=sys.stderr,
        )
        sys.exit(1)


def _generate_split(
    output_dir: Path,
    split: str,
    num_samples: int,
    param_spec: str,
    plugin_path: str,
    preset_path: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    sample_batch_size: int,
) -> Path:
    """Run generate_vst_dataset.py for one split, return path to the .h5 file."""
    out_file = output_dir / f"{split}.h5"
    cmd = [
        _HEADLESS_WRAPPER,
        "python",
        _GENERATE_SCRIPT,
        str(out_file),
        str(num_samples),
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
    print(f"[generate] {split}: {num_samples} samples -> {out_file}", flush=True)
    subprocess.run(cmd, check=True)  # nosec B603
    return out_file


def _compute_stats(train_h5: Path) -> None:
    """Run get_dataset_stats.py on the train split to produce stats.npz."""
    print(f"[stats] computing normalization stats from {train_h5}", flush=True)
    subprocess.run(["python", _STATS_SCRIPT, str(train_h5)], check=True)  # nosec B603 B607


def _write_metadata(
    output_dir: Path,
    param_spec: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    r2_prefix: str | None,
    git_sha: str,
    git_dirty: bool | None = None,
    git_ref_source: str = "unknown",
    plugin_path: str = _DEFAULT_PLUGIN_PATH,
    preset_path: str = _DEFAULT_PRESET_PATH,
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -55.0,
    sample_batch_size: int = 32,
) -> Path:
    """Write metadata.json describing this generation run.

    git_dirty / git_ref_source fields record code provenance so dataset
    consumers can distinguish clean production builds from dev runs:

      git_ref_source: "baked"  — code was downloaded as a tarball at build time
                                 (dev-snapshot or prod image). git_sha is authoritative.
                     "local"   — code was mounted or cloned at runtime (dev-live image).
                                 git_sha reflects HEAD at run time; git_dirty indicates
                                 whether the working tree had uncommitted changes.
                     "unknown" — provenance could not be determined.

      git_dirty:     true  — working tree had uncommitted changes at generation time.
                     false — working tree was clean.
                     null  — could not be determined (e.g. no .git in container).

    A dataset is production-quality when git_ref_source == "baked" and git_dirty == false.
    """
    meta = {
        # TODO: add dataset_schema_version once src/data/dataset_version.py is wired
        # through generate_vst_dataset.py (writes HDF5 attr) and surge_datamodule.py
        # (validates on load). See src/data/dataset_version.py.
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_ref_source": git_ref_source,
        "git_dirty": git_dirty,
        "param_spec": param_spec,
        "param_spec_num_params": _PARAM_SPEC_NUM_PARAMS.get(param_spec, None),
        "splits": {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        },
        "r2_prefix": r2_prefix,
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
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[metadata] written to {meta_path}", flush=True)
    return meta_path


def run_pipeline(
    param_spec: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    output_dir: Path,
    uploader: DatasetUploader | None,
    r2_prefix: str | None = None,
    plugin_path: str = _DEFAULT_PLUGIN_PATH,
    preset_path: str = _DEFAULT_PRESET_PATH,
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -55.0,
    sample_batch_size: int = 32,
    git_sha: str = "unknown",
    git_dirty: bool | None = None,
    git_ref_source: str = "unknown",
) -> None:
    """Generate all dataset splits, compute stats, write metadata, then upload.

    This function is the testable core of the pipeline. Tests can pass a LocalFakeUploader to
    exercise the full flow without real R2 credentials. Pass uploader=None to skip upload entirely
    (useful for local smoke runs).
    """
    _validate_param_spec(param_spec)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_kwargs = dict(
        param_spec=param_spec,
        plugin_path=plugin_path,
        preset_path=preset_path,
        sample_rate=sample_rate,
        channels=channels,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        min_loudness=min_loudness,
        sample_batch_size=sample_batch_size,
    )

    for split, n in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        _generate_split(output_dir=output_dir, split=split, num_samples=n, **shared_kwargs)

    _compute_stats(output_dir / "train.h5")
    _write_metadata(
        output_dir=output_dir,
        param_spec=param_spec,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        r2_prefix=r2_prefix,
        git_sha=git_sha,
        git_dirty=git_dirty,
        git_ref_source=git_ref_source,
        plugin_path=plugin_path,
        preset_path=preset_path,
        sample_rate=sample_rate,
        channels=channels,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        min_loudness=min_loudness,
        sample_batch_size=sample_batch_size,
    )

    if uploader is not None and r2_prefix is not None:
        print(f"[upload] uploading {output_dir} -> {r2_prefix}", flush=True)
        uploader.upload(output_dir, r2_prefix)
        print("[upload] done", flush=True)
    else:
        print("[upload] skipped (no uploader or r2_prefix configured)", flush=True)


@click.command()
@click.option(
    "--param-spec",
    default="surge_simple",
    show_default=True,
    help="Param spec name: 'surge_xt' (189 params) or 'surge_simple' (92 params).",
)
@click.option("--train-samples", default=10000, show_default=True)
@click.option("--val-samples", default=1000, show_default=True)
@click.option("--test-samples", default=1000, show_default=True)
@click.option(
    "--output-dir",
    default="data/surge_simple",
    show_default=True,
    help="Local directory to write HDF5 files into.",
)
@click.option(
    "--r2-prefix",
    default=None,
    help="R2 path prefix (e.g. 'runs/surge_simple/abc1234'). " "Omit to skip upload.",
)
@click.option(
    "--r2-bucket",
    default=None,
    envvar="R2_BUCKET",
    help="R2 bucket name. Reads from R2_BUCKET env var if not set.",
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
    "--dry-run-upload",
    is_flag=True,
    default=False,
    help="Pass --dry-run to rclone (verify config without actually uploading).",
)
@click.option(
    "--git-ref-source",
    default="unknown",
    type=click.Choice(["baked", "local", "unknown"]),
    help="Code provenance: 'baked' (tarball, prod/dev-snapshot), "
    "'local' (mounted/cloned at runtime, dev-live), or 'unknown'.",
)
@click.option(
    "--git-dirty",
    default=None,
    type=click.BOOL,
    help="Whether the working tree had uncommitted changes at generation time. "
    "Set by docker_entrypoint.sh. Omit if indeterminate.",
)
def main(
    param_spec: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    output_dir: str,
    r2_prefix: str | None,
    r2_bucket: str | None,
    plugin_path: str,
    preset_path: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    sample_batch_size: int,
    dry_run_upload: bool,
    git_ref_source: str,
    git_dirty: bool | None,
) -> None:
    """CLI entry point: parse options, build the uploader, and invoke run_pipeline."""
    git_sha = os.environ.get("SYNTH_PERMUTATIONS_GIT_REF", "unknown")

    uploader: DatasetUploader | None = None
    if r2_prefix is not None:
        if not r2_bucket:
            print(
                "ERROR: --r2-prefix given but no R2 bucket configured. "
                "Pass --r2-bucket or set R2_BUCKET env var.",
                file=sys.stderr,
            )
            sys.exit(1)
        uploader = RcloneUploader(bucket=r2_bucket, dry_run=dry_run_upload)

    run_pipeline(
        param_spec=param_spec,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        output_dir=Path(output_dir),
        uploader=uploader,
        r2_prefix=r2_prefix,
        plugin_path=plugin_path,
        preset_path=preset_path,
        sample_rate=sample_rate,
        channels=channels,
        velocity=velocity,
        signal_duration_seconds=signal_duration_seconds,
        min_loudness=min_loudness,
        sample_batch_size=sample_batch_size,
        git_sha=git_sha,
        git_dirty=git_dirty,
        git_ref_source=git_ref_source,
    )


if __name__ == "__main__":
    main()
