#!/usr/bin/env python3
"""Run the paired Vocal Imitation Set × Surge sketch CFG suite."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import click
import lance
from pydantic import BaseModel, ConfigDict, field_validator
from sh import Command

from synth_setter.cli.sketch_render import (
    cfg_arm_name,
    cfg_grid,
    load_audio_file,
    load_render_config,
)
from synth_setter.data.vst.core import write_wav
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import _retry_lance_read
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.utils.logging_utils import resolve_git_sha

VOCAL_DATASET = "r2://experiments/third_party/VocalImitationSet/test.lance"
CONTENT_DATASET = "r2://experiments/third_party/NSynth/test.lance"
DEFAULT_CHECKPOINT = (
    "r2://intermediate-data/checkpoints/flow_sketch_prelim/"
    "flow_sketch_prelim-20260902T044048985Z-"
    "eed5063da1164b1e92ac62a55ffc17b3/last.ckpt"
)
DEFAULT_CHECKPOINT_SHA256 = "d20cd4c3c86ae062a206f05596072b230c8aa86334920c775c2b4fec04aefc9e"
DEFAULT_STATS = (
    "r2://experiments/data/surge-simple-surgepy-lance-2m-40k-10k/"
    "surge-simple-surgepy-lance-2m-40k-10k-20260824T195308545Z/stats.npz"
)
DEFAULT_STATS_SHA256 = "c0c45d75a8b77004b3802c761bc77b5b34e7709a08343b2cf70fee04b7f52a19"
VOCAL_DATASET_VERSION = 1
CONTENT_DATASET_VERSION = 1
DEFAULT_CFG_STRENGTHS = (0.0, 1.0, 2.0)
SUITE_SIZE = 50
_METRICS = ("mss", "wmfcc", "sot", "rms")
_MAX_TORCH_SEED = 2**64 - 1
_PAIR_FIELDS = (
    "pair_index",
    "arm",
    "content_cfg",
    "sketch_cfg",
    "seed",
    *_METRICS,
    "r2_uri",
)
_Scalar = str | int | float
_PairValue = str | int


class _ArmMetrics(BaseModel):
    """Strict child-process metrics report.

    .. attribute :: model_config

        Strict Pydantic boundary validation.

    .. attribute :: content_cfg

        Content guidance strength.

    .. attribute :: sketch_cfg

        Sketch guidance strength.

    .. attribute :: seed

        Pair sampling seed.

    .. attribute :: mss

        Multi-scale spectrogram distance.

    .. attribute :: wmfcc

        Warped MFCC distance.

    .. attribute :: sot

        Spectral optimal-transport score.

    .. attribute :: rms

        RMS-envelope similarity.

    .. attribute :: r2_uri

        Published arm prefix.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    content_cfg: float
    sketch_cfg: float
    seed: int
    mss: float
    wmfcc: float
    sot: float
    rms: float
    r2_uri: str

    @field_validator("content_cfg", "sketch_cfg", "mss", "wmfcc", "sot", "rms")
    @classmethod
    def _finite_float(cls, value: float) -> float:
        """Require finite numeric report values.

        :param value: Parsed report value.
        :returns: Finite value.
        :raises ValueError: The value is non-finite.
        """
        if not math.isfinite(value):
            raise ValueError("metric report values must be finite")
        return value

    @field_validator("seed")
    @classmethod
    def _nonnegative_seed(cls, value: int) -> int:
        """Require a nonnegative pair seed.

        :param value: Parsed seed.
        :returns: Nonnegative seed.
        :raises ValueError: The seed is negative.
        """
        if value < 0:
            raise ValueError("metric report seed must be nonnegative")
        return value

    @field_validator("r2_uri")
    @classmethod
    def _r2_provenance(cls, value: str) -> str:
        """Require uploaded-arm provenance.

        :param value: Parsed arm URI.
        :returns: R2 URI.
        :raises ValueError: The report lacks an R2 URI.
        """
        if not r2_io.is_r2_uri(value):
            raise ValueError("metric report r2_uri must use r2://")
        return value


_AGGREGATE_FIELDS = (
    "arm",
    "content_cfg",
    "sketch_cfg",
    "count",
    *(f"{metric}_{stat}" for metric in _METRICS for stat in ("mean", "std_population")),
)


def select_vocal_rows(
    rows: Sequence[Mapping[str, object]], count: int
) -> tuple[tuple[int, Mapping[str, object]], ...]:
    """Select curated vocal imitations in immutable stored order.

    :param rows: Metadata rows in Lance scan order.
    :param count: Number of rows required.
    :returns: Original row indices paired with selected metadata.
    :raises ValueError: Fewer than ``count`` curated rows decode successfully.
    """
    eligible = tuple(
        (index, row)
        for index, row in enumerate(rows)
        if row["row_type"] == "imitation" and row["included"] is True
    )
    if len(eligible) < count:
        raise ValueError(f"requested {count} curated vocal imitations, found {len(eligible)}")
    selected = eligible[:count]
    for index, row in selected:
        if row["audio_decode_status"] != "decoded":
            raise ValueError(f"vocal row {index} is not decoded")
    return selected


def aggregate_metrics(rows: Sequence[Mapping[str, _Scalar]]) -> list[dict[str, _Scalar]]:
    """Aggregate per-pair audio metrics independently for each CFG arm.

    :param rows: Long-form pair/arm metric rows.
    :returns: Arm rows sorted by arm name with population statistics.
    :raises ValueError: A per-pair metric is non-finite.
    """
    arm_names = sorted({str(row["arm"]) for row in rows})
    aggregates: list[dict[str, _Scalar]] = []
    for arm in arm_names:
        arm_rows = [row for row in rows if row["arm"] == arm]
        first = arm_rows[0]
        aggregate: dict[str, _Scalar] = {
            "arm": arm,
            "content_cfg": first.get("content_cfg", ""),
            "sketch_cfg": first.get("sketch_cfg", ""),
            "count": len(arm_rows),
        }
        for metric in _METRICS:
            values = [float(row[metric]) for row in arm_rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"finite {metric} values are required for arm {arm}")
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std_population"] = statistics.pstdev(values)
        aggregates.append(aggregate)
    return aggregates


def _open_dataset(uri: str, version: int) -> lance.LanceDataset:
    """Open one immutable R2 Lance snapshot.

    :param uri: Dataset R2 URI.
    :param version: Pinned Lance version.
    :returns: Open Lance dataset.
    """
    target, storage_options = r2_io.lance_target(uri)
    return _retry_lance_read(
        "sketch_suite_open",
        lambda: lance.dataset(target, version=version, storage_options=storage_options),
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write mappings with stable column order.

    :param path: CSV destination.
    :param fieldnames: Ordered field names.
    :param rows: Rows to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as string-valued mappings.

    :param path: CSV source.
    :returns: Rows in source order.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_blob_wav(data: bytes, destination: Path, render: RenderConfig) -> None:
    """Decode one stored audio blob into the suite render grid.

    :param data: Encoded source audio.
    :param destination: PCM WAV destination.
    :param render: Effective sketch CLI audio grid.
    """
    encoded_path = destination.with_suffix(".source.wav")
    encoded_path.write_bytes(data)
    try:
        audio = load_audio_file(
            encoded_path,
            sample_rate=render.sample_rate,
            channels=render.channels,
            num_samples=int(render.sample_rate * render.signal_duration_seconds),
        )
    finally:
        encoded_path.unlink(missing_ok=True)
    write_wav(audio, str(destination), render.sample_rate, render.channels)


def _materialize_pairs(
    output_dir: Path, count: int, render: RenderConfig
) -> list[dict[str, _PairValue]]:
    """Materialize index-zipped vocal-imitation and NSynth test rows.

    :param output_dir: Suite workspace.
    :param count: Number of paired rows.
    :param render: Effective sketch CLI audio grid.
    :returns: Pair provenance rows with local input paths.
    :raises ValueError: Dataset row counts or decode states violate the suite contract.
    """
    vocal = _open_dataset(VOCAL_DATASET, VOCAL_DATASET_VERSION)
    content = _open_dataset(CONTENT_DATASET, CONTENT_DATASET_VERSION)
    if _retry_lance_read("sketch_suite_content_count", content.count_rows) < count:
        raise ValueError(f"content dataset has fewer than {count} rows")

    vocal_columns = [
        "row_id",
        "row_type",
        "included",
        "audio_decode_status",
        "audio_sha256",
        "source_path",
    ]
    vocal_metadata = _retry_lance_read(
        "sketch_suite_vocal_metadata",
        lambda: vocal.to_table(columns=vocal_columns).to_pylist(),
    )
    vocal_rows = select_vocal_rows(vocal_metadata, count)
    vocal_indices = [index for index, _ in vocal_rows]
    vocal_blobs = _retry_lance_read(
        "sketch_suite_vocal_audio",
        lambda: dict(vocal.read_blobs("audio", indices=vocal_indices, preserve_order=True)),
    )
    content_indices = list(range(count))
    content_columns = [
        "instrument_family_str",
        "instrument_source_str",
        "note_str",
        "pitch",
        "velocity",
        "sample_rate",
        "wav_sha256",
    ]
    content_metadata = _retry_lance_read(
        "sketch_suite_content_metadata",
        lambda: content.take(content_indices, columns=content_columns).to_pylist(),
    )
    content_blobs = _retry_lance_read(
        "sketch_suite_content_audio",
        lambda: dict(content.read_blobs("audio", indices=content_indices, preserve_order=True)),
    )

    pairs: list[dict[str, _PairValue]] = []
    for pair_index, ((vocal_index, vocal_row), content_row) in enumerate(
        zip(vocal_rows, content_metadata, strict=True)
    ):
        input_dir = output_dir / "inputs" / f"sample_{pair_index:03d}"
        input_dir.mkdir(parents=True, exist_ok=True)
        sketch_path = input_dir / "sketch.wav"
        content_path = input_dir / "content.wav"
        if not sketch_path.is_file():
            _write_blob_wav(vocal_blobs[vocal_index], sketch_path, render)
        if not content_path.is_file():
            _write_blob_wav(content_blobs[pair_index], content_path, render)
        pairs.append(
            {
                "pair_index": pair_index,
                "vocal_row_index": vocal_index,
                "vocal_row_id": str(vocal_row["row_id"]),
                "vocal_audio_sha256": str(vocal_row["audio_sha256"]),
                "vocal_source_path": str(vocal_row["source_path"]),
                "content_row_index": pair_index,
                "content_wav_sha256": str(content_row["wav_sha256"]),
                "content_instrument_family": str(content_row["instrument_family_str"]),
                "content_instrument_source": str(content_row["instrument_source_str"]),
                "content_note": str(content_row["note_str"]),
                "content_pitch": int(content_row["pitch"]),
                "content_velocity": int(content_row["velocity"]),
                "content_sample_rate": int(content_row["sample_rate"]),
                "sketch_path": str(sketch_path),
                "content_path": str(content_path),
            }
        )
    return pairs


def _render_pair(
    pair: Mapping[str, _PairValue],
    *,
    output_dir: Path,
    destination: str,
    checkpoint: str,
    checkpoint_sha256: str,
    stats: str,
    stats_sha256: str,
    render: RenderConfig,
    content_cfg: Sequence[float],
    sketch_cfg: Sequence[float],
    sample_steps: int,
    seed: int,
    device: str,
    timeout_seconds: int,
) -> None:
    """Render and upload every CFG arm for one input pair.

    :param pair: Materialized pair provenance.
    :param output_dir: Suite workspace.
    :param destination: Suite R2 prefix.
    :param checkpoint: Sketch checkpoint source.
    :param checkpoint_sha256: Trusted checkpoint digest.
    :param stats: Matching content mel-statistics source.
    :param stats_sha256: Trusted statistics digest.
    :param render: Immutable audio grid forwarded to the child CLI.
    :param content_cfg: Content CFG axis.
    :param sketch_cfg: Sketch CFG axis.
    :param sample_steps: Flow integration steps.
    :param seed: Pair seed shared across arms.
    :param device: Inference device.
    :param timeout_seconds: Per-pair subprocess wall-clock limit.
    :raises FileExistsError: The pair output path already exists.
    """
    pair_index = int(pair["pair_index"])
    pair_name = f"sample_{pair_index:03d}"
    pair_output = output_dir / "audio" / pair_name
    if pair_output.exists():
        raise FileExistsError(f"pair requires a fresh output directory: {pair_output}")

    args = [
        str(pair["sketch_path"]),
        str(pair["content_path"]),
        "--checkpoint",
        checkpoint,
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--stats",
        stats,
        "--stats-sha256",
        stats_sha256,
        "--sample-steps",
        str(sample_steps),
        "--seed",
        str(seed),
        "--output-dir",
        str(pair_output),
        "--upload-prefix",
        f"{destination}/audio/{pair_name}",
        "--device",
        device,
        "--sample-rate",
        str(render.sample_rate),
        "--channels",
        str(render.channels),
        "--duration",
        str(render.signal_duration_seconds),
        "--upload",
    ]
    for strength in content_cfg:
        args.extend(("--content-cfg", str(strength)))
    for strength in sketch_cfg:
        args.extend(("--sketch-cfg", str(strength)))
    Command("synth-setter-sketch")(
        *args,
        _out=sys.stdout,
        _err=sys.stderr,
        _timeout=timeout_seconds,
    )


def _collect_metrics(
    output_dir: Path,
    pair_count: int,
    grid: Sequence[tuple[float, float]],
    base_seed: int,
) -> list[dict[str, _Scalar]]:
    """Collect every pair/arm metrics row.

    :param output_dir: Suite workspace.
    :param pair_count: Number of rendered pairs.
    :param grid: Ordered CFG arms.
    :param base_seed: First pair seed.
    :returns: Long-form metric rows.
    :raises ValueError: An arm has no unique metrics row.
    """
    rows: list[dict[str, _Scalar]] = []
    for pair_index in range(pair_count):
        for content_strength, sketch_strength in grid:
            arm = cfg_arm_name(content_strength, sketch_strength)
            metrics_path = (
                output_dir / "audio" / f"sample_{pair_index:03d}" / "arms" / arm / "metrics.csv"
            )
            metrics_rows = _read_csv(metrics_path)
            if len(metrics_rows) != 1:
                raise ValueError(f"expected one metrics row in {metrics_path}")
            metrics = _ArmMetrics.model_validate_strings(metrics_rows[0], strict=True)
            if metrics.content_cfg != content_strength or metrics.sketch_cfg != sketch_strength:
                raise ValueError(f"CFG provenance mismatch in {metrics_path}")
            if metrics.seed != base_seed + pair_index:
                raise ValueError(f"seed provenance mismatch in {metrics_path}")
            rows.append({"pair_index": pair_index, "arm": arm, **metrics.model_dump()})
    return rows


def _validate_seed_range(base_seed: int, count: int) -> None:
    """Require every derived pair seed to fit PyTorch's unsigned seed domain.

    :param base_seed: First pair seed.
    :param count: Number of sequential pair seeds.
    :raises ValueError: Any derived seed lies outside the supported domain.
    """
    if base_seed < 0 or base_seed + count - 1 > _MAX_TORCH_SEED:
        raise ValueError("derived pair seeds must be between 0 and 2**64 - 1")


def _require_fresh_run(output_dir: Path, destination: str) -> None:
    """Reject local or remote prefixes that could mix two suite runs.

    :param output_dir: Local suite root.
    :param destination: Exact R2 run prefix.
    :raises FileExistsError: Either run location already contains artifacts.
    """
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if r2_io.r2_directory_exists(destination):
        raise FileExistsError(f"R2 destination is not empty: {destination}")


def _claim_destination(destination: str) -> None:
    """Atomically reserve an empty R2 run prefix with an immutable token.

    :param destination: Exact R2 run prefix.
    """
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as claim:
        claim.write(f"{uuid4()}\n")
        claim.flush()
        subprocess.run(  # noqa: S603 — fixed rclone command without a shell
            [  # noqa: S607 — rclone resolves from the operator environment
                "rclone",
                "copyto",
                "--checksum",
                "--immutable",
                claim.name,
                r2_io.to_rclone_path(f"{destination}/.run-claim"),
            ],
            check=True,
            timeout=60,
        )


def _parse_args() -> argparse.Namespace:
    """Parse suite inputs and publication destination.

    :returns: Command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--stats-sha256", default=DEFAULT_STATS_SHA256)
    parser.add_argument("--content-cfg", type=float, action="append")
    parser.add_argument("--sketch-cfg", type=float, action="append")
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-timeout-seconds", type=int, default=3600)
    parser.add_argument("--count", type=int, default=SUITE_SIZE, choices=range(1, SUITE_SIZE + 1))
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    return parser.parse_args()


def main() -> None:
    """Render the deterministic paired suite and publish all artifacts.

    :raises ValueError: URIs or numeric arguments violate the suite contract.
    """
    args = _parse_args()
    if not r2_io.is_r2_uri(args.checkpoint):
        raise ValueError(f"checkpoint must use r2://, got {args.checkpoint}")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint SHA-256 must contain 64 hex characters")
    try:
        int(args.checkpoint_sha256, 16)
    except ValueError as exc:
        raise ValueError("checkpoint SHA-256 must be hexadecimal") from exc
    if len(args.stats_sha256) != 64:
        raise ValueError("statistics SHA-256 must contain 64 hex characters")
    try:
        int(args.stats_sha256, 16)
    except ValueError as exc:
        raise ValueError("statistics SHA-256 must be hexadecimal") from exc
    if not r2_io.is_r2_uri(args.stats):
        raise ValueError(f"stats must use r2://, got {args.stats}")
    if not r2_io.is_r2_uri(args.destination):
        raise ValueError(f"destination must use r2://, got {args.destination}")
    if args.sample_steps <= 0:
        raise ValueError("sample steps must be positive")
    if args.pair_timeout_seconds <= 0:
        raise ValueError("pair timeout must be positive")
    _validate_seed_range(args.seed, args.count)
    content_cfg = tuple(args.content_cfg or DEFAULT_CFG_STRENGTHS)
    sketch_cfg = tuple(args.sketch_cfg or DEFAULT_CFG_STRENGTHS)
    grid = cfg_grid(content_cfg, sketch_cfg)
    destination = args.destination.rstrip("/")
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    r2_io.ensure_r2_env_loaded()
    _require_fresh_run(output_dir, destination)
    _claim_destination(destination)

    render = load_render_config()
    pairs = _materialize_pairs(output_dir, args.count, render)
    pair_manifest = output_dir / "pair_manifest.csv"
    _write_csv(pair_manifest, tuple(pairs[0]), pairs)
    for index, pair in enumerate(pairs, start=1):
        _render_pair(
            pair,
            output_dir=output_dir,
            destination=destination,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256.lower(),
            stats=args.stats,
            stats_sha256=args.stats_sha256.lower(),
            render=render,
            content_cfg=content_cfg,
            sketch_cfg=sketch_cfg,
            sample_steps=args.sample_steps,
            seed=args.seed + int(pair["pair_index"]),
            device=args.device,
            timeout_seconds=args.pair_timeout_seconds,
        )
        click.echo(f"[{index:03d}/{len(pairs):03d}] pair {int(pair['pair_index']):03d}")

    paired_rows = _collect_metrics(output_dir, len(pairs), grid, args.seed)
    paired_path = output_dir / "paired_results.csv"
    aggregate_path = output_dir / "aggregate_comparison.csv"
    manifest_path = output_dir / "comparison_manifest.csv"
    _write_csv(paired_path, _PAIR_FIELDS, paired_rows)
    _write_csv(aggregate_path, _AGGREGATE_FIELDS, aggregate_metrics(paired_rows))
    _write_csv(
        manifest_path,
        ("key", "value"),
        [
            {"key": "created_at", "value": datetime.now(UTC).isoformat()},
            {"key": "run_id", "value": destination.rsplit("/", maxsplit=1)[-1]},
            {"key": "git_commit", "value": resolve_git_sha()},
            {"key": "checkpoint", "value": args.checkpoint},
            {"key": "checkpoint_sha256", "value": args.checkpoint_sha256.lower()},
            {"key": "stats", "value": args.stats},
            {"key": "stats_sha256", "value": args.stats_sha256.lower()},
            {"key": "vocal_dataset", "value": VOCAL_DATASET},
            {"key": "vocal_dataset_version", "value": VOCAL_DATASET_VERSION},
            {"key": "content_dataset", "value": CONTENT_DATASET},
            {"key": "content_dataset_version", "value": CONTENT_DATASET_VERSION},
            {"key": "content_cfg", "value": ",".join(map(str, content_cfg))},
            {"key": "sketch_cfg", "value": ",".join(map(str, sketch_cfg))},
            {"key": "sample_steps", "value": args.sample_steps},
            {"key": "base_seed", "value": args.seed},
            {"key": "pair_count", "value": len(pairs)},
            {"key": "render_sample_rate_hz", "value": render.sample_rate},
            {"key": "render_channels", "value": render.channels},
            {
                "key": "render_signal_duration_seconds",
                "value": render.signal_duration_seconds,
            },
            {
                "key": "render_num_samples",
                "value": int(render.sample_rate * render.signal_duration_seconds),
            },
            {"key": "destination", "value": destination},
        ],
    )
    for path, relative in (
        (pair_manifest, "manifests/pairs.csv"),
        (paired_path, "metrics/paired_results.csv"),
        (aggregate_path, "metrics/aggregate_comparison.csv"),
        (manifest_path, "comparison_manifest.csv"),
    ):
        r2_io.upload(path, f"{destination}/{relative}")
    click.echo(destination)


if __name__ == "__main__":
    main()
