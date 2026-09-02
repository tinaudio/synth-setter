#!/usr/bin/env python3
"""Run the paired Vocal Imitation Set × Surge sketch CFG suite."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import click
import lance
import numpy as np
import pyarrow as pa
from sh import Command

from synth_setter.cli.sketch_render import cfg_arm_name, cfg_grid, load_audio_file
from synth_setter.data.vst.core import write_wav
from synth_setter.pipeline import r2_io

VOCAL_DATASET = "r2://experiments/third_party/VocalImitationSet/test.lance"
CONTENT_DATASET = (
    "r2://experiments/data/surge-simple-surgepy-lance-2m-40k-10k/"
    "surge-simple-surgepy-lance-2m-40k-10k-20260824T195308545Z/test.lance"
)
DEFAULT_CHECKPOINT = (
    "r2://intermediate-data/checkpoints/flow_sketch_prelim/"
    "flow_sketch_prelim-20260902T044048985Z-"
    "eed5063da1164b1e92ac62a55ffc17b3/last.ckpt"
)
VOCAL_DATASET_VERSION = 1
CONTENT_DATASET_VERSION = 9
DEFAULT_CFG_STRENGTHS = (0.0, 1.0, 2.0)
SUITE_SIZE = 50
SAMPLE_RATE = 44_100
CHANNELS = 2
NUM_SAMPLES = 176_400
_METRICS = ("mss", "wmfcc", "sot", "rms")
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
    """Select decoded Vocal Imitation rows in immutable stored order.

    :param rows: Metadata rows in Lance scan order.
    :param count: Number of rows required.
    :returns: Original row indices paired with selected metadata.
    :raises ValueError: Fewer than ``count`` rows decoded successfully.
    """
    if len(rows) < count:
        raise ValueError(f"requested {count} vocal rows, found {len(rows)}")
    selected = tuple(enumerate(rows[:count]))
    for index, row in selected:
        if row["audio_decode_status"] != "decoded":
            raise ValueError(f"vocal row {index} is not decoded")
    return selected


def aggregate_metrics(rows: Sequence[Mapping[str, _Scalar]]) -> list[dict[str, _Scalar]]:
    """Aggregate per-pair audio metrics independently for each CFG arm.

    :param rows: Long-form pair/arm metric rows.
    :returns: Arm rows sorted by arm name with population statistics.
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
    return lance.dataset(target, version=version, storage_options=storage_options)


def _tensor_column(table: pa.Table, name: str) -> np.ndarray:
    """Convert one fixed-shape tensor column to a dense array.

    :param table: Source Arrow table.
    :param name: Tensor column name.
    :returns: Dense values preserving row order.
    """
    chunks = table[name].chunks
    return np.concatenate([chunk.to_numpy_ndarray() for chunk in chunks], axis=0)


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


def _materialize_pairs(output_dir: Path, count: int) -> list[dict[str, _PairValue]]:
    """Materialize index-zipped vocal and Surge test rows.

    :param output_dir: Suite workspace.
    :param count: Number of paired rows.
    :returns: Pair provenance rows with local input paths.
    :raises ValueError: Dataset row counts or decode states violate the suite contract.
    """
    vocal = _open_dataset(VOCAL_DATASET, VOCAL_DATASET_VERSION)
    content = _open_dataset(CONTENT_DATASET, CONTENT_DATASET_VERSION)
    if content.count_rows() < count:
        raise ValueError(f"content dataset has fewer than {count} rows")

    vocal_metadata = vocal.to_table(
        columns=["row_id", "audio_decode_status", "audio_sha256", "source_path"]
    ).to_pylist()
    vocal_rows = select_vocal_rows(vocal_metadata, count)
    vocal_blobs = dict(
        vocal.read_blobs(
            "audio",
            indices=[index for index, _ in vocal_rows],
            preserve_order=True,
        )
    )
    content_table = content.take(
        list(range(count)),
        columns=["audio", "param_array", "audio_uuid", "debug"],
    )
    content_audio = _tensor_column(content_table, "audio").astype(np.float32)
    content_params = _tensor_column(content_table, "param_array").astype(np.float32)
    content_uuids = content_table["audio_uuid"].to_pylist()
    content_debug = [json.loads(value) for value in content_table["debug"].to_pylist()]

    pairs: list[dict[str, _PairValue]] = []
    for pair_index, ((vocal_index, vocal_row), audio, params, audio_uuid, debug) in enumerate(
        zip(vocal_rows, content_audio, content_params, content_uuids, content_debug, strict=True)
    ):
        input_dir = output_dir / "inputs" / f"sample_{pair_index:03d}"
        input_dir.mkdir(parents=True, exist_ok=True)
        sketch_path = input_dir / "sketch.wav"
        content_path = input_dir / "content.wav"
        params_path = input_dir / "content.params.npy"
        if not sketch_path.is_file():
            encoded_path = input_dir / "sketch-source.wav"
            encoded_path.write_bytes(vocal_blobs[vocal_index])
            try:
                sketch = load_audio_file(
                    encoded_path,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    num_samples=NUM_SAMPLES,
                )
            finally:
                encoded_path.unlink(missing_ok=True)
            write_wav(sketch, str(sketch_path), SAMPLE_RATE, CHANNELS)
        if not content_path.is_file():
            write_wav(
                np.asarray(audio, dtype=np.float32), str(content_path), SAMPLE_RATE, CHANNELS
            )
        if not params_path.is_file():
            np.save(params_path, params)
        pairs.append(
            {
                "pair_index": pair_index,
                "vocal_row_index": vocal_index,
                "vocal_row_id": str(vocal_row["row_id"]),
                "vocal_audio_sha256": str(vocal_row["audio_sha256"]),
                "vocal_source_path": str(vocal_row["source_path"]),
                "content_row_index": pair_index,
                "content_audio_uuid": str(audio_uuid),
                "content_master_seed": int(debug["master_seed"]),
                "content_sample_idx": int(debug["sample_idx"]),
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
    content_cfg: Sequence[float],
    sketch_cfg: Sequence[float],
    sample_steps: int,
    seed: int,
    device: str,
) -> None:
    """Render and upload every CFG arm for one input pair.

    :param pair: Materialized pair provenance.
    :param output_dir: Suite workspace.
    :param destination: Suite R2 prefix.
    :param checkpoint: Sketch checkpoint source.
    :param content_cfg: Content CFG axis.
    :param sketch_cfg: Sketch CFG axis.
    :param sample_steps: Flow integration steps.
    :param seed: Pair seed shared across arms.
    :param device: Inference device.
    """
    pair_index = int(pair["pair_index"])
    pair_name = f"sample_{pair_index:03d}"
    pair_output = output_dir / "audio" / pair_name
    expected = [
        pair_output / "arms" / cfg_arm_name(content_strength, sketch_strength) / "metrics.csv"
        for content_strength, sketch_strength in cfg_grid(content_cfg, sketch_cfg)
    ]
    if all(path.is_file() for path in expected):
        r2_io.upload_dir(pair_output, f"{destination}/audio/{pair_name}")
        return
    if pair_output.exists():
        shutil.rmtree(pair_output)

    args = [
        str(pair["sketch_path"]),
        str(pair["content_path"]),
        "--checkpoint",
        checkpoint,
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
    ]
    for strength in content_cfg:
        args.extend(("--content-cfg", str(strength)))
    for strength in sketch_cfg:
        args.extend(("--sketch-cfg", str(strength)))
    Command("synth-setter-sketch")(*args, _out=sys.stdout, _err=sys.stderr)


def _collect_metrics(
    output_dir: Path,
    pair_count: int,
    grid: Sequence[tuple[float, float]],
) -> list[dict[str, _Scalar]]:
    """Collect every pair/arm metrics row.

    :param output_dir: Suite workspace.
    :param pair_count: Number of rendered pairs.
    :param grid: Ordered CFG arms.
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
            rows.append({"pair_index": pair_index, "arm": arm, **metrics_rows[0]})
    return rows


def _parse_args() -> argparse.Namespace:
    """Parse suite inputs and publication destination.

    :returns: Command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--content-cfg", type=float, action="append")
    parser.add_argument("--sketch-cfg", type=float, action="append")
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
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
    if not r2_io.is_r2_uri(args.destination):
        raise ValueError(f"destination must use r2://, got {args.destination}")
    if args.sample_steps <= 0:
        raise ValueError("sample steps must be positive")
    content_cfg = tuple(args.content_cfg or DEFAULT_CFG_STRENGTHS)
    sketch_cfg = tuple(args.sketch_cfg or DEFAULT_CFG_STRENGTHS)
    grid = cfg_grid(content_cfg, sketch_cfg)
    destination = args.destination.rstrip("/")
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    r2_io.ensure_r2_env_loaded()

    pairs = _materialize_pairs(output_dir, args.count)
    pair_manifest = output_dir / "pair_manifest.csv"
    _write_csv(pair_manifest, tuple(pairs[0]), pairs)
    for index, pair in enumerate(pairs, start=1):
        _render_pair(
            pair,
            output_dir=output_dir,
            destination=destination,
            checkpoint=args.checkpoint,
            content_cfg=content_cfg,
            sketch_cfg=sketch_cfg,
            sample_steps=args.sample_steps,
            seed=args.seed + int(pair["pair_index"]),
            device=args.device,
        )
        click.echo(f"[{index:03d}/{len(pairs):03d}] pair {int(pair['pair_index']):03d}")

    paired_rows = _collect_metrics(output_dir, len(pairs), grid)
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
            {"key": "checkpoint", "value": args.checkpoint},
            {"key": "vocal_dataset", "value": VOCAL_DATASET},
            {"key": "vocal_dataset_version", "value": VOCAL_DATASET_VERSION},
            {"key": "content_dataset", "value": CONTENT_DATASET},
            {"key": "content_dataset_version", "value": CONTENT_DATASET_VERSION},
            {"key": "content_cfg", "value": ",".join(map(str, content_cfg))},
            {"key": "sketch_cfg", "value": ",".join(map(str, sketch_cfg))},
            {"key": "sample_steps", "value": args.sample_steps},
            {"key": "base_seed", "value": args.seed},
            {"key": "pair_count", "value": len(pairs)},
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
