#!/usr/bin/env python3
"""Run a paired CLAP prompt-suite checkpoint comparison and publish it to R2.

The destination follows the validation audio-probe shape: each ``audio/sample_*``
directory contains baseline and candidate WAV/CSV pairs, while aggregate and
per-prompt comparisons live under ``metrics/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import click
from sh import Command

from synth_setter.cli.clap_render import summarize_cosine_distances, write_summary_csv
from synth_setter.pipeline import r2_io

BASELINE_SUITE = "r2://experiments/clap-renders/suites/clap-suite-20260731T211818136757Z"
BASELINE_CHECKPOINT = (
    "r2://intermediate-data/checkpoints/flow_simple_440k_1m_clap/"
    "flow_simple_440k_1m_clap-20260730T215504153Z-"
    "588c02237a964b0aad982370cf347086/last.ckpt"
)
SUMMARY_FIELDS = (
    "count",
    "mean",
    "std_population",
    "min",
    "p25",
    "median",
    "p75",
    "max",
)
PAIR_FIELDS = (
    "index",
    "prompt",
    "baseline_cosine_similarity",
    "baseline_cosine_distance",
    "candidate_cosine_similarity",
    "candidate_cosine_distance",
    "distance_delta_candidate_minus_baseline",
    "winner",
)


def build_paired_row(
    index: int,
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
) -> dict[str, int | float | str]:
    """Build one paired metric row with lower cosine distance as the winner.

    :param index: Stable one-based prompt index.
    :param baseline: Baseline CLAP comparison row.
    :param candidate: Candidate CLAP comparison row.
    :returns: Paired metrics and winning arm.
    :raises ValueError: Prompt identities differ between arms.
    """
    if baseline["prompt"] != candidate["prompt"]:
        raise ValueError(f"prompt mismatch at index {index}")
    baseline_distance = float(baseline["cosine_distance"])
    candidate_distance = float(candidate["cosine_distance"])
    delta = round(candidate_distance - baseline_distance, 12)
    winner = "tie"
    if delta < 0:
        winner = "candidate"
    elif delta > 0:
        winner = "baseline"
    return {
        "index": index,
        "prompt": baseline["prompt"],
        "baseline_cosine_similarity": float(baseline["cosine_similarity"]),
        "baseline_cosine_distance": baseline_distance,
        "candidate_cosine_similarity": float(candidate["cosine_similarity"]),
        "candidate_cosine_distance": candidate_distance,
        "distance_delta_candidate_minus_baseline": delta,
        "winner": winner,
    }


def write_aggregate_comparison(
    path: Path,
    paired_rows: Sequence[Mapping[str, int | float | str]],
) -> None:
    """Write per-arm distance statistics and paired win counts.

    :param path: Aggregate CSV destination.
    :param paired_rows: Per-prompt paired comparison rows.
    """
    fieldnames: list[str] = ["arm", *SUMMARY_FIELDS, "wins", "ties"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for arm in ("baseline", "candidate"):
            distances = [float(row[f"{arm}_cosine_distance"]) for row in paired_rows]
            writer.writerow(
                {
                    "arm": arm,
                    **summarize_cosine_distances(distances),
                    "wins": sum(row["winner"] == arm for row in paired_rows),
                    "ties": sum(row["winner"] == "tie" for row in paired_rows),
                }
            )


def _render_candidate(*args: str) -> None:
    """Run one isolated CLAP render with streamed output.

    :param *args: Arguments forwarded to ``synth-setter-clap``.
    """
    Command("synth-setter-clap")(*args, _out=sys.stdout, _err=sys.stderr)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as string-keyed rows.

    :param path: CSV source.
    :returns: Rows in source order.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write rows with a stable field order.

    :param path: CSV destination.
    :param fieldnames: Ordered column names.
    :param rows: Rows to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _publish(path: Path, destination: str) -> None:
    """Upload one local artifact to an exact R2 URI.

    :param path: Local source file.
    :param destination: Exact R2 object URI.
    """
    r2_io.upload(path, destination)


def _parse_args() -> argparse.Namespace:
    """Parse comparison sources and destinations.

    :returns: Validated command-line argument namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-suite", default=BASELINE_SUITE)
    parser.add_argument("--baseline-checkpoint", default=BASELINE_CHECKPOINT)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    """Render and publish one paired CLAP checkpoint comparison.

    :raises ValueError: An input URI or baseline manifest is invalid.
    """
    args = _parse_args()
    for value in (
        args.baseline_checkpoint,
        args.baseline_suite,
        args.candidate_checkpoint,
        args.destination,
    ):
        if not r2_io.is_r2_uri(value):
            raise ValueError(f"expected r2:// URI, got {value}")

    destination = args.destination.rstrip("/")
    baseline_suite = args.baseline_suite.rstrip("/")
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = output_dir / "source-manifest.csv"
    r2_io.download_to_path(f"{baseline_suite}/manifest.csv", source_manifest)
    baseline_rows = _read_csv(source_manifest)
    if not baseline_rows:
        raise ValueError("baseline manifest is empty")

    paired_rows: list[dict[str, int | float | str]] = []
    baseline_manifest: list[dict[str, object]] = []
    candidate_manifest: list[dict[str, object]] = []
    prompts: list[str] = []

    for index, baseline in enumerate(baseline_rows, start=1):
        prompt = baseline["prompt"]
        prompts.append(prompt)
        source_stem = Path(baseline["wav_r2_uri"]).stem
        sample_name = f"sample_{index:03d}_{source_stem.partition('-')[2]}"
        sample_dir = output_dir / "audio" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_uri = f"{destination}/audio/{sample_name}"

        baseline_wav_uri = f"{sample_uri}/baseline.wav"
        baseline_csv_uri = f"{sample_uri}/baseline.csv"
        r2_io.upload(baseline["wav_r2_uri"], baseline_wav_uri)
        r2_io.upload(baseline["csv_r2_uri"], baseline_csv_uri)

        candidate_wav = sample_dir / "candidate.wav"
        candidate_csv = sample_dir / "candidate.csv"
        candidate_wav_uri = f"{sample_uri}/candidate.wav"
        candidate_csv_uri = f"{sample_uri}/candidate.csv"
        if not candidate_wav.is_file() or not candidate_csv.is_file():
            candidate_wav.unlink(missing_ok=True)
            candidate_csv.unlink(missing_ok=True)
            _render_candidate(
                prompt,
                "--checkpoint",
                args.candidate_checkpoint,
                "--output",
                str(candidate_wav),
                "--upload-uri",
                candidate_wav_uri,
                "--device",
                args.device,
                "--seed",
                "0",
            )
        else:
            _publish(candidate_wav, candidate_wav_uri)
            _publish(candidate_csv, candidate_csv_uri)

        candidate = _read_csv(candidate_csv)[0]
        paired_rows.append(build_paired_row(index, baseline, candidate))
        baseline_manifest.append(
            {
                "index": index,
                **baseline,
                "wav_r2_uri": baseline_wav_uri,
                "csv_r2_uri": baseline_csv_uri,
            }
        )
        candidate_manifest.append({"index": index, **candidate})
        click.echo(f"[{index:03d}/{len(baseline_rows):03d}] {prompt}")

    prompts_path = output_dir / "prompts.txt"
    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    baseline_manifest_path = output_dir / "baseline_manifest.csv"
    candidate_manifest_path = output_dir / "candidate_manifest.csv"
    paired_path = output_dir / "paired_results.csv"
    aggregate_path = output_dir / "aggregate_comparison.csv"
    provenance_path = output_dir / "comparison_manifest.csv"
    manifest_fields = ("index", *baseline_rows[0].keys())
    _write_csv(baseline_manifest_path, manifest_fields, baseline_manifest)
    _write_csv(candidate_manifest_path, manifest_fields, candidate_manifest)
    _write_csv(paired_path, PAIR_FIELDS, paired_rows)
    write_aggregate_comparison(aggregate_path, paired_rows)
    _write_csv(
        provenance_path,
        ("key", "value"),
        [
            {"key": "created_at", "value": datetime.now(UTC).isoformat()},
            {"key": "baseline_suite", "value": baseline_suite},
            {"key": "baseline_checkpoint", "value": args.baseline_checkpoint},
            {"key": "candidate_checkpoint", "value": args.candidate_checkpoint},
            {"key": "destination", "value": destination},
            {"key": "seed", "value": 0},
            {"key": "prompt_count", "value": len(prompts)},
        ],
    )

    for path, relative in (
        (prompts_path, "prompts.txt"),
        (provenance_path, "comparison_manifest.csv"),
        (baseline_manifest_path, "manifests/baseline.csv"),
        (candidate_manifest_path, "manifests/candidate.csv"),
        (paired_path, "metrics/paired_results.csv"),
        (aggregate_path, "metrics/aggregate_comparison.csv"),
    ):
        _publish(path, f"{destination}/{relative}")
    baseline_summary = summarize_cosine_distances(
        [float(row["baseline_cosine_distance"]) for row in paired_rows]
    )
    candidate_summary = summarize_cosine_distances(
        [float(row["candidate_cosine_distance"]) for row in paired_rows]
    )
    for arm, summary in (("baseline", baseline_summary), ("candidate", candidate_summary)):
        path = output_dir / f"{arm}_aggregate_stats.csv"
        write_summary_csv(path, summary)
        _publish(path, f"{destination}/metrics/{arm}_aggregate_stats.csv")
    click.echo(destination)


if __name__ == "__main__":
    main()
