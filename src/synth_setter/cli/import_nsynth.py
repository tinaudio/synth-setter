"""Packaged CLI for immutable NSynth ingest and byte-level verification."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from synth_setter.pipeline.data.nsynth_import import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_REMOTE_ROOT,
    OFFICIAL_EXPECTED_COUNTS,
    SPLITS,
    download_and_verify_nsynth,
    ingest_nsynth,
)

_EXPECTED_COUNTS_HELP = (
    "Override official counts as train=N,valid=N,test=N; intended for tiny fixtures."
)


def _parse_expected_counts(value: str | None) -> dict[str, int]:
    """Parse the explicit three-split count override or return official counts.

    :param value: Comma-separated ``split=count`` assignments, or ``None``.
    :returns: Positive count for each official split.
    :raises click.BadParameter: The override is malformed, incomplete, or non-positive.
    """
    if value is None:
        return dict(OFFICIAL_EXPECTED_COUNTS)
    assignments = value.split(",")
    counts: dict[str, int] = {}
    try:
        for assignment in assignments:
            split, raw_count = assignment.split("=", maxsplit=1)
            if split in counts:
                raise click.BadParameter(
                    f"duplicate split {split!r}", param_hint="--expected-counts"
                )
            counts[split] = int(raw_count)
    except ValueError as exc:
        raise click.BadParameter(
            "expected train=N,valid=N,test=N", param_hint="--expected-counts"
        ) from exc
    if set(counts) != set(SPLITS) or any(count < 1 for count in counts.values()):
        raise click.BadParameter(
            "expected each of train, valid, and test exactly once with a positive count",
            param_hint="--expected-counts",
        )
    return counts


def _counts_text(counts: dict[str, int]) -> str:
    """Render split counts in canonical order.

    :param counts: Count keyed by official split.
    :returns: Human-readable comma-separated counts.
    """
    return ", ".join(f"{split}={counts[split]}" for split in SPLITS)


@click.group()
def main() -> None:
    """Import official NSynth extracts into immutable Lance Blob-v2 datasets."""


@main.command("ingest", help="Build and upload all three official splits from SOURCE_ROOT.")
@click.argument(
    "source_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("output_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--remote-root",
    default=DEFAULT_REMOTE_ROOT,
    show_default=True,
    help="Immutable R2 prefix receiving datasets, sidecars, then manifest.",
)
@click.option(
    "--batch-size",
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum metadata rows and WAV payloads held per Arrow batch.",
)
@click.option("--expected-counts", default=None, help=_EXPECTED_COUNTS_HELP)
def ingest_command(
    source_root: Path,
    output_root: Path,
    remote_root: str,
    batch_size: int,
    expected_counts: str | None,
) -> None:
    """Build and upload all three official splits from SOURCE_ROOT.

    :param source_root: Parent of ``nsynth-{train,valid,test}``.
    :param output_root: New local root for the completed import.
    :param remote_root: Immutable R2 destination prefix.
    :param batch_size: Maximum records and WAVs held per batch.
    :param expected_counts: Optional explicit three-split count override.
    :raises click.ClickException: Source validation, local publication, or upload fails.
    """
    counts = _parse_expected_counts(expected_counts)
    try:
        manifest = ingest_nsynth(
            source_root,
            output_root,
            remote_root=remote_root,
            expected_counts=counts,
            batch_size=batch_size,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Imported {manifest.total_count} NSynth rows ({_counts_text(counts)}); "
        f"uploaded immutable artifacts to {remote_root.rstrip('/')} with manifest last."
    )


@main.command("verify", help="Download the remote import and compare it with SOURCE_ROOT.")
@click.argument(
    "source_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("download_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--remote-root",
    default=DEFAULT_REMOTE_ROOT,
    show_default=True,
    help="Immutable R2 prefix downloaded in full before verification.",
)
@click.option(
    "--batch-size",
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum metadata rows and Blob-v2 handles scanned together.",
)
@click.option("--expected-counts", default=None, help=_EXPECTED_COUNTS_HELP)
def verify_command(
    source_root: Path,
    download_root: Path,
    remote_root: str,
    batch_size: int,
    expected_counts: str | None,
) -> None:
    """Download the remote import and compare it with SOURCE_ROOT.

    :param source_root: Parent of ``nsynth-{train,valid,test}``.
    :param download_root: New local root receiving the complete remote prefix.
    :param remote_root: Immutable R2 source prefix.
    :param batch_size: Maximum metadata rows and blob handles scanned together.
    :param expected_counts: Optional explicit three-split count override.
    :raises click.ClickException: Download, strict parsing, or any comparison fails.
    """
    counts = _parse_expected_counts(expected_counts)
    try:
        summary = download_and_verify_nsynth(
            source_root,
            download_root,
            remote_root=remote_root,
            expected_counts=counts,
            batch_size=batch_size,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Verified {summary.total_count} NSynth rows ({_counts_text(summary.split_counts)}): "
        f"{summary.mismatches} mismatches; source metadata, JSON sidecars, and WAV blobs match."
    )
