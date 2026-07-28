"""Run ``synth-setter-bbc-sfx --help`` for the BBC import workflow."""

from pathlib import Path

import click

from synth_setter.pipeline.data.bbc_sfx import (
    DEFAULT_SHARD_TARGET_BYTES,
    DEFAULT_VERIFY_BATCH_SIZE,
    REMOTE_URI,
    convert_release,
    download_source,
    upload_release,
    verify_release,
)


@click.group()
def main() -> None:
    """Import and verify the BBC Sound Effects Internet Archive collection."""


@main.command("download")
@click.argument("source_root", type=click.Path(path_type=Path, file_okay=False))
def download_command(source_root: Path) -> None:
    """Download WAVs and snapshot IA metadata below SOURCE_ROOT.

    :param source_root: Destination directory for the Internet Archive item.
    """
    snapshot = download_source(source_root)
    click.echo(f"download inventory: {len(snapshot.files)} WAV files")


@main.command("convert")
@click.argument("source_root", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.argument("release_root", type=click.Path(path_type=Path, file_okay=False))
@click.option(
    "--shard-target-bytes",
    type=click.IntRange(min=1),
    default=DEFAULT_SHARD_TARGET_BYTES,
    show_default=True,
)
def convert_command(source_root: Path, release_root: Path, shard_target_bytes: int) -> None:
    """Convert SOURCE_ROOT into immutable Lance shards below RELEASE_ROOT.

    :param source_root: Directory containing the downloaded Internet Archive item.
    :param release_root: New directory to receive the completed release.
    :param shard_target_bytes: Approximate maximum source bytes assigned to each shard.
    """
    manifest = convert_release(source_root, release_root, shard_target_bytes=shard_target_bytes)
    click.echo(
        f"converted {manifest.total_rows} rows and {manifest.total_source_bytes} bytes "
        f"into {len(manifest.shards)} shards"
    )


@main.command("upload")
@click.argument("release_root", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--remote-uri", default=REMOTE_URI, show_default=True)
def upload_command(release_root: Path, remote_uri: str) -> None:
    """Upload RELEASE_ROOT immutably without publishing completion.

    :param release_root: Completed local release to upload.
    :param remote_uri: Destination object-store prefix.
    """
    manifest = upload_release(release_root, remote_uri=remote_uri)
    click.echo(f"uploaded {len(manifest.release_files)} immutable release files")


@main.command("verify")
@click.argument("source_root", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.argument("download_root", type=click.Path(path_type=Path, file_okay=False))
@click.option(
    "--release-manifest",
    "release_manifest_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--remote-uri", default=REMOTE_URI, show_default=True)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_VERIFY_BATCH_SIZE,
    show_default=True,
)
def verify_command(
    source_root: Path,
    download_root: Path,
    *,
    release_manifest_path: Path,
    remote_uri: str,
    batch_size: int,
) -> None:
    """Verify downloaded release bytes and publish manifest.json last.

    :param source_root: Directory containing the authoritative source item.
    :param download_root: New directory to receive the remote release.
    :param release_manifest_path: Local completion candidate to verify and publish.
    :param remote_uri: Source prefix and completion-manifest destination.
    :param batch_size: Maximum metadata rows read per verification batch.
    """
    result = verify_release(
        source_root,
        download_root,
        release_manifest_path,
        batch_size=batch_size,
        remote_uri=remote_uri,
    )
    click.echo(
        f"verified {result.rows} rows and {result.source_bytes} bytes; "
        f"{result.mismatches} mismatches"
    )


if __name__ == "__main__":
    main()
