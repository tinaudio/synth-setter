"""Generate a report of shards stored in an R2 bucket via rclone."""

from __future__ import annotations

import re
import subprocess
from typing import NamedTuple, TypedDict

import click


class RcloneFile(NamedTuple):
    """A file entry from rclone ls output."""

    size_bytes: int
    filename: str


class ShardFileInfo(TypedDict):
    """Info about a single h5 file in the shard report."""

    filename: str
    size_bytes: int
    size_human: str
    is_suspect: bool


class ShardReport(TypedDict):
    """Typed analysis result from analyze_shards."""

    h5_count: int
    json_count: int
    other_count: int
    logical_shard_count: int
    total_h5_bytes: int
    total_h5_human: str
    threshold_gib: float
    h5_files: list[ShardFileInfo]
    suspect_files: list[ShardFileInfo]


def format_size(size_bytes: int) -> str:
    """Human-readable file size with binary prefixes."""
    if size_bytes == 0:
        return "0 B"

    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KiB"

    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MiB"

    return f"{size_bytes / 1024**3:.2f} GiB"


def parse_rclone_ls_output(output: str) -> list[RcloneFile]:
    """Parse rclone ls stdout into a list of RcloneFile entries."""
    files: list[RcloneFile] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Malformed rclone ls line (expected 'SIZE FILENAME'): {line!r}"
            )
        size_str, filename = parts
        try:
            size_bytes = int(size_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid size in rclone ls line (expected integer bytes): {line!r}"
            ) from exc
        files.append(RcloneFile(size_bytes=size_bytes, filename=filename))
    return files


def analyze_shards(files: list[RcloneFile], threshold_gib: float) -> ShardReport:
    """Classify files and build a shard report."""
    h5_files: list[RcloneFile] = []
    json_count = 0
    other_count = 0

    for f in files:
        if f.filename.endswith(".h5"):
            h5_files.append(f)
        elif f.filename.endswith(".json"):
            json_count += 1
        else:
            other_count += 1

    # Count logical shards by stripping chunk suffix and extension
    chunk_pattern = re.compile(r"-\d{4,}\.h5$")
    logical_ids: set[str] = set()
    for f in h5_files:
        logical_id = chunk_pattern.sub("", f.filename)
        # If the pattern didn't match (no chunk suffix), strip just the extension
        if logical_id == f.filename:
            logical_id = f.filename.removesuffix(".h5")
        logical_ids.add(logical_id)

    total_h5_bytes = sum(f.size_bytes for f in h5_files)
    threshold_bytes = threshold_gib * 1024**3

    h5_file_infos: list[ShardFileInfo] = []
    suspect_files: list[ShardFileInfo] = []

    for f in h5_files:
        is_suspect = f.size_bytes < threshold_bytes
        info = ShardFileInfo(
            filename=f.filename,
            size_bytes=f.size_bytes,
            size_human=format_size(f.size_bytes),
            is_suspect=is_suspect,
        )
        h5_file_infos.append(info)
        if is_suspect:
            suspect_files.append(info)

    return ShardReport(
        h5_count=len(h5_files),
        json_count=json_count,
        other_count=other_count,
        logical_shard_count=len(logical_ids),
        total_h5_bytes=total_h5_bytes,
        total_h5_human=format_size(total_h5_bytes),
        threshold_gib=threshold_gib,
        h5_files=h5_file_infos,
        suspect_files=suspect_files,
    )


def format_report(report: ShardReport, prefix: str) -> str:
    """Render a ShardReport to plain text."""
    lines: list[str] = []

    lines.append("=== R2 Shard Report ===")
    lines.append(f"Prefix: {prefix}")
    lines.append("")

    # Summary
    lines.append("Summary:")
    lines.append(f"  H5 files:       {report['h5_count']}")
    lines.append(f"  JSON files:     {report['json_count']}")
    lines.append(f"  Logical shards: {report['logical_shard_count']}")
    lines.append(f"  Total H5 size:  {report['total_h5_human']}")
    lines.append(f"  Suspect files:  {len(report['suspect_files'])}")
    lines.append("")

    # H5 files table
    lines.append("H5 Files:")
    if report["h5_files"]:
        # Determine column widths
        max_name = max(len(f["filename"]) for f in report["h5_files"])
        max_size = max(len(f["size_human"]) for f in report["h5_files"])
        max_name = max(max_name, len("Filename"))
        max_size = max(max_size, len("Size"))

        header = f"  {'Filename':<{max_name}}  {'Size':>{max_size}}  Status"
        lines.append(header)
        lines.append(f"  {'-' * max_name}  {'-' * max_size}  ------")

        for f in report["h5_files"]:
            status = "SUSPECT" if f["is_suspect"] else "OK"
            lines.append(f"  {f['filename']:<{max_name}}  {f['size_human']:>{max_size}}  {status}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Suspect files section
    if report["suspect_files"]:
        lines.append(f"Suspect Files (below {report['threshold_gib']:.2f} GiB):")
        for f in report["suspect_files"]:
            lines.append(f"  {f['filename']}  ({f['size_human']})")
        lines.append("")

    return "\n".join(lines)


def run_rclone_ls(prefix: str) -> str:
    """Run rclone ls and return stdout."""
    result = subprocess.run(
        ["rclone", "ls", prefix],  # noqa: S603, S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@click.command()
@click.argument("prefix")
@click.option("--size-threshold-gib", "-t", type=click.FloatRange(min=0), default=1.0)
def main(prefix: str, size_threshold_gib: float) -> None:
    """Generate a report of shards in an R2 bucket."""
    output = run_rclone_ls(prefix)
    files = parse_rclone_ls_output(output)
    report = analyze_shards(files, size_threshold_gib)
    text = format_report(report, prefix)
    click.echo(text)


if __name__ == "__main__":
    main()
