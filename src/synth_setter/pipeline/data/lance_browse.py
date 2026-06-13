"""Export single-file Lance shards into SmooSense-browsable Lance datasets.

The pipeline writes shards and finalized splits via ``LanceFileWriter`` — the
single-file Lance v2 format :mod:`synth_setter.pipeline.data.lance_shard`
emits. SmooSense's ``sense db`` browses Lance *datasets* (directories carrying
a manifest), not single-file shards, so this module re-materializes a shard's
batches as a dataset the browser can open. Schema metadata (the embedded
:class:`~synth_setter.pipeline.schemas.shard_metadata.ShardMetadata`) rides
along so the browser's metadata view shows the render parameters.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import lance
import pyarrow as pa
from lance.file import LanceFileReader


def export_shard_to_dataset(shard_path: Path | str, dataset_dir: Path | str) -> Path:
    """Re-materialize one single-file Lance shard as a browsable Lance dataset.

    Non-atomic: the destination is wiped and rewritten, so concurrent calls on
    the same ``dataset_dir`` may corrupt it.

    :param shard_path: Source ``.lance`` shard file (the single-file format the
        pipeline writes), never a Lance dataset directory.
    :param dataset_dir: Destination dataset directory; replaced if it exists.
    :returns: The browsable Lance dataset directory just written.
    :raises ValueError: ``shard_path`` is a directory (already a dataset layout).
    :raises FileNotFoundError: ``shard_path`` does not exist.
    """
    shard_path = Path(shard_path)
    if shard_path.is_dir():
        raise ValueError(
            f"expected a single-file Lance shard, got a directory "
            f"(already a Lance dataset?): {shard_path}"
        )
    if not shard_path.is_file():
        raise FileNotFoundError(f"Lance shard file not found: {shard_path}")

    reader = LanceFileReader(str(shard_path))
    schema = reader.metadata().schema

    dataset_dir = Path(dataset_dir)
    # Wipe first so repeated exports don't leave stale Lance version fragments behind.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    batches = pa.RecordBatchReader.from_batches(schema, reader.read_all().to_batches())
    lance.write_dataset(batches, str(dataset_dir), mode="create")
    return dataset_dir


def duplicate_stems(paths: Sequence[Path | str]) -> list[str]:
    """Return the filename stems shared by two or more ``paths``, sorted.

    Each browse-db table is named by its source's stem, so a repeated stem is a table-name
    collision.

    :param paths: Source paths/URIs to check for stem collisions.
    :returns: Sorted stems that occur more than once; empty when all distinct.
    """
    names = [Path(p).stem for p in paths]
    return sorted(name for name, count in Counter(names).items() if count > 1)


def build_browse_db(shard_paths: Sequence[Path | str], db_dir: Path | str) -> list[Path]:
    """Export each shard into ``<db_dir>/<stem>.lance`` so ``sense db`` can open them.

    On an unexpected failure mid-export, ``db_dir`` is left holding the tables
    written so far — re-running overwrites them, so a partial run is recoverable.

    :param shard_paths: One or more single-file ``.lance`` shards or splits;
        each becomes a table named after its filename stem.
    :param db_dir: Browse-db root passed to ``sense db``; created if missing.
    :returns: The dataset directories written, in input order.
    :raises ValueError: ``shard_paths`` is empty, or two sources share a stem
        (which would collide on one table name).
    """
    if not shard_paths:
        raise ValueError("build_browse_db needs at least one shard to browse")
    duplicates = duplicate_stems(shard_paths)
    if duplicates:
        raise ValueError(f"duplicate table name(s) across sources: {duplicates}")

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return [
        export_shard_to_dataset(shard_path, db_dir / f"{Path(shard_path).stem}.lance")
        for shard_path in shard_paths
    ]
