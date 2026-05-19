"""``input_spec.json`` write/upload/discover helpers.

The frozen ``DatasetSpec`` is written to two well-known locations every run:

  - local: ``<repo_root>/data/<task_name>/<run_id>/metadata/input_spec.json``
    (the runner anchors at ``_REPO_ROOT``; see :func:`local_spec_path`'s
    ``output_dir`` parameter for why ``cfg.paths.output_dir`` is not used)
  - R2:    ``spec.r2.input_spec_uri()`` (see ``R2Location.input_spec_uri``)

The local path anticipates the ``docs/design/storage-provenance-spec.md`` §3a
*target* layout, which places ``input_spec.json`` under a ``metadata/``
subdirectory. The R2 destination is the MVP flat shape that
``R2Location.input_spec_uri()`` returns today
(``<r2_prefix_root>/<task_name>/<run_id>/<INPUT_SPEC_FILENAME>``); migrating
the R2 object under ``metadata/`` is tracked by #385.

``find_input_specs`` is the inverse of ``local_spec_path`` — given the local
``data/`` root, it discovers every spec written there.

The write/upload helpers are idempotent: re-running the same operator command
rewrites the same bytes to the same path / key.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.r2_io import upload_to_uri
from synth_setter.pipeline.schemas.spec import DatasetSpec

__all__ = [
    "find_input_specs",
    "local_spec_path",
    "upload_spec",
    "write_spec_locally",
]

# Top-level local directory that mirrors the R2 ``prefix_root``. Hard-coded
# per ``storage-provenance-spec.md`` §3a — the local mirror always sits under
# ``output_dir/data/`` regardless of where R2 puts the prefix root.
_LOCAL_DATA_DIRNAME = "data"


def local_spec_path(spec: DatasetSpec, output_dir: Path) -> Path:
    """Return the local path for ``spec``'s ``input_spec.json``.

    :param spec: The frozen DatasetSpec.
    :param output_dir: Operator-side artifact root. The runner
        (``cli/generate_dataset.py::main()``) passes ``_REPO_ROOT`` —
        ``cfg.paths.output_dir`` is pinned to the same value as a shim
        for ``${hydra:runtime.output_dir}`` resolution, but is not the
        anchor read back here.
    :returns: ``<output_dir>/data/<task_name>/<run_id>/metadata/input_spec.json``
        — i.e. ``<repo_root>/data/...`` when invoked from the runner.
    """
    return (
        output_dir
        / _LOCAL_DATA_DIRNAME
        / spec.task_name
        / spec.run_id
        / "metadata"
        / INPUT_SPEC_FILENAME
    )


def write_spec_locally(spec: DatasetSpec, output_dir: Path) -> Path:
    """Serialize ``spec`` to its local path; create parent dirs.

    :param spec: The frozen DatasetSpec to serialize.
    :param output_dir: Operator-side artifact root; see
        :func:`local_spec_path` for the runner's anchor convention.
    :returns: The path written.
    """
    target = local_spec_path(spec, output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return target


def find_input_specs(data_dir: Path) -> list[Path]:
    """Return every ``input_spec.json`` under ``data_dir/<task>/<run>/metadata/``.

    Globs the canonical local layout
    ``<data_dir>/<task_name>/<run_id>/metadata/input_spec.json``
    (mirrors ``local_spec_path``'s structure under ``output_dir/data/``).
    Intended for the @hydra.main entrypoint to discover already-materialized
    specs to re-upload or re-validate; not yet wired in ``cli/generate_dataset.py``.

    :param data_dir: Local ``data/`` directory (typically
        ``cfg.paths.output_dir / "data"``).
    :returns: Sorted list of matching ``input_spec.json`` paths. Empty if
        ``data_dir`` does not exist or has no matches.
    """
    return sorted(data_dir.glob(f"*/*/metadata/{INPUT_SPEC_FILENAME}"))


def upload_spec(spec: DatasetSpec) -> str:
    """Upload ``spec`` to its R2 URI; return that URI.

    Serializes to a NamedTemporaryFile, copies via ``r2_io.upload_to_uri``
    (which uses ``rclone copyto`` with checksum / timeout / retry flags), and
    removes the temp file. Idempotent at the R2 key — same content + same
    key = no-op object overwrite. A non-zero rclone exit propagates as
    ``subprocess.CalledProcessError`` from ``r2_io.upload_to_uri``.

    :param spec: The frozen DatasetSpec to upload.
    :returns: The R2 URI (``spec.r2.input_spec_uri()``).
    """
    r2_uri = spec.r2.input_spec_uri()
    # NamedTemporaryFile(delete=False) returns an open file with .name set
    # synchronously, so tmp_path is available before any write that could raise.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — manual cleanup in finally
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(spec.model_dump_json(indent=2))
        upload_to_uri(tmp_path, r2_uri)
    finally:
        tmp_path.unlink(missing_ok=True)
    return r2_uri
