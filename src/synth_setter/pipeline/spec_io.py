"""``input_spec.json`` write/upload/discover helpers.

The frozen ``DatasetSpec`` is written to two well-known locations every run:

  - local: ``<output_dir>/data/<task_name>/<run_id>/metadata/input_spec.json``
    where ``output_dir`` is ``cfg.paths.output_dir`` (the Hydra per-run dir,
    resolved from ``${hydra:runtime.output_dir}``)
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
from urllib.parse import urlparse

from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.file_uri import file_uri_to_path
from synth_setter.pipeline.r2_io import downloaded_to_tempfile, upload_to_uri
from synth_setter.pipeline.schemas.spec import DatasetSpec

__all__ = [
    "find_input_specs",
    "load_spec_from_uri",
    "local_spec_path",
    "read_spec_text",
    "upload_spec",
    "write_spec_locally",
    "write_spec_to_path",
]

# Top-level local directory that mirrors the R2 ``prefix_root``. Hard-coded
# per ``storage-provenance-spec.md`` §3a — the local mirror always sits under
# ``output_dir/data/`` regardless of where R2 puts the prefix root.
_LOCAL_DATA_DIRNAME = "data"

# Schemes ``read_spec_text`` knows how to fetch. Bare arguments (no scheme,
# e.g. ``./data/spec.json`` or ``/abs/spec.json``) are treated as local paths
# and read against the process CWD — matching the convention used by rclone,
# fsspec, and similar libraries.
_LOCAL_FILESYSTEM_SCHEMES: frozenset[str] = frozenset({"", "file"})
_REMOTE_OBJECT_SCHEMES: frozenset[str] = frozenset({"r2"})


def read_spec_text(spec_uri: str) -> str:  # noqa: DOC502
    """Read spec JSON text from a bare path, ``file://`` URI, or ``r2://`` URI.

    Front-of-pipeline dispatcher: parses the scheme via :func:`urllib.parse.urlparse`
    and routes to the matching backend. Inputs without a scheme are treated as
    local paths (relative paths resolve against the process CWD, same as
    ``rclone`` / ``fsspec`` / Arrow). Unsupported schemes raise ``ValueError``
    so a typo (e.g. ``s3://``) fails loudly instead of being silently passed
    to :class:`~pathlib.Path`.

    :param spec_uri: Local filesystem path, ``file://`` URI, or ``r2://`` URI.
    :returns: The JSON text content of the spec file.
    :raises ValueError: ``spec_uri`` carries a scheme other than ``file://``
        or ``r2://``, or is a malformed ``file://`` URI (propagated from
        :func:`~synth_setter.pipeline.file_uri.file_uri_to_path`).
    """
    scheme = urlparse(spec_uri).scheme
    if scheme in _LOCAL_FILESYSTEM_SCHEMES:
        local_path = file_uri_to_path(spec_uri) if scheme == "file" else Path(spec_uri)
        return local_path.read_text()
    if scheme in _REMOTE_OBJECT_SCHEMES:
        with downloaded_to_tempfile(spec_uri) as fetched:
            return fetched.read_text()
    raise ValueError(
        f"unsupported spec_uri scheme {scheme!r}: {spec_uri!r}. "
        f"Supported: bare local paths, ``file://``, ``r2://``."
    )


def load_spec_from_uri(spec_uri: str) -> DatasetSpec:  # noqa: DOC502
    """Load a ``DatasetSpec`` from a local path, ``file://`` URI, or ``r2://`` URI.

    Thin wrapper that composes :func:`read_spec_text` with
    :meth:`DatasetSpec.model_validate_json` so callers don't have to pull
    in the cli runner (and its Hydra/workspace bootstrap) just to parse a
    spec.

    :param spec_uri: Local filesystem path, ``file://`` URI, or ``r2://`` URI.
    :returns: The parsed spec.
    :raises ValueError: ``spec_uri`` carries an unsupported scheme (propagated
        from :func:`read_spec_text`); also the base class of
        ``pydantic.ValidationError`` for malformed/stale spec JSON.
    """
    return DatasetSpec.model_validate_json(read_spec_text(spec_uri))


def local_spec_path(spec: DatasetSpec, output_dir: Path) -> Path:
    """Return the local path for ``spec``'s ``input_spec.json``.

    :param spec: The frozen DatasetSpec.
    :param output_dir: Operator-side artifact root. The runner
        (``cli/generate_dataset.py::main()``) passes
        ``Path(cfg.paths.output_dir)`` — the Hydra per-run dir.
    :returns: ``<output_dir>/data/<task_name>/<run_id>/metadata/input_spec.json``
        — i.e. under the Hydra per-run dir when invoked from the runner.
    """
    return (
        output_dir
        / _LOCAL_DATA_DIRNAME
        / spec.task_name
        / spec.run_id
        / "metadata"
        / INPUT_SPEC_FILENAME
    )


def write_spec_to_path(spec: DatasetSpec, target: Path) -> Path:
    """Serialize ``spec`` to an explicit filesystem path; create parent dirs.

    Used by callers that need a spec written at a specific location (e.g.
    ``cli.finalize_dataset.finalize_hdf5`` writes ``input_spec.json`` flat
    inside its scratch work_dir so reshard's default discovery — see
    ``data.reshard._load_spec`` — resolves it without a ``--spec`` override).
    :func:`write_spec_locally` is the operator-side counterpart that places
    the same bytes under the canonical ``<output_dir>/data/<task>/<run>/metadata/``
    layout.

    :param spec: The frozen DatasetSpec to serialize.
    :param target: Destination file path; must end in
        :data:`~synth_setter.pipeline.constants.INPUT_SPEC_FILENAME` so the
        write is discoverable by other tools that glob the standard name.
    :returns: The path written (same as ``target``).
    :raises ValueError: ``target.name`` is not
        :data:`~synth_setter.pipeline.constants.INPUT_SPEC_FILENAME`.
    """
    if target.name != INPUT_SPEC_FILENAME:
        raise ValueError(
            f"write_spec_to_path requires target.name == {INPUT_SPEC_FILENAME!r}; "
            f"got {target.name!r}."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return target


def write_spec_locally(spec: DatasetSpec, output_dir: Path) -> Path:
    """Serialize ``spec`` to its canonical operator-side path; create parent dirs.

    Thin wrapper that resolves the nested ``<output_dir>/data/<task>/<run>/metadata/``
    layout via :func:`local_spec_path` and delegates to
    :func:`write_spec_to_path` for the actual write. Callers that need to
    bypass the nested layout (e.g. write flat inside a tempdir) use
    :func:`write_spec_to_path` directly.

    :param spec: The frozen DatasetSpec to serialize.
    :param output_dir: Operator-side artifact root; see
        :func:`local_spec_path` for the runner's anchor convention.
    :returns: The path written.
    """
    return write_spec_to_path(spec, local_spec_path(spec, output_dir))


def find_input_specs(data_dir: Path) -> list[Path]:
    """Return every ``input_spec.json`` under ``data_dir/<task>/<run>/metadata/``.

    Globs the canonical local layout
    ``<data_dir>/<task_name>/<run_id>/metadata/input_spec.json``
    (mirrors ``local_spec_path``'s structure under ``output_dir/data/``).
    Called from ``skypilot_launch.main`` to discover the spec produced by the
    inner generator command before resolving its canonical R2 URI; also
    intended for any @hydra.main entrypoint that needs to re-upload or
    re-validate already-materialized specs.

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
