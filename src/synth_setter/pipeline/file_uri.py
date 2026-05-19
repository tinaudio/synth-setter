"""``file://`` URI helpers for spec_uri consumers.

Accepts RFC 8089 forms ``file:///abs/path`` and ``file://localhost/abs/path``;
rejects other authorities and empty paths so callers never silently
dereference the CWD. The r2/file/bare dispatch lives in
:func:`synth_setter.pipeline.spec_io.read_spec_text`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from synth_setter.pipeline.constants import FILE_URI_SCHEME

__all__ = [
    "FILE_URI_SCHEME",
    "file_uri_to_path",
    "is_file_uri",
    "local_path_from_arg",
]


def is_file_uri(uri: str) -> bool:
    """Return True if ``uri`` starts with ``file://``.

    :param uri: Candidate URI / path string.
    :return: ``True`` if ``uri`` begins with the ``file://`` scheme prefix.
    """
    return uri.startswith(FILE_URI_SCHEME)


def file_uri_to_path(file_uri: str) -> Path:
    """Convert a ``file://`` URI to a local filesystem ``Path``.

    Accepts ``file:///abs/path`` (RFC 8089 §2, empty authority) and
    ``file://localhost/abs/path`` (RFC 8089 §3, the only non-empty authority
    we resolve locally). Percent-encoded path segments are decoded.

    :param file_uri: Canonical ``file://`` URI string.
    :return: Absolute :class:`~pathlib.Path` on the local filesystem.
    :raises ValueError: ``file_uri`` is not a ``file://`` URI, names a remote
        host other than ``localhost``, or carries an empty path.
    """
    if not is_file_uri(file_uri):
        raise ValueError(f"not a file:// URI: {file_uri!r}")
    parsed = urlparse(file_uri)
    if parsed.netloc and parsed.netloc != "localhost":
        raise ValueError(
            f"file:// URI host must be empty or 'localhost', got {parsed.netloc!r}: {file_uri!r}"
        )
    path = unquote(parsed.path)
    if not path.startswith("/"):
        raise ValueError(f"file:// URI must carry an absolute path: {file_uri!r}")
    return Path(path)


def local_path_from_arg(spec_arg: str) -> Path:  # noqa: DOC502
    """Resolve a non-``r2://`` spec argument to a local filesystem ``Path``.

    Branches on ``file://`` to dispatch through :func:`file_uri_to_path`;
    bare arguments are passed through to :class:`~pathlib.Path` unchanged so
    pre-existing local-path callers keep working.

    :param spec_arg: Either a ``file://`` URI or a bare filesystem path.
    :return: Local :class:`~pathlib.Path`.
    :raises ValueError: ``spec_arg`` is a malformed ``file://`` URI (propagated
        from :func:`file_uri_to_path`).
    """
    if is_file_uri(spec_arg):
        return file_uri_to_path(spec_arg)
    return Path(spec_arg)
