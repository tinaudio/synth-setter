"""Canonical metadata keys and safe source resolution for materialized checkpoints.

Typical use calls :func:`canonical_base_checkpoint_source` before logging a
materialized checkpoint's source alongside its local path and SHA-256.
"""

from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

BASE_CHECKPOINT_SOURCE_ENV = "SYNTH_SETTER_BASE_CHECKPOINT_SOURCE"
BASE_CHECKPOINT_RESOLVED_SOURCE_KEY = "model/base_checkpoint/resolved_source"
BASE_CHECKPOINT_MATERIALIZED_PATH_KEY = "model/base_checkpoint/materialized_path"
BASE_CHECKPOINT_SHA256_KEY = "model/base_checkpoint/sha256"
BASE_CHECKPOINT_IDENTITY_KEYS = (
    BASE_CHECKPOINT_RESOLVED_SOURCE_KEY,
    BASE_CHECKPOINT_MATERIALIZED_PATH_KEY,
    BASE_CHECKPOINT_SHA256_KEY,
)


def redact_checkpoint_source(source: str) -> str:
    """Remove URI credentials, query parameters, and fragments before logging.

    :param source: Canonical local or remote checkpoint source.
    :returns: Source safe to publish as run metadata.
    """
    parsed = urlsplit(source)
    if not parsed.scheme:
        return source
    host = parsed.netloc.rsplit("@", maxsplit=1)[-1]
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def canonical_base_checkpoint_source(materialized_path: Path, source: str | Path | None) -> str:
    """Resolve local checkpoint sources to strict file URIs and preserve remote sources.

    :param materialized_path: Local checkpoint loaded when no separate source is supplied.
    :param source: Original checkpoint source, or ``None`` for ``materialized_path``.
    :returns: Canonical local file URI or remote source without credentials.
    :raises ValueError: An explicit source is blank or a file URI names a non-local host.
    """
    if source is None:
        return materialized_path.expanduser().resolve(strict=True).as_uri()
    if isinstance(source, Path):
        return source.expanduser().resolve(strict=True).as_uri()
    if not source.strip():
        raise ValueError("base_checkpoint_source cannot be blank")

    redacted_source = redact_checkpoint_source(source)
    parsed = urlsplit(redacted_source)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"base_checkpoint_source file URI must be local, got host {parsed.netloc!r}"
            )
        return Path(unquote(parsed.path)).expanduser().resolve(strict=True).as_uri()
    if parsed.scheme:
        return redacted_source
    return Path(source).expanduser().resolve(strict=True).as_uri()
