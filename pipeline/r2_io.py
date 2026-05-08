"""R2 object-store I/O primitives shared across pipeline stages.

Wraps `rclone` with a small set of typed helpers so worker, launcher, and CI
validation code can share one implementation. Resolution of `r2:` is left to
the caller's environment — the standard `RCLONE_CONFIG_R2_*` vars must be set
when any of these functions runs.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

R2_URI_SCHEME = "r2://"


def is_r2_uri(uri: str) -> bool:
    """Return True if `uri` is an `r2://bucket/key` URI."""
    return uri.startswith(R2_URI_SCHEME)


def _to_rclone_path(r2_uri: str) -> str:
    """Convert an `r2://bucket/key` URI to rclone's `r2:bucket/key` syntax.

    Raises ValueError if `r2_uri` is not an r2:// URI — callers should branch
    on `is_r2_uri` before calling.
    """
    if not is_r2_uri(r2_uri):
        raise ValueError(f"not an r2:// URI: {r2_uri!r}")
    return "r2:" + r2_uri[len(R2_URI_SCHEME) :]


def download_to_path(r2_uri: str, dest_path: Path) -> None:
    """Download an R2 object to a specific local file path.

    Uses `rclone copyto` (file→file) so the destination filename is preserved
    exactly — `rclone copy` would treat `dest_path` as a directory and write
    the source basename inside it.
    """
    args = [  # noqa: S607 — rclone resolved by image's PATH
        "rclone",
        "copyto",
        "--checksum",
        _to_rclone_path(r2_uri),
        str(dest_path),
    ]
    subprocess.check_call(args)  # noqa: S603 — args from validated URI


def upload_to_uri(local_path: Path, r2_uri: str) -> None:
    """Upload a local file to a specific R2 object URI.

    Uses `rclone copyto` so the destination filename matches the URI exactly (not the source
    basename). Connection-level timeouts and retries are rclone's job: bounds the TCP connect phase
    and the per-request timeout, retries the whole copy on transient failure, and emits per-request
    debug logs so a CI failure leaves actionable evidence in stdout.
    """
    args = [  # noqa: S607 — rclone resolved by image's PATH
        "rclone",
        "copyto",
        "-vv",
        "--checksum",
        "--contimeout=30s",
        "--timeout=300s",
        "--retries=3",
        str(local_path),
        _to_rclone_path(r2_uri),
    ]
    subprocess.check_call(args)  # noqa: S603 — args from validated URI


@contextmanager
def downloaded_to_tempfile(r2_uri: str) -> Iterator[Path]:
    """Download an R2 object to a tempdir; yield local Path; clean up on exit.

    The local filename matches the URI's basename. Caller uses the yielded Path for reads;
    everything is removed when the context exits.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / Path(r2_uri).name
        download_to_path(r2_uri, local_path)
        yield local_path
