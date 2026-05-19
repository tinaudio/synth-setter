"""R2 object-store I/O primitives shared across pipeline stages.

Wraps `rclone` with a small set of typed helpers so worker, launcher, and CI
validation code can share one implementation. Resolution of `r2:` is left to
the caller's environment — the standard `RCLONE_CONFIG_R2_*` vars must be set
when any of these functions runs; ``ensure_r2_env_loaded`` is the load + validate
+ auth-check entry point callers use to set them up.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from synth_setter.pipeline.constants import R2_URI_SCHEME, RCLONE_REMOTE

__all__ = [
    "R2_URI_SCHEME",
    "download_to_path",
    "downloaded_to_tempfile",
    "ensure_r2_env_loaded",
    "is_r2_uri",
    "object_size",
    "shard_uri",
    "to_rclone_path",
    "upload_to_uri",
]

# Keys rclone needs to authenticate to R2. Type and provider are defaulted by the
# launcher (s3 / Cloudflare); the three below are secrets that must come from a
# .env file or process env — no built-in default.
_SECRET_R2_ENV_KEYS: tuple[str, ...] = (
    "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "RCLONE_CONFIG_R2_ENDPOINT",
)


def ensure_r2_env_loaded(env_file: Path | None = None) -> None:
    """Load ``RCLONE_CONFIG_R2_*`` from ``env_file`` into ``os.environ``; validate.

    Two-step pre-flight that callers run once before invoking any other helper
    in this module:

    1. If ``env_file`` is provided and exists on disk, mirror every key
       prefixed with ``RCLONE_CONFIG_R2_`` from that dotenv file into
       ``os.environ`` (dotenv values overwrite — matches the launcher's
       precedence so the same view applies on every entry point).
    2. Verify the three secret keys in ``_SECRET_R2_ENV_KEYS`` are present in
       process env, then run ``rclone lsd r2:`` as an auth ping. Either check
       failing raises ``RuntimeError`` with an actionable message.

    No-op on the dotenv step if ``env_file`` is ``None`` or doesn't exist; the
    presence+auth check still runs against whatever ``os.environ`` already has.

    :param env_file: Optional dotenv file to merge into ``os.environ`` first
        (typically ``sky_cfg.env_file``).
    :raises RuntimeError: A required secret key is unset after the load, or
        ``rclone lsd r2:`` exits non-zero (bad creds, network, etc.).
    """
    if env_file is not None and env_file.is_file():
        # Lazy import so importing this module (e.g. from validate_spec.py in
        # the lean --no-deps CI env) does not pull python-dotenv. Refs #1120.
        from dotenv import dotenv_values

        for key, value in dotenv_values(env_file).items():
            if key and key.startswith("RCLONE_CONFIG_R2_") and value is not None:
                os.environ[key] = value

    missing = [k for k in _SECRET_R2_ENV_KEYS if k not in os.environ]
    if missing:
        where = str(env_file) if env_file is not None else "<env_file not set>"
        raise RuntimeError(
            f"R2 credentials missing from process env after dotenv load: {', '.join(missing)}. "
            f"Set RCLONE_CONFIG_R2_* in process env (e.g. `docker run -e ...=...`) "
            f"or populate {where}."
        )

    # Auth ping — fail fast on bad creds instead of letting the first real
    # operation fail several seconds in. Cheap (<1 RTT to R2) and unambiguous:
    # `rclone lsd r2:` lists buckets visible to the configured creds.
    result = subprocess.run(  # noqa: S603 — args are literal strings
        ["rclone", "lsd", "r2:", "--contimeout=10s", "--timeout=30s"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_excerpt = result.stderr.strip().splitlines()[-1][:200] if result.stderr else ""
        raise RuntimeError(
            "rclone failed to authenticate to R2 with the resolved credentials "
            f"(exit {result.returncode}): {stderr_excerpt}"
        )


def is_r2_uri(uri: str) -> bool:
    """Return True if `uri` is an `r2://bucket/key` URI."""
    return uri.startswith(R2_URI_SCHEME)


def to_rclone_path(r2_uri: str) -> str:
    """Convert an `r2://bucket/key` URI to rclone's `r2:bucket/key` syntax.

    Callers should branch on `is_r2_uri` before calling.

    :param r2_uri: Canonical ``r2://bucket/key`` URI string.
    :return: ``r2:bucket/key`` rclone-form path string.
    :raises ValueError: ``r2_uri`` is not an ``r2://`` URI.
    """
    if not is_r2_uri(r2_uri):
        raise ValueError(f"not an r2:// URI: {r2_uri!r}")
    return f"{RCLONE_REMOTE}:" + r2_uri[len(R2_URI_SCHEME) :]


# Backward-compatible alias for the previously-private translator.
_to_rclone_path = to_rclone_path


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


def shard_uri(bucket: str, prefix: str, shard_filename: str) -> str:
    """Build the canonical R2 URI for a shard object: ``r2://{bucket}/{prefix}{filename}``.

    Centralizes the convention so the worker's skip-existing probe, the worker's upload, and the
    CI validator agree on one URI shape — protects resumability and reconciliation from
    prefix-format drift.
    """
    return f"{R2_URI_SCHEME}{bucket}/{prefix}{shard_filename}"


def object_size(r2_uri: str) -> int | None:
    """Return the size in bytes of the R2 object at ``r2_uri``, or ``None`` if it does not exist.

    Uses ``rclone lsf --format=s`` for a size-only single-line listing: integer stdout if the
    object exists, empty stdout if it does not. A non-zero rclone exit (auth, network, etc.) raises
    ``subprocess.CalledProcessError`` so callers fail fast on environmental issues rather than
    silently masking them. ``--checksum`` does not apply to listings.

    A zero-size object exists and returns ``0``. Callers that want to treat zero-size as absent
    (e.g. defending against half-uploaded objects) test ``size and size > 0`` themselves.
    """
    args = [  # noqa: S607 — rclone resolved by image's PATH
        "rclone",
        "lsf",
        "--format=s",
        _to_rclone_path(r2_uri),
    ]
    result = subprocess.run(  # noqa: S603 — args from validated URI
        args, check=True, capture_output=True, text=True
    )
    out = result.stdout.strip()
    return int(out) if out else None


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
