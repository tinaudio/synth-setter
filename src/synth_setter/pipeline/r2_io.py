"""R2 object-store I/O primitives shared across pipeline stages.

Wraps `rclone` with a small set of typed helpers so worker, launcher, and CI
validation code can share one implementation. Resolution of `r2:` is left to
the caller's environment — the standard `RCLONE_CONFIG_R2_*` vars must be set
when any of these functions runs; ``ensure_r2_env_loaded`` is the load + validate
+ auth-check entry point callers use to set them up.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from synth_setter.pipeline.constants import R2_URI_SCHEME, RCLONE_REMOTE

__all__ = [
    "R2_URI_SCHEME",
    "RemoteEntry",
    "download_dir_no_overwrite",
    "download_to_path",
    "downloaded_to_tempfile",
    "ensure_r2_env_loaded",
    "from_s3_uri",
    "is_r2_reachable",
    "is_r2_uri",
    "lance_target",
    "list_entries",
    "object_size",
    "purge_prefix",
    "r2_directory_exists",
    "r2_storage_options",
    "shard_uri",
    "to_rclone_path",
    "to_s3_uri",
    "upload",
    "upload_dir",
    "upload_to_uri",
]

# Keys rclone needs to authenticate to R2. The three below are secrets that must
# come from a .env file or process env — no built-in default.
_SECRET_R2_ENV_KEYS: tuple[str, ...] = (
    "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
    "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
    "RCLONE_CONFIG_R2_ENDPOINT",
)

# Structural keys rclone's env-override convention needs to assemble a complete
# remote definition. Without them `rclone lsd r2:` reports
# "didn't find section in config file" even when the three secrets above are
# populated. Defaulted here (not required from callers) so steps that wire only
# the secrets — e.g. the skypilot-local matrix in `generate-dataset-shards.yaml`
# — still get a working `r2:` remote. setdefault preserves caller overrides.
_R2_STRUCTURAL_DEFAULTS: dict[str, str] = {
    "RCLONE_CONFIG_R2_TYPE": "s3",
    "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
}

_CHECKOUT_MARKER = ".project-root"
_WORKSPACE_ENV = "SYNTH_SETTER_WORKSPACE"


def _default_env_file() -> Path:
    """Resolve the workspace dotenv path without importing optional launcher deps.

    :returns: ``$SYNTH_SETTER_WORKSPACE/.env`` when set, otherwise the checkout
        marker root's ``.env`` or the current directory's ``.env`` fallback.
    """
    workspace = os.environ.get(_WORKSPACE_ENV, "").strip()
    if workspace:
        return Path(workspace).resolve() / ".env"
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _CHECKOUT_MARKER).is_file():
            return candidate / ".env"
    return Path.cwd().resolve() / ".env"


_DEFAULT_ENV_FILE = _default_env_file()

# rclone ``--timeout`` is the IO idle timeout, not a wall-clock cap. A whole eval
# run dir — rendered audio, predictions, metrics — can stream far longer than a
# single shard, so the directory upload bounds it generously rather than tripping
# a healthy large transfer at the 5-minute default the single-file helpers use.
_UPLOAD_DIR_TIMEOUT = "3h"


def _rclone_argv(verb: str, *operands: str, timeout: str = "300s") -> list[str]:
    """Build an rclone argv with the shared reliability-flag block, then operands.

    Centralizes ``-vv --checksum --contimeout=30s --timeout=<timeout> --retries=3``
    so every transfer helper retries transient blips identically. ``--timeout`` is
    the IO idle timeout, not a wall-clock cap; only directory uploads widen it past
    the 300s single-file default.

    :param verb: rclone subcommand (``copy`` / ``copyto``).
    :param \\*operands: Per-call args (extra flags like ``--immutable`` plus the
        source/destination paths) appended verbatim after the shared flags.
    :param timeout: Value for ``--timeout`` (IO idle timeout).
    :returns: The full ``["rclone", verb, ...flags, *operands]`` argv list.
    """
    return [
        "rclone",
        verb,
        "-vv",
        "--checksum",
        "--contimeout=30s",
        f"--timeout={timeout}",
        "--retries=3",
        *operands,
    ]


def _resolve_env_file(env_file: Path | None) -> Path:
    """Resolve the dotenv file used by R2 preflight.

    :param env_file: Explicit dotenv path supplied by a caller, or ``None`` for
        the default dotenv lookup.
    :returns: The explicit path when provided, otherwise the resolved default dotenv path.
    """
    return env_file if env_file is not None else _DEFAULT_ENV_FILE


def _load_r2_env_file(env_file: Path) -> None:
    """Mirror RCLONE_CONFIG_R2_* keys from an existing dotenv file into process env.

    :param env_file: Resolved dotenv path; missing files are intentionally ignored.
    """
    if not env_file.is_file():
        return
    for key, value in dotenv_values(env_file).items():
        if key and key.startswith("RCLONE_CONFIG_R2_") and value is not None:
            os.environ[key] = value


def _env_value_is_set(key: str) -> bool:
    """Return whether an env var is present with a non-blank value.

    :param key: Environment variable name to inspect.
    :returns: ``True`` when the key exists and is not empty/whitespace.
    """
    return bool(os.environ.get(key, "").strip())


def ensure_r2_env_loaded(env_file: Path | None = None) -> None:
    """Load ``RCLONE_CONFIG_R2_*`` from dotenv/process env; validate access.

    Three-step pre-flight that callers run once before invoking any other helper
    in this module:

    1. Mirror every key prefixed with ``RCLONE_CONFIG_R2_`` from ``env_file``
       into ``os.environ`` when that file exists. If ``env_file`` is ``None``,
       use the default dotenv lookup: ``$SYNTH_SETTER_WORKSPACE/.env``, the
       checkout marker root's ``.env``, then cwd ``.env``. Dotenv values
       overwrite process env, matching the launcher's precedence.
    2. Default ``RCLONE_CONFIG_R2_TYPE=s3`` and ``RCLONE_CONFIG_R2_PROVIDER=Cloudflare``
       into ``os.environ`` if unset. rclone's env-override convention needs both to
       assemble a complete remote definition; without them ``rclone lsd r2:``
       reports ``didn't find section in config file``. Caller values win.
    3. Verify the three secret keys in ``_SECRET_R2_ENV_KEYS`` are present in
       process env, then run ``rclone lsd r2:`` as an auth ping. Either check
       failing raises ``RuntimeError`` with an actionable message.

    No-op on the dotenv step if the resolved file doesn't exist; the defaulting
    + presence+auth checks still run against whatever ``os.environ`` already has.

    :param env_file: Optional dotenv file to merge into ``os.environ`` first.
        ``None`` means the resolved default dotenv path.
    :raises RuntimeError: A required secret key is unset after the load, or
        ``rclone lsd r2:`` exits non-zero (bad creds, network, etc.).
    """
    resolved_env_file = _resolve_env_file(env_file)
    _load_r2_env_file(resolved_env_file)

    for key, default in _R2_STRUCTURAL_DEFAULTS.items():
        os.environ.setdefault(key, default)

    missing = [k for k in _SECRET_R2_ENV_KEYS if not _env_value_is_set(k)]
    if missing:
        raise RuntimeError(
            f"R2 credentials missing from process env after dotenv load: {', '.join(missing)}. "
            f"Set RCLONE_CONFIG_R2_* in process env (e.g. `docker run -e ...=...`) "
            f"or populate {resolved_env_file}."
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


def is_r2_reachable() -> bool:
    """Return ``True`` iff every :func:`ensure_r2_env_loaded` precondition holds.

    Tests gate ``@pytest.mark.integration_r2`` cases on this helper. The
    predicate has to match :func:`ensure_r2_env_loaded`'s contract — if it
    returns ``True`` only because a user's local rclone config makes
    ``rclone lsd r2:`` succeed while the secret env keys are unset, the
    test then calls :func:`ensure_r2_env_loaded` and hits a hard
    ``RuntimeError`` instead of the intended auto-skip.

    Loads the default dotenv file and structural R2 defaults into process env
    before checking secrets, matching the preflight mutation semantics.

    :returns: ``True`` when rclone is on PATH AND all three ``_SECRET_R2_ENV_KEYS``
        are present with non-blank values after default dotenv loading AND a
        credentialled ``rclone lsd r2:`` exits 0; ``False`` otherwise.
    """
    if shutil.which("rclone") is None:
        return False
    _load_r2_env_file(_DEFAULT_ENV_FILE)
    for key, default in _R2_STRUCTURAL_DEFAULTS.items():
        os.environ.setdefault(key, default)
    if not all(_env_value_is_set(key) for key in _SECRET_R2_ENV_KEYS):
        return False
    try:
        subprocess.run(  # noqa: S603 — args are literal strings
            ["rclone", "lsd", "r2:", "--contimeout=10s", "--timeout=30s"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def r2_storage_options() -> dict[str, str]:
    """Build Lance's object-store ``storage_options`` for the R2 bucket from env.

    Reads the same ``RCLONE_CONFIG_R2_*`` vars rclone uses (call
    :func:`ensure_r2_env_loaded` first); S3-compatible stores require both keys.

    :returns: ``{access_key_id, secret_access_key, endpoint, aws_endpoint, region}``
        for ``lance.dataset`` / ``lance.write_dataset``.
    :raises RuntimeError: A required secret key is unset in ``os.environ``.
    """
    # Treat empty/whitespace as missing: a blank env var would otherwise build a
    # partial dict that fails opaquely on the first S3/Lance call.
    values = {key: os.environ.get(key, "").strip() for key in _SECRET_R2_ENV_KEYS}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"R2 credentials missing from process env: {', '.join(missing)}. "
            "Call ensure_r2_env_loaded() or set RCLONE_CONFIG_R2_* first."
        )
    endpoint = values["RCLONE_CONFIG_R2_ENDPOINT"]
    return {
        "access_key_id": values["RCLONE_CONFIG_R2_ACCESS_KEY_ID"],
        "secret_access_key": values["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"],
        "endpoint": endpoint,
        # Lance detects R2 multipart constraints through the AWS-prefixed alias.
        "aws_endpoint": endpoint,
        # R2 ignores region, but object-store requires it set for S3-compatible
        # stores; "auto" is R2's documented placeholder.
        "region": "auto",
    }


def lance_target(r2_uri: str) -> tuple[str, dict[str, str] | None]:
    """Resolve an ``r2://`` URI to the ``(uri, storage_options)`` pair Lance opens.

    Normally returns ``(s3://bucket/key, r2_storage_options())``. When the
    ``r2:`` remote is env-configured as rclone's ``local`` backend (the local
    compute mode dev/tests use — see ``fake_r2_remote``), Lance must read the
    same bytes rclone writes, so the URI resolves to the cwd-relative
    ``<bucket>/<key>`` path rclone's local backend uses, with no options.

    :param r2_uri: Canonical ``r2://bucket/key`` URI string.
    :returns: ``(uri_or_path, storage_options)`` for ``lance.dataset`` /
        ``LanceFragment.create`` / ``LanceDataset.commit``.
    :raises ValueError: ``r2_uri`` is not an ``r2://`` URI — fails fast on both
        backends instead of resolving a nonsense cwd-relative local path.
    """
    if not is_r2_uri(r2_uri):
        raise ValueError(f"not an r2:// URI: {r2_uri!r}")
    if os.environ.get("RCLONE_CONFIG_R2_TYPE", "").strip().lower() == "local":
        return str(Path.cwd() / r2_uri[len(R2_URI_SCHEME) :]), None
    return to_s3_uri(r2_uri), r2_storage_options()


@dataclass(frozen=True)
class RemoteEntry:
    """One object from a :func:`list_entries` listing.

    .. attribute :: path

        Object key relative to the listed directory (``/``-joined for nested
        entries under a recursive listing).

    .. attribute :: mtime

        Storage-assigned last-modified timestamp (R2 ``LastModified``; local
        backend: file mtime).

    .. attribute :: size

        Object size in bytes.
    """

    path: str
    mtime: datetime
    size: int


# Listing probes share rclone's retry/contimeout reliability flags with the
# transfer helpers; ``--checksum`` and ``--timeout`` don't apply to listings.
_PROBE_RELIABILITY_FLAGS = ("--retries=3", "--contimeout=30s")


def _run_listing_probe(args: Sequence[str]) -> str | None:
    """Run an rclone listing probe, normalizing a missing directory to absent.

    S3 backends list a missing key/prefix as empty output; the local backend
    exits 3 for a missing directory. Only that local-backend result becomes
    ``None``: on S3 the same exit code can mean a missing bucket, which is an
    infrastructure failure rather than an absent key.

    :param args: Full rclone argv for a listing subcommand (``lsf`` / ``lsjson``).
    :returns: The probe's stdout, or ``None`` when a local-backend target directory is absent.
    :raises subprocess.CalledProcessError: Any S3 failure, or a local-backend failure other than
        a missing directory.
    """
    result = subprocess.run(  # noqa: S603 — args from validated URIs
        args, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        remote_type = os.environ.get("RCLONE_CONFIG_R2_TYPE", "").strip().lower()
        if result.returncode == 3 and remote_type == "local":
            return None
        raise subprocess.CalledProcessError(
            result.returncode, args, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


# DOC502: the documented CalledProcessError propagates from _run_listing_probe.
def list_entries(r2_uri: str, *, recursive: bool = False) -> list[RemoteEntry]:  # noqa: DOC502
    """List files under an ``r2://`` directory with storage-assigned mtimes.

    Uses ``rclone lsjson --files-only`` so the mtime is the storage server's
    ``LastModified`` — the single-authority timestamp winner selection trusts
    (design doc §7.6). A missing directory lists as empty rather than raising:
    absence of staged work is a normal reconciliation answer, not an error.

    :param r2_uri: Directory ``r2://bucket/key/`` URI to list.
    :param recursive: List nested entries (``-R``) instead of one level.
    :returns: Entries sorted by ``path``; empty when the directory is absent.
    :raises subprocess.CalledProcessError: rclone failed for a reason other
        than a missing directory (auth, network, config).
    """
    args = [  # noqa: S607 — rclone resolved by the image's PATH.
        "rclone",
        "lsjson",
        "--files-only",
        "--use-server-modtime",
        *_PROBE_RELIABILITY_FLAGS,
    ]
    if recursive:
        args.append("-R")
    args.append(_to_rclone_path(r2_uri))
    stdout = _run_listing_probe(args)
    if stdout is None:
        return []
    entries = [
        RemoteEntry(
            path=item["Path"],
            mtime=datetime.fromisoformat(item["ModTime"]),
            size=int(item["Size"]),
        )
        for item in json.loads(stdout)
    ]
    return sorted(entries, key=lambda entry: entry.path)


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


def to_s3_uri(r2_uri: str) -> str:
    """Rewrite an `r2://bucket/key` URI to the `s3://` scheme W&B references record.

    R2 exposes an S3-compatible API; only the scheme differs, so the
    bucket/key path is preserved verbatim. ``storage-provenance-spec.md`` §4
    logs artifact references as ``s3://``.

    :param r2_uri: Canonical ``r2://bucket/key`` URI string.
    :returns: The same location as ``s3://bucket/key``.
    :raises ValueError: ``r2_uri`` is not an ``r2://`` URI.
    """
    if not is_r2_uri(r2_uri):
        raise ValueError(f"not an r2:// URI: {r2_uri!r}")
    return "s3://" + r2_uri[len(R2_URI_SCHEME) :]


def from_s3_uri(s3_uri: str) -> str:
    """Rewrite an `s3://bucket/key` URI back to canonical `r2://bucket/key`.

    Inverse of :func:`to_s3_uri`: R2 exposes an S3-compatible API, so only the
    scheme differs. Used to recover the rclone-reachable ``r2://`` location from
    the ``s3://`` reference W&B records on a model artifact.

    :param s3_uri: ``s3://bucket/key`` URI string.
    :returns: The same location as ``r2://bucket/key``.
    :raises ValueError: ``s3_uri`` is not an ``s3://`` URI.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {s3_uri!r}")
    return R2_URI_SCHEME + s3_uri[len("s3://") :]


def download_dir_no_overwrite(r2_uri: str, dest_path: Path) -> None:
    """Copy every object under an R2 prefix into a local directory, never clobbering.

    Unlike :func:`download_to_path` (single object → file), this is a directory
    copy. ``--immutable`` hard-fails if a destination file already exists with a
    different size/mtime/checksum (rather than silently overwriting or skipping),
    so re-running against a populated dataset root surfaces drift instead of
    masking it. Reliability flags mirror the upload helpers so a transient blip
    retries instead of failing the eval outright.

    :param r2_uri: ``r2://`` directory prefix; every object beneath it is copied.
    :param dest_path: Local destination directory, created by rclone if absent.
    """
    args = _rclone_argv("copy", "--immutable", _to_rclone_path(r2_uri), str(dest_path))
    subprocess.check_call(args)  # noqa: S603 — args from validated URI


def download_to_path(r2_uri: str, dest_path: Path) -> None:
    """Download an R2 object to a specific local file path.

    Uses `rclone copyto` (file→file) so the destination filename is preserved
    exactly — `rclone copy` would treat `dest_path` as a directory and write
    the source basename inside it. Reliability flags mirror the upload/dir
    helpers so a transient blip on the per-shard copy-source fetch retries
    instead of failing the render outright.
    """
    args = _rclone_argv("copyto", _to_rclone_path(r2_uri), str(dest_path))
    subprocess.check_call(args)  # noqa: S603 — args from validated URI


def upload_to_uri(local_path: Path, r2_uri: str) -> None:
    """Upload a local file to a specific R2 object URI.

    Uses `rclone copyto` so the destination filename matches the URI exactly (not the source
    basename). Connection-level timeouts and retries are rclone's job: bounds the TCP connect phase
    and the per-request timeout, retries the whole copy on transient failure, and emits per-request
    debug logs so a CI failure leaves actionable evidence in stdout.
    """
    args = _rclone_argv("copyto", str(local_path), _to_rclone_path(r2_uri))
    subprocess.check_call(args)  # noqa: S603 — args from validated URI


def upload_dir(local_dir: Path, r2_uri: str, exclude: str | None = None) -> None:
    """Copy a local directory tree into an R2 prefix (upload mirror of the dir download).

    ``rclone copy`` walks ``local_dir`` and writes each file under ``r2_uri``,
    preserving the relative tree. ``--checksum`` skips files already present with
    a matching hash, so a re-run is idempotent; the connect-timeout and retry
    flags match the other helpers so a transient blip retries instead of failing
    the caller, while the IO timeout is widened to :data:`_UPLOAD_DIR_TIMEOUT`
    because a whole run dir can stream past the single-file default. Unlike the
    download helper there is no ``--immutable`` — the caller is pushing its own
    freshly-produced directory, not guarding an immutable dataset.

    :param local_dir: Local directory whose contents land directly under
        ``r2_uri`` (the directory itself is not nested under its own name).
    :param r2_uri: ``r2://`` destination prefix; created implicitly by rclone.
    :param exclude: Optional rclone ``--exclude`` glob; lets a caller stage a
        subtree last by excluding it from a first pass.
    """
    operands = [f"--exclude={exclude}"] if exclude is not None else []
    operands += [str(local_dir), _to_rclone_path(r2_uri)]
    args = _rclone_argv("copy", *operands, timeout=_UPLOAD_DIR_TIMEOUT)
    subprocess.check_call(args)  # noqa: S603 — args from validated URI


def upload(source: str | Path, destination_uri: str) -> None:
    """Copy ``source`` to ``destination_uri``; ``source`` is a local path or ``r2://`` URI.

    R2-source dispatches to ``rclone copyto`` (R2→R2 promotion); local-source
    delegates to :func:`upload_to_uri` so the reliability-flag set lives in
    one place.

    :param source: Local filesystem path (``str`` or ``Path``) or ``r2://`` URI
        as a ``str`` — a ``Path`` whose text starts with ``r2://`` is rejected
        because the type signature carries no URI semantics.
    :param destination_uri: Destination ``r2://`` URI.
    :raises TypeError: ``source`` is a ``Path`` whose textual form begins with
        ``r2://``; pass the URI as ``str`` so dispatch is unambiguous.
    """
    if isinstance(source, Path) and str(source).startswith(("r2://", "r2:/")):
        # ``Path("r2://bucket/key")`` collapses the double slash to ``"r2:/bucket/key"``,
        # so both forms have to be guarded; either way the caller meant a URI.
        raise TypeError(
            f"upload() received Path({str(source)!r}); pass r2:// URIs as str "
            f"so the source-type dispatch is unambiguous."
        )
    if isinstance(source, str) and is_r2_uri(source):
        args = _rclone_argv("copyto", _to_rclone_path(source), _to_rclone_path(destination_uri))
        subprocess.check_call(args)  # noqa: S603 — args from validated URIs
        return
    upload_to_uri(Path(source), destination_uri)


def shard_uri(bucket: str, prefix: str, shard_filename: str) -> str:
    """Build the canonical R2 URI for a shard object: ``r2://{bucket}/{prefix}{filename}``.

    Centralizes the convention so the worker's skip-existing probe, the worker's upload, and the
    CI validator agree on one URI shape — protects resumability and reconciliation from
    prefix-format drift.
    """
    return f"{R2_URI_SCHEME}{bucket}/{prefix}{shard_filename}"


# DOC503: the documented CalledProcessError propagates from _run_listing_probe.
def object_size(r2_uri: str) -> int | None:  # noqa: DOC503
    """Return the size in bytes of the R2 object at ``r2_uri``, or ``None`` if it does not exist.

    Uses ``rclone lsf --format=s`` for a size-only single-line listing: integer stdout if the
    object exists, empty stdout if it does not. A missing parent directory also reads as absent —
    S3 backends list it empty while the local backend (local compute mode) errors, so both
    normalize to ``None``. ``--checksum`` does not apply to listings.

    A zero-size object exists and returns ``0``. Callers that want to treat zero-size as absent
    (e.g. defending against half-uploaded objects) test ``size and size > 0`` themselves.

    :raises subprocess.CalledProcessError: rclone exited non-zero for a reason other than a
        missing directory (auth, network, config) — callers fail fast on environmental issues.
    :raises RuntimeError: stdout is non-empty but not an integer; chained to the
        underlying ``ValueError`` so the probed URI survives the failure path.
    """
    args = [  # noqa: S607 — rclone resolved by image's PATH
        "rclone",
        "lsf",
        "--format=s",
        *_PROBE_RELIABILITY_FLAGS,
        _to_rclone_path(r2_uri),
    ]
    stdout = _run_listing_probe(args)
    if stdout is None:
        return None
    out = stdout.strip()
    if not out:
        return None
    try:
        return int(out)
    except ValueError as exc:
        # Chain the parse failure so the probed URI and rclone payload survive
        # the generate failure path instead of a bare ``invalid literal``.
        raise RuntimeError(
            f"rclone lsf --format=s returned unparsable size {out!r} for {r2_uri}"
        ) from exc


# DOC502: the documented CalledProcessError propagates from _run_listing_probe.
def r2_directory_exists(r2_uri: str) -> bool:  # noqa: DOC502
    """Return whether any object exists under the ``r2_uri`` prefix.

    Directory counterpart of :func:`object_size` — a missing prefix reads as
    ``False`` on both the S3 and local backends.

    :param r2_uri: Canonical ``r2://bucket/prefix`` URI of the directory.
    :returns: ``True`` if the prefix contains at least one object.
    :raises subprocess.CalledProcessError: rclone exited non-zero for a reason
        other than a missing directory (auth, network, config).
    """
    # rclone lsf is non-recursive by default, so this lists only the prefix's
    # immediate entries — an O(1) boolean probe, not a full-tree enumeration.
    args = [  # noqa: S607 — rclone resolved by image's PATH
        "rclone",
        "lsf",
        *_PROBE_RELIABILITY_FLAGS,
        _to_rclone_path(r2_uri),
    ]
    stdout = _run_listing_probe(args)
    return bool(stdout and stdout.strip())


def purge_prefix(bucket: str, prefix: str) -> None:
    """Recursively delete every object under ``r2://{bucket}/{prefix}`` (best-effort).

    Intended for integration-test teardown that runs in a ``finally`` block:
    rclone's exit status is intentionally ignored so a transient purge failure
    does not mask the test's real result (e.g. an assertion that fired before
    cleanup). Pair with a unique per-run ``prefix`` so a partial purge cannot
    leak shards across concurrent runs.

    :param bucket: R2 bucket name (no scheme, no trailing slash).
    :param prefix: Key prefix to wipe; must be non-empty and end in ``/`` so the
        rclone target is unambiguously a directory and an absent or accidental
        bare ``"/"`` cannot purge the whole bucket.
    :raises ValueError: ``prefix`` is empty, is ``"/"``, or lacks a trailing
        ``/`` — guards against ``rclone purge r2:{bucket}/`` wiping the bucket.
    """
    stripped = prefix.strip()
    if not stripped or stripped == "/" or not prefix.endswith("/"):
        raise ValueError(
            f"purge_prefix refuses bucket-wide or single-object target: prefix={prefix!r} "
            "must be non-empty, not '/', and end with '/'"
        )
    subprocess.run(  # noqa: S603 — args from validated bucket + prefix
        [  # noqa: S607
            "rclone",
            "purge",
            f"{RCLONE_REMOTE}:{bucket}/{prefix}",
            "--contimeout=10s",
            "--timeout=60s",
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )


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
