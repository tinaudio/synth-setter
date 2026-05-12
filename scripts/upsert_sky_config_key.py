"""Per-key YAML upsert for ``~/.sky/config.yaml`` (shared across SkyPilot providers).

The file is shared by multiple providers (OCI writes ``oci:``; local writes
``jobs:`` for the managed-jobs controller resource floor). Earlier "skip if file
exists" guards meant running for one provider after another would silently leave
the second provider without its section. The semantics are: replace exactly the
top-level key we manage, preserve every other top-level key the user (or another
provider's run) already populated. Within the managed top-level key the entire
mapping is replaced wholesale — hand-added nested keys under the managed key
are NOT preserved.

The file is re-serialized via PyYAML ``safe_dump``, so comments and original key
ordering / formatting are dropped — only mapping data round-trips. This is
acceptable because ``~/.sky/config.yaml`` is bootstrap-owned in CI; hand-managed
local-dev configs that rely on comments should be edited outside this helper.

Secret-borne fragments (e.g. an ``oci:`` block carrying ``OCI_COMPARTMENT_OCID``)
are passed to this module's CLI via the ``SYNTH_UPSERT_FRAGMENT`` environment
variable rather than argv: ``/proc/<pid>/cmdline`` is world-readable on Linux but
``/proc/<pid>/environ`` is owner-readable, so an env-borne secret can't be
observed by other users on the runner via ``ps``. See #876.

Used by:
- ``scripts/skypilot_write_provider_creds.sh::upsert_sky_config_key`` (bash wrapper)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_FRAGMENT_ENV_VAR = "SYNTH_UPSERT_FRAGMENT"
_OUTPUT_FILE_MODE = 0o600


class UpsertError(Exception):
    """Operator-facing failure during ``upsert_sky_config_key``.

    The message is formatted for direct emission to stderr: it starts with
    ``upsert_sky_config_key[<key>]:`` for grep-ability and names both the
    failure reason and (when relevant) the offending path. Callers should not
    wrap or re-raise — the CLI wrapper catches this exception and exits non-zero
    with the message verbatim on stderr (no traceback).
    """


def upsert_sky_config_key(key: str, fragment_text: str, path: Path) -> None:
    """Upsert ``key`` from ``fragment_text`` into the YAML config at ``path``.

    Reads ``path`` if it exists, replaces the top-level ``key`` mapping wholesale
    with the one in ``fragment_text``, preserves every other top-level key,
    writes back, and chmods to ``0o600``. Raises :class:`UpsertError` (propagated
    from :func:`_load_existing_config` or :func:`_parse_fragment`) on any malformed
    input — see those helpers' ``:raises:`` sections for the full failure taxonomy.

    :param key: Top-level YAML key to manage (e.g. ``"oci"``, ``"jobs"``).
    :param fragment_text: YAML text — must parse to a mapping with ``key`` at the top level.
    :param path: Path to the config file. Created if it doesn't exist.
    """
    existing = _load_existing_config(key, path)
    fragment_doc = _parse_fragment(key, fragment_text)
    existing[key] = fragment_doc[key]
    path.write_text(yaml.safe_dump(existing, sort_keys=False))
    os.chmod(path, _OUTPUT_FILE_MODE)


def _load_existing_config(key: str, path: Path) -> dict[str, object]:
    """Read and parse ``path``, validating it's a YAML mapping or empty/non-existent.

    :param key: Managed top-level YAML key (used in the error message prefix).
    :param path: Path to the existing config file. Treated as empty if missing or zero-sized.
    :returns: parsed top-level mapping, or an empty dict for a non-existent / empty file.
    :rtype: dict[str, object]
    :raises UpsertError: If the file is non-UTF-8, unparsable YAML, or a top-level non-mapping.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        raw = path.read_text()
    except UnicodeDecodeError as exc:
        raise UpsertError(
            f"upsert_sky_config_key[{key}]: {path} is not valid UTF-8 ({exc}); "
            f"refusing to upsert. Fix or remove the file."
        ) from exc
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise UpsertError(
            f"upsert_sky_config_key[{key}]: {path} is not valid YAML "
            f"({exc.__class__.__name__}); refusing to upsert. Fix or remove the file."
        ) from exc
    if not isinstance(parsed, dict):
        raise UpsertError(
            f"upsert_sky_config_key[{key}]: {path} is not a YAML mapping at the top level "
            f"(got {type(parsed).__name__}); refusing to upsert. Fix or remove the file."
        )
    return parsed


def _parse_fragment(key: str, fragment_text: str) -> dict[str, object]:
    """Parse the caller-supplied fragment and validate it carries ``key`` at the top level.

    :param key: Managed top-level YAML key the fragment must contain.
    :param fragment_text: YAML text supplied by the caller.
    :returns: parsed top-level mapping, guaranteed to contain ``key``.
    :rtype: dict[str, object]
    :raises UpsertError: If the fragment is unparsable, not a mapping, or missing ``key``.
    """
    try:
        parsed = yaml.safe_load(fragment_text) or {}
    except yaml.YAMLError as exc:
        raise UpsertError(
            f"upsert_sky_config_key[{key}]: fragment is not valid YAML ({exc.__class__.__name__})"
        ) from exc
    if not isinstance(parsed, dict):
        raise UpsertError(
            f"upsert_sky_config_key[{key}]: fragment is not a YAML mapping "
            f"(got {type(parsed).__name__})"
        )
    if key not in parsed:
        raise UpsertError(f"upsert_sky_config_key[{key}]: fragment missing top-level {key!r}")
    return parsed


def _main(argv: list[str]) -> int:
    """CLI entry point. Reads fragment from env, writes via :func:`upsert_sky_config_key`.

    :param argv: Positional CLI arguments after ``sys.argv[0]`` — must be ``[key, path]``.
    :returns: process exit code (0 on success, 1 on UpsertError, 2 on usage error).
    :rtype: int
    """
    if len(argv) != 2:
        sys.stderr.write(
            f"upsert_sky_config_key: usage: {sys.argv[0]} <key> <path> "
            f"(fragment supplied via {_FRAGMENT_ENV_VAR} env var)\n"
        )
        return 2
    key, path_str = argv
    fragment = os.environ.pop(_FRAGMENT_ENV_VAR, None)
    if fragment is None:
        sys.stderr.write(
            f"upsert_sky_config_key[{key}]: {_FRAGMENT_ENV_VAR} env var is not set "
            f"(fragment carriage is via env, not argv, to keep secrets out of "
            f"/proc/<pid>/cmdline)\n"
        )
        return 1
    try:
        upsert_sky_config_key(key, fragment, Path(path_str))
    except UpsertError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
