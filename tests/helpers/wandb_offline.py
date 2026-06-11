"""Helpers for asserting on wandb offline-run artifacts.

The wandb offline runtime writes one binary protobuf file per run
(``run-*.wandb``) using ``wandb.sdk.internal.datastore`` — JSON history
mirrors only materialize after ``wandb sync``. Tests that need to assert on
``log_metrics`` payloads in an offline run therefore have to decode the
binary directly.

Both ``tests/test_generate_dataset_wandb.py`` and
``tests/integration/test_generate_dataset_cli_wandb_e2e.py`` need this
decoder; this module is the single owner so a wandb upgrade only requires
updating one site.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

import wandb

_T = TypeVar("_T")

# Verified against wandb 0.26.x. ``wandb.proto.wandb_internal_pb2.Record`` and
# ``wandb.sdk.internal.datastore.DataStore`` are wandb internals — a future
# release that moves them lands here as an ImportError and the consumer test
# module skips with a loud, pointable reason rather than red-CI-ing.
try:
    from wandb.proto import wandb_internal_pb2 as wandb_pb
    from wandb.sdk.internal import datastore as wandb_datastore
except ImportError as exc:  # pragma: no cover — guards a future wandb upgrade
    pytest.skip(
        f"wandb internals moved (wandb=={wandb.__version__}); update tests/helpers/wandb_offline.py: {exc}",
        allow_module_level=True,
    )


_FLUSH_TIMEOUT_S = 10.0
_FLUSH_POLL_S = 0.05


def _poll_until(
    read_once: Callable[[], _T],
    until: Callable[[_T], bool] | None,
    timeout_s: float,
) -> _T:
    """Re-invoke ``read_once`` until ``until`` holds, or return its last result at ``timeout_s``.

    The offline writer flushes the ``run-*.wandb`` datastore asynchronously, so
    a read right after ``wandb.finish()`` can race ahead and capture a
    not-yet-flushed binary (the conda-only 0-records flake). On timeout the last
    read is returned, not raised, so the caller's own assertion reports the
    shortfall.

    :param read_once: Single read; must be safe to call repeatedly.
    :param until: Predicate over a read; when ``None`` ``read_once`` runs exactly
        once (no polling).
    :param timeout_s: Upper bound on polling; ignored when ``until`` is ``None``.
    :returns: The result of the final ``read_once`` call.
    """
    if until is None:
        return read_once()
    deadline = time.monotonic() + timeout_s
    while True:
        result = read_once()
        if until(result) or time.monotonic() >= deadline:
            return result
        time.sleep(_FLUSH_POLL_S)


def read_run_binary(
    wandb_binary: Path,
    *,
    until: Callable[[bytes], bool] | None = None,
    timeout_s: float = _FLUSH_TIMEOUT_S,
) -> bytes:
    """Read an offline ``run-*.wandb`` binary, optionally polling until flushed.

    Raw-bytes sibling of ``read_history_rows`` for assertions on records the
    datastore decoder doesn't surface (e.g. artifact name/type). Pass ``until``
    to re-read until the predicate holds — see ``_poll_until`` for the
    flush-race rationale and the on-timeout semantics.

    :param wandb_binary: Path to the offline ``run-*.wandb`` file.
    :param until: Predicate over the raw bytes; when omitted the file is read
        exactly once (no polling).
    :param timeout_s: Upper bound on flush polling; ignored when ``until`` is
        ``None``.
    :returns: Bytes from the final read (the last poll on timeout).
    """
    return _poll_until(wandb_binary.read_bytes, until, timeout_s)


def read_history_rows(
    wandb_binary: Path,
    *,
    until: Callable[[list[dict[str, str]]], bool] | None = None,
    timeout_s: float = _FLUSH_TIMEOUT_S,
) -> list[dict[str, str]]:
    """Decode history records in a wandb offline ``run-*.wandb`` binary.

    Slash-paths arrive as ``nested_key`` (e.g. ``['shard', 'bytes']``); the
    rejoiner reconstructs the keys callers passed to ``log_metrics`` so the
    caller's assertions read like the production payload. Pass ``until`` to
    re-scan until the predicate holds — see ``_poll_until`` for the flush-race
    rationale and the on-timeout semantics.

    :param wandb_binary: Path to the offline ``run-*.wandb`` file.
    :param until: Predicate over the decoded rows; when omitted the binary is
        scanned exactly once (no polling).
    :param timeout_s: Upper bound on flush polling; ignored when ``until`` is
        ``None``.
    :returns: One dict per history record; values are JSON-encoded strings
        (as the datastore stores them).
    """
    return _poll_until(lambda: _scan_history_rows(wandb_binary), until, timeout_s)


def _scan_history_rows(wandb_binary: Path) -> list[dict[str, str]]:
    """Single-pass decode of the datastore binary's history records.

    :param wandb_binary: Path to the offline ``run-*.wandb`` file.
    :returns: One dict per history record found in this scan.
    """
    ds = wandb_datastore.DataStore()
    ds.open_for_scan(str(wandb_binary))
    # ``open_for_scan`` opens a file handle; close it each pass so the polling
    # loop in ``read_history_rows`` can't leak one handle per re-scan.
    try:
        rows: list[dict[str, str]] = []
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = wandb_pb.Record()  # pyright: ignore[reportAttributeAccessIssue]
            rec.ParseFromString(data)
            if not rec.HasField("history"):
                continue
            row: dict[str, str] = {}
            for item in rec.history.item:
                key = item.key if item.key else "/".join(item.nested_key)
                row[key] = item.value_json
            rows.append(row)
        return rows
    finally:
        ds.close()


def read_run_config(
    wandb_binary: Path,
    *,
    until: Callable[[dict[str, str]], bool] | None = None,
    timeout_s: float = _FLUSH_TIMEOUT_S,
) -> dict[str, str]:
    """Merge every ``wandb.config`` update in a wandb offline ``run-*.wandb`` binary, last-wins.

    Config records are surfaced separately from history; use this to assert on
    provenance keys (``github_sha``, ``image_tag``, ``command``) that land in
    config, not history.

    :param wandb_binary: The offline ``run-*.wandb`` file to decode.
    :param until: Re-scan until this predicate over the merged config holds; when
        ``None``, scan exactly once. See ``_poll_until`` for the flush-race and
        on-timeout semantics.
    :param timeout_s: Polling upper bound; unused when ``until`` is ``None``.
    :returns: Merged config; values are JSON-encoded strings as the datastore stores them.
    """
    return _poll_until(lambda: _scan_run_config(wandb_binary), until, timeout_s)


def _scan_run_config(wandb_binary: Path) -> dict[str, str]:
    """Single-pass decode of the datastore binary's config records, merged.

    :param wandb_binary: Path to the offline ``run-*.wandb`` file.
    :returns: Merged config mapping over every config update in this scan.
    """
    ds = wandb_datastore.DataStore()
    ds.open_for_scan(str(wandb_binary))
    # ``open_for_scan`` opens a file handle; close it each pass so the polling
    # loop in ``read_run_config`` can't leak one handle per re-scan.
    try:
        config: dict[str, str] = {}
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = wandb_pb.Record()  # pyright: ignore[reportAttributeAccessIssue]
            rec.ParseFromString(data)
            if not rec.HasField("config"):
                continue
            for item in rec.config.update:
                key = item.key if item.key else "/".join(item.nested_key)
                config[key] = item.value_json
        return config
    finally:
        ds.close()
