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

from pathlib import Path

import pytest
import wandb

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


def read_history_rows(wandb_binary: Path) -> list[dict[str, str]]:
    """Decode history records in a wandb offline ``run-*.wandb`` binary.

    Slash-paths arrive as ``nested_key`` (e.g. ``['shard', 'bytes']``); the
    rejoiner reconstructs the keys callers passed to ``log_metrics`` so the
    caller's assertions read like the production payload.

    :param wandb_binary: Path to the offline ``run-*.wandb`` file.
    :returns: One dict per history record; values are JSON-encoded strings
        (as the datastore stores them).
    """
    ds = wandb_datastore.DataStore()
    ds.open_for_scan(str(wandb_binary))
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
