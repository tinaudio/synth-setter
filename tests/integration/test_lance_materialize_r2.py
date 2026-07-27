"""Live end-to-end materialization of a txid-pinned subset from a real R2 Lance dataset.

No fakes, no mocks, no local-backend remote: the source dataset lives in
Cloudflare R2 and every read goes over the network through the same
``r2_io.lance_target()`` + credentials path production hydration uses. The
test is read-only on R2 — the materialized output lands on local disk only.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pytest

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_materialize import (
    MaterializeManifest,
    materialize_lance_subset,
    request_hash,
    resolve_txid_version,
    sidecar_path,
)

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

# Small (1k-row) production-written train split; read-only fixture for this test.
_SOURCE_URI = (
    "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
    "surge-simple-lance-1k-2k-2k-20260716T163226347Z/train.lance"
)
_COLUMNS = ("param_array",)
_LIMIT = 8


def test_materialize_live_r2_dataset_txid_pinned_subset_round_trips(
    tmp_path: Path,
) -> None:
    """Full production path: pin a live R2 dataset's txid and materialize a subset.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip(
            "R2 not reachable (rclone missing, RCLONE_CONFIG_R2_* env vars missing, "
            "or rclone lsd r2: failed)"
        )
    r2_io.ensure_r2_env_loaded()

    # Pin the live dataset's current version by its real transaction uuid.
    open_uri, storage_options = r2_io.lance_target(_SOURCE_URI)
    source = lance.dataset(open_uri, storage_options=storage_options)
    pinned = source.read_transaction(source.version)
    assert pinned is not None
    txid = pinned.uuid
    assert resolve_txid_version(source, txid) == source.version

    dest = tmp_path / "train.lance"
    result = materialize_lance_subset(
        _SOURCE_URI, dest, txid=txid, columns=_COLUMNS, limit=_LIMIT, batch_size=_LIMIT
    )
    assert result == dest

    out = lance.dataset(str(dest))
    assert out.schema.names == list(_COLUMNS)
    assert out.count_rows() == _LIMIT
    txn = out.read_transaction(out.version)
    assert txn is not None and txn.transaction_properties is not None
    assert txn.transaction_properties["cloned_from_txn"] == txid

    manifest = MaterializeManifest.model_validate_json(
        sidecar_path(dest).read_text(encoding="utf-8")
    )
    assert manifest.request_hash == request_hash(
        _SOURCE_URI, txid, manifest.resolved_version, _COLUMNS, _LIMIT
    )
