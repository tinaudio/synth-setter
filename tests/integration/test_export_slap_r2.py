"""Production-path SLAP retrieval export against real Cloudflare R2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import lance
import numpy as np
import pytest
from omegaconf import OmegaConf

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.export_slap import SLAP_SOURCE_POINTER_KEY
from tests.pipeline.data.test_export_slap import _checkpoint, _config, _source

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_R2_BUCKET = "intermediate-data"
_CLI_TIMEOUT_SECONDS = 300


def _unique_test_prefix() -> str:
    """Return a purge-safe prefix owned by one test invocation.

    :returns: Unique trailing-slash-terminated R2 key prefix.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"ci-export-slap/{run_id}/{run_attempt}/{uuid.uuid4().hex[:8]}/"


def _open_remote(uri: str) -> lance.LanceDataset:
    """Open an R2 or S3-compatible Lance URI through production storage options.

    :param uri: Remote Lance dataset URI.
    :returns: Open remote dataset.
    """
    canonical = r2_io.from_s3_uri(uri) if uri.startswith("s3://") else uri
    target, storage_options = r2_io.lance_target(canonical)
    return lance.dataset(target, storage_options=storage_options)


@pytest.mark.parametrize("uri_scheme", ["r2", "s3"])
def test_export_slap_cli_real_r2_round_trip_is_searchable_and_idempotent(
    tmp_path: Path, uri_scheme: str
) -> None:
    """The installed CLI exports, joins, points, searches, and reuses real R2 data.

    :param tmp_path: Isolated local staging root.
    :param uri_scheme: Public spelling used for source and output roots.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone missing, credentials missing, or rclone lsd failed)")
    r2_io.ensure_r2_env_loaded()

    prefix = _unique_test_prefix()
    source_key = f"{prefix}source"
    output_key = f"{prefix}output"
    canonical_source_root = f"r2://{_R2_BUCKET}/{source_key}"
    canonical_output_root = f"r2://{_R2_BUCKET}/{output_key}"
    source_root = (
        canonical_source_root if uri_scheme == "r2" else r2_io.to_s3_uri(canonical_source_root)
    )
    output_root = (
        canonical_output_root if uri_scheme == "r2" else r2_io.to_s3_uri(canonical_output_root)
    )

    local_source = tmp_path / "source"
    local_source.mkdir()
    source_uuids = [str(uuid.uuid4()) for _ in range(4)]
    version = _source(local_source / "train.lance", rows=4, row_uuid=source_uuids)
    checkpoint = tmp_path / "model.ckpt"
    _checkpoint(checkpoint)
    config = _config(
        tmp_path / "unused-source",
        tmp_path / "unused-output",
        checkpoint,
        version,
        batch_size=2,
    ).model_dump(mode="json")
    config.update(
        source_root_uri=source_root,
        output_root_uri=output_root,
        synth={"name": "test"},
    )
    config_path = tmp_path / "export.yaml"
    config_path.write_text(OmegaConf.to_yaml(config))
    command = [
        str(Path(sys.executable).with_name("synth-setter-export-slap")),
        "--config-dir",
        str(tmp_path),
        "--config-name",
        "export",
    ]

    try:
        r2_io.upload_dir(local_source, canonical_source_root)

        first = subprocess.run(  # noqa: S603 - installed script and fixed trusted arguments
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
        )
        assert first.returncode == 0, first.stderr

        output = _open_remote(f"{output_root}/train.lance")
        table = output.to_table()
        assert table.num_rows == 8
        query = np.asarray(table["slap"][0].as_py(), dtype=np.float32)
        nearest = output.to_table(nearest={"column": "slap", "q": query, "k": 1})
        matched_uuid = nearest["row_uuid"][0].as_py()
        source = _open_remote(f"{source_root}/train.lance")
        source_match = source.to_table(filter=f"row_uuid = '{matched_uuid}'", columns=["payload"])
        assert source_match.num_rows == 1
        pointer = json.loads(source.schema.metadata[SLAP_SOURCE_POINTER_KEY])
        assert pointer["output_uri"] == f"{output_root}/train.lance"
        source_version = source.version
        output_version = output.version

        second = subprocess.run(  # noqa: S603 - installed script and fixed trusted arguments
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
        )
        assert second.returncode == 0, second.stderr
        assert _open_remote(f"{source_root}/train.lance").version == source_version
        assert _open_remote(f"{output_root}/train.lance").version == output_version
    finally:
        r2_io.purge_prefix(_R2_BUCKET, prefix)
