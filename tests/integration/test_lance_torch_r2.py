"""End-to-end ``lance.torch`` dataloader streaming against real Cloudflare R2.

Writes a real Lance dataset straight to R2 through the pipeline writer, then streams it back
through both new dataloaders — no fakes, no mocks, no local copy of the dataset on the read side.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from synth_setter.data.lance_torch import (
    lance_tensor_iterable_dataloader,
    lance_tensor_map_dataloader,
)
from synth_setter.pipeline import r2_io
from tests.helpers.lance_torch_datasets import FIELD_SHAPES, write_random_lance_dataset

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_R2_BUCKET = "intermediate-data"


def _unique_test_prefix() -> str:
    """Build a purge-safe R2 prefix for one dataloader streaming run.

    :returns: Trailing-slash-terminated R2 key prefix.
    """
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"ci-lance-torch-loader/{run_id}/{run_attempt}/{uuid.uuid4().hex[:8]}/"


@pytest.fixture(scope="module")
def r2_lance_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """Upload one real Lance dataset to R2 and purge it afterwards.

    The dataset is written locally through the pipeline writer, uploaded as a
    directory tree, and served to the tests as an ``s3://`` URI so every read
    goes over the network.

    :param tmp_path_factory: Pytest factory providing the staging dir.
    :yields: ``(s3_uri, source_arrays)`` pair.
    :ytype: tuple[str, dict[str, np.ndarray]]
    """
    if not r2_io.is_r2_reachable():
        pytest.skip(
            "R2 not reachable (rclone missing, RCLONE_CONFIG_R2_* env vars missing, "
            "or rclone lsd r2: failed)"
        )
    r2_io.ensure_r2_env_loaded()
    prefix = _unique_test_prefix()
    local = tmp_path_factory.mktemp("lance_torch_r2") / "train.lance"
    arrays = write_random_lance_dataset(local)
    dataset_r2_uri = f"r2://{_R2_BUCKET}/{prefix}train.lance"
    r2_io.upload_dir(local, dataset_r2_uri)
    try:
        yield r2_io.to_s3_uri(dataset_r2_uri), arrays
    finally:
        r2_io.purge_prefix(_R2_BUCKET, prefix)


def test_iterable_dataloader_streams_lance_dataset_from_real_r2(
    r2_lance_dataset: tuple[str, dict[str, np.ndarray]],
) -> None:
    """The iterable loader streams every row from R2 with values and shapes intact.

    :param r2_lance_dataset: Uploaded dataset URI and its source arrays.
    """
    s3_uri, arrays = r2_lance_dataset
    loader = lance_tensor_iterable_dataloader(
        s3_uri, batch_size=8, storage_options=r2_io.r2_storage_options()
    )

    batches = list(loader)

    assert batches[0]["mel_spec"].shape == (8, *FIELD_SHAPES["mel_spec"][1:])
    for field, source in arrays.items():
        streamed = torch.cat([batch[field] for batch in batches]).numpy()
        np.testing.assert_array_equal(streamed, source)


def test_map_dataloader_reads_lance_dataset_from_real_r2(
    r2_lance_dataset: tuple[str, dict[str, np.ndarray]],
) -> None:
    """The map-style loader random-accesses R2 directly with values intact.

    :param r2_lance_dataset: Uploaded dataset URI and its source arrays.
    """
    s3_uri, arrays = r2_lance_dataset
    loader = lance_tensor_map_dataloader(
        s3_uri,
        batch_size=8,
        columns=["param_array"],
        storage_options=r2_io.r2_storage_options(),
        shuffle=False,
    )

    read = torch.cat([batch["param_array"] for batch in loader]).numpy()

    np.testing.assert_array_equal(read, arrays["param_array"])
