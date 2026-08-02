"""Fixtures shared by the model-component tests."""

from pathlib import Path

import pytest

from synth_setter.same import SAME_SAMPLE_RATE
from tests.helpers.same_reference import write_tiny_same_checkpoint


@pytest.fixture(scope="session")
def tiny_same_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize a CPU-sized SAME checkpoint with a real gradient path.

    :param tmp_path_factory: Session-scoped temporary directory factory.
    :returns: Loadable SAME checkpoint directory.
    """
    return write_tiny_same_checkpoint(tmp_path_factory.mktemp("same"), SAME_SAMPLE_RATE)
