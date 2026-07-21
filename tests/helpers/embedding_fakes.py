"""Deterministic, training-shaped fake encoders for ``add_embeddings`` e2e tests.

Unlike the deliberately-broken fakes in ``tests/pipeline/data/test_add_embeddings.py``,
these produce embeddings that are a fixed injective-enough function of the audio, so
a model conditioned on them can memorize the fixture (overtrain) meaningfully.
"""

from pathlib import Path

import numpy as np

# Matches the embedpool encoder's default embed_dim and the real m2l C*D (2 ch x 64 dims).
FAKE_M2L_CHANNELS = 128
FAKE_M2L_TIME = 4
FAKE_CLAP_DIM = 512


def fake_m2l_encode(audio: np.ndarray) -> np.ndarray:
    """Project an audio batch to a deterministic ``(B, 128, 4)`` latent sequence.

    :param audio: ``(B, C, T)`` audio batch; ``T`` must be divisible by ``FAKE_M2L_TIME``.
    :returns: ``(B, FAKE_M2L_CHANNELS, FAKE_M2L_TIME)`` float32 latent batch.
    """
    batch, channels, samples = audio.shape
    chunks = audio.astype(np.float32).reshape(
        batch, channels * samples // FAKE_M2L_TIME, FAKE_M2L_TIME
    )
    projection = np.random.default_rng(0).standard_normal(
        (chunks.shape[1], FAKE_M2L_CHANNELS), dtype=np.float32
    )
    return np.einsum("bct,cd->bdt", chunks, projection)


def fake_clap_encode(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Project a mono batch to deterministic L2-normalized ``(B, 512)`` vectors.

    :param mono: ``(B, T)`` mono audio batch.
    :param sample_rate: Unused; present to satisfy the ``ClapEncodeFn`` contract.
    :returns: ``(B, FAKE_CLAP_DIM)`` float32 unit-norm embedding batch.
    """
    projection = np.random.default_rng(1).standard_normal(
        (mono.shape[1], FAKE_CLAP_DIM), dtype=np.float32
    )
    clap = mono.astype(np.float32) @ projection
    return clap / np.linalg.norm(clap, axis=1, keepdims=True)


def assert_embedding_columns(
    dataset_path: Path,
    source: dict[str, np.ndarray],
    *,
    expected_m2l: np.ndarray | None = None,
) -> None:
    """Assert an ``add_embeddings``-augmented split carries correct, searchable columns.

    :param dataset_path: Augmented ``.lance`` dataset directory.
    :param source: Pre-augmentation column arrays the split was written from.
    :param expected_m2l: When set, the exact ``m2l`` values the encoder must have stored.
    """
    # Local imports: lance/pyarrow are absent from the Docker VST CI images.
    import lance
    import pyarrow as pa

    from synth_setter.pipeline.data.add_embeddings import CLAP_EMBEDDING_DIM

    dataset = lance.dataset(str(dataset_path))
    clap_type = dataset.schema.field("clap").type
    assert pa.types.is_fixed_size_list(clap_type), clap_type
    assert clap_type.list_size == CLAP_EMBEDDING_DIM
    assert clap_type.value_type == pa.float32()

    table = dataset.to_table()
    num_rows = len(source["param_array"])
    m2l = table["m2l"].combine_chunks().to_numpy_ndarray()
    clap = np.asarray(table["clap"].combine_chunks().flatten()).reshape(
        num_rows, CLAP_EMBEDDING_DIM
    )
    assert m2l.dtype == np.float32
    assert np.isfinite(m2l).all()
    assert np.isfinite(clap).all()
    if expected_m2l is not None:
        np.testing.assert_array_equal(m2l, expected_m2l)

    np.testing.assert_array_equal(
        table["audio"].combine_chunks().to_numpy_ndarray(), source["audio"]
    )
    np.testing.assert_array_equal(
        table["param_array"].combine_chunks().to_numpy_ndarray(), source["param_array"]
    )

    # Distinct rows must resolve to themselves under the same metric the production
    # IVF_PQ index would use; below MIN_ROWS_FOR_INDEX this is Lance's exact scan.
    for row in range(num_rows):
        hit = dataset.to_table(
            columns=["param_array"],
            nearest={"column": "clap", "q": clap[row], "k": 1, "metric": "cosine"},
        )
        np.testing.assert_array_equal(
            hit["param_array"].combine_chunks().to_numpy_ndarray()[0],
            source["param_array"][row],
        )
