"""Exact contracts for held-out paired embedding retrieval."""

import math

import pytest
import torch
from torch.utils.data import DistributedSampler, TensorDataset

from synth_setter.evaluation.paired_retrieval import paired_retrieval_metrics


@pytest.mark.parametrize("direction", ["audio_to_param", "param_to_audio"])
@pytest.mark.parametrize("metric", ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"])
def test_retrieval_orthogonal_pairs_scores_perfect(direction: str, metric: str) -> None:
    """Unique aligned maxima recover every pair even when K exceeds gallery size.

    :param direction: Query modality and candidate modality.
    :param metric: Ranking statistic to verify.
    """
    embeddings = torch.eye(3)
    result = paired_retrieval_metrics(embeddings, embeddings, [2, 0, 1])
    assert result[f"{direction}/{metric}"] == pytest.approx(1.0)


@pytest.mark.parametrize("direction", ["audio_to_param", "param_to_audio"])
@pytest.mark.parametrize("metric, expected", [("recall_at_1", 0), ("mrr", 0.5)])
def test_retrieval_swapped_pair_scores_wrong(direction: str, metric: str, expected: float) -> None:
    """A reversed gallery puts the positive behind the incorrect candidate.

    :param direction: Query modality and candidate modality.
    :param metric: Ranking statistic to verify.
    :param expected: Exact score when every positive is ranked second.
    """
    result = paired_retrieval_metrics(torch.eye(2), torch.eye(2).flip(0), [0, 1])
    assert result[f"{direction}/{metric}"] == pytest.approx(expected)


@pytest.mark.parametrize("direction", ["audio_to_param", "param_to_audio"])
@pytest.mark.parametrize(
    "metric, expected",
    [
        ("recall_at_1", 1 / 12),
        ("recall_at_5", 5 / 12),
        ("recall_at_10", 10 / 12),
        ("mrr", 0.25860088985088986),
    ],
)
def test_retrieval_collapsed_gallery_scores_chance(
    direction: str, metric: str, expected: float
) -> None:
    """Equal scores cannot exploit row order to recover the positive.

    :param direction: Query modality and candidate modality.
    :param metric: Ranking statistic to verify.
    :param expected: Chance expectation under uniform tie-breaking.
    """
    result = paired_retrieval_metrics(torch.ones(12, 2), torch.ones(12, 2), list(range(12)))
    assert result[f"{direction}/{metric}"] == pytest.approx(expected)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 20])
def test_retrieval_chunk_sizes_preserve_scores(chunk_size: int) -> None:
    """Partial query blocks preserve the complete candidate gallery.

    :param chunk_size: Query block length, including ragged-tail cases.
    """
    audio = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    params = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]])
    result = paired_retrieval_metrics(audio, params, [0, 1, 2, 3], chunk_size=chunk_size)
    assert result == pytest.approx(
        paired_retrieval_metrics(audio, params, [0, 1, 2, 3], chunk_size=4)
    )


def test_diagnostics_orthogonal_pairs_measure_population_variance_and_derangement() -> None:
    """Diagnostics use normalized predictions and exclude self-pairs from mismatch."""
    result = paired_retrieval_metrics(torch.eye(3), 2 * torch.eye(3), [2, 0, 1])
    assert result["audio/embedding_variance"] == pytest.approx(2 / 9)
    assert result["param/embedding_variance"] == pytest.approx(2 / 9)
    assert result["matched_similarity"] == pytest.approx(1.0)
    assert result["mismatched_similarity"] == pytest.approx(0.0)


def test_retrieval_empty_gallery_omits_undefined_metrics() -> None:
    """No observations produce a count rather than NaN metrics."""
    assert paired_retrieval_metrics(torch.empty(0, 2), torch.empty(0, 2), []) == {
        "gallery_size": 0.0
    }


def test_retrieval_singleton_gallery_omits_mismatch() -> None:
    """A singleton has a trivial rank but no possible negative."""
    result = paired_retrieval_metrics(torch.ones(1, 2), torch.ones(1, 2), [3])
    assert result["audio_to_param/mrr"] == 1.0
    assert result["audio/embedding_variance"] == 0.0
    assert "mismatched_similarity" not in result


def test_retrieval_repeated_sample_ids_deduplicate_padding() -> None:
    """Repeating an observation cannot reweight the gallery statistics."""
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = paired_retrieval_metrics(vectors, vectors, [7, 2, 7])
    assert result["gallery_size"] == 2.0
    assert result["mismatched_similarity"] == 0.0
    assert result == pytest.approx(paired_retrieval_metrics(torch.eye(2), torch.eye(2), [7, 2]))


def test_retrieval_repeated_ids_with_conflicting_vectors_rejects_ambiguity() -> None:
    """Conflicting repeated observations cannot silently select an arbitrary vector."""
    with pytest.raises(ValueError, match="conflicting"):
        paired_retrieval_metrics(torch.eye(2), torch.ones(2, 2), [3, 3])


def test_retrieval_loader_order_preserves_deterministic_mismatch() -> None:
    """Identity order, not batch arrival, determines the negative pairing."""
    audio = torch.eye(4, 5)
    params = torch.eye(4, 5)[[3, 0, 1, 2]]
    order = [2, 0, 3, 1]
    result = paired_retrieval_metrics(audio[order], params[order], order)
    assert result["mismatched_similarity"] == pytest.approx(1.0)
    assert result == pytest.approx(paired_retrieval_metrics(audio, params, [0, 1, 2, 3]))


@pytest.mark.parametrize("modality", ["audio", "params"])
def test_retrieval_nonfinite_predictions_rejects_invalid_metrics(modality: str) -> None:
    """Invalid predictions cannot disappear inside rank comparisons.

    :param modality: Predictor output receiving the invalid value.
    """
    inputs = {"audio": torch.eye(2), "params": torch.eye(2)}
    inputs[modality][0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        paired_retrieval_metrics(inputs["audio"], inputs["params"], [0, 1])


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_retrieval_nonpositive_chunk_rejects_invalid_budget(chunk_size: int) -> None:
    """An invalid memory budget fails before scoring any query.

    :param chunk_size: Nonpositive query block length.
    """
    with pytest.raises(ValueError, match="chunk_size"):
        paired_retrieval_metrics(torch.eye(2), torch.eye(2), [0, 1], chunk_size=chunk_size)


@pytest.mark.parametrize("direction", ["audio_to_param", "param_to_audio"])
def test_retrieval_tied_positive_below_winner_averages_reciprocal_ranks(direction: str) -> None:
    """Expected reciprocal rank is not the reciprocal of the mean tied rank.

    :param direction: Query modality and candidate modality.
    """
    result = paired_retrieval_metrics(torch.eye(3), torch.eye(3)[[1, 2, 0]], [0, 1, 2])
    assert result[f"{direction}/mrr"] == pytest.approx(5 / 12)


def test_retrieval_distributed_sampler_padding_preserves_global_gallery() -> None:
    """Globally assembled padded sampler rows recover the original split metrics."""
    vectors = torch.eye(5)
    dataset = TensorDataset(vectors)
    rank0 = list(DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False))
    rank1 = list(DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False))
    ids = rank0 + rank1
    result = paired_retrieval_metrics(vectors[ids], vectors[ids], ids)
    assert result == pytest.approx(paired_retrieval_metrics(vectors, vectors, [0, 1, 2, 3, 4]))


@pytest.mark.parametrize("metric", ["audio/embedding_variance", "param/embedding_variance"])
def test_diagnostics_zero_predictions_have_no_variance(metric: str) -> None:
    """Zero outputs remain finite collapsed vectors after normalization.

    :param metric: Modality-specific population variance statistic.
    """
    result = paired_retrieval_metrics(torch.zeros(2, 3), torch.zeros(2, 3), [0, 1])
    assert result[metric] == 0.0


def test_retrieval_missing_identity_rejects_misaligned_rows() -> None:
    """Every observation needs a source identity before deduplication."""
    with pytest.raises(ValueError, match="sample_ids"):
        paired_retrieval_metrics(torch.eye(2), torch.eye(2), [0])


@pytest.mark.parametrize("params", [torch.ones(2, 3), torch.ones(3, 2), torch.ones(2)])
def test_retrieval_incompatible_shapes_rejects_inputs(params: torch.Tensor) -> None:
    """Both modalities must share paired row counts and predictor width.

    :param params: Parameter matrix with an incompatible shape.
    """
    with pytest.raises(ValueError, match="shape"):
        paired_retrieval_metrics(torch.eye(2), params, [0, 1])


def test_diagnostics_coordinate_means_differ_measures_between_observation_variance() -> None:
    """Within-vector coordinate variation is not embedding diversity."""
    vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    result = paired_retrieval_metrics(vectors, vectors, [0, 1])
    assert result["audio/embedding_variance"] == 0.0
    assert result["param/embedding_variance"] == 0.0


def test_retrieval_scalar_predictors_accepts_nonzero_width() -> None:
    """Signed scalar predictions can identify opposite paired rows."""
    vectors = torch.tensor([[1.0], [-1.0]])
    result = paired_retrieval_metrics(vectors, vectors, [0, 1])
    assert result["audio_to_param/recall_at_1"] == 1.0


def test_retrieval_zero_width_rejects_undefined_cosine() -> None:
    """Rows without predictor coordinates cannot define retrieval scores."""
    with pytest.raises(ValueError, match="shape"):
        paired_retrieval_metrics(torch.empty(2, 0), torch.empty(2, 0), [0, 1])
