"""Full-gallery retrieval in online predictor space, independent of ANN exports."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

# Only detached CPU predictions and split-local IDs cross rank boundaries.
type RetrievalBatches = dict[int, list[tuple[Tensor, Tensor, Tensor]]]


def gathered_retrieval_metrics(batches: RetrievalBatches) -> dict[int, dict[str, float]]:
    """Gather variable-length rank observations and score each loader independently.

    :param batches: CPU audio predictions, parameter predictions, and int64 IDs per loader.
    :returns: Full-gallery metrics keyed by loader index, identical on every rank.
    """
    ranks = [batches]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ranks = [{} for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(ranks, batches)
    loader_ids = sorted({index for rank in ranks for index in rank})
    result = {}
    for index in loader_ids:
        rows = [batch for rank in ranks for batch in rank.get(index, [])]
        if not rows:
            result[index] = {"gallery_size": 0.0}
            continue
        audio, params, ids = (torch.cat(values) for values in zip(*rows, strict=True))
        result[index] = paired_retrieval_metrics(audio, params, ids.tolist())
    return result


def _direction_metrics(query: Tensor, gallery: Tensor, chunk_size: int) -> dict[str, float]:
    """Average retrieval scores with uniform tie-breaking over equal cosine scores.

    :param query: Normalized query rows paired with gallery rows.
    :param gallery: Normalized gallery rows.
    :param chunk_size: Maximum number of queries scored at once.
    :returns: Recall at 1/5/10 and expected reciprocal rank.
    """
    count = len(query)
    harmonic = torch.cat((torch.zeros(1), (1 / torch.arange(1, count + 1).double()).cumsum(0)))
    totals: dict[str, float] = dict.fromkeys(
        ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"), 0.0
    )
    for start in range(0, count, chunk_size):
        scores = query[start : start + chunk_size] @ gallery.T
        positive = scores[torch.arange(len(scores)), torch.arange(start, start + len(scores))]
        better = (scores > positive[:, None]).sum(1)
        tied = (scores == positive[:, None]).sum(1)
        for k in (1, 5, 10):
            totals[f"recall_at_{k}"] += (
                ((k - better).clamp(min=0).minimum(tied) / tied).sum().item()
            )
        totals["mrr"] += ((harmonic[better + tied] - harmonic[better]) / tied).sum().item()
    return {key: value / count for key, value in totals.items()}


def paired_retrieval_metrics(
    audio: Tensor,
    params: Tensor,
    sample_ids: Sequence[int],
    *,
    chunk_size: int = 256,
) -> dict[str, float]:
    """Score paired online predictions against the complete held-out gallery.

    Exact cosine ties receive expected Recall/MRR under uniform tie-breaking. Each unique ID
    contributes one query and one candidate. Repeats retain the first rank/batch observation;
    normalized L2 distances up to 0.01 (cosine distance 5e-5 for unit vectors) allow bf16 batch-
    shape roundoff even when arms return float32. Larger conflicts fail. Distinct IDs remain
    negatives. The caller must aggregate every rank's observations before calling this function.

    :param audio: Audio predictor outputs, one row per observation.
    :param params: Parameter predictor outputs paired row-for-row with audio.
    :param sample_ids: Stable split-local source row identities.
    :param chunk_size: Maximum score-matrix rows, bounding memory to O(chunk_size * N).
    :returns: Bidirectional metrics, mean population variance per coordinate, matched cosine
        similarity, and mismatched cosine from a one-position cyclic shift in sorted-ID order.
        Empty galleries return only size; singleton galleries omit mismatch. Zero vectors remain
        zero.
    :raises ValueError: If shapes, identities, values, or chunk size are invalid, or repeated
        identities have conflicting normalized predictions.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if audio.ndim != 2 or params.shape != audio.shape or audio.shape[1] == 0:
        raise ValueError("audio and params must have the same nonzero-width matrix shape")
    if len(sample_ids) != len(audio):
        raise ValueError("sample_ids must contain one identity per paired row")
    if not torch.isfinite(audio).all() or not torch.isfinite(params).all():
        raise ValueError("predictions must be finite")
    audio = F.normalize(audio.detach().cpu().double(), dim=1)
    params = F.normalize(params.detach().cpu().double(), dim=1)
    first_rows: dict[int, int] = {}
    for row, identity in enumerate(sample_ids):
        if identity in first_rows:
            first = first_rows[identity]
            if any(
                torch.linalg.vector_norm(values[row] - values[first]) > 0.01
                for values in (audio, params)
            ):
                raise ValueError(f"sample ID {identity} has conflicting predictions")
        else:
            first_rows[identity] = row
    rows = [first_rows[identity] for identity in sorted(first_rows)]
    if not rows:
        return {"gallery_size": 0.0}
    audio, params = audio[rows], params[rows]
    result = {
        f"{direction}/{name}": value
        for direction, query, gallery in (
            ("audio_to_param", audio, params),
            ("param_to_audio", params, audio),
        )
        for name, value in _direction_metrics(query, gallery, chunk_size).items()
    }
    result.update(
        {
            "gallery_size": float(len(audio)),
            "audio/embedding_variance": audio.var(dim=0, correction=0).mean().item(),
            "param/embedding_variance": params.var(dim=0, correction=0).mean().item(),
            "matched_similarity": (audio * params).sum(1).mean().item(),
        }
    )
    if len(rows) > 1:
        result["mismatched_similarity"] = (audio * params.roll(-1, dims=0)).sum(1).mean().item()
    return result
