"""Rank-global retrieval handles padding, empty ranks, and ragged observations."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch.multiprocessing.spawn import spawn

from synth_setter.evaluation.paired_retrieval import (
    gathered_retrieval_metrics,
    paired_retrieval_metrics,
)


def _gather_worker(rank: int, rendezvous: str, output: str) -> None:
    """Gather uneven loader coverage in a real CPU process group.

    :param rank: Process-group rank.
    :param rendezvous: File-based process-group address.
    :param output: Directory receiving per-rank metric results.
    """
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        ids = torch.tensor([0, 2, 4]) if rank == 0 else torch.tensor([1, 3, 0])
        rows = torch.ones(len(ids), 3)
        batches = {0: [(rows, rows, ids)]}
        if rank == 0:
            batches[1] = [(torch.eye(2), torch.eye(2), torch.tensor([0, 1]))]
        result = gathered_retrieval_metrics(batches)
        torch.save(result, Path(output) / f"rank-{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def test_gather_two_cpu_processes_deduplicates_padding_and_handles_empty_rank(
    tmp_path: Path,
) -> None:
    """Every rank sees the unique global gallery, including a rank with no loader-1 rows.

    :param tmp_path: Rendezvous and result directory.
    """
    spawn(
        _gather_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )
    first = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    second = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert first == second
    assert first[0]["gallery_size"] == 5
    assert first[0]["audio_to_param/recall_at_1"] == pytest.approx(0.2)
    assert first[1]["gallery_size"] == 2
    assert first[1]["audio_to_param/recall_at_1"] == 1


def test_gather_empty_loader_reports_zero_gallery() -> None:
    """An empty observed loader has no fabricated retrieval score."""
    assert gathered_retrieval_metrics({0: []}) == {0: {"gallery_size": 0.0}}


@pytest.mark.parametrize("modality", ["audio", "params"])
def test_duplicate_roundoff_below_tolerance_keeps_first_observation(modality: str) -> None:
    """Roundoff is accepted without averaging or replacing the first observation.

    :param modality: Arm carrying the perturbed repeated prediction.
    """
    audio = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    params = audio.clone()
    values = audio if modality == "audio" else params
    values[2, 1] = 0.009
    result = paired_retrieval_metrics(audio, params, [0, 1, 0])
    assert result["matched_similarity"] == 1
    assert result["audio/embedding_variance"] == 0.25


@pytest.mark.parametrize("modality", ["audio", "params"])
def test_duplicate_direction_conflict_above_tolerance_rejected(modality: str) -> None:
    """Either arm can invalidate a reused source identity beyond the roundoff budget.

    :param modality: Arm carrying the conflicting repeated prediction.
    """
    audio = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    params = audio.clone()
    values = audio if modality == "audio" else params
    values[1, 1] = 0.011
    with pytest.raises(ValueError, match="conflicting"):
        paired_retrieval_metrics(audio, params, [0, 0])
