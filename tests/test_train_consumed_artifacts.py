"""Unit tests for ``train._consumed_artifact_refs`` and the lineage call seam.

The pure helper maps the datamodule's local dataset root to the
``(name, alias)`` lineage edge training feeds to ``record_input_lineage``
(``storage-provenance-spec.md`` §5), plus the unresolved roots that call marks
the run for (#2424). The seam tests below drive the real ``train(cfg)`` with its
heavy collaborators stubbed and pin that the entrypoint actually calls
``record_input_lineage`` with those edges, gated on
``train``/``test`` — coverage the isolated helper tests cannot give. The two
offline-wandb tests at the end close the remaining gap: they drive the real
entrypoint against a real ``WandbLogger`` and read the incompleteness marker
back out of wandb's own datastore binary. All kept out of the canonical
``test_train.py`` per ``tests/_meta/test_entrypoint_test_modules.py``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import _consumed_artifact_refs, train
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import write_spec_to_path
from synth_setter.utils.logging_utils import LINEAGE_INCOMPLETE_TAG
from tests.helpers.wandb_offline import read_run_binary


def _seam_cfg(
    output_dir: Path,
    *,
    dataset_root: Path,
    download_dataset_root_uri: str | None = None,
    train_flag: bool,
    test_flag: bool,
) -> DictConfig:
    """Build a minimal train cfg that drives ``train(cfg)`` under stubbed collaborators.

    Only the keys ``train`` reads are populated; every instantiated object is a
    mock, so no real datamodule/model/trainer is built.

    :param output_dir: ``paths.output_dir`` — read only by ``task_wrapper``'s
        finally-block log line, never written to.
    :param dataset_root: Local root whose frozen spec declares the dataset ID.
    :param download_dataset_root_uri: Optional remote root to hydrate before use.
    :param train_flag: ``cfg.train`` — gates ``trainer.fit`` and the lineage edge.
    :param test_flag: ``cfg.test`` — gates ``trainer.test`` and the lineage edge.
    :returns: A ``DictConfig`` accepted by ``train``.
    """
    return OmegaConf.create(
        {
            "seed": None,
            "datamodule": {
                "_target_": "stub.Datamodule",
                "dataset_root": str(dataset_root),
                "download_dataset_root_uri": download_dataset_root_uri,
            },
            "model": {"_target_": "stub.Model"},
            "trainer": {"_target_": "stub.Trainer"},
            "callbacks": None,
            "logger": {"wandb": {"_target_": "stub.WandbLogger"}},
            "watch_gradients": False,
            "train": train_flag,
            "test": test_flag,
            "ckpt_path": None,
            "paths": {"output_dir": str(output_dir)},
        }
    )


@contextlib.contextmanager
def _stub_train_collaborators(logger_sentinel: object) -> Iterator[MagicMock]:
    """Patch ``train``'s heavy collaborators and yield the ``record_input_lineage`` spy.

    ``instantiate_loggers`` returns ``logger_sentinel`` so the test can assert the
    exact object handed to ``record_input_lineage``; the trainer is a mock whose
    ``fit``/``test`` are inert, and the hyperparameter/provenance writers are
    no-ops so nothing touches wandb or disk.

    :param logger_sentinel: Object ``instantiate_loggers`` is stubbed to return.
    :yields MagicMock: The patched ``record_input_lineage`` mock for call assertions.
    """
    # The trainer is the third ``instantiate`` call; ``train`` merges its
    # ``callback_metrics`` into a dict, so back the attribute with a real dict.
    instantiated = MagicMock()
    instantiated.callback_metrics = {}
    instantiated.checkpoint_callback.best_model_path = "ckpt.ckpt"
    with (
        patch("synth_setter.cli.train.record_input_lineage") as spy,
        patch("synth_setter.cli.train.hydra.utils.instantiate", return_value=instantiated),
        patch("synth_setter.cli.train.instantiate_callbacks", return_value=[]),
        patch("synth_setter.cli.train.instantiate_loggers", return_value=logger_sentinel),
        patch("synth_setter.cli.train.log_hyperparameters"),
        patch("synth_setter.cli.train.log_wandb_provenance"),
        patch("synth_setter.cli.train.pin_wandb_run_id"),
        patch("synth_setter.cli.train.make_wandb_run_id", return_value="rid"),
        patch("synth_setter.cli.train.resolve_run_config_id", return_value="cid"),
    ):
        yield spy


def test_train_calls_record_input_lineage_with_discovered_dataset_edge(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """``train`` hands the logger the dataset edge found from its local root.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    logger_sentinel = MagicMock(name="loggers")
    write_spec_to_path(
        dataset_spec_factory(
            task_name="diva-v1",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        tmp_path / "input_spec.json",
    )
    cfg = _seam_cfg(tmp_path, dataset_root=tmp_path, train_flag=True, test_flag=False)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(
        logger_sentinel, [("data-diva-v1", "diva-v1-20260520T000000000Z")], []
    )


def test_train_calls_record_input_lineage_with_unresolved_root_without_provenance(
    tmp_path: Path,
) -> None:
    """A local root without provenance records no edge and reports the root as unresolved.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(tmp_path, dataset_root=tmp_path, train_flag=True, test_flag=False)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(logger_sentinel, [], [f"dataset root {tmp_path}"])


def test_train_remote_provenance_precedes_local_dataset_spec(
    tmp_path: Path,
    dataset_spec_factory: Callable[..., DatasetSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A training run records the configured remote dataset rather than stale local bytes.

    :param tmp_path: Local roots for both frozen specs and train output.
    :param dataset_spec_factory: Factory producing valid frozen dataset specs.
    :param monkeypatch: Bypasses credential setup so the real filesystem specs drive the test.
    """
    local_root = tmp_path / "local-dataset"
    remote_root = tmp_path / "remote-dataset"
    write_spec_to_path(
        dataset_spec_factory(
            task_name="local-lineage",
            run_id="local-lineage-20260713T170000000Z",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        local_root / "input_spec.json",
    )
    write_spec_to_path(
        dataset_spec_factory(
            task_name="remote-lineage",
            run_id="remote-lineage-20260713T180000000Z",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        remote_root / "input_spec.json",
    )
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda: None)
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(
        tmp_path,
        dataset_root=local_root,
        download_dataset_root_uri=str(remote_root),
        train_flag=True,
        test_flag=False,
    )

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(
        logger_sentinel,
        [("data-remote-lineage", "remote-lineage-20260713T180000000Z")],
        [],
    )


def test_train_records_lineage_when_only_test_is_true(
    tmp_path: Path, dataset_spec_factory: Callable[..., DatasetSpec]
) -> None:
    """A test-only run (``train=False, test=True``) still records the dataset edge.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    logger_sentinel = MagicMock(name="loggers")
    write_spec_to_path(
        dataset_spec_factory(
            task_name="diva-v1",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        tmp_path / "input_spec.json",
    )
    cfg = _seam_cfg(tmp_path, dataset_root=tmp_path, train_flag=False, test_flag=True)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_called_once_with(
        logger_sentinel, [("data-diva-v1", "diva-v1-20260520T000000000Z")], []
    )


def test_train_skips_lineage_when_train_and_test_both_false(tmp_path: Path) -> None:
    """Neither ``train`` nor ``test`` set means the lineage gate stays shut.

    :param tmp_path: Pytest tmp dir wired to ``paths.output_dir``.
    """
    logger_sentinel = MagicMock(name="loggers")
    cfg = _seam_cfg(tmp_path, dataset_root=tmp_path, train_flag=False, test_flag=False)

    with _stub_train_collaborators(logger_sentinel) as spy:
        train(cfg)

    spy.assert_not_called()


def test_consumed_artifact_refs_missing_dataset_root_returns_empty() -> None:
    """A datamodule with no root at all consumes nothing, so nothing is unresolved."""
    assert _consumed_artifact_refs(OmegaConf.create({"datamodule": {}})) == ([], [])


def test_consumed_artifact_refs_unreadable_spec_reports_the_configured_root(
    tmp_path: Path,
) -> None:
    """A configured root whose frozen spec will not parse is reported, not silently dropped.

    :param tmp_path: Empty dataset root standing in for one with an unreadable spec.
    """
    cfg = OmegaConf.create({"datamodule": {"dataset_root": str(tmp_path)}})

    assert _consumed_artifact_refs(cfg) == ([], [f"dataset root {tmp_path}"])


def _attach_offline_wandb_logger(cfg: DictConfig, save_dir: Path) -> None:
    """Swap ``cfg.logger`` for a real offline ``WandbLogger`` group rooted at ``save_dir``.

    :param cfg: Train cfg, mutated in place to carry ``logger.wandb``.
    :param save_dir: Directory the offline run's ``wandb/`` tree is written under.
    """
    with open_dict(cfg):
        cfg.logger = {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "offline": True,
                "save_dir": str(save_dir),
                "id": None,
                "job_type": "",
                "project": "train-lineage-marker-test-project",
            }
        }


def _run_offline_train(cfg: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Drive ``train(cfg)`` against a real offline ``WandbLogger`` and return the run binary.

    :param cfg: Train cfg, mutated to carry the offline logger group.
    :param tmp_path: Hosts the offline ``wandb/`` tree.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env.
    :returns: Path to the offline run's ``run-*.wandb`` datastore binary.
    """
    for key in [k for k in os.environ if k.startswith("WANDB_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    wandb.teardown()
    _attach_offline_wandb_logger(cfg, tmp_path)

    train(cfg)

    offline_dirs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(offline_dirs) == 1, f"expected one offline-run dir, found {offline_dirs}"
    binaries = list(offline_dirs[0].glob("run-*.wandb"))
    assert len(binaries) == 1, f"expected one .wandb binary, found {binaries}"
    return binaries[0]


@pytest.mark.slow
def test_train_unresolvable_dataset_root_marks_the_wandb_run_incomplete(
    cfg_train_lance: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real run over a root with no frozen spec carries the durable lineage marker (#2424).

    Drives the real entrypoint against a real ``WandbLogger`` and decodes the
    datastore binary the live client wrote, so the marker is read back from
    wandb's own bytes rather than from a stub.

    :param cfg_train_lance: CPU-cheap Lance train cfg whose root has no ``input_spec.json``.
    :param tmp_path: Hosts the dataset, offline run dir, and outputs.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env.
    """
    dataset_root = Path(cfg_train_lance.datamodule.dataset_root)
    assert not (dataset_root / "input_spec.json").exists()

    binary = _run_offline_train(cfg_train_lance, tmp_path, monkeypatch)

    payload = read_run_binary(binary, until=lambda data: b"lineage_incomplete" in data)
    assert b"lineage_incomplete" in payload, "summary flag not recorded on the run"
    assert LINEAGE_INCOMPLETE_TAG.encode() in payload, "marker tag not recorded on the run"
    assert str(dataset_root).encode() in payload, "unresolved root not named in the run summary"


@pytest.mark.slow
def test_train_resolvable_dataset_root_leaves_the_wandb_run_unmarked(
    cfg_train_lance: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """A root with a readable frozen spec is never marked incomplete (#2424).

    Also pins that an offline run — where ``use_artifact`` is unavailable by
    design — is not reported as a lineage gap it could never close.

    :param cfg_train_lance: CPU-cheap Lance train cfg.
    :param tmp_path: Hosts the dataset, offline run dir, and outputs.
    :param monkeypatch: Pins a hermetic offline ``WANDB_*`` env.
    :param dataset_spec_factory: Factory producing a valid frozen dataset spec.
    """
    write_spec_to_path(
        dataset_spec_factory(
            task_name="diva-v1",
            train_val_test_sizes=[4, 4, 0],
            r2={"bucket": "intermediate-data"},
            render={"samples_per_shard": 4},
        ),
        Path(cfg_train_lance.datamodule.dataset_root) / "input_spec.json",
    )

    binary = _run_offline_train(cfg_train_lance, tmp_path, monkeypatch)

    payload = read_run_binary(binary)
    assert b"lineage_incomplete" not in payload, "a resolvable root must not mark the run"
    assert LINEAGE_INCOMPLETE_TAG.encode() not in payload, "marker tag written without a gap"
