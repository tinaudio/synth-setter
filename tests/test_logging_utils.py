"""Tests for log_wandb_provenance() in src/synth_setter/utils/logging_utils.py.

Uses fakes (not mocks) for wandb, real subprocess where possible, and state assertions throughout.
See python-testing.md §Fakes.
"""

import os
import subprocess
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf

from synth_setter.utils.logging_utils import (
    log_wandb_provenance,
    pin_wandb_run_id,
    resolve_run_config_id,
    use_input_artifacts,
)

# ---------------------------------------------------------------------------
# Fake wandb module — captures config updates as inspectable state
# ---------------------------------------------------------------------------


class FakeWandbConfig:
    """Fake wandb.config that stores updates for state testing."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def update(self, d: dict, **kwargs: object) -> None:
        self.data.update(d)


def make_fake_wandb(*, has_run: bool = True) -> SimpleNamespace:
    """Build a fake wandb module with inspectable config state."""
    return SimpleNamespace(
        run=object() if has_run else None,
        config=FakeWandbConfig(),
        __spec__=object(),
    )


# ---------------------------------------------------------------------------
# Happy-path behavior tests (real subprocess, fake wandb only)
# ---------------------------------------------------------------------------


class TestLogWandbProvenanceHappyPath:
    """Provenance fields are logged correctly when all dependencies are available."""

    def test_logs_git_sha_as_valid_hex(self) -> None:
        """github_sha is the real 40-char hex SHA from the current git repo."""
        fake = make_fake_wandb()

        with patch.dict("sys.modules", {"wandb": fake}):
            log_wandb_provenance()

        sha = fake.config.data["github_sha"]
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_logs_image_tag_from_env(self) -> None:
        """image_tag matches the IMAGE_TAG environment variable."""
        fake = make_fake_wandb()

        with (
            patch.dict("sys.modules", {"wandb": fake}),
            patch.dict(os.environ, {"IMAGE_TAG": "v1.2.3"}),
        ):
            log_wandb_provenance()

        assert fake.config.data["image_tag"] == "v1.2.3"

    def test_logs_command_from_argv(self) -> None:
        """Command is a non-empty string derived from sys.argv."""
        fake = make_fake_wandb()

        with patch.dict("sys.modules", {"wandb": fake}):
            log_wandb_provenance()

        assert isinstance(fake.config.data["command"], str)
        assert len(fake.config.data["command"]) > 0


# ---------------------------------------------------------------------------
# Fallback behavior tests
# ---------------------------------------------------------------------------


class TestLogWandbProvenanceFallbacks:
    """Graceful fallbacks when git or IMAGE_TAG are unavailable."""

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("git not found"),
            subprocess.CalledProcessError(128, "git"),
        ],
        ids=["git_not_installed", "not_a_git_repo"],
    )
    def test_git_sha_unknown_on_subprocess_error(self, error: Exception) -> None:
        """github_sha falls back to 'unknown' when git rev-parse fails."""
        fake = make_fake_wandb()

        with (
            patch.dict("sys.modules", {"wandb": fake}),
            patch(
                "synth_setter.utils.logging_utils.subprocess.check_output",
                side_effect=error,
            ),
        ):
            log_wandb_provenance()

        assert fake.config.data["github_sha"] == "unknown"

    def test_image_tag_unknown_when_env_unset(self) -> None:
        """image_tag falls back to 'unknown' when IMAGE_TAG is absent."""
        fake = make_fake_wandb()
        env_without_image_tag = {k: v for k, v in os.environ.items() if k != "IMAGE_TAG"}

        with (
            patch.dict("sys.modules", {"wandb": fake}),
            patch.dict(os.environ, env_without_image_tag, clear=True),
        ):
            log_wandb_provenance()

        assert fake.config.data["image_tag"] == "unknown"


# ---------------------------------------------------------------------------
# Guard clause tests
# ---------------------------------------------------------------------------


class TestLogWandbProvenanceGuards:
    """Safe noop when wandb is unavailable or no run is active."""

    def test_noop_when_wandb_not_installed(self) -> None:
        """No crash when wandb is not installed."""
        with patch.dict("sys.modules", {"wandb": None}):
            log_wandb_provenance()

    def test_noop_when_no_active_run(self) -> None:
        """No config update when wandb.run is None."""
        fake = make_fake_wandb(has_run=False)

        with patch.dict("sys.modules", {"wandb": fake}):
            log_wandb_provenance()

        assert fake.config.data == {}


# ---------------------------------------------------------------------------
# resolve_run_config_id — Hydra experiment choice, fallback task_name
# ---------------------------------------------------------------------------


class TestResolveRunConfigId:
    """config_id resolution for training/eval runs."""

    def test_experiment_choice_basename_wins(self) -> None:
        """A grouped experiment choice resolves to its basename, not task_name."""
        cfg = OmegaConf.create({"task_name": "train"})
        fake_hydra_cfg = SimpleNamespace(
            runtime=SimpleNamespace(choices={"experiment": "surge/flow_simple"})
        )

        with patch(
            "synth_setter.utils.logging_utils.HydraConfig.get", return_value=fake_hydra_cfg
        ):
            assert resolve_run_config_id(cfg) == "flow_simple"

    def test_falls_back_to_task_name_when_experiment_absent(self) -> None:
        """A missing experiment choice resolves to task_name."""
        cfg = OmegaConf.create({"task_name": "train"})
        fake_hydra_cfg = SimpleNamespace(runtime=SimpleNamespace(choices={"experiment": None}))

        with patch(
            "synth_setter.utils.logging_utils.HydraConfig.get", return_value=fake_hydra_cfg
        ):
            assert resolve_run_config_id(cfg) == "train"

    def test_falls_back_to_task_name_when_experiment_is_null_string(self) -> None:
        """The literal ``"null"`` choice (Hydra's null default) resolves to task_name."""
        cfg = OmegaConf.create({"task_name": "train"})
        fake_hydra_cfg = SimpleNamespace(runtime=SimpleNamespace(choices={"experiment": "null"}))

        with patch(
            "synth_setter.utils.logging_utils.HydraConfig.get", return_value=fake_hydra_cfg
        ):
            assert resolve_run_config_id(cfg) == "train"

    def test_falls_back_to_task_name_without_hydra_context(self) -> None:
        """Outside a @hydra.main run (no HydraConfig), task_name is used."""
        cfg = OmegaConf.create({"task_name": "eval"})

        with patch(
            "synth_setter.utils.logging_utils.HydraConfig.get",
            side_effect=ValueError("HydraConfig was not set"),
        ):
            assert resolve_run_config_id(cfg) == "eval"


# ---------------------------------------------------------------------------
# pin_wandb_run_id — write run id + job_type before logger instantiation
# ---------------------------------------------------------------------------


class TestPinWandbRunId:
    """Pinning the deterministic run id and job_type into the wandb logger cfg."""

    def test_sets_run_id_and_job_type(self) -> None:
        """A wandb logger cfg gets the given run id and job_type verbatim."""
        cfg = OmegaConf.create({"logger": {"wandb": {"id": None, "job_type": ""}}})

        pin_wandb_run_id(cfg, "flow_simple-20260313T100000000Z", "training")

        assert cfg.logger.wandb.id == "flow_simple-20260313T100000000Z"
        assert cfg.logger.wandb.job_type == "training"

    def test_noop_when_wandb_logger_absent(self) -> None:
        """A non-wandb logger group is left untouched (no KeyError)."""
        cfg = OmegaConf.create({"logger": {"tensorboard": {"save_dir": "logs"}}})

        pin_wandb_run_id(cfg, "flow_simple", "training")

        assert "wandb" not in cfg.logger


# ---------------------------------------------------------------------------
# use_input_artifacts — consumed-artifact lineage edges (spec §5)
# ---------------------------------------------------------------------------


class FakeWandbRun:
    """Fake wandb run recording every ``use_artifact`` call as inspectable state."""

    def __init__(self, raises: bool = False) -> None:
        """:param raises: When true, ``use_artifact`` raises after recording the call."""
        self.consumed: list[str] = []
        self._raises = raises

    def use_artifact(self, name_alias: str) -> None:
        """Record the requested ``name:alias`` (or raise to model a wandb outage).

        :param name_alias: The ``{name}:{alias}`` lineage edge requested.
        :raises RuntimeError: when the fake was built with ``raises=True``.
        """
        self.consumed.append(name_alias)
        if self._raises:
            raise RuntimeError("wandb down")


class FakeWandbLogger(WandbLogger):
    """A ``WandbLogger`` whose ``experiment`` is a fake run; bypasses ``wandb.init``."""

    def __init__(self, run: FakeWandbRun) -> None:
        """:param run: Fake run returned in place of the live wandb run."""
        self._fake_run = run

    @property
    def experiment(self) -> FakeWandbRun:  # type: ignore[override]
        """:returns: The injected fake run standing in for the live wandb run."""
        return self._fake_run


class TestUseInputArtifacts:
    """Recording consumed-artifact edges on WandbLoggers for the lineage DAG."""

    def test_wandb_logger_present_records_name_alias_edge(self) -> None:
        """A set ref forwards ``name:alias`` to the run's ``use_artifact``."""
        run = FakeWandbRun()

        use_input_artifacts([FakeWandbLogger(run)], [("data-diva-v1", "latest")])

        assert run.consumed == ["data-diva-v1:latest"]

    def test_multiple_refs_record_one_edge_each(self) -> None:
        """Eval consumes both model and dataset — one edge recorded per ref."""
        run = FakeWandbRun()

        use_input_artifacts(
            [FakeWandbLogger(run)],
            [("model-flow-simple", "best"), ("data-diva-v1", "latest")],
        )

        assert run.consumed == ["model-flow-simple:best", "data-diva-v1:latest"]

    def test_non_wandb_logger_records_no_edge(self) -> None:
        """A logger list without a WandbLogger is a silent no-op."""
        run = FakeWandbRun()
        non_wandb_logger = cast(Logger, SimpleNamespace(experiment=run))

        use_input_artifacts([non_wandb_logger], [("data-diva-v1", "latest")])

        assert run.consumed == []

    def test_empty_refs_records_no_edge(self) -> None:
        """No refs means no lineage edge, so use_artifact is never called."""
        run = FakeWandbRun()

        use_input_artifacts([FakeWandbLogger(run)], [])

        assert run.consumed == []

    def test_generator_refs_record_on_every_wandb_logger(self) -> None:
        """A one-shot generator ``refs`` records on both loggers, not just the first."""
        run_a, run_b = FakeWandbRun(), FakeWandbRun()
        refs = (edge for edge in [("data-diva-v1", "latest")])

        use_input_artifacts([FakeWandbLogger(run_a), FakeWandbLogger(run_b)], refs)

        assert run_a.consumed == ["data-diva-v1:latest"]
        assert run_b.consumed == ["data-diva-v1:latest"]

    def test_use_artifact_failure_is_swallowed(self) -> None:
        """A wandb failure is swallowed so a run is never aborted by lineage."""
        run = FakeWandbRun(raises=True)

        use_input_artifacts([FakeWandbLogger(run)], [("data-diva-v1", "latest")])

        assert run.consumed == ["data-diva-v1:latest"]
