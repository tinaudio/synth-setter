"""Tests for synth_setter.cli.generate_dataset — spec-driven run.

The entrypoint's public surface:

- ``main()``: launcher-side orchestrator. Composes the cfg, writes the local
  ``input_spec.json`` mirror, runs ``r2_io.ensure_r2_env_loaded`` (dotenv +
  auth ping), uploads the canonical spec via ``spec_io.upload_spec``, then
  either calls ``generate(spec, work_dir, loggers)`` inline (local-run) or dispatches
  to a SkyPilot worker pod.
- ``generate(spec, work_dir, loggers)``: per-rank renderer. For each owned shard in
  ``spec.shards``, shells out to ``generate_vst_dataset.py`` writing into
  ``work_dir``, then uploads the shard to R2 at ``r2:{bucket}/{prefix}/``;
  rendered shards are retained under ``work_dir`` for downstream consumption.
  ``main()`` writes the canonical spec to R2 once on the launcher host.

``TestRun`` tests share a ``patched_subprocess`` fixture that pulls in
``fake_r2_remote`` (see ``tests/pipeline/conftest.py``) and patches
``_check_call_streamed`` with ``stub_renderer``: renderer calls write a
validation-shaped Lance shard directory (mirroring the contract of
``generate_vst_dataset.py``); rclone calls fall through to the real binary
against the local-typed remote. Orchestration assertions (call counts,
render/stage ordering, partitioning) are state-based — a rendered shard's
completion is probed via ``shard_has_complete_attempt`` (the worker skip-probe,
which reads the shard's staged ``.valid`` marker under ``fake_r2_remote``).
``main()`` tests still stub ``write_spec_locally`` / ``upload_spec`` /
``ensure_r2_env_loaded`` to keep the cfg-composition surface isolated from R2.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.cli.generate_dataset import (
    _RENDERER_SCRIPT,
    build_generate_args,
    generate,
)
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.data.lance_staging import shard_has_complete_attempt
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.pipeline.shard_claims import ShardClaims
from synth_setter.resources import vst_headless_wrapper
from tests.helpers.dummy_shards import stub_renderer
from tests.helpers.finalize_shards import write_minimal_lance_shard
from tests.helpers.subprocess_args import find_script_index

VST_HEADLESS_WRAPPER = str(vst_headless_wrapper())


def _write_lance_split(path: Path, num_rows: int) -> None:
    """Write a minimal ``.lance`` split dataset with ``num_rows`` rows.

    ``_run_oracle_eval_subprocess`` reads this row count via ``count_rows`` to
    scale the eval timeout, so only the row count is load-bearing here.

    :param path: Destination ``.lance`` dataset directory.
    :param num_rows: Row count the dataset is given.
    """
    import lance
    import pyarrow as pa

    lance.write_dataset(pa.table({"audio": pa.array(range(num_rows))}), str(path))


def _render_valid_shard(args: list[str], spec: DatasetSpec) -> None:
    """Write the validation-shaped Lance shard the renderer promises, for ``spec``.

    Custom ``_check_call_streamed`` dispatchers (retry, fail-fast, parallel)
    call this on the renderer branch so the staged shard passes worker-side
    validation before ``stage_lance_shard_attempt`` uploads it.

    :param args: argv list passed to the patched ``_check_call_streamed``.
    :param spec: Dataset spec whose render shape/dtypes define the shard contract.
    """
    write_minimal_lance_shard(Path(args[find_script_index(args) + 1]), spec)


# Reusable VST3 bundle with a real Contents/moduleinfo.json so
# extract_renderer_version (called by generate) returns a deterministic version
# without loading any .so via pedalboard. Version inside is "1.0.0-test" — the
# specs built in this file pin renderer_version to the same string so the
# constraint check passes.
TEST_PLUGIN_VST3 = Path(__file__).resolve().parent.parent / "fixtures" / "TestPlugin.vst3"
TEST_PLUGIN_VERSION = "1.0.0-test"


def _call_hydra_main(main_fn: Callable[..., object]) -> None:
    """Invoke a Hydra-decorated main whose static signature still names ``cfg``.

    :param main_fn: Hydra-decorated main function to call.
    """
    main_fn()


def _renderer_argv_lists(mock: MagicMock) -> list[list[str]]:
    """Return argv lists from non-rclone calls recorded by a patched ``_check_call_streamed``.

    The dispatcher routes both renderer and rclone invocations through one
    ``_check_call_streamed`` mock, so tests that want to introspect just the
    renderer args (script path, flag set, headless wrapper) filter the
    interleaved call list through this helper.

    :param mock: A patched ``_check_call_streamed`` mock.
    :returns: argv lists from invocations whose first element is not ``"rclone"``.
    """
    return [
        call.args[0]
        for call in mock.call_args_list
        if not (call.args and call.args[0] and call.args[0][0] == "rclone")
    ]


def _base_spec_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """Return valid DatasetSpec kwargs for direct construction."""
    kwargs: dict[str, object] = {
        "task_name": "test-dataset",
        "run_id": "test-dataset-20260328T120000000Z",
        "created_at": datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC),
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": "lance",
        # Smoke-sized so ``stub_renderer`` can write a validation-shaped Lance
        # shard per test: worker-side validation checks the row count equals
        # ``samples_per_shard`` and the audio shape derives from the render
        # config, so a full-size shard would be gigabytes.
        "train_val_test_sizes": [2, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "intermediate-data",
            "prefix": "data/test-dataset/test-dataset-20260328T120000000Z/",
        },
        "render": {
            "plugin_path": str(TEST_PLUGIN_VST3),
            "plugin_state_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": TEST_PLUGIN_VERSION,
            "sample_rate": 8000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 0.01,
            "min_loudness": -55.0,
            "samples_per_render_batch": 2,
            "samples_per_shard": 2,
            "gui_toggle_cadence": "never",
        },
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def spec(tmp_path: Path) -> DatasetSpec:
    """Return a valid single-shard DatasetSpec."""
    return DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]


def _multi_shard_spec(tmp_path: Path, n: int = 3) -> DatasetSpec:
    """Return a DatasetSpec with ``n`` shards (deterministic filenames/seeds)."""
    kwargs = _base_spec_kwargs(
        tmp_path,
        train_val_test_sizes=[2 * n, 0, 0],
    )
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


def test_build_generate_args_passes_shard_seed_as_base_seed(tmp_path: Path) -> None:
    """build_generate_args gives each shard its own ``--base_seed`` (#884).

    Argv-shape contract pin, backed end-to-end by
    ``test_distinct_shard_seeds_render_distinct_reproducible_rows``.

    :param tmp_path: Output dir build_generate_args composes shard paths under.
    """
    spec = _multi_shard_spec(tmp_path, n=3)
    passed_seeds = []
    for shard in spec.shards:
        args = build_generate_args(spec, shard, tmp_path)
        idx = args.index("--base_seed")
        passed_seeds.append(args[idx + 1])
    assert len(passed_seeds) == 3
    assert passed_seeds == ["42", "43", "44"]


def test_build_generate_args_passes_split_local_sample_offset(tmp_path: Path) -> None:
    """Renderer argv carries each shard's split-local seed position.

    :param tmp_path: Output dir build_generate_args composes shard paths under.
    """
    kwargs = _base_spec_kwargs(
        tmp_path,
        train_val_test_sizes=[4, 2, 2],
        train_val_test_seeds=[101, 202, 303],
    )
    spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]

    seed_offsets = []
    for shard in spec.shards:
        args = build_generate_args(spec, shard, tmp_path)
        seed_offsets.append(
            (
                args[args.index("--base_seed") + 1],
                args[args.index("--sample_offset") + 1],
            )
        )

    assert seed_offsets == [("101", "0"), ("101", "2"), ("202", "0"), ("303", "0")]


# ---------------------------------------------------------------------------
# load_spec_from_uri — local path, file:// URI, r2:// URI dispatch
# ---------------------------------------------------------------------------


class TestLoadSpecFromUri:
    """``load_spec_from_uri`` accepts bare paths, ``file://`` URIs, and ``r2://`` URIs."""

    def test_bare_local_path_is_read_directly(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """A non-URI argument is treated as a filesystem path.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        from synth_setter.pipeline.spec_io import load_spec_from_uri

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(spec.model_dump_json())

        loaded = load_spec_from_uri(str(spec_path))

        assert loaded.task_name == spec.task_name

    def test_file_uri_is_read_from_local_disk(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """A ``file://`` URI is decoded to a local path and read directly.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        from synth_setter.pipeline.spec_io import load_spec_from_uri

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(spec.model_dump_json())

        loaded = load_spec_from_uri(spec_path.as_uri())

        assert loaded.task_name == spec.task_name


class TestLoadSpecFromRoot:
    """``load_spec_from_root`` joins ``input_spec.json`` under a dataset-root URI."""

    @pytest.mark.parametrize(
        "root",
        ["r2://bucket/data/task/run/", "r2://bucket/data/task/run"],
        ids=["trailing_slash", "no_trailing_slash"],
    )
    def test_joins_spec_filename_onto_root_collapsing_trailing_slash(
        self, root: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both root forms resolve to exactly ``<root>/input_spec.json`` (no double slash).

        Captures the URI handed to ``load_spec_from_uri`` so the assertion pins
        the scheme-agnostic join — dropping the ``rstrip('/')`` would leak a
        ``//`` into the trailing-slash case, which a ``file://`` round-trip would
        silently normalize away.

        :param root: Dataset-root URI with and without a trailing slash.
        :param monkeypatch: Used to capture ``load_spec_from_uri``'s argument.
        """
        import synth_setter.pipeline.spec_io as spec_io

        captured: list[str] = []
        monkeypatch.setattr(spec_io, "load_spec_from_uri", lambda uri: captured.append(uri))

        spec_io.load_spec_from_root(root)

        assert captured == [f"r2://bucket/data/task/run/{INPUT_SPEC_FILENAME}"]

    def test_round_trips_spec_through_local_root(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """A local dataset-root URI re-hydrates the spec written under it.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir hosting the run prefix.
        """
        from synth_setter.pipeline.spec_io import load_spec_from_root

        (tmp_path / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())

        loaded = load_spec_from_root(tmp_path.as_uri())

        assert loaded.task_name == spec.task_name


class TestSpecUriCliMain:
    """``generate_dataset_from_spec_uri.main`` — parse spec URI and W&B opt-out."""

    def test_single_positional_runs_that_spec_uri_with_wandb_enabled(self) -> None:
        """The sole positional enables the default grouped W&B logging path."""
        import synth_setter.cli.generate_dataset_from_spec_uri as cli

        with patch.object(cli, "run_from_spec_uri") as mock_run:
            cli.main(["r2://bucket/run/input_spec.json"])

        mock_run.assert_called_once_with("r2://bucket/run/input_spec.json", enable_wandb=True)

    def test_no_wandb_flag_disables_wandb_logging(self) -> None:
        """``--no-wandb`` skips only W&B auth/logging for repair runs."""
        import synth_setter.cli.generate_dataset_from_spec_uri as cli

        with patch.object(cli, "run_from_spec_uri") as mock_run:
            cli.main(["--no-wandb", "r2://bucket/run/input_spec.json"])

        mock_run.assert_called_once_with("r2://bucket/run/input_spec.json", enable_wandb=False)

    @pytest.mark.parametrize(
        "argv",
        [[], ["a/input_spec.json", "b/input_spec.json"]],
        ids=["missing_uri", "two_uris"],
    )
    def test_wrong_positional_count_exits_with_usage_error(self, argv: list[str]) -> None:
        """Zero or two positionals exit via argparse's usage error (code 2).

        :param argv: argv tail (no program name).
        """
        import synth_setter.cli.generate_dataset_from_spec_uri as cli

        with patch.object(cli, "run_from_spec_uri") as mock_run:
            with pytest.raises(SystemExit) as excinfo:
                cli.main(argv)

        assert excinfo.value.code == 2
        mock_run.assert_not_called()


class TestSpecUriWandbSettings:
    """``generate_dataset_from_spec_uri`` W&B settings defaults."""

    def test_settings_disable_wandb_without_api_key_or_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing W&B auth selects disabled mode instead of prompting.

        :param monkeypatch: Clears W&B auth and mode environment variables.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import _wandb_mode_override

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)

        assert _wandb_mode_override() == "disabled"

    def test_settings_preserve_explicit_wandb_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit ``WANDB_MODE`` overrides the no-auth disabled fallback.

        :param monkeypatch: Sets W&B mode and clears W&B auth.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import _wandb_mode_override

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")

        assert _wandb_mode_override() == "offline"

    def test_settings_keep_default_mode_with_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Present W&B auth leaves mode unspecified for normal online logging.

        :param monkeypatch: Sets W&B auth and clears W&B mode.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import _wandb_mode_override

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        monkeypatch.delenv("WANDB_MODE", raising=False)

        assert _wandb_mode_override() is None

    def test_settings_disable_wandb_for_invalid_mode_without_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unsupported ``WANDB_MODE`` falls back to disabled mode without auth.

        :param monkeypatch: Sets invalid W&B mode and clears W&B auth.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import _wandb_mode_override

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "bogus")

        assert _wandb_mode_override() == "disabled"

    def test_settings_keep_default_for_invalid_mode_with_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unsupported ``WANDB_MODE`` falls back to W&B's default with auth.

        :param monkeypatch: Sets W&B auth and invalid W&B mode.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import _wandb_mode_override

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        monkeypatch.setenv("WANDB_MODE", "bogus")

        assert _wandb_mode_override() is None


class TestRunFromSpecUri:
    """``run_from_spec_uri`` — load a spec by URI, then render/upload its shards."""

    @pytest.fixture(autouse=True)
    def _single_worker_full_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin rank=0/world=1 and force every shard absent in R2 (full render path).

        :param monkeypatch: Pytest fixture used to set env vars and force the worker skip-probe to
            report every shard as un-staged.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            lambda *_a, **_k: False,
        )

    @pytest.fixture()
    def patched_env_and_subprocess(
        self, fake_r2_remote: Path, spec: DatasetSpec
    ) -> Iterator[MagicMock]:
        """Patch the render/rclone seam and the R2 env pre-flight.

        ``ensure_r2_env_loaded`` is stubbed (same isolation the ``main()``
        tests use) because the fake local-typed remote has no creds to
        validate; renderer calls write a validation-shaped Lance shard and
        rclone staging still lands on ``fake_r2_remote`` via the passthrough.

        :param fake_r2_remote: Local-typed R2 remote root (chdirs into the
            tmp dir, so relative work dirs also land there).
        :param spec: Fixture-provided single-shard ``DatasetSpec`` whose render
            shape the ``stub_renderer`` shard writer honours.
        :yields MagicMock: Patched ``_check_call_streamed`` mock.
        """
        with patch("synth_setter.pipeline.r2_io.ensure_r2_env_loaded"):
            with patch(
                "synth_setter.cli.generate_dataset._check_call_streamed",
                side_effect=stub_renderer(spec),
            ) as mock_check_call:
                yield mock_check_call

    def test_local_spec_path_renders_and_uploads_shards(
        self,
        patched_env_and_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """A bare local spec path drives the full render → R2 upload flow.

        :param patched_env_and_subprocess: Render/rclone dispatcher seam.
        :param fake_r2_remote: Fake R2 root; uploaded shards materialize here.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import run_from_spec_uri

        spec_path = tmp_path / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json())

        run_from_spec_uri(str(spec_path), enable_wandb=False)

        shard = spec.shards[0]
        assert shard_has_complete_attempt(spec, shard.shard_id)

    def test_r2_spec_uri_downloads_spec_then_renders(
        self,
        patched_env_and_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
    ) -> None:
        """An ``r2://`` spec URI is fetched from the remote before rendering.

        The spec is placed in the fake remote via the production
        ``spec_io.upload_spec`` path, then re-loaded through the URI the
        launcher advertises to workers (``spec.r2.input_spec_uri()``).

        :param patched_env_and_subprocess: Render/rclone dispatcher seam.
        :param fake_r2_remote: Fake R2 root backing both spec and shards.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import run_from_spec_uri
        from synth_setter.pipeline.spec_io import upload_spec

        spec_uri = upload_spec(spec)

        run_from_spec_uri(spec_uri, enable_wandb=False)

        shard = spec.shards[0]
        assert shard_has_complete_attempt(spec, shard.shard_id)

    def test_work_dir_derives_from_run_id_under_cwd(
        self,
        patched_env_and_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """Rendered shards persist under ``logs/generate_dataset/from_spec_uri/<run_id>/``.

        :param patched_env_and_subprocess: Render/rclone dispatcher seam.
        :param fake_r2_remote: Fake R2 root; the fixture chdirs into it, so the
            relative work dir lands inside the test's tmp dir.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import run_from_spec_uri

        spec_path = tmp_path / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json())

        run_from_spec_uri(str(spec_path), enable_wandb=False)

        work_dir = Path.cwd() / "logs" / "generate_dataset" / "from_spec_uri" / spec.run_id
        assert (work_dir / spec.shards[0].filename).is_dir()

    def test_wandb_enabled_passes_resume_loggers_to_generate(
        self,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """The default runner creates a W&B resume logger list for ``generate``.

        :param fake_r2_remote: Fake R2 root (unused; activates rclone skip gate).
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        import synth_setter.cli.generate_dataset_from_spec_uri as cli

        spec_path = tmp_path / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json())
        loggers = [MagicMock()]

        with patch("synth_setter.pipeline.r2_io.ensure_r2_env_loaded"):
            with patch.object(cli, "_resume_loggers", return_value=loggers):
                with patch.object(cli, "generate") as mock_generate:
                    cli.run_from_spec_uri(str(spec_path))

        work_dir = Path("logs") / "generate_dataset" / "from_spec_uri" / spec.run_id
        mock_generate.assert_called_once_with(spec, work_dir, loggers)

    def test_wandb_enabled_uses_no_loggers_when_wandb_missing(
        self,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """A wandb-free install still renders through the default CLI path.

        :param fake_r2_remote: Fake R2 root (unused; activates rclone skip gate).
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        import synth_setter.cli.generate_dataset_from_spec_uri as cli

        spec_path = tmp_path / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json())

        with patch("synth_setter.pipeline.r2_io.ensure_r2_env_loaded"):
            with patch.object(cli, "find_spec", return_value=None):
                with patch.object(cli, "generate") as mock_generate:
                    cli.run_from_spec_uri(str(spec_path))

        work_dir = Path("logs") / "generate_dataset" / "from_spec_uri" / spec.run_id
        mock_generate.assert_called_once_with(spec, work_dir, [])

    def test_wandb_enabled_builds_grouped_repair_run(
        self,
        fake_r2_remote: Path,
        monkeypatch: pytest.MonkeyPatch,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """The default runner creates a grouped W&B run for the repair attempt.

        :param fake_r2_remote: Fake R2 root (unused; activates rclone skip gate).
        :param monkeypatch: Clears W&B env vars so defaults are hermetic.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        import synth_setter.cli.generate_dataset_from_spec_uri as cli

        monkeypatch.setenv("WANDB_PROJECT", "")
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        spec_path = tmp_path / INPUT_SPEC_FILENAME
        spec_path.write_text(spec.model_dump_json())
        settings = MagicMock(name="settings")
        wandb_logger = MagicMock(name="wandb_logger")

        with patch("synth_setter.pipeline.r2_io.ensure_r2_env_loaded"):
            with patch("wandb.Settings", return_value=settings) as mock_settings:
                with patch(
                    "lightning.pytorch.loggers.wandb.WandbLogger",
                    return_value=wandb_logger,
                ) as mock_wandb_logger:
                    with patch.object(cli, "generate") as mock_generate:
                        cli.run_from_spec_uri(str(spec_path))

        work_dir = Path("logs") / "generate_dataset" / "from_spec_uri" / spec.run_id
        mock_settings.assert_called_once_with(
            code_dir=".",
            console="wrap",
            console_multipart=True,
            mode="disabled",
        )
        mock_wandb_logger.assert_called_once_with(
            save_dir=str(work_dir),
            name=f"resume-{spec.task_name}-{spec.run_id}",
            project="synth-setter",
            entity=None,
            group=spec.run_id,
            job_type="data-generation-resume",
            tags=["from-spec-uri", "resume", spec.task_name],
            log_model=False,
            settings=settings,
        )
        mock_generate.assert_called_once_with(spec, work_dir, [wandb_logger])

    def test_r2_env_preflight_runs_before_spec_fetch(
        self,
        fake_r2_remote: Path,
        spec: DatasetSpec,
    ) -> None:
        """``ensure_r2_env_loaded`` gates the run — its failure aborts before any fetch.

        :param fake_r2_remote: Fake R2 root (unused; activates rclone skip gate).
        :param spec: Fixture-provided single-shard ``DatasetSpec`` (unused body).
        """
        from synth_setter.cli.generate_dataset_from_spec_uri import run_from_spec_uri

        with patch(
            "synth_setter.pipeline.r2_io.ensure_r2_env_loaded",
            side_effect=RuntimeError("no creds"),
        ):
            with pytest.raises(RuntimeError, match="no creds"):
                run_from_spec_uri("r2://bucket/never-fetched/input_spec.json")


class RenderSeamFixtures:
    """Shared render/rclone-seam fixtures for ``generate()`` dispatch test classes."""

    @pytest.fixture(autouse=True)
    def _default_shard_absent_in_r2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the worker skip-probe to report every shard un-staged (full render path).

        Lance shards resume off a staged ``.valid`` marker; defaulting
        ``shard_has_complete_attempt`` to ``False`` insulates the render-path
        tests from any host state. Tests that exercise the skip-existing path
        override this with their own ``monkeypatch.setattr``.

        :param monkeypatch: Pytest fixture used to patch the completion probe.
        """
        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            lambda *_a, **_k: False,
        )

    @pytest.fixture()
    def patched_subprocess(self, fake_r2_remote: Path, spec: DatasetSpec) -> Iterator[MagicMock]:  # noqa: ARG002
        """Patch ``_check_call_streamed`` with the ``stub_renderer`` dispatcher.

        Pulls in ``fake_r2_remote`` (consumed by the rclone staging passthrough)
        so staging rclone copies land on the local-typed remote rooted at the
        tmp dir instead of hitting real R2. Renderer calls write a
        validation-shaped Lance shard for ``spec``. Yielding the mock lets tests
        introspect ``call_args_list`` (typically via ``_renderer_argv_lists`` to
        filter out interleaved rclone calls) and override ``side_effect``
        per-test when a failure or no-write renderer is needed.

        :param fake_r2_remote: Local-typed R2 remote root (fixture-activation
            only — referenced via the ARG002 noqa).
        :param spec: Fixture-provided ``DatasetSpec`` whose render shape the
            ``stub_renderer`` shard writer honours.
        :yields MagicMock: Patched ``_check_call_streamed`` mock.
        """
        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=stub_renderer(spec),
        ) as mock_check_call:
            yield mock_check_call


class TestRun(RenderSeamFixtures):
    """Render → upload, per owned shard.

    No spec upload — ``main()`` writes it once.
    """

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin rank=0/world=1 explicitly so partition-agnostic tests are insulated from host env.

        ``read_rank_world_from_env`` defaults to ``(0, 1)`` when both vars are
        absent, but this fixture also overwrites any value the developer's shell
        may have exported (e.g. an in-flight multi-worker debugging session) so
        the partition-agnostic tests in this class stay deterministic. Tests
        that probe multi-worker partitioning override via ``monkeypatch.setenv``;
        the default-fallback test overrides via ``monkeypatch.delenv``.

        :param monkeypatch: Pytest fixture used to set env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    def test_invokes_generate_vst_dataset_with_spec_derived_args(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """The render seam invokes generate_vst_dataset.py with spec-derived args.

        :param patched_subprocess: Subprocess dispatcher used to introspect the
            single renderer call's argv.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        args = renderer_calls[0]
        # args = [VST_HEADLESS_WRAPPER (linux only), python, generate_vst_dataset.py, ...]
        assert any("generate_vst_dataset.py" in a for a in args)
        assert str(spec.render.samples_per_shard) in args

    def test_dawdreamer_worker_runtime_failure_precedes_render_side_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """An incompatible DawDreamer worker fails before logging or filesystem writes.

        :param monkeypatch: Replaces the runtime guard and first logging side effect.
        :param spec: Base spec copied to select the DawDreamer backend.
        :param tmp_path: Work directory that must remain absent.
        """
        import synth_setter.cli.generate_dataset as gd

        dawdreamer_spec = spec.model_copy(
            update={"render": spec.render.model_copy(update={"renderer_backend": "dawdreamer"})}
        )
        log_mock = MagicMock()
        monkeypatch.setattr(gd, "_log_hyperparams", log_mock)
        monkeypatch.setattr(
            gd,
            "ensure_dawdreamer_runtime",
            MagicMock(side_effect=RuntimeError("unsupported DawDreamer worker")),
        )
        work_dir = tmp_path / "never-created"

        with pytest.raises(RuntimeError, match="unsupported DawDreamer worker"):
            generate(dawdreamer_spec, work_dir, [])

        log_mock.assert_not_called()
        assert not work_dir.exists()

    def test_shard_generation_runs_under_headless_vst_wrapper(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """Prefix the VST subprocess with ``run-linux-vst-headless.sh`` on Linux.

        X11 bootstrap lives at the audio-rendering boundary (this subprocess), keeping the outer
        pipeline X11-agnostic. The wrapper is Linux-only (Xvfb is a Linux X11 server); on macOS and
        other platforms the generator is invoked directly without a wrapper prefix.

        :param patched_subprocess: Subprocess dispatcher used to introspect the
            renderer argv (looking for the headless-wrapper prefix on Linux).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        args = renderer_calls[0]
        if sys.platform == "linux":
            assert args[0] == VST_HEADLESS_WRAPPER
            renderer_script = Path(args[2])
        else:
            assert VST_HEADLESS_WRAPPER not in args
            renderer_script = Path(args[1])
        assert renderer_script.as_posix().endswith("synth_setter/data/vst/generate_vst_dataset.py")
        assert renderer_script.is_absolute()
        assert renderer_script.is_file()

    def test_uploads_shard_to_r2_after_generation(
        self,
        spec: DatasetSpec,
        fake_r2_remote: Path,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """Shard stages a complete attempt under ``spec.r2`` after generation.

        State-based: the real rclone staging runs against the fake-local R2
        remote rooted at ``fake_r2_remote``, and the test asserts the shard's
        staged ``.valid`` marker via ``shard_has_complete_attempt``. The renderer
        subprocess ``_check_call_streamed`` is patched via the shared
        ``patched_subprocess`` fixture so we don't shell out to the VST generator;
        its renderer branch writes the validation-shaped Lance shard directory
        the renderer would.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param patched_subprocess: Fixture-activation only (handles the
            ``_check_call_streamed`` patch).
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

        assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)

    def test_subprocess_failure_propagates(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """CalledProcessError from generate_vst_dataset propagates to caller.

        :param patched_subprocess: Subprocess dispatcher; overridden here to
            unconditionally raise so the renderer call short-circuits.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote — asserted empty since
            no shard should land when the renderer fails.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        patched_subprocess.side_effect = subprocess.CalledProcessError(
            1, "generate_vst_dataset.py"
        )

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

        # Renderer failed before validation/staging: no complete attempt is staged.
        assert not shard_has_complete_attempt(spec, spec.shards[0].shard_id)

    def test_rclone_failure_propagates(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """CalledProcessError from rclone (shard upload path) propagates to caller.

        Forces a real rclone failure by pointing the ``r2:`` remote at a
        nonexistent backend type. The renderer side of the dispatcher still
        materializes the shard file (so the source-existence check passes);
        ``rclone copy`` then raises ``CalledProcessError`` when it tries to
        construct the destination backend.

        :param patched_subprocess: Fixture-activation only (the renderer path
            still materializes the shard so the rclone source exists).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to invalidate the ``r2:`` remote type so the
            real rclone subprocess exits non-zero on the copy.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "this-backend-does-not-exist")

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

    def test_run_with_three_shards_renders_each_shard(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Multi-shard run invokes generate_vst_dataset.py once per shard, in order.

        :param patched_subprocess: Dispatcher mock; ``_renderer_argv_lists``
            filters out rclone calls so we can introspect the per-shard
            output-path argv.
        :param fake_r2_remote: All three shards should stage into this remote.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)

        generate(spec, tmp_path, [])

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 3
        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name for args in renderer_calls
        ]
        assert rendered_filenames == [s.filename for s in spec.shards]
        # State-based proof: every shard staged a complete attempt.
        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id)

    def test_each_shard_staged_after_its_render(
        self,
        fake_r2_remote: Path,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render and stage are interleaved per shard: render0, stage0, render1, stage1, ...

        :param fake_r2_remote: Fixture-activation only — the render's own
            ``write_rendering_marker`` staging needs the local-typed remote.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Records the ``stage_lance_shard_attempt`` events.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        events: list[str] = []

        def _record_render(args: list[str]) -> None:
            events.append("renderer")
            _render_valid_shard(args, spec)

        def _record_stage(*_a: object, **_k: object) -> None:
            events.append("stage")

        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.stage_lance_shard_attempt", _record_stage
        )
        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=_record_render,
        ):
            generate(spec, tmp_path, [])

        assert events == [
            "renderer",  # shard 0
            "stage",
            "renderer",  # shard 1
            "stage",
            "renderer",  # shard 2
            "stage",
        ]

    def test_shards_persist_after_upload(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Every rendered shard remains at ``work_dir / shard.filename`` after staging.

        Pins the post-stage retention contract: ``finalize_dataset`` and
        post-mortem consumers expect shards to outlive the render+stage step.

        :param fake_r2_remote: Local-typed R2 remote — asserted to hold a
            complete staged attempt per shard alongside the on-disk copies.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)

        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=stub_renderer(spec),
        ):
            generate(spec, tmp_path, [])

        for shard in spec.shards:
            # Lance shards are directories; the rendered shard is retained on disk.
            assert (tmp_path / shard.filename).is_dir()
            assert shard_has_complete_attempt(spec, shard.shard_id)

    def test_subprocess_failure_in_second_shard_propagates_immediately(
        self,
        fake_r2_remote: Path,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """Mid-loop subprocess failure raises immediately; later shards are not attempted.

        :param fake_r2_remote: Fixture-activation only — the render's staging
            markers need the local-typed remote.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        renderer_call_count = 0

        def _side_effect(args: list[str]) -> None:
            nonlocal renderer_call_count
            renderer_call_count += 1
            if renderer_call_count == 2:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            _render_valid_shard(args, spec)

        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=_side_effect,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                generate(spec, tmp_path, [])

        assert renderer_call_count == 2
        # State-based proof of fail-fast: only shard 0 staged a complete attempt.
        assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)
        assert not shard_has_complete_attempt(spec, spec.shards[1].shard_id)
        assert not shard_has_complete_attempt(spec, spec.shards[2].shard_id)

    def test_subprocess_exits_zero_without_writing_shard_raises(
        self,
        fake_r2_remote: Path,  # noqa: ARG002
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """If the renderer exits 0 but never wrote the expected shard dataset, fail loudly.

        Catches a generator bug at the rendering boundary instead of letting it surface as a less-
        direct rclone "source not found" further down the pipeline.

        :param fake_r2_remote: Fixture-activation only — the pre-render staging
            marker needs the local-typed remote.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        # Renderer-only side effect: succeed without writing the shard dataset,
        # so the ``shard_path.is_dir()`` guard raises before validation/staging.
        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="did not write expected shard"):
                generate(spec, tmp_path, [])

        assert not shard_has_complete_attempt(spec, spec.shards[0].shard_id)

    def test_renderer_version_mismatch_raises_before_uploads(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Fail before any rclone/subprocess work when plugin version disagrees with spec.

        This prevents emitting a shard tagged with the wrong renderer_version.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "renderer_version": "999.999.999"}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Renderer version mismatch"):
            generate(spec, tmp_path, [])
        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_run_defaults_to_single_worker_when_skypilot_env_absent(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No partition env → rank=0/world=1, the lone worker renders every shard.

        Underwrites the dev-experience contract that ``synth-setter-generate-dataset`` works
        out of the box without exporting ``SYNTH_SETTER_WORKER_RANK`` / ``SYNTH_SETTER_NUM_WORKERS``.

        :param patched_subprocess: Subprocess dispatcher used to introspect renderer argv per shard.
        :param fake_r2_remote: Local-typed R2 remote — asserted to contain every shard.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to unset the rank/world env vars.
        """
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.delenv("SYNTH_SETTER_NUM_WORKERS", raising=False)
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)

        generate(spec, tmp_path, [])

        rendered_filenames = {
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        }
        assert rendered_filenames == {shard.filename for shard in spec.shards}
        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id)

    def test_run_raises_on_partial_partition_env(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only world set (rank dropped) → ValueError before any rclone or subprocess work.

        Partial partition env almost always means a launcher dropped half its env injection;
        silently coercing it to single-worker would duplicate every shard across every node
        (#763). The default-to-(0, 1) fallback is gated on BOTH vars being absent.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to set world but unset rank.
        """
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)

        with pytest.raises(ValueError) as excinfo:
            generate(spec, tmp_path, [])
        message = str(excinfo.value)
        assert "SYNTH_SETTER_WORKER_RANK" in message
        assert "SYNTH_SETTER_NUM_WORKERS" in message
        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    @pytest.mark.parametrize(
        ("rank", "world", "expected_indices"),
        [
            pytest.param(0, 2, [0, 1], id="rank0-of-2-renders-first-half"),
            pytest.param(1, 2, [2], id="rank1-of-2-renders-remainder"),
            pytest.param(3, 4, [], id="excess-worker-renders-none"),
        ],
    )
    def test_worker_renders_only_its_partition_of_shards(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        rank: int,
        world: int,
        expected_indices: list[int],
    ) -> None:
        """A worker renders (and uploads) exactly its contiguous slice of the 3 shards.

        Covers in-range ranks that own a non-empty slice and an excess rank
        (``world`` > num_shards) whose empty range renders nothing and makes no
        rclone calls.

        :param patched_subprocess: Subprocess dispatcher used to introspect
            renderer argv; never invoked when ``expected_indices`` is empty.
        :param fake_r2_remote: Local-typed R2 remote — asserted to contain only
            the worker's shards.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to set the rank/world env vars.
        :param rank: This worker's ``SYNTH_SETTER_WORKER_RANK``.
        :param world: Partition size ``SYNTH_SETTER_NUM_WORKERS``.
        :param expected_indices: Shard indices the worker should render and upload.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", str(rank))
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", str(world))
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)

        generate(spec, tmp_path, [])

        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        ]
        assert rendered_filenames == [spec.shards[i].filename for i in expected_indices]
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        if not expected_indices:
            patched_subprocess.assert_not_called()
            assert not bucket_prefix.exists()
            return
        for index, shard in enumerate(spec.shards):
            if index in expected_indices:
                assert shard_has_complete_attempt(spec, shard.shard_id)
            else:
                assert not shard_has_complete_attempt(spec, shard.shard_id)

    # Skip-existing-shards — see #750.

    def test_run_skips_render_when_shard_already_staged(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A complete staged attempt → renderer is not invoked, no new staging happens.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty (the
            probe stub claims "complete" without seeding an actual attempt).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to override the skip-probe to claim the shard
            is already staged.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            lambda *_a, **_k: True,
        )

        generate(spec, tmp_path, [])

        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_run_renders_when_shard_absent(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """An un-staged shard (skip-probe False) is rendered exactly once and staged.

        :param patched_subprocess: Subprocess dispatcher; renderer is asserted
            to fire exactly once (the autouse fixture forces the skip-probe False).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)

    def test_run_skip_path_probes_each_assigned_shard_id(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The skip-probe is called once per assigned shard, with that shard's id.

        :param patched_subprocess: Fixture-activation only (renderer stages
            shards so the staging probe has a valid source).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to install the skip-probe capture stub.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)
        probed_ids: list[int] = []

        def _probe(_spec: DatasetSpec, shard_id: int) -> bool:
            probed_ids.append(shard_id)
            return False

        monkeypatch.setattr("synth_setter.cli.generate_dataset.shard_has_complete_attempt", _probe)

        generate(spec, tmp_path, [])

        assert probed_ids == [shard.shard_id for shard in spec.shards]

    def test_run_renders_only_absent_shards_in_mixed_run(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mid-run resumption: shard 0 already staged, shards 1 and 2 absent → render only 1 and 2.

        :param patched_subprocess: Subprocess dispatcher used to introspect
            renderer argv.
        :param fake_r2_remote: Fixture-activation only — shards 1 and 2 stage
            here; shard 0 does not (it was reported "complete" by the probe).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to install the per-shard skip-probe stub.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)

        def _complete_only_for_shard_0(_spec: DatasetSpec, shard_id: int) -> bool:
            return shard_id == 0

        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            _complete_only_for_shard_0,
        )

        generate(spec, tmp_path, [])

        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        ]
        assert rendered_filenames == ["shard-000001.lance", "shard-000002.lance"]
        # shard 0 was skip-simulated (never rendered); shards 1 and 2 staged.
        assert not shard_has_complete_attempt(spec, spec.shards[0].shard_id)
        assert shard_has_complete_attempt(spec, spec.shards[1].shard_id)
        assert shard_has_complete_attempt(spec, spec.shards[2].shard_id)

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_run_logs_summary_with_rendered_and_skipped_counts(
        self,
        mock_logger: MagicMock,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-of-run summary reports rendered/skipped counts over the assigned range.

        :param mock_logger: Patched ``generate_dataset.logger`` for capturing
            the summary line.
        :param patched_subprocess: Fixture-activation only (renderer stages
            shards so the staging probe has a valid source).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to install the per-shard skip-probe stub.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)

        def _complete_only_for_shard_0(_spec: DatasetSpec, shard_id: int) -> bool:
            return shard_id == 0

        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            _complete_only_for_shard_0,
        )

        generate(spec, tmp_path, [])

        info_calls = mock_logger.info.call_args_list
        summary_calls = [
            c for c in info_calls if "rendered=" in str(c.args[0]) and "skipped=" in str(c.args[0])
        ]
        assert len(summary_calls) == 1, f"expected exactly one summary line, got: {info_calls}"
        assert summary_calls[0].kwargs["rendered"] == 2
        assert summary_calls[0].kwargs["skipped"] == 1
        assert "of 3 assigned" in summary_calls[0].kwargs["assignment"]

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_run_logs_generation_speed_from_rendered_samples(
        self,
        mock_logger: MagicMock,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Speed log line counts only ``rendered`` shards, not skipped ones.

        :param mock_logger: Captures the speed log line's content.
        :param patched_subprocess: Fixture-activation only.
        :param tmp_path: Used by ``_multi_shard_spec``.
        :param monkeypatch: Installs the skip-probe stub that skips shard 0.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        patched_subprocess.side_effect = stub_renderer(spec)

        def _complete_only_for_shard_0(_spec: DatasetSpec, shard_id: int) -> bool:
            return shard_id == 0

        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            _complete_only_for_shard_0,
        )

        generate(spec, tmp_path, [])

        info_calls = mock_logger.info.call_args_list
        speed_calls = [c for c in info_calls if "generation speed:" in str(c.args[0])]
        assert len(speed_calls) == 1, f"expected one speed line, got: {info_calls}"
        assert speed_calls[0].kwargs["samples"] == 2 * spec.render.samples_per_shard
        assert "samples/s" in str(speed_calls[0].args[0])

    def test_run_probe_failure_propagates(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A non-zero rclone exit during the skip-probe propagates as CalledProcessError.

        :param patched_subprocess: Subprocess dispatcher; asserted never
            invoked (the probe failure raises before any render/stage).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to install the raising skip-probe stub.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """

        def _raise(*_a: object, **_k: object) -> None:
            raise subprocess.CalledProcessError(1, ["rclone", "lsf"])

        monkeypatch.setattr("synth_setter.cli.generate_dataset.shard_has_complete_attempt", _raise)

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

        patched_subprocess.assert_not_called()

    def test_render_retries_transient_failure_when_max_retries_set(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """``max_retries=1`` + flaky renderer (1 fail, then success) → shard stages in R2.

        The renderer-subprocess retry loop covers transient X11 / Xvfb init races
        and pedalboard load hiccups on first call into a fresh subprocess. Staging
        sits outside the loop; only the renderer call is wrapped.

        :param fake_r2_remote: Fixture-activation only — the render's staging
            markers need the local-typed remote.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "max_retries": 1}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        renderer_calls = 0

        def _flaky_dispatcher(args: list[str]) -> None:
            nonlocal renderer_calls
            renderer_calls += 1
            if renderer_calls == 1:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            _render_valid_shard(args, spec)

        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=_flaky_dispatcher,
        ):
            generate(spec, tmp_path, [])

        assert renderer_calls == 2
        assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)

    def test_parallel_render_uses_thread_pool_and_uploads_all_shards(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``render.parallel=True`` + 4 shards → ≥2 worker threads; every shard stages.

        Pins ``available_cpus`` to 8 so the dispatch heuristic
        ``min(max(1, available_cpus() // 2), len(my_range))`` resolves to 4
        workers regardless of CI runner CPU count. The dispatcher stub blocks
        each render until the second thread enters, forcing the pool to
        actually parallelize.

        :param fake_r2_remote: Fixture-activation only — the parallel staging
            markers need the local-typed remote.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        :param monkeypatch: Pins ``available_cpus`` so pool size is deterministic.
        """
        monkeypatch.setattr("synth_setter.cli.generate_dataset.available_cpus", lambda: 8)
        kwargs = _base_spec_kwargs(tmp_path, train_val_test_sizes=[8, 0, 0])
        kwargs["render"] = {**kwargs["render"], "parallel": True}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        assert len(spec.shards) == 4
        thread_ids: set[int] = set()
        lock = threading.Lock()
        two_threads_seen = threading.Event()

        def _thread_recording_dispatcher(args: list[str]) -> None:
            with lock:
                thread_ids.add(threading.get_ident())
                if len(thread_ids) >= 2:
                    two_threads_seen.set()
            two_threads_seen.wait(timeout=5.0)
            _render_valid_shard(args, spec)

        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=_thread_recording_dispatcher,
        ):
            generate(spec, tmp_path, [])

        assert len(thread_ids) >= 2
        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id)

    def test_parallel_render_propagates_subprocess_failure(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One failing render in a parallel pool → CalledProcessError surfaces fail-fast.

        Pins ``available_cpus`` to 4 so pool size resolves to 2 workers. The
        first thread to enter fails; the in-flight peer can complete, but
        the remaining two futures get ``cancel_futures=True``-aborted before
        they start. Net effect: renderer fires at most twice (1 fail + ≤1
        in-flight peer), and at most one shard lands in R2.

        :param fake_r2_remote: Fixture-activation only — used for the
            state-based cancellation check.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        :param monkeypatch: Pins ``available_cpus`` so pool size is deterministic.
        """
        monkeypatch.setattr("synth_setter.cli.generate_dataset.available_cpus", lambda: 4)
        kwargs = _base_spec_kwargs(tmp_path, train_val_test_sizes=[8, 0, 0])
        kwargs["render"] = {**kwargs["render"], "parallel": True}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        renderer_call_count = 0
        lock = threading.Lock()

        def _one_failing(args: list[str]) -> None:
            nonlocal renderer_call_count
            with lock:
                renderer_call_count += 1
                this_attempt = renderer_call_count
            if this_attempt == 1:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            _render_valid_shard(args, spec)

        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=_one_failing,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                generate(spec, tmp_path, [])

        landed = sum(
            1 for shard in spec.shards if shard_has_complete_attempt(spec, shard.shard_id)
        )
        assert landed <= 1
        assert renderer_call_count <= 2

    def test_render_raises_after_exhausting_max_retries(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """``max_retries=2`` + always-failing renderer → 3 attempts then propagate.

        Confirms the retry budget is bounded: ``max_retries + 1`` total attempts,
        then ``CalledProcessError`` surfaces and no shard is staged.

        :param fake_r2_remote: Fixture-activation only — the pre-render staging
            marker needs the local-typed remote.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "max_retries": 2}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        renderer_calls = 0

        def _always_fails(_args: list[str]) -> None:
            nonlocal renderer_calls
            renderer_calls += 1
            raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")

        with patch(
            "synth_setter.cli.generate_dataset._check_call_streamed",
            side_effect=_always_fails,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                generate(spec, tmp_path, [])

        assert renderer_calls == 3
        assert not shard_has_complete_attempt(spec, spec.shards[0].shard_id)

    def test_shard_lands_in_work_dir_before_staging(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Caller-supplied ``work_dir`` hosts the rendered shard at the staging boundary.

        Stubs ``stage_lance_shard_attempt`` to snapshot the shard path and its
        on-disk existence at the staging moment — proving the shard landed at
        ``work_dir / shard.filename`` (as a Lance directory) before staging runs.

        :param patched_subprocess: Fixture-activation only; the renderer side of
            the dispatcher writes the shard dataset into ``work_dir``.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for this run.
        :param monkeypatch: Used to install the staging stub that captures the
            shard path + existence at the staging moment.
        """
        captured: dict[str, object] = {}

        def _capture_stage(
            _spec: object, _shard: object, shard_path: Path, **_kwargs: object
        ) -> None:
            captured["src"] = shard_path
            captured["existed_at_stage"] = shard_path.is_dir()

        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.stage_lance_shard_attempt", _capture_stage
        )

        generate(spec, tmp_path, [])

        assert captured["src"] == tmp_path / spec.shards[0].filename
        assert captured["existed_at_stage"] is True

    def test_provenance_not_stamped_when_no_wandb_logger(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``log_wandb_provenance`` is skipped when ``loggers`` owns no ``WandbLogger``.

        Provenance mutates the process-global ``wandb.run``; gating the call on a
        locally-owned ``WandbLogger`` keeps an empty-logger run from stamping a
        foreign in-process run, mirroring the ``_close_loggers`` ownership guard.

        :param patched_subprocess: Fixture-activation only; the renderer
            materializes the shard so the run reaches its summary.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        :param monkeypatch: Installs the ``log_wandb_provenance`` spy.
        """
        provenance = MagicMock()
        monkeypatch.setattr("synth_setter.cli.generate_dataset.log_wandb_provenance", provenance)

        generate(spec, tmp_path, [])

        provenance.assert_not_called()


# generate() with use_shard_queue — dynamic claim-table dispatch


def _claims_spec(tmp_path: Path, n: int = 3) -> DatasetSpec:
    """Return an ``n``-shard DatasetSpec that opts into dynamic claims.

    :param tmp_path: Per-test dir used for the spec's plugin/preset paths.
    :param n: Number of train shards (10000 samples each).
    :returns: Validated spec with ``use_shard_queue=True``.
    """
    kwargs = _base_spec_kwargs(
        tmp_path,
        train_val_test_sizes=[2 * n, 0, 0],
        use_shard_queue=True,
    )
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


def test_shard_claims_for_spec_targets_s3_table_under_run_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claims seam binds env storage credentials to the run's table URI.

    :param tmp_path: Per-test dir used for the spec's plugin/preset paths.
    :param monkeypatch: Injects the canonical ``SYNTH_SETTER_STORAGE_*`` env.
    """
    from synth_setter.cli.generate_dataset import _shard_claims_for_spec

    monkeypatch.delenv("RCLONE_CONFIG_R2_TYPE", raising=False)
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("SYNTH_SETTER_STORAGE_ENDPOINT_URL", "https://endpoint.example")
    spec = _claims_spec(tmp_path)

    claims = _shard_claims_for_spec(spec)

    assert claims.uri == f"s3://{spec.r2.bucket}/{spec.r2.prefix}metadata/shard-claims.lance"
    assert claims.storage_options is not None
    assert claims.storage_options["endpoint"] == "https://endpoint.example"
    assert claims.storage_options["access_key_id"] == "ak"


def test_shard_claims_for_spec_local_remote_resolves_under_cwd(
    tmp_path: Path,
    fake_r2_remote: Path,  # noqa: ARG001 — activates the local-typed remote
) -> None:
    """With the local-typed ``r2:`` remote, the table sits inside the fake R2 root.

    :param tmp_path: Per-test dir used for the spec's plugin/preset paths.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote
        (fixture-activation only — referenced via the ARG001 noqa).
    """
    from synth_setter.cli.generate_dataset import _shard_claims_for_spec

    spec = _claims_spec(tmp_path)

    claims = _shard_claims_for_spec(spec)

    expected = Path.cwd() / spec.r2.bucket / spec.r2.prefix / "metadata/shard-claims.lance"
    assert claims.uri == str(expected)
    assert claims.storage_options is None


class TestRunWithShardClaims(RenderSeamFixtures):
    """``generate()`` with ``use_shard_queue=True`` claims shards from the run table.

    No seam is patched: ``_shard_claims_for_spec`` resolves through the real
    ``lance_target`` against the local-typed remote the ``patched_subprocess``
    fixture activates, so the worker and these tests read one real Lance
    table inside the fake R2 root.
    """

    def _populate(self, spec: DatasetSpec, shard_ids: list[int]) -> ShardClaims:
        """Seed the run's claims table the way the operator does.

        :param spec: Spec locating the table under the fake R2 root.
        :param shard_ids: Rows to seed.
        :returns: A peer claims handle over the same table, for assertions.
        """
        from synth_setter.cli.generate_dataset import _shard_claims_for_spec

        claims = _shard_claims_for_spec(spec)
        claims.populate(shard_ids)
        return claims

    def test_generate_renders_every_claimed_shard_and_completes_all(
        self,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """All populated shards render, upload, and are marked done.

        :param patched_subprocess: Renderer stub; the real rclone upload runs.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        spec = _claims_spec(tmp_path, n=3)
        claims = self._populate(spec, [shard.shard_id for shard in spec.shards])
        patched_subprocess.side_effect = stub_renderer(spec)

        generate(spec, tmp_path, [])

        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id)
        assert claims.claim() is None
        assert claims.status_counts() == {"done": 3}

    def test_generate_renders_only_claimed_shards_not_full_spec_range(
        self,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """The claims table, not the spec's shard range, decides the assignment.

        :param patched_subprocess: Renderer stub; the real rclone upload runs.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        spec = _claims_spec(tmp_path, n=3)
        self._populate(spec, [1])
        patched_subprocess.side_effect = stub_renderer(spec)

        generate(spec, tmp_path, [])

        assert shard_has_complete_attempt(spec, spec.shards[1].shard_id)
        assert not shard_has_complete_attempt(spec, spec.shards[0].shard_id)
        assert not shard_has_complete_attempt(spec, spec.shards[2].shard_id)

    @pytest.mark.parametrize("invalid_shard_id", [-1, 3, 10_000])
    def test_generate_out_of_range_claimed_shard_id_fails_before_rendering(
        self,
        invalid_shard_id: int,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Persisted claim rows outside the spec range fail before rendering.

        :param invalid_shard_id: Negative, upper-bound, or far-out shard ID.
        :param patched_subprocess: Renderer spy that must remain unused.
        :param tmp_path: Per-test work directory.
        """
        spec = _claims_spec(tmp_path, n=3)
        self._populate(spec, [invalid_shard_id])

        with pytest.raises(
            ValueError,
            match=rf"shard_id {invalid_shard_id}.*outside.*\[0, 3\)",
        ):
            generate(spec, tmp_path, [])

        patched_subprocess.assert_not_called()

    def test_generate_render_failure_leaves_claim_held_and_propagates(
        self,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A failed render fails the run with its claim still held.

        Holding the claim (rather than releasing it) keeps a deterministically
        failing shard from cascading across the live fleet: peers cannot
        re-claim it until the lease lapses.

        :param patched_subprocess: Overridden to raise on the renderer call.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        spec = _claims_spec(tmp_path, n=1)
        claims = self._populate(spec, [0])
        patched_subprocess.side_effect = subprocess.CalledProcessError(
            1, "generate_vst_dataset.py"
        )

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

        assert claims.status_counts() == {"claimed": 1}
        assert claims.claim() is None  # live lease shields the poison shard

    def test_generate_keyboard_interrupt_leaves_claim_held_and_propagates(
        self,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Ctrl-C mid-render propagates; the held claim recovers via lease lapse.

        :param patched_subprocess: Overridden to raise ``KeyboardInterrupt``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        spec = _claims_spec(tmp_path, n=1)
        claims = self._populate(spec, [0])
        patched_subprocess.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            generate(spec, tmp_path, [])

        assert claims.status_counts() == {"claimed": 1}

    def test_generate_claims_mode_ignores_render_parallel_with_warning(
        self,
        patched_subprocess: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``render.parallel=true`` + claims mode warns and never enters the thread pool.

        :param patched_subprocess: Renderer stub; renders succeed.
        :param monkeypatch: Guards the parallel dispatcher against being called.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        base = _claims_spec(tmp_path, n=2)
        spec = base.model_copy(
            update={"render": base.render.model_copy(update={"parallel": True})}
        )
        claims = self._populate(spec, [0, 1])
        patched_subprocess.side_effect = stub_renderer(spec)

        def _parallel_must_not_fire(*_args: object, **_kwargs: object) -> tuple[int, int]:
            raise AssertionError("_dispatch_shards_parallel must not run in claims mode")

        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset._dispatch_shards_parallel",
            _parallel_must_not_fire,
        )
        from loguru import logger as loguru_logger

        warnings: list[str] = []
        sink_id = loguru_logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        try:
            generate(spec, tmp_path, [])
        finally:
            loguru_logger.remove(sink_id)

        assert any("render.parallel=true is ignored" in w for w in warnings)
        assert shard_has_complete_attempt(spec, spec.shards[0].shard_id)
        assert shard_has_complete_attempt(spec, spec.shards[1].shard_id)
        assert claims.status_counts() == {"done": 2}

    def test_generate_shard_already_in_r2_completes_without_render(
        self,
        patched_subprocess: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The R2 skip-probe stays authoritative in claims mode: present → done, no render.

        :param patched_subprocess: Introspected to assert no renderer call ran.
        :param monkeypatch: Marks the shard present in R2.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        monkeypatch.setattr(
            "synth_setter.cli.generate_dataset.shard_has_complete_attempt",
            lambda *_a, **_k: True,
        )
        spec = _claims_spec(tmp_path, n=1)
        claims = self._populate(spec, [0])

        generate(spec, tmp_path, [])

        assert _renderer_argv_lists(patched_subprocess) == []
        assert claims.status_counts() == {"done": 1}


# ---------------------------------------------------------------------------
# build_generate_args — arg construction from spec + shard
# ---------------------------------------------------------------------------


class TestBuildGenerateArgs:
    """build_generate_args() produces correct CLI arg lists from spec + shard."""

    def test_output_file_uses_shard_filename(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Output file path is {output_dir}/{shard.filename}."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, tmp_path)

        assert args[2] == str(tmp_path / "shard-000000.lance")

    def test_samples_per_shard_passed_as_option(self, spec: DatasetSpec) -> None:
        """samples_per_shard is emitted as ``--samples_per_shard <count>`` flag.

        The CLI no longer takes a positional ``num_samples`` — every renderer
        config field is exposed as a flag, including the per-shard sample count.
        """
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        flag_idx = args.index("--samples_per_shard")
        assert args[flag_idx + 1] == str(spec.render.samples_per_shard)

    def test_all_render_config_fields_passed_as_options(self, spec: DatasetSpec) -> None:
        """The flag set equals ``RenderConfig.model_fields`` — auto-derived parity guard."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        option_keys: set[str] = {arg.lstrip("-") for arg in args if arg.startswith("--")}

        assert option_keys == set(RenderConfig.model_fields.keys())

    def test_args_start_with_python_and_script(self, spec: DatasetSpec) -> None:
        """First arg is the Python executable, second is the generation script."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        assert Path(args[1]) == _RENDERER_SCRIPT
        assert _RENDERER_SCRIPT.is_file()

    def test_script_path_resolves_from_any_working_directory(
        self, spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shard dispatch survives a cwd far from the repo root (import-anchored script path).

        The script argv entry must resolve to an existing absolute file
        regardless of process cwd — e.g. under ``fake_r2_remote``'s ``chdir``.

        :param spec: Smoke dataset spec fixture.
        :param tmp_path: Working directory unrelated to the repo checkout.
        :param monkeypatch: Changes the process cwd for the call.
        """
        monkeypatch.chdir(tmp_path)

        args = build_generate_args(spec, spec.shards[0], tmp_path / "out")

        assert Path(args[1]).is_absolute()
        assert Path(args[1]).is_file()

    def test_missing_renderer_script_raises_at_build_time(
        self, spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vanished renderer script fails loudly at arg-build, not inside a worker.

        :param spec: Smoke dataset spec fixture.
        :param tmp_path: Holds the nonexistent script path.
        :param monkeypatch: Points the module at the nonexistent script.
        """
        from synth_setter.cli import generate_dataset

        missing_script = tmp_path / "gone.py"
        monkeypatch.setattr(generate_dataset, "_RENDERER_SCRIPT", missing_script)

        with pytest.raises(RuntimeError, match=f"renderer script not found: {missing_script}"):
            build_generate_args(spec, spec.shards[0], tmp_path / "out")

    def test_renderer_script_directory_raises_at_build_time(
        self, spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A renderer directory fails at arg-build rather than reaching a worker.

        :param spec: Smoke dataset spec fixture.
        :param tmp_path: Holds the directory substituted for the script.
        :param monkeypatch: Points the module at the substituted directory.
        """
        from synth_setter.cli import generate_dataset

        renderer_directory = tmp_path / "renderer"
        renderer_directory.mkdir()
        monkeypatch.setattr(generate_dataset, "_RENDERER_SCRIPT", renderer_directory)

        with pytest.raises(
            RuntimeError,
            match=f"renderer script not found: {renderer_directory}",
        ):
            build_generate_args(spec, spec.shards[0], tmp_path / "out")


# ---------------------------------------------------------------------------
# spec_from_cfg — Hydra-composed cfg → DatasetSpec
# ---------------------------------------------------------------------------


class TestSpecFromCfg:
    """``spec_from_cfg`` drops Hydra-only groups and constructs a DatasetSpec."""

    def test_drops_non_spec_groups(self, valid_dataset_spec_kwargs: dict[str, object]) -> None:
        """``data``, ``paths``, ``hydra`` are dropped so strict validation passes.

        DatasetSpec is configured with ``extra="forbid"``; if any of these groups leaked through,
        construction would raise on the unknown field. The assertion is implicit in the absence
        of a ValidationError. After the ``R2Location`` migration ``r2`` is *not* dropped —
        it composes from ``configs/r2/default.yaml`` directly into ``DatasetSpec.r2``.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        cfg_dict: dict[str, object] = dict(valid_dataset_spec_kwargs)
        cfg_dict["datamodule"] = {"sample_rate": 44100}
        cfg_dict["paths"] = {"root_dir": "/fake-root"}
        cfg_dict["hydra"] = {"runtime": {"output_dir": "/fake-out"}}

        spec = spec_from_cfg(OmegaConf.create(cfg_dict))

        assert spec.task_name == valid_dataset_spec_kwargs["task_name"]

    def test_r2_group_flows_into_nested_r2_field(
        self, valid_dataset_spec_kwargs: dict[str, object]
    ) -> None:
        """The ``r2`` group composes directly into ``DatasetSpec.r2`` (no flat-key indirection).

        Mirrors the production composition after the ``R2Location`` migration:
        ``configs/dataset.yaml`` no longer interpolates flat ``r2_bucket`` /
        ``r2_prefix_root`` keys — the group's content lands at ``cfg.r2`` and
        passes through to ``DatasetSpec.r2``.

        :param valid_dataset_spec_kwargs: Baseline spec kwargs from conftest.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        kwargs = dict(valid_dataset_spec_kwargs)
        kwargs["r2"] = {"bucket": "from-group-bucket", "prefix_root": "data"}

        spec = spec_from_cfg(OmegaConf.create(kwargs))

        assert spec.r2.bucket == "from-group-bucket"
        assert spec.r2.prefix_root == "data"


# PROJECT_ROOT-bootstrap behavior is exercised end-to-end by tests/pipeline/configs/
# test_experiment_yamls.py — those tests fail with an InterpolationResolutionError if the
# launcher's import-time `operator_workspace()` ever stops setting PROJECT_ROOT.


# ---------------------------------------------------------------------------
# _build_worker_cmd — shell-quoted cmd injection for sky.Task.run
# ---------------------------------------------------------------------------


class TestBuildWorkerCmd:
    """The worker cmd reconstructs the operator's Hydra invocation under bash."""

    @pytest.fixture()
    def spec(self, tmp_path: Path) -> DatasetSpec:
        """Reusable DatasetSpec for worker-cmd construction (no I/O — pure kwargs).

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :return: A ``DatasetSpec`` built from base kwargs.
        """
        return DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]

    def test_cmd_uses_from_hydra_console_script(self, spec: DatasetSpec) -> None:
        """The worker reproduces the composition by re-entering the from_hydra entry point.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd(["experiment=foo"], spec)
        assert "synth-setter-generate-dataset-from-hydra" in cmd
        assert "experiment=foo" in cmd

    def test_cmd_cds_to_worker_repo_root_not_launcher_repo(self, spec: DatasetSpec) -> None:
        """Cd target is the worker checkout, not the launcher's path.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _WORKER_REPO_ROOT, _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        assert cmd.startswith(f"cd {_WORKER_REPO_ROOT}")
        assert _WORKER_REPO_ROOT == "/home/build/synth-setter"

    def test_cmd_runs_sync_worker_checkout_before_exec(self, spec: DatasetSpec) -> None:
        """sync_worker_checkout.sh bypasses dev-snapshot bake-lag when WORKER_GIT_REF is set.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        sync_idx = cmd.find("bash scripts/sync_worker_checkout.sh")
        exec_idx = cmd.find("exec synth-setter-generate-dataset-from-hydra")
        assert sync_idx != -1, f"sync step missing from cmd: {cmd!r}"
        assert exec_idx != -1, f"exec step missing from cmd: {cmd!r}"
        assert sync_idx < exec_idx, "sync_worker_checkout must run before exec"

    def test_cmd_repairs_stale_worker_python_before_sync(self, spec: DatasetSpec) -> None:
        """The PR checkout can raise the Python floor before the image is rebuilt.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        sync_idx = cmd.find("bash scripts/sync_worker_checkout.sh")
        repair_idx = cmd.find("source scripts/ensure_worker_python.sh")
        exec_idx = cmd.find("exec synth-setter-generate-dataset-from-hydra")
        assert repair_idx != -1, f"python repair step missing from cmd: {cmd!r}"
        assert "uv venv --python 3.12.13" in cmd
        assert "bash scripts/sync_worker_checkout.sh --python-ready" in cmd
        assert repair_idx < sync_idx < exec_idx

    def test_cmd_pins_spec_created_at_via_hydra_override(self, spec: DatasetSpec) -> None:
        """Worker compose must inherit launcher's created_at to land on the same r2.prefix.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        # `+key=value` is Hydra's add-key syntax; spec.created_at.isoformat() goes in verbatim
        # (no surrounding quotes added by shlex when the value has no shell metachars).
        assert f"+created_at={spec.created_at.isoformat()}" in cmd

    def test_cmd_shell_quotes_overrides_with_spaces(self, spec: DatasetSpec) -> None:
        """Spaces and special chars in an override survive bash interpretation in run:.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd(["task_name=value with space"], spec)
        # shlex.quote wraps the whole assignment in single quotes; the bare-word form
        # would be split into two argv items by bash.
        assert "'task_name=value with space'" in cmd

    def test_cmd_handles_empty_operator_overrides(self, spec: DatasetSpec) -> None:
        """No operator overrides → cmd is just cd + sync + exec + pinned-runtime override.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        assert cmd.startswith("cd ")
        assert " && exec synth-setter-generate-dataset-from-hydra " in cmd
        # No bash-interpretable trailing whitespace that would surface as an empty argv item.
        assert cmd == cmd.rstrip()


# ---------------------------------------------------------------------------
# main — dispatching CLI entry: local vs SkyPilot
# ---------------------------------------------------------------------------


class TestMainDispatchBranches:
    """``main()`` composes the dataset cfg from argv, then dispatches local or via SkyPilot."""

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pin single-worker rank/world + isolate Hydra's per-run dir to ``tmp_path``.

        ``@hydra.main`` resolves ``${paths.log_dir}`` from ``${oc.env:PROJECT_ROOT}``;
        redirecting PROJECT_ROOT keeps the per-run dir under the test tree.

        :param monkeypatch: Pytest fixture used to set env vars.
        :param tmp_path: Per-test tmp dir hosting PROJECT_ROOT.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    @pytest.fixture(autouse=True)
    def _stub_spec_io_in_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the spec_io helpers + ``ensure_r2_env_loaded`` so ``main()`` doesn't shell out.

        Tests access the mocks via ``gd.write_spec_locally``,
        ``gd.upload_spec``, and ``gd.r2_io.ensure_r2_env_loaded`` to keep test
        signatures stable for pydoclint.

        :param monkeypatch: Pytest fixture used to patch the helpers.
        """
        import synth_setter.cli.generate_dataset as gd

        write_mock = MagicMock(side_effect=lambda spec, out: Path(out) / "input_spec.json")
        upload_mock = MagicMock(return_value="r2://stub-bucket/stub-key/input_spec.json")
        monkeypatch.setattr("synth_setter.cli.generate_dataset.write_spec_locally", write_mock)
        monkeypatch.setattr("synth_setter.cli.generate_dataset.upload_spec", upload_mock)
        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", MagicMock(return_value=None))

    def test_compute_template_null_calls_run_locally(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``compute_template=null`` calls ``generate(spec, work_dir, loggers)`` inline.

        Dispatch (``dispatch_via_skypilot``) is never reached on this branch.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        # Use a real experiment so cfg.skypilot_launch resolves; override the plugin path
        # to the test VST3 so generate() — which we replace below — sees the right spec shape.
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, object] = {}

        def _fake_run(spec: object, _work_dir: object, _loggers: object) -> None:
            recorded["spec"] = spec

        def _dispatch_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dispatch_via_skypilot must not be called on the local branch")

        monkeypatch.setattr(gd, "generate", _fake_run)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", _dispatch_must_not_fire)

        _call_hydra_main(gd.main)

        spec = recorded.get("spec")
        assert isinstance(spec, DatasetSpec)
        assert spec.render.plugin_path == str(TEST_PLUGIN_VST3)

    def test_local_dawdreamer_runtime_failure_precedes_spec_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A local DawDreamer run validates this process before materializing its spec.

        :param monkeypatch: Selects the DawDreamer experiment and fails its runtime guard.
        """
        import synth_setter.cli.generate_dataset as gd

        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-generate-dataset",
                "experiment=generate_dataset/surge-xt-dawdreamer-smoke",
                f"render.plugin_path={TEST_PLUGIN_VST3}",
            ],
        )
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        monkeypatch.setattr(
            gd,
            "ensure_dawdreamer_runtime",
            MagicMock(side_effect=RuntimeError("unsupported DawDreamer worker")),
        )

        with pytest.raises(RuntimeError, match="unsupported DawDreamer worker"):
            _call_hydra_main(gd.main)

        gd.write_spec_locally.assert_not_called()
        gd.upload_spec.assert_not_called()

    def test_local_run_applies_extras_writing_tags_and_config_tree(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``main()`` runs ``extras(cfg)`` before generating, materializing its artifacts.

        ``dataset.yaml`` composes ``extras: default`` (``enforce_tags`` +
        ``print_config`` true) and a non-empty ``tags``, so ``extras(cfg)``
        exports ``tags.log`` and ``config_tree.log`` to ``cfg.paths.output_dir``.
        Asserting those files exist verifies the entrypoint applied extras via
        its observable side effects rather than mocking the call.

        :param monkeypatch: Pytest fixture used to patch argv + ``generate``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, Path] = {}

        def _fake_run(_spec: object, work_dir: Path, _loggers: object) -> None:
            recorded["work_dir"] = work_dir

        monkeypatch.setattr(gd, "generate", _fake_run)

        _call_hydra_main(gd.main)

        output_dir = recorded["work_dir"]
        for artifact in ("tags.log", "config_tree.log"):
            path = output_dir / artifact
            assert path.is_file(), f"extras did not write {artifact}"
            assert path.stat().st_size > 0, f"{artifact} is empty"

    def test_use_shard_queue_local_run_populates_claims_before_generate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``use_shard_queue=true`` seeds every claim row before generating.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        :param tmp_path: Hosts the run's local claims table.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "use_shard_queue=true",
        ]
        monkeypatch.setattr("sys.argv", argv)

        claims = ShardClaims.for_run(str(tmp_path / "shard-claims.lance"), None)
        monkeypatch.setattr(gd, "_shard_claims_for_spec", lambda _spec: claims)

        recorded: dict[str, object] = {}

        def _fake_run(spec: DatasetSpec, _work_dir: object, _loggers: object) -> None:
            recorded["spec"] = spec
            recorded["claim_during_generate"] = claims.claim()

        monkeypatch.setattr(gd, "generate", _fake_run)

        _call_hydra_main(gd.main)

        spec = recorded["spec"]
        assert isinstance(spec, DatasetSpec)
        assert recorded["claim_during_generate"] is not None, (
            "claims must be populated before generate runs"
        )
        drained = 0
        while claims.claim() is not None:
            drained += 1
        assert drained == spec.num_shards - 1

    def test_use_shard_queue_dispatch_branch_populates_claims_before_launch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The SkyPilot branch seeds the claims table before any worker is launched.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
            "use_shard_queue=true",
        ]
        monkeypatch.setattr("sys.argv", argv)

        claims = ShardClaims.for_run(str(tmp_path / "shard-claims.lance"), None)
        monkeypatch.setattr(gd, "_shard_claims_for_spec", lambda _spec: claims)

        recorded: dict[str, object] = {}

        def _fake_dispatch(_sky_cfg: object) -> None:
            recorded["claim_at_dispatch"] = claims.claim()

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        _call_hydra_main(gd.main)

        assert recorded["claim_at_dispatch"] is not None, (
            "claims must be populated before workers are dispatched"
        )
        uploaded_spec = gd.upload_spec.call_args.args[0]  # type: ignore[attr-defined]
        drained = 0
        while claims.claim() is not None:
            drained += 1
        assert drained == uploaded_spec.num_shards - 1

    def test_compute_template_set_calls_dispatch_via_skypilot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """compute_template=<path> routes through dispatch_via_skypilot with cmd populated.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        # A bare-minimum compute YAML the loader will accept (resources + envs, no run:).
        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, object] = {}

        def _fake_dispatch(sky_cfg: object) -> None:
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        def _run_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("generate must not be called on the dispatch branch")

        monkeypatch.setattr(gd, "generate", _run_must_not_fire)

        _call_hydra_main(gd.main)

        assert "sky_cfg" in recorded
        sky_cfg = recorded["sky_cfg"]
        assert sky_cfg.compute_template == str(template)  # type: ignore[attr-defined]
        assert sky_cfg.cmd is not None  # type: ignore[attr-defined]
        # Every operator-supplied override (sans argv[0]) round-trips into the worker cmd
        # so the worker reproduces this composition byte-for-byte.
        for override in argv[1:]:
            assert override in sky_cfg.cmd, (  # type: ignore[attr-defined]
                f"override {override!r} missing from worker cmd: {sky_cfg.cmd!r}"  # type: ignore[attr-defined]
            )

    def test_remote_dawdreamer_dispatch_does_not_validate_launcher_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Remote dispatch defers DawDreamer runtime validation to the worker.

        :param monkeypatch: Selects remote dispatch and records runtime validation.
        :param tmp_path: Holds the minimal SkyPilot compute template.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-generate-dataset",
                "experiment=generate_dataset/surge-xt-dawdreamer-smoke",
                f"render.plugin_path={TEST_PLUGIN_VST3}",
                f"skypilot_launch.compute_template={template}",
            ],
        )
        runtime_mock = MagicMock(side_effect=AssertionError("launcher runtime was validated"))
        dispatch_mock = MagicMock()
        monkeypatch.setattr(gd, "ensure_dawdreamer_runtime", runtime_mock)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", dispatch_mock)

        _call_hydra_main(gd.main)

        runtime_mock.assert_not_called()
        dispatch_mock.assert_called_once()

    def test_operator_supplied_cmd_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `+skypilot_launch.cmd=…` override is rejected before any dispatch fires.

        Uses Hydra's `+key=value` add-syntax because the key isn't in
        configs/skypilot_launch/default.yaml (struct-mode would otherwise reject it before our
        guard runs). ``HYDRA_FULL_ERROR=1`` makes ``@hydra.main`` re-raise the launcher-side
        ``ValueError`` instead of converting it to ``SystemExit(1)``, so the assertion pins the
        launcher contract directly rather than coupling to Hydra's error-handler formatting.

        :param monkeypatch: Pytest fixture used to set ``sys.argv`` and ``HYDRA_FULL_ERROR``.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "+skypilot_launch.cmd=rm -rf /",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")

        def _run_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("generate must not be called when cmd is rejected")

        def _dispatch_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dispatch_via_skypilot must not be called when cmd is rejected")

        monkeypatch.setattr(gd, "generate", _run_must_not_fire)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", _dispatch_must_not_fire)

        with pytest.raises(ValueError, match="skypilot_launch.cmd is launcher-internal"):
            _call_hydra_main(gd.main)

    def test_main_finalize_inline_true_invokes_finalize_from_spec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """finalize_inline=true on the local-run branch invokes finalize_from_spec.

        Composes a real ``smoke-shard`` experiment with the new flag set,
        stubs ``generate`` to a no-op, and replaces ``finalize_from_spec``
        with a mock so the test pins the wire (call + spec identity)
        without needing real rclone against a finalize-shaped remote. The
        end-to-end marker upload is already covered by the Phase 1
        ``test_finalize_from_spec_uploads_stats_then_marker_at_canonical_uris``
        sibling test.

        :param monkeypatch: Pytest fixture used to patch argv +
            ``generate`` + ``finalize_from_spec``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)

        captured: dict[str, object] = {}

        def _capture_spec(spec: object, _work_dir: Path, _loggers: object) -> None:
            captured["spec"] = spec

        monkeypatch.setattr(gd, "generate", _capture_spec)
        finalize_mock = MagicMock()
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)

        _call_hydra_main(gd.main)

        finalize_mock.assert_called_once()
        called_spec, called_work_dir = finalize_mock.call_args[0]
        assert isinstance(called_spec, DatasetSpec)
        assert called_spec is captured["spec"]
        assert isinstance(called_work_dir, Path)

    def test_main_finalize_inline_default_false_skips_finalize(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default ``finalize_inline=false`` leaves the existing local-run shape unchanged.

        Pins the opt-in invariant — no finalize fires when the operator
        omits the override.

        :param monkeypatch: Pytest fixture used to patch argv +
            ``generate`` + ``finalize_from_spec``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        finalize_mock = MagicMock()
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)

        _call_hydra_main(gd.main)

        finalize_mock.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_main_finalize_inline_ignored_in_dispatch_branch(
        self,
        mock_logger: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """finalize_inline=true is ignored (with INFO log) when dispatching to SkyPilot.

        SkyPilot delegation hands the run to a worker pod; finalize must
        run out-of-band via the finalize-dataset workflow rather than fire
        from the launcher process. Pins both halves: ``finalize_from_spec``
        is not called, and an INFO log fires (wording unpinned).

        :param mock_logger: Patched ``generate_dataset.logger`` — the
            established loguru capture pattern in this file.
        :param monkeypatch: Pytest fixture used to patch argv + dispatch +
            ``finalize_from_spec`` (asserted unreached).
        :param tmp_path: Pytest fixture providing a fresh test directory for
            the minimal compute template.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
            "finalize_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)
        monkeypatch.setattr(
            gd,
            "generate",
            lambda *_a, **_k: pytest.fail("generate must not fire on dispatch branch"),
        )
        finalize_mock = MagicMock()
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)

        _call_hydra_main(gd.main)

        finalize_mock.assert_not_called()
        # State assertion above is the contract. The log check matches stable
        # tokens (knob name + "ignored") so a reworded message survives, but a
        # vacuous ``assert_called`` would not — ``main`` always emits INFO logs.
        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        ignored_lines = [m for m in info_messages if "finalize_inline=" in m and "ignored" in m]
        assert len(ignored_lines) == 1, (
            f"expected one INFO log marking the override ignored; got: {info_messages!r}"
        )

    def test_main_oracle_eval_inline_true_invokes_subprocess(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """oracle_eval_inline=true fires the eval subprocess once per split.

        Asserts the eval helper fires once per split, reading data **in place**
        from ``cfg.paths.output_dir`` (no download), with each Hydra run dir
        isolated under ``output_dir/oracle_eval/<split>/<run_id>/``.

        :param monkeypatch: Patches argv + the three module-level seams.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            # Override smoke-shard's [12, 0, 0] — the zero-size guard rejects
            # train_val_test_sizes with any zero split for oracle_eval_inline.
            "train_val_test_sizes=[12, 4, 4]",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        monkeypatch.setattr(gd, "finalize_from_spec", MagicMock())
        # finalize writes each split to R2; ``main`` materializes them locally for
        # the eval. Stub that download so no real rclone runs against a bare remote.
        monkeypatch.setattr(gd.r2_io, "download_dir_no_overwrite", MagicMock())
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        # Capture the resolved output_dir so the eval's dataset_root can be
        # pinned to the exact dir generate+finalize wrote the shards into.
        observed: dict[str, object] = {}
        real_spec_from_cfg = gd.spec_from_cfg

        def _capture_output_dir(cfg: object) -> DatasetSpec:
            observed["output_dir"] = Path(cfg.paths.output_dir)  # type: ignore[attr-defined]
            observed["num_workers"] = cfg.datamodule.num_workers  # type: ignore[attr-defined]
            return real_spec_from_cfg(cfg)  # type: ignore[arg-type]

        monkeypatch.setattr(gd, "spec_from_cfg", _capture_output_dir)

        _call_hydra_main(gd.main)

        # One invocation per split.
        assert oracle_mock.call_count == 3
        output_dir = observed["output_dir"]
        assert isinstance(output_dir, Path)
        splits = ("train", "val", "test")
        split_lances = ("train.lance", "val.lance", "test.lance")
        # test stays bare ``audio/*``; train/val are namespaced so the shared
        # wandb run keeps one summary key per split instead of overwriting.
        split_prefixes = ("train/", "val/", "")
        for call, split, split_lance, prefix in zip(
            oracle_mock.call_args_list, splits, split_lances, split_prefixes
        ):
            dataset_root, run_dir, _run_id = call[0]
            # The whole generation RenderConfig flows through (keyword-only) so
            # the eval re-renders through the same spec; smoke-shard is surge_simple.
            render_arg = call.kwargs["render"]
            assert render_arg.param_spec_name == "surge_simple"
            # The eval inherits the generate run's datamodule worker count verbatim,
            # so a Darwin override (num_workers=0) reaches the predict DataLoader.
            assert call.kwargs["num_workers"] == observed["num_workers"]
            assert render_arg.plugin_state_path == "presets/surge-simple.vstpreset"
            # plugin_path is the TEST_PLUGIN_VST3 this test overrode at generation —
            # proving a non-default plugin flows through to the eval re-render.
            assert render_arg.plugin_path == str(TEST_PLUGIN_VST3)
            # The eval reads in place from the Hydra output_dir where the shards and
            # VDS splits already live — not a downloaded copy under oracle_eval/.
            assert dataset_root == output_dir
            # predict_file targets this split's Lance dataset.
            assert call.kwargs["predict_file"] == output_dir / split_lance
            # Run dir: oracle_eval/<split>/<run_id>
            assert run_dir.parent.parent.name == "oracle_eval", (
                f"eval run dir should land under "
                f"<output_dir>/oracle_eval/<split>/<run_id>/; got {run_dir!r}"
            )
            assert run_dir.parent.name == split
            assert run_dir.parent.parent.parent == dataset_root
            assert call.kwargs["metric_prefix"] == prefix

    def test_run_oracle_eval_subprocess_builds_expected_argv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """Calls ``synth_setter.cli.eval`` as a subprocess and pins the contract argv.

        Pins the load-bearing overrides (``experiment=surge/fake_oracle``,
        ``datamodule.dataset_root``, ``ckpt_path=null``, ``mode=predict``), the
        wandb-resume trio that routes the eval's ``audio/*`` metrics onto the
        generate run, and every render field passed through from the generation
        ``RenderConfig`` so the eval re-renders identically (here a surge_xt spec).
        Runs the helper directly so cfg-resolution noise can't mask an argv drift.

        :param monkeypatch: Patches the module's ``_check_call_streamed``.
        :param tmp_path: Roots the distinct dataset-root and eval run dirs.
        :param spec: Source of a valid ``RenderConfig`` to derive the eval render from.
        """
        import synth_setter.cli.generate_dataset as gd

        streamed_call_mock = MagicMock()
        monkeypatch.setattr(gd, "_check_call_streamed", streamed_call_mock)

        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for name in ("train.lance", "val.lance", "stats.npz"):
            (dataset_root / name).touch()
        run_dir = tmp_path / "oracle_eval" / "some-run-id"
        render = spec.render.model_copy(
            update={
                "param_spec_name": "surge_xt",
                "plugin_state_path": "presets/surge-base.vstpreset",
                "plugin_path": "plugins/Surge XT.vst3",
            }
        )
        predict_file = dataset_root / "test.lance"
        n_samples = 4
        _write_lance_split(predict_file, n_samples)
        gd._run_oracle_eval_subprocess(
            dataset_root,
            run_dir,
            "some-run-id",
            render=render,
            num_workers=7,
            predict_file=predict_file,
        )

        streamed_call_mock.assert_called_once()
        # Hard-coded (not mirroring the module constants) so a wrong constant fails;
        # the timeout bounds an otherwise-unbounded eval hang (#735) and now scales
        # with the split: 600 overhead + 120/sample * 4 rows = 1080.
        assert streamed_call_mock.call_args.kwargs["timeout"] == 1080.0
        called_argv = streamed_call_mock.call_args[0][0]
        assert "-m" in called_argv
        assert "synth_setter.cli.eval" in called_argv
        assert "experiment=surge/fake_oracle" in called_argv
        # dataset_root and run_dir are distinct: split virtual datasets are
        # read in place beside their shards; eval outputs land in run_dir.
        assert f"datamodule.dataset_root={dataset_root}" in called_argv
        assert f"hydra.run.dir={run_dir}" in called_argv
        assert dataset_root != run_dir
        assert "ckpt_path=null" in called_argv
        # The eval resumes the generate run rather than opening a fresh one, so
        # its audio/* metrics share the run id (logger=null crashed Hydra — see #1331).
        assert "logger=wandb" in called_argv
        # id exists in logger/wandb.yaml (plain override); resume is absent (+append).
        assert "logger.wandb.id=some-run-id" in called_argv
        assert "+logger.wandb.resume=must" in called_argv
        # render_vst=true re-renders predicted params; surge_simple supplies the
        # group structure, while every render field predict_vst_audio renders with
        # is overridden from the generation RenderConfig so the re-render matches it.
        assert "render=surge_simple" in called_argv
        assert "render.param_spec_name=surge_xt" in called_argv
        assert "render.plugin_state_path=presets/surge-base.vstpreset" in called_argv
        assert "render.plugin_path=plugins/Surge XT.vst3" in called_argv
        assert f"render.sample_rate={render.sample_rate}" in called_argv
        assert f"render.channels={render.channels}" in called_argv
        assert f"render.velocity={render.velocity}" in called_argv
        assert f"render.signal_duration_seconds={render.signal_duration_seconds}" in called_argv
        # batch_size=1 keeps the smoke-sized test split (4 samples) from
        # flooring to zero batches under the 128 default — see #1331.
        assert "datamodule.batch_size=1" in called_argv
        # Sentinel 7 (no config default) proves the value is forwarded, not hardcoded.
        assert "datamodule.num_workers=7" in called_argv
        assert "mode=predict" in called_argv
        # predict_file routes the datamodule to this split's Lance dataset.
        assert f"datamodule.predict_file={predict_file}" in called_argv
        # Default (test split) carries no prefix override: its keys stay bare
        # ``audio/*`` so existing sweeps/dashboards keep working.
        assert not any(a.startswith("+evaluation.metric_prefix=") for a in called_argv)

    def test_run_oracle_eval_subprocess_metric_prefix_adds_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """A non-empty ``metric_prefix`` appends the eval override that namespaces audio keys.

        The override routes through to ``cfg.evaluation.metric_prefix`` in the
        eval subprocess, which prepends it to every ``audio/*`` key so the
        per-split passes don't overwrite each other on the shared wandb run.

        :param monkeypatch: Patches the module's ``_check_call_streamed``.
        :param tmp_path: Roots the dataset-root and eval run dirs.
        :param spec: Source of a valid ``RenderConfig``.
        """
        import synth_setter.cli.generate_dataset as gd

        streamed_call_mock = MagicMock()
        monkeypatch.setattr(gd, "_check_call_streamed", streamed_call_mock)

        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for name in ("val.lance", "test.lance", "stats.npz"):
            (dataset_root / name).touch()
        predict_file = dataset_root / "train.lance"
        _write_lance_split(predict_file, 4)
        gd._run_oracle_eval_subprocess(
            dataset_root,
            tmp_path / "oracle_eval" / "train" / "some-run-id",
            "some-run-id",
            render=spec.render,
            num_workers=0,
            predict_file=predict_file,
            metric_prefix="train/",
        )

        called_argv = streamed_call_mock.call_args[0][0]
        # ``+`` appends the key: it is absent from eval.yaml's evaluation group.
        assert "+evaluation.metric_prefix=train/" in called_argv

    def test_run_oracle_eval_subprocess_timeout_grows_with_split_sample_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """A larger predict split yields a strictly larger eval timeout.

        Pins the scaling itself (not just one formula point): a flat ceiling
        would return the same budget for both splits and fail this.

        :param monkeypatch: Patches ``_check_call_streamed`` to capture the timeout.
        :param tmp_path: Roots the two distinct dataset roots.
        :param spec: Source of a valid ``RenderConfig``.
        """
        import synth_setter.cli.generate_dataset as gd

        def _timeout_for_split(root: Path, num_samples: int) -> float:
            root.mkdir()
            for name in ("train.lance", "val.lance", "stats.npz"):
                (root / name).touch()
            predict_file = root / "test.lance"
            _write_lance_split(predict_file, num_samples)
            mock = MagicMock()
            monkeypatch.setattr(gd, "_check_call_streamed", mock)
            gd._run_oracle_eval_subprocess(
                root,
                root / "run",
                "some-run-id",
                render=spec.render,
                num_workers=1,
                predict_file=predict_file,
            )
            return mock.call_args.kwargs["timeout"]

        small = _timeout_for_split(tmp_path / "small", 4)
        large = _timeout_for_split(tmp_path / "large", 40)
        assert large > small

    def test_run_oracle_eval_subprocess_missing_local_artifacts_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """Unpopulated ``dataset_root`` ⇒ clear ``FileNotFoundError``, no eval subprocess.

        ``finalize_from_spec`` short-circuits when R2 already holds the
        ``dataset.complete`` marker, leaving ``output_dir`` without the splits
        on a resume; the preflight turns the downstream low-signal Lance read
        error into an actionable one before shelling out.

        :param monkeypatch: Patches ``_check_call_streamed`` to assert it never fires.
        :param tmp_path: Empty stand-in for an unpopulated ``output_dir``.
        :param spec: Source of a valid ``RenderConfig`` for the call signature.
        """
        import synth_setter.cli.generate_dataset as gd

        streamed_call_mock = MagicMock()
        monkeypatch.setattr(gd, "_check_call_streamed", streamed_call_mock)

        with pytest.raises(FileNotFoundError, match=r"test\.lance"):
            gd._run_oracle_eval_subprocess(
                tmp_path,
                tmp_path / "oracle_eval" / "test" / "rid",
                "rid",
                render=spec.render,
                num_workers=0,
                predict_file=tmp_path / "test.lance",
            )

        streamed_call_mock.assert_not_called()

    def test_run_oracle_eval_subprocess_missing_predict_file_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """Non-existent ``predict_file`` ⇒ ``FileNotFoundError`` before subprocess.

        All required artifacts are present in ``dataset_root`` so the existing
        preflight passes; the ``predict_file``-specific check then catches the
        absent path before shelling out.

        :param monkeypatch: Patches ``_check_call_streamed`` to assert it never fires.
        :param tmp_path: Roots the dataset dir and a missing predict path.
        :param spec: Source of a valid ``RenderConfig`` for the call signature.
        """
        import synth_setter.cli.generate_dataset as gd

        streamed_call_mock = MagicMock()
        monkeypatch.setattr(gd, "_check_call_streamed", streamed_call_mock)

        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for name in ("train.lance", "val.lance", "test.lance", "stats.npz"):
            (dataset_root / name).touch()

        with pytest.raises(FileNotFoundError, match=r"predict_file"):
            gd._run_oracle_eval_subprocess(
                dataset_root,
                tmp_path / "oracle_eval" / "test" / "rid",
                "rid",
                render=spec.render,
                num_workers=0,
                predict_file=tmp_path / "nonexistent_split.lance",
            )

        streamed_call_mock.assert_not_called()

    def test_main_oracle_eval_inline_default_false_skips(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opt-in invariant: default false ⇒ the eval subprocess never fires.

        :param monkeypatch: Patches argv + the three module-level seams.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        monkeypatch.setattr(gd, "finalize_from_spec", MagicMock())
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        _call_hydra_main(gd.main)

        oracle_mock.assert_not_called()

    def test_main_always_on_with_render_reload_skips_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``gui_toggle_cadence=always_on`` + ``plugin_reload_cadence=render`` is a no-op skip.

        The render arm of a cadence grid sweep hits this schema-invalid cell; ``main``
        logs a warning and returns before building the spec (which would otherwise raise
        in ``RenderConfig``), so the wandb trial completes instead of crashing.

        :param monkeypatch: Patches argv, the warning sink, and ``generate``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "render.plugin_reload_cadence=render",
            "render.gui_toggle_cadence=always_on",
        ]
        monkeypatch.setattr("sys.argv", argv)
        warn_mock = MagicMock()
        monkeypatch.setattr(gd.logger, "warning", warn_mock)
        generate_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)

        _call_hydra_main(gd.main)

        # Skipped before generation, with a warning (exact wording unpinned).
        generate_mock.assert_not_called()
        warn_mock.assert_called_once()

    def test_main_oracle_eval_inline_requires_finalize_inline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``oracle_eval_inline=true`` without ``finalize_inline=true`` raises pre-``generate()``.

        :param monkeypatch: Patches argv + the three seams the test asserts are unreached.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "oracle_eval_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        generate_mock = MagicMock()
        finalize_mock = MagicMock()
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        with pytest.raises(ValueError, match="requires finalize_inline=true"):
            _call_hydra_main(gd.main)
        generate_mock.assert_not_called()
        finalize_mock.assert_not_called()
        oracle_mock.assert_not_called()

    def test_main_oracle_eval_inline_rejects_zero_size_split(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fail-fast guard: ``oracle_eval_inline=true`` rejects ``[N, 0, 0]``-style sizes.

        ``VSTDataModule.setup()`` opens train/val/test ``.lance`` unconditionally
        regardless of stage, so any zero-size split would FileNotFoundError
        deep inside Lightning. The launcher catches the misconfig up front.

        :param monkeypatch: Patches argv and the ``generate`` / ``finalize_from_spec``
            / oracle-eval seams; the test asserts none of them is reached.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            # smoke-shard now defaults to [4, 4, 4]; pin a zero split to exercise the guard.
            "train_val_test_sizes=[4,0,0]",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        generate_mock = MagicMock()
        finalize_mock = MagicMock()
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        with pytest.raises(ValueError, match="train_val_test_sizes > 0"):
            _call_hydra_main(gd.main)
        generate_mock.assert_not_called()
        finalize_mock.assert_not_called()
        oracle_mock.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_main_oracle_eval_inline_ignored_in_dispatch_branch(
        self,
        mock_logger: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Dispatch branch: ``oracle_eval_inline=true`` is logged-and-ignored, not raised.

        SkyPilot hands the run to a worker pod; oracle eval runs out-of-band
        via its own workflow. Asserts no eval subprocess fires and the INFO
        log mentions the override was ignored.

        :param mock_logger: Patched ``generate_dataset.logger`` — the
            established loguru capture pattern in this file.
        :param monkeypatch: Patches argv + dispatch + the oracle-eval seam.
        :param tmp_path: Holds the minimal compute template the dispatch
            branch reads from disk.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
            "oracle_eval_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)
        monkeypatch.setattr(
            gd,
            "generate",
            lambda *_a, **_k: pytest.fail("generate must not fire on dispatch branch"),
        )
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        _call_hydra_main(gd.main)

        oracle_mock.assert_not_called()
        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        ignored_lines = [
            m for m in info_messages if "oracle_eval_inline=True" in m and "ignored" in m
        ]
        assert len(ignored_lines) == 1, (
            f"expected exactly one INFO log mentioning 'oracle_eval_inline=True' + 'ignored'; "
            f"got messages: {info_messages!r}"
        )


class TestMainSpecPersistence:
    """``main()`` writes the local spec, loads R2 env, uploads the canonical spec on every path.

    The R2 upload is launcher-side and happens once per ``main()`` invocation:
    after the local write, before the local-run / dispatch branch is taken.
    Workers in the dispatch path no longer re-upload the spec (the worker's
    ``generate(spec, work_dir, loggers)`` writes shards only); the canonical R2 object
    exists before any worker boots.
    """

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pin single-worker rank/world + isolate Hydra's per-run dir to ``tmp_path``.

        ``@hydra.main`` resolves ``${paths.log_dir}`` from ``${oc.env:PROJECT_ROOT}``;
        redirecting PROJECT_ROOT keeps the per-run dir under the test tree.

        :param monkeypatch: Pytest fixture used to set env vars.
        :param tmp_path: Per-test tmp dir hosting PROJECT_ROOT.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    @pytest.fixture(autouse=True)
    def _stub_run_and_spec_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``generate()``, the spec_io helpers, and ``r2_io.ensure_r2_env_loaded``.

        Tests assert via the module-level mocks ``gd.write_spec_locally``,
        ``gd.upload_spec``, and ``gd.r2_io.ensure_r2_env_loaded`` to keep test
        signatures stable for pydoclint.

        :param monkeypatch: Pytest fixture used to patch module-level callables.
        """
        import synth_setter.cli.generate_dataset as gd

        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        monkeypatch.setattr(
            gd,
            "write_spec_locally",
            MagicMock(side_effect=lambda spec, out: Path(out) / "input_spec.json"),
        )
        monkeypatch.setattr(
            gd,
            "upload_spec",
            MagicMock(return_value="r2://stub-bucket/stub-key/input_spec.json"),
        )
        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", MagicMock(return_value=None))

    @staticmethod
    def _dispatch_argv(template_path: Path) -> list[str]:
        """Build argv that triggers the dispatch branch of ``main()``.

        :param template_path: Path to a minimal SkyPilot compute template the
            ``skypilot_launch`` cfg loader will accept.
        :return: ``sys.argv`` overrides setting ``compute_template``.
        """
        return [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template_path}",
        ]

    @staticmethod
    def _write_minimal_template(tmp_path: Path) -> Path:
        """Write the bare-minimum compute template YAML the loader accepts.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :return: Path to the written template.
        """
        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        return template

    def test_main_writes_local_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``main()`` calls ``write_spec_locally`` with ``Path(cfg.paths.output_dir)``.

        Pinned by cross-reference: ``main()`` passes the same value to both
        ``write_spec_locally`` and ``generate()`` (the local-run shard
        scratch dir), so equality with the captured ``generate`` arg
        anchors the source without hard-coding the timestamped Hydra dir.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd

        generate_mock = MagicMock(return_value=None)
        monkeypatch.setattr(gd, "generate", generate_mock)

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        _call_hydra_main(gd.main)

        gd.write_spec_locally.assert_called_once()  # type: ignore[attr-defined]
        called_spec, called_out = gd.write_spec_locally.call_args[0]  # type: ignore[attr-defined]
        assert isinstance(called_spec, DatasetSpec)
        assert isinstance(called_out, Path)
        generate_mock.assert_called_once()
        _, generate_work_dir, _ = generate_mock.call_args[0]
        assert called_out == generate_work_dir

    def test_local_run_uploads_spec_from_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local-run branch uploads the spec from ``main()`` exactly once.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        _call_hydra_main(gd.main)

        gd.upload_spec.assert_called_once()  # type: ignore[attr-defined]
        called_spec = gd.upload_spec.call_args[0][0]  # type: ignore[attr-defined]
        assert isinstance(called_spec, DatasetSpec)

    def test_dispatch_branch_uploads_spec_from_main(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Dispatch branch also uploads from ``main()`` — worker no longer re-uploads.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)

        _call_hydra_main(gd.main)

        gd.upload_spec.assert_called_once()  # type: ignore[attr-defined]
        gd.write_spec_locally.assert_called_once()  # type: ignore[attr-defined]

    def test_main_uploads_spec_with_projected_rclone_env_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``upload_spec`` sees projected rclone env after ``ensure_r2_env_loaded``.

        Asserts the observable invariant — backend env is present in process
        env when the upload fires — rather than the internal call order. A
        benign re-ordering that still loads creds before uploading passes; a
        regression that uploads before ``ensure_r2_env_loaded`` populates the
        env fails because the stub records an absent key.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd
        from synth_setter.pipeline.schemas.object_storage import RCLONE_REQUIRED_ENV_KEYS

        probe_key = RCLONE_REQUIRED_ENV_KEYS[0]
        monkeypatch.delenv(probe_key, raising=False)

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        def _load_creds(*_a: object, **_k: object) -> None:
            # setenv (not raw os.environ) so monkeypatch restores it on teardown.
            monkeypatch.setenv(probe_key, "stub-access-key-id")

        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", _load_creds)

        creds_present_at_upload: dict[str, bool] = {}

        def _record_env(*_a: object, **_k: object) -> str:
            creds_present_at_upload["present"] = probe_key in os.environ
            return "r2://stub-bucket/stub-key/input_spec.json"

        monkeypatch.setattr("synth_setter.cli.generate_dataset.upload_spec", _record_env)

        _call_hydra_main(gd.main)

        assert creds_present_at_upload.get("present") is True

    def test_dispatch_branch_passes_canonical_spec_uri_via_extra_envs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``main()`` forwards ``spec.r2.input_spec_uri()`` via ``sky_cfg.extra_envs``.

        The canonical spec URI (with run prefix) lands in
        ``sky_cfg.extra_envs[WORKER_SPEC_URI_ENV]`` so each rank reads the same
        R2 object ``main()`` just uploaded.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl
        from synth_setter.pipeline.constants import WORKER_SPEC_URI_ENV

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))

        recorded: dict[str, object] = {}

        def _fake_dispatch(sky_cfg: object) -> None:
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        _call_hydra_main(gd.main)

        sky_cfg = recorded["sky_cfg"]
        spec = gd.write_spec_locally.call_args[0][0]  # type: ignore[attr-defined]
        assert isinstance(spec, DatasetSpec)
        assert sky_cfg.extra_envs[WORKER_SPEC_URI_ENV] == spec.r2.input_spec_uri()  # type: ignore[attr-defined]

    def test_main_does_not_emit_spec_uri_sentinel(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``main()`` must not print the ``::synth-setter-spec-uri::`` marker on stdout.

        CI derives the URI via ``synth-setter-spec-uri`` (Hydra-compose) — see #1154.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)

        _call_hydra_main(gd.main)

        assert "::synth-setter-spec-uri::" not in capsys.readouterr().out

    def test_generate_dataset_pins_smoke_job_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``main()`` pins the dataset-specific job-name stem before dispatching.

        The launcher is domain-neutral; the dataset-specific
        ``synth-setter-smoke-<task[:8]>`` stem lives on the caller side so the
        worker job name still encodes the task.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))

        recorded: dict[str, object] = {}

        def _fake_dispatch(sky_cfg: object) -> None:
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        _call_hydra_main(gd.main)

        sky_cfg = recorded["sky_cfg"]
        spec_call = gd.write_spec_locally.call_args[0][0]  # type: ignore[attr-defined]
        assert sky_cfg.job_name == gd._smoke_job_name(spec_call)  # type: ignore[attr-defined]


class TestMainHydraOutputDir:
    """``cfg.paths.output_dir`` resolves to Hydra's per-run dir inside ``main()``.

    Pins the @hydra.main decoration contract for the launcher entrypoint.
    """

    @pytest.fixture(autouse=True)
    def _isolate_hydra_output_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Redirect PROJECT_ROOT to tmp so Hydra writes the per-run dir under the test tree.

        :param monkeypatch: Pytest fixture used to override env vars.
        :param tmp_path: Per-test tmp dir hosting the synthetic checkout root.
        """
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    @pytest.fixture(autouse=True)
    def _stub_run_and_spec_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the launcher's R2 + dispatch surface so main() runs without I/O.

        :param monkeypatch: Pytest fixture used to patch module-level callables.
        """
        import synth_setter.cli.generate_dataset as gd

        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        monkeypatch.setattr(
            gd,
            "write_spec_locally",
            MagicMock(side_effect=lambda spec, out: Path(out) / "input_spec.json"),
        )
        monkeypatch.setattr(
            gd,
            "upload_spec",
            MagicMock(return_value="r2://stub-bucket/stub-key/input_spec.json"),
        )
        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", MagicMock(return_value=None))

    def test_main_resolves_output_dir_under_hydra_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside main(), cfg.paths.output_dir equals HydraConfig.get().runtime.output_dir.

        Pins the @hydra.main decoration contract: the per-run dir is supplied by
        Hydra runtime rather than pinned by the launcher to a hand-picked anchor.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + capture cfg.
        """
        from hydra.core.hydra_config import HydraConfig

        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        observed: dict[str, str] = {}
        real_spec_from_cfg = gd.spec_from_cfg

        def _capture_then_build(cfg: object) -> DatasetSpec:
            observed["output_dir"] = cfg.paths.output_dir  # type: ignore[attr-defined]
            observed["runtime_output_dir"] = HydraConfig.get().runtime.output_dir
            return real_spec_from_cfg(cfg)  # type: ignore[arg-type]

        monkeypatch.setattr(gd, "spec_from_cfg", _capture_then_build)

        _call_hydra_main(gd.main)

        assert observed["output_dir"] == observed["runtime_output_dir"]


def test_smoke_job_name_rejects_unsafe_task_name() -> None:
    """``_smoke_job_name`` raises with a task-name-aware diagnostic on malformed task_name.

    Pins the dataset-aware error message that the launcher's
    ``_JOB_NAME_RE`` validator would otherwise surface without spec context.
    """
    from synth_setter.cli.generate_dataset import _smoke_job_name

    bad_spec = SimpleNamespace(task_name="bad.task.name")
    with pytest.raises(ValueError, match=r"fix spec.task_name or pin"):
        _smoke_job_name(bad_spec)  # type: ignore[arg-type]


def test_worker_id_sanitizes_hostname_for_object_key_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characters outside ``[A-Za-z0-9._-]`` in the hostname become ``-``.

    :param monkeypatch: Pins ``platform.node`` to a hostile hostname.
    """
    from synth_setter.cli import generate_dataset

    monkeypatch.setattr(generate_dataset.platform, "node", lambda: "pod@host:1/x")

    assert generate_dataset._worker_id() == "pod-host-1-x"


def test_worker_id_empty_hostname_falls_back_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``platform.node()`` yields the ``worker`` fallback, never ``""``.

    :param monkeypatch: Pins ``platform.node`` to return an empty string.
    """
    from synth_setter.cli import generate_dataset

    monkeypatch.setattr(generate_dataset.platform, "node", lambda: "")

    assert generate_dataset._worker_id() == "worker"
