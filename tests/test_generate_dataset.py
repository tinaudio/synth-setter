"""Tests for the ``synth-setter-generate-dataset`` CLI entrypoint.

Covers tests that exercise ``from_hydra`` or the CLI subprocess end-to-end:
an ``integration_r2``-gated end-to-end render that drives ``from_hydra``
against ``cfg_dataset`` and asserts every shard lands at the spec-derived R2
URI in real Cloudflare R2; an ``integration_r2`` subprocess run of the
``smoke-shard-with-oracle-eval`` experiment that asserts the inline oracle
eval's per-split ``metrics.json`` holds bounded audio metrics under the bare
``audio/*`` key for ``test`` and the namespaced ``<split>/audio/*`` key for
``train``/``val``; and a variant with ``param_sample_cadence=shard`` that
asserts the ``shuffled_audio/*`` group also appears (under the same per-split
prefix) when all sample dirs share uniform ``params.csv`` (#489); and an
``integration_r2`` multi-process contention run over the Lance shard-claims
table in real R2 that proves each claim generation is granted exactly once
under R2's conditional-put commit protocol. The integration tests auto-skip
when ``rclone`` / R2 creds are absent.

Keep this module to tests that drive ``from_hydra`` or the real CLI subprocess.
Config-composition and ``spec_from_cfg`` unit tests live in
``tests/pipeline/configs/``; direct-call unit tests for ``generate`` / ``main``
and the arg-builders live in
``tests/pipeline/entrypoints/test_generate_dataset_unit.py``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import lance
import numpy as np
import pytest
from omegaconf import DictConfig, open_dict

from synth_setter.cli.finalize_dataset import finalize_lance
from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2
from synth_setter.pipeline.data.lance_staging import shard_has_complete_attempt
from synth_setter.pipeline.schemas.skypilot_launch import SkypilotLaunchConfig
from synth_setter.pipeline.schemas.spec import DatasetSpec, Split
from tests._vst import (
    PLUGIN_PATH,
)
from tests.evaluation._oracle_helpers import ORACLE_AUDIO_METRIC_BOUNDS
from tests.helpers.dummy_shards import stub_renderer

# The predict-mode oracle eval (surge/fake_oracle) dumps one mean+std per audio
# metric; predict leaves ``trainer.callback_metrics`` empty, so these are the
# only keys in ``metrics.json`` (see ``synth_setter.evaluation.compute_audio_metrics``).
_ORACLE_AUDIO_METRICS = ("mss", "wmfcc", "sot", "rms")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_PLUGIN_VST3 = (
    Path(PLUGIN_PATH) if Path(PLUGIN_PATH).is_absolute() else _REPO_ROOT / PLUGIN_PATH
).resolve()

# Moduleinfo-only VST3 bundle: extract_renderer_version reads its
# Contents/moduleinfo.json and returns the pinned version without loading any
# .so, so generate()'s renderer-version guard passes with no real plugin.
_TEST_PLUGIN_VST3 = Path(__file__).resolve().parent / "pipeline" / "fixtures" / "TestPlugin.vst3"
_TEST_PLUGIN_VERSION = "1.0.0-test"


def test_cfg_dataset_composes_and_validates_as_dataset_spec(
    cfg_dataset: DictConfig,
) -> None:
    """The new fixture composes ``dataset.yaml`` and round-trips through ``DatasetSpec``.

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    spec = spec_from_cfg(cfg_dataset)
    assert isinstance(spec, DatasetSpec)
    assert spec.num_shards >= 1
    assert spec.render.samples_per_shard >= 1


def test_cfg_dataset_dawdreamer_error_precedes_darwin_guard(
    cfg_dataset: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generation config reports DawDreamer's backend constraint on Darwin.

    :param cfg_dataset: Function-scoped dataset generation configuration.
    :param monkeypatch: Stubs platform detection to exercise the overlapping guards.
    """
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._current_platform", lambda: "darwin")
    with open_dict(cfg_dataset):
        cfg_dataset.render.renderer_backend = "dawdreamer"
        cfg_dataset.render.gui_toggle_cadence = "render"

    with pytest.raises(ValueError, match='DawDreamer requires gui_toggle_cadence="never"'):
        spec_from_cfg(cfg_dataset)


def test_cfg_dataset_render_obxf_resolves_param_spec_through_spec_from_cfg(
    cfg_dataset_obxf: DictConfig,
) -> None:
    """``render=obxf`` resolves its registered spec through the ``spec_from_cfg`` entrypoint path.

    ``num_params`` is ``len(param_specs[param_spec_name])`` — the registry lookup
    the shard writer makes — so a resolving width proves the entrypoint reaches the
    OB-Xf spec without a ``KeyError`` (P31 e2e gate for the new ``render`` group).

    :param cfg_dataset_obxf: Function-scoped fixture composing ``dataset.yaml`` with
        the smoke-shard experiment, ``render=obxf``, and ``tmp_path``-pinned paths.
    """
    spec = spec_from_cfg(cfg_dataset_obxf)
    assert spec.render.param_spec_name == "obxf"
    assert spec.num_params == 187


@pytest.mark.fake_vst
def test_from_hydra_renders_every_shard_to_fake_r2_then_resume_skips(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hydra`` renders every owned shard to a fake R2, then skips them on resume.

    Drives the real worker entrypoint end-to-end on the CPU-fast loop: the Surge
    VST3 subprocess is replaced by ``stub_renderer`` (writes a deterministic
    validation-shaped Lance shard) and ``r2:`` is a local-filesystem rclone remote
    via ``fake_r2_remote``, so the partition → render → stage → skip-existing probe
    loop (#750) runs with no real plugin and no real R2. Asserts (1) ``smoke-shard``
    partitions into one shard per split, (2) every ``.lance`` shard stages a
    complete attempt (sidecar + stats + ``.valid``) with its fragment data under
    the assigned split dataset (#1776), and (3) a second ``from_hydra`` pass
    renders nothing because the probe finds all shards already staged.

    :param cfg_dataset: Hydra cfg composed with ``generate_dataset/smoke-shard``
        and ``tmp_path``-pinned paths (the same ``tmp_path`` ``fake_r2_remote``
        backs ``r2:`` against).
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote.
    :param monkeypatch: Pins the single-worker rank/world env contract and the
        moduleinfo-only plugin so the renderer-version guard passes.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        # Pin r2.prefix so the spec built here for assertions and the one
        # from_hydra rebuilds internally derive the same shard URIs — an unpinned
        # created_at would fire its default factory twice and diverge the run_id.
        cfg_dataset.r2.prefix = "fake-r2/test-run/"
        # Disable the default wandb logger: generate() would call wandb.init() and block.
        cfg_dataset.logger = None

    spec = spec_from_cfg(cfg_dataset)
    # smoke-shard partitions into one shard per split, so the stub covers train→val→test.
    assert spec.split_shard_ranges == {"train": (0, 1), "val": (1, 2), "test": (2, 3)}

    render_shard = stub_renderer(spec)
    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=render_shard,
    ):
        from_hydra(cfg_dataset)

    # fake_r2_remote materializes r2://<bucket>/<key> at <root>/<bucket>/<key>.
    run_root = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
    split_of = {
        shard_id: split
        for split, (lo, hi) in spec.split_shard_ranges.items()
        for shard_id in range(lo, hi)
    }
    for shard in spec.shards:
        assert shard.filename.endswith(".lance")
        # Require one shared attempt identity; independent suffix matches can combine partial attempts — see #1776.
        staging = run_root / "metadata" / "workers" / "shards" / f"shard-{shard.shard_id:06d}"
        staged_names = [p.name for p in staging.iterdir()]
        bases_by_suffix = {
            suffix: {n.removesuffix(suffix) for n in staged_names if n.endswith(suffix)}
            for suffix in (".fragment.json", ".shard-stats.npz", ".valid", ".rendering")
        }
        shared_bases = set.intersection(*bases_by_suffix.values())
        assert len(shared_bases) == 1, (
            f"expected one complete staged attempt for {shard.filename}, "
            f"got files: {sorted(staged_names)}"
        )
        split_data = run_root / f"{split_of[shard.shard_id]}.lance" / "data"
        assert list(split_data.glob("*.lance")), (
            f"no fragment data under {split_data} for {shard.filename}"
        )

    renderer_invocations = 0

    def _count_renderer(args: list[str]) -> None:
        nonlocal renderer_invocations
        if not (args and args[0] == "rclone"):
            renderer_invocations += 1
        render_shard(args)

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=_count_renderer,
    ):
        from_hydra(cfg_dataset)
    assert renderer_invocations == 0, (
        f"resume re-rendered {renderer_invocations} shard(s) already present in R2"
    )


@pytest.mark.fake_vst
def test_from_hydra_claims_mode_renders_claimed_shards_and_completes_all(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
) -> None:
    """``from_hydra`` with ``use_shard_queue=true`` claims, renders, and completes every shard.

    Drives the real worker entrypoint end-to-end like its static-partition
    sibling above, but with the run's Lance shard-claims table as the shard
    source. Nothing is patched around the claims path: the worker resolves the
    table through the real ``lance_target`` to the sanctioned
    ``metadata/shard-claims.lance`` key inside the fake R2 root, exactly as it
    does against real R2. Asserts every claimed shard lands at its
    spec-derived R2 URI and every claim row ends ``done``.

    :param cfg_dataset: Hydra cfg composed with ``generate_dataset/smoke-shard``
        and ``tmp_path``-pinned paths.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote
        (also where ``lance_target`` resolves the claims table).
    """
    from synth_setter.pipeline.r2_io import lance_target
    from synth_setter.pipeline.shard_claims import ShardClaims

    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.use_shard_queue = True
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        cfg_dataset.r2.prefix = "fake-r2/test-run/"
        # Disable the default wandb logger: generate() would call wandb.init() and block.
        cfg_dataset.logger = None

    spec = spec_from_cfg(cfg_dataset)
    claims = ShardClaims.for_run(*lance_target(spec.r2.shard_claims_uri()))
    claims.populate([shard.shard_id for shard in spec.shards])

    render_shard = stub_renderer(spec)
    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=render_shard,
    ):
        from_hydra(cfg_dataset)

    table_dir = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "metadata/shard-claims.lance"
    assert table_dir.is_dir(), "claims table must live at its sanctioned key in the R2 layout"
    for shard in spec.shards:
        assert shard_has_complete_attempt(spec, shard.shard_id)
    assert claims.claim() is None, "no claimable rows may remain after the run"
    assert claims.status_counts() == {"done": spec.num_shards}


@pytest.mark.fake_vst
def test_from_hydra_claims_mode_crashed_claim_rerenders_only_after_lease_lapse(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,  # noqa: ARG001 — activates the local-typed remote
) -> None:
    """A crashed claim is shielded while its lease lives, then recovered by relaunch.

    Full crash-recovery loop through the real worker entrypoint, no claims
    seam patched: run 1's renderer fails and the run dies with the claim
    held; run 2 inside the lease window renders nothing (a poison shard
    cannot cascade across relaunches); after the lease is aged out, run 3
    re-claims at the next generation, renders, and completes the shard.

    :param cfg_dataset: Hydra cfg composed with ``generate_dataset/smoke-shard``
        and ``tmp_path``-pinned paths.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote
        (fixture-activation only — referenced via the ARG001 noqa).
    """
    from synth_setter.pipeline.r2_io import lance_target
    from synth_setter.pipeline.shard_claims import ShardClaims

    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.use_shard_queue = True
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        cfg_dataset.r2.prefix = "fake-r2/crash-run/"
        cfg_dataset.logger = None

    spec = spec_from_cfg(cfg_dataset)
    claims = ShardClaims.for_run(*lance_target(spec.r2.shard_claims_uri()))
    claims.populate([0])
    render_shard = stub_renderer(spec)

    def _crash_renderer(args: list[str]) -> None:
        if args and args[0] == "rclone":
            render_shard(args)
            return
        raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=_crash_renderer,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            from_hydra(cfg_dataset)
    assert claims.status_counts() == {"claimed": 1}

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=render_shard,
    ) as inside_lease:
        from_hydra(cfg_dataset)
    renderer_calls = [c for c in inside_lease.call_args_list if c.args[0][0] != "rclone"]
    assert renderer_calls == [], "a live lease must shield the crashed claim from relaunch"
    assert not shard_has_complete_attempt(spec, 0)

    # Age the crashed claim's lease in place of waiting out the real 2 hours.
    lance.dataset(claims.uri).update({"lease_expiry_s": "0"}, where="status = 'claimed'")

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=render_shard,
    ):
        from_hydra(cfg_dataset)
    assert shard_has_complete_attempt(spec, 0)
    assert claims.status_counts() == {"done": 1}
    row = lance.dataset(claims.uri).to_table().to_pylist()[0]
    assert row["claim_gen"] == 2, "recovery must advance the fencing generation"
    assert row["attempts"] == 2


@pytest.mark.fake_vst
def test_from_hydra_lance_render_failing_local_validation_never_stages_a_valid_marker(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt render fails loudly before staging — no ``.valid`` ever lands (#1776).

    Drives the real worker entrypoint with a stub renderer that writes a shard
    holding double the spec's rows. Worker-side validation must reject it, the
    run must fail, and the staging directory must hold no ``.valid`` marker or
    sidecar (the shard stays "missing" for the next reconciliation pass).

    :param cfg_dataset: Hydra cfg composed with the smoke-shard dataset.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote.
    :param monkeypatch: Pins the single-worker rank/world env and the
        moduleinfo-only plugin so the renderer-version guard passes.
    """
    from synth_setter.pipeline.data.lance_shard import (
        lance_schema,
        record_batch_from_arrays,
        write_lance_dataset,
    )
    from tests.helpers.subprocess_args import find_script_index

    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        cfg_dataset.r2.prefix = "fake-r2/invalid-run/"
        cfg_dataset.logger = None
    spec = spec_from_cfg(cfg_dataset)

    def _render_oversized_shard(args: list[str]) -> None:
        from synth_setter.data.vst.shapes import (
            DATASET_FIELD_DTYPES,
            dataset_field_shapes,
        )

        output_file = Path(args[find_script_index(args) + 1])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        oversized = spec.render.model_copy(
            update={"samples_per_shard": spec.render.samples_per_shard * 2}
        )
        shapes = dataset_field_shapes(oversized, spec.num_params)
        schema = lance_schema(shapes, oversized.shard_metadata())
        arrays = {
            field: np.zeros(shape, dtype=DATASET_FIELD_DTYPES[field])
            for field, shape in shapes.items()
        }
        write_lance_dataset(output_file, schema, [record_batch_from_arrays(arrays, schema)])

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=_render_oversized_shard,
    ):
        with pytest.raises(RuntimeError, match="failed local validation"):
            from_hydra(cfg_dataset)

    staging_root = (
        fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "metadata" / "workers" / "shards"
    )
    staged = [p.name for p in staging_root.rglob("*") if p.is_file()]
    assert not [name for name in staged if name.endswith((".valid", ".fragment.json"))], (
        f"invalid render must not stage a complete attempt, found: {staged}"
    )
    # The attempt-start marker is the only allowed trace of the failed attempt.
    assert staged
    assert all(name.endswith(".rendering") for name in staged)


@pytest.mark.requires_vst
@pytest.mark.slow
def test_from_hydra_claims_mode_real_vst_writes_consumable_shard(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    tmp_path: Path,
) -> None:
    """Claims mode stages one real VST Lance shard that validates from fake R2.

    :param cfg_dataset: Hydra dataset config reduced to one sample and shard.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote
        (also where ``lance_target`` resolves the claims table).
    :param tmp_path: Scratch directory holding Hydra's worktree-relative links.
    """
    from synth_setter.pipeline.r2_io import lance_target
    from synth_setter.pipeline.shard_claims import ShardClaims

    (tmp_path / "src").symlink_to(_REPO_ROOT / "src", target_is_directory=True)
    (tmp_path / "presets").symlink_to(_REPO_ROOT / "presets", target_is_directory=True)
    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.train_val_test_sizes = [1, 0, 0]
        cfg_dataset.use_shard_queue = True
        cfg_dataset.render.plugin_path = str(_REAL_PLUGIN_VST3)
        cfg_dataset.render.samples_per_render_batch = 1
        cfg_dataset.render.samples_per_shard = 1
        cfg_dataset.r2.prefix = "fake-r2/real-vst-claims/"
        cfg_dataset.logger = None

    spec = spec_from_cfg(cfg_dataset)
    claims = ShardClaims.for_run(*lance_target(spec.r2.shard_claims_uri()))
    claims.populate(shard.shard_id for shard in spec.shards)

    from_hydra(cfg_dataset)

    staging_root = (
        fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "metadata" / "workers" / "shards"
    )
    assert len(list(staging_root.rglob("*.valid"))) == 1
    assert validate_all_shards_from_r2(spec) == []
    assert claims.claim() is None
    assert claims.status_counts() == {"done": spec.num_shards}


@pytest.mark.requires_vst
@pytest.mark.slow
def test_from_hydra_real_vst_lance_render_stages_then_resume_skips(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real VST Lance render stages complete attempts that resume skips.

    :param cfg_dataset: Hydra cfg composed with the smoke-shard dataset.
    :param fake_r2_remote: Local-filesystem root backing the real rclone process.
    :param monkeypatch: Pins the single-worker rank and world size.
    :param tmp_path: Scratch directory for finalize sidecars and statistics.
    """
    (tmp_path / "src").symlink_to(_REPO_ROOT / "src", target_is_directory=True)
    (tmp_path / "presets").symlink_to(_REPO_ROOT / "presets", target_is_directory=True)
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.train_val_test_sizes = [1, 1, 1]
        cfg_dataset.render.plugin_path = str(_REAL_PLUGIN_VST3)
        cfg_dataset.render.samples_per_render_batch = 1
        cfg_dataset.render.samples_per_shard = 1
        cfg_dataset.r2.prefix = "fake-r2/real-vst-lance-run/"
        cfg_dataset.logger = None
    spec = spec_from_cfg(cfg_dataset)

    from_hydra(cfg_dataset)

    staging_root = (
        fake_r2_remote / spec.r2.bucket / spec.r2.prefix / "metadata" / "workers" / "shards"
    )
    first_attempts = sorted(path.name for path in staging_root.rglob("*.valid"))
    assert len(first_attempts) == len(spec.shards)

    from_hydra(cfg_dataset)

    resumed_attempts = sorted(path.name for path in staging_root.rglob("*.valid"))
    assert resumed_attempts == first_attempts

    assert validate_all_shards_from_r2(spec) == []


def test_from_hydra_passes_per_shard_base_seed_to_renderer(
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each shard's render subprocess carries its own ``--base_seed`` end-to-end (#884).

    Drives the real worker entrypoint; the renderer subprocess is stubbed but its
    composed argv is captured, pinning that ``build_generate_args`` injected
    ``ShardSpec.seed`` per shard through the full ``from_hydra`` path — the behavior
    the argv-shape unit test only asserts in isolation.

    :param cfg_dataset: Hydra cfg composed with the smoke-shard dataset.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote.
    :param monkeypatch: Pins the single-worker env and the moduleinfo-only plugin
        so the renderer-version guard passes.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "lance"
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        cfg_dataset.r2.prefix = "fake-r2/seed-run/"
        cfg_dataset.logger = None

    spec = spec_from_cfg(cfg_dataset)
    render_shard = stub_renderer(spec)
    captured: list[list[str]] = []

    def _capture(args: list[str]) -> None:
        if not (args and args[0] == "rclone"):
            captured.append(args)
        render_shard(args)

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=_capture,
    ):
        from_hydra(cfg_dataset)

    for shard in spec.shards:
        argv = next(a for a in captured if any(shard.filename in tok for tok in a))
        assert argv[argv.index("--base_seed") + 1] == str(shard.seed)


def test_from_hydra_dawdreamer_experiment_forwards_backend_and_uploads_shard(
    cfg_dataset_dawdreamer: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed DawDreamer smoke experiment reaches renderer argv and fake R2.

    :param cfg_dataset_dawdreamer: Hydra cfg composed from the DawDreamer smoke experiment.
    :param fake_r2_remote: Local-filesystem root backing the ``r2:`` remote.
    :param monkeypatch: Pins the worker contract and plugin metadata.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with open_dict(cfg_dataset_dawdreamer):
        cfg_dataset_dawdreamer.output_format = "lance"
        cfg_dataset_dawdreamer.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset_dawdreamer.render.renderer_version = _TEST_PLUGIN_VERSION
        cfg_dataset_dawdreamer.r2.prefix = "fake-r2/dawdreamer-run/"
        cfg_dataset_dawdreamer.logger = None

    spec = spec_from_cfg(cfg_dataset_dawdreamer)
    captured_renderer_argv: list[str] = []
    render_shard = stub_renderer(spec)

    def _capture(args: list[str]) -> None:
        if not (args and args[0] == "rclone"):
            captured_renderer_argv.extend(args)
        render_shard(args)

    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=_capture,
    ):
        from_hydra(cfg_dataset_dawdreamer)

    assert spec.render.renderer_backend == "dawdreamer"
    backend_index = captured_renderer_argv.index("--renderer_backend")
    assert captured_renderer_argv[backend_index + 1] == "dawdreamer"
    shard = spec.shards[0]
    # The rendered Lance shard stages a complete attempt (sidecar + stats + .valid).
    staging = (
        fake_r2_remote
        / spec.r2.bucket
        / spec.r2.prefix
        / "metadata"
        / "workers"
        / "shards"
        / f"shard-{shard.shard_id:06d}"
    )
    assert list(staging.glob("*.valid")), f"shard missing in fake R2: {shard.filename}"


def test_from_hydra_applies_extras_writing_tags_and_config_tree(
    cfg_dataset: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hydra`` runs ``extras(cfg)`` before rendering, materializing its artifacts.

    Drives the worker entrypoint end-to-end with ``generate`` stubbed so no VST/R2
    is needed. ``dataset.yaml`` composes ``extras: default`` (``enforce_tags`` +
    ``print_config`` true) and a non-empty ``tags``, so ``extras`` exports
    ``tags.log`` and ``config_tree.log`` to ``cfg.paths.output_dir``. Asserting both
    files exist and are non-empty verifies the entrypoint applied extras via its
    observable side effects rather than mocking the call.

    :param cfg_dataset: Composed dataset cfg; ``logger`` is nulled so stubbing
        ``generate`` does not leave a wandb run to instantiate.
    :param monkeypatch: Stubs ``generate`` to a no-op so only the ``extras`` side
        effects are exercised.
    """
    with open_dict(cfg_dataset):
        cfg_dataset.logger = None
    output_dir = Path(cfg_dataset.paths.output_dir)

    monkeypatch.setattr("synth_setter.cli.generate_dataset.generate", lambda *_a, **_k: None)

    from_hydra(cfg_dataset)

    for artifact in ("tags.log", "config_tree.log"):
        path = output_dir / artifact
        assert path.is_file(), f"extras did not write {artifact}"
        assert path.stat().st_size > 0, f"{artifact} is empty"


@pytest.mark.slow
def test_main_skips_schema_invalid_cadence_cell_without_failing(
    tmp_path: Path,
) -> None:
    """The CLI no-ops (exit 0) on a cadence grid cell ``RenderConfig`` rejects.

    ``gui_toggle_cadence=always_on`` is valid only with ``plugin_reload_cadence=once``,
    so the render arm of a cadence grid sweep hits an invalid cell. ``main`` logs a
    warning and returns instead of raising, which a wandb grid relies on to keep the
    trial from failing. Driven as a real subprocess so the whole entrypoint — Hydra
    compose, ``RenderConfig`` validation, and the skip guard — is exercised; the skip
    returns before any render, so no VST/R2 is needed.

    :param tmp_path: Hydra run dir, kept out of the repo tree.
    """
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell, trusted entrypoint
        [
            sys.executable,
            "-m",
            "synth_setter.cli.generate_dataset",
            "experiment=generate_dataset/smoke-shard",
            "render.plugin_reload_cadence=render",
            "render.gui_toggle_cadence=always_on",
            f"hydra.run.dir={tmp_path}",
        ],
        cwd=_REPO_ROOT,
        # Prepend this worktree's src so the subprocess imports the same
        # synth_setter / configs as the collected test, not a sibling editable install.
        env={
            **os.environ,
            "PROJECT_ROOT": str(_REPO_ROOT),
            "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
            "WANDB_MODE": "disabled",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "skipping run" in (result.stdout + result.stderr)


def _write_executable(path: Path, body: str) -> None:
    """Write a small executable used by the worker-command integration test.

    :param path: Executable path to create.
    :param body: Complete shell-script contents.
    """
    path.write_text(body)
    path.chmod(0o755)


def _make_stale_worker_checkout(tmp_path: Path) -> Path:
    """Create the pre-helper checkout shape found in an old worker image.

    :param tmp_path: Scratch root for the worker checkout.
    :returns: Worker checkout containing only the legacy sync script.
    """
    worker_root = tmp_path / "worker"
    scripts_dir = worker_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "sync_worker_checkout.sh").write_text(
        '#!/bin/bash\nset -euo pipefail\n[[ -z "${WORKER_GIT_REF:-}" ]] && exit 0\n'
    )
    return worker_root


def _make_fake_worker_runtime(tmp_path: Path, trace: Path) -> Path:
    """Create a worker runtime that delegates venv creation to the real ``uv``.

    :param tmp_path: Scratch root for the fake executable directory.
    :param trace: File receiving install and entrypoint invocations.
    :returns: Directory to prepend to ``PATH``.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    real_uv = shutil.which("uv")
    assert real_uv is not None
    _write_executable(
        fake_bin / "uv",
        "#!/bin/bash\n"
        f'if [[ "$1" == "venv" ]]; then exec {shlex.quote(real_uv)} "$@"; fi\n'
        f'printf "install:%s\\n" "$*" >> {shlex.quote(str(trace))}\n',
    )
    _write_executable(
        fake_bin / "synth-setter-generate-dataset-from-hydra",
        f'#!/bin/bash\nprintf "exec:%s\\n" "$*" >> {shlex.quote(str(trace))}\n',
    )
    return fake_bin


@pytest.fixture()
def remote_worker_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Path, Path, Path], tuple[subprocess.CompletedProcess[str], Path]]:
    """Build a launcher that executes its generated worker command locally.

    :param tmp_path: Scratch paths for the compute template and generated command.
    :param monkeypatch: Redirects storage and SkyPilot boundaries.
    :returns: Callable accepting worker root, worker venv, and fake runtime directory.
    """
    import synth_setter.cli.generate_dataset as generate_dataset_cli
    import synth_setter.pipeline.skypilot_launch as skypilot_launch

    def _dispatch(
        worker_root: Path, worker_venv: Path, fake_bin: Path
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        compute_template = tmp_path / "compute.yaml"
        compute_template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("VIRTUAL_ENV", str(worker_venv))
        monkeypatch.delenv("WORKER_GIT_REF", raising=False)
        monkeypatch.setattr(generate_dataset_cli, "_WORKER_REPO_ROOT", str(worker_root))
        monkeypatch.setattr(generate_dataset_cli, "_WORKER_VENV", str(worker_venv))
        monkeypatch.setattr(
            generate_dataset_cli,
            "write_spec_locally",
            lambda _spec, output_dir: Path(output_dir) / "input_spec.json",
        )
        monkeypatch.setattr(
            generate_dataset_cli,
            "upload_spec",
            lambda _spec: "r2://test/input_spec.json",
        )
        monkeypatch.setattr(
            generate_dataset_cli.r2_io,
            "ensure_r2_env_loaded",
            lambda _env_file: None,
        )

        completed: list[subprocess.CompletedProcess[str]] = []

        def _execute_worker(sky_cfg: SkypilotLaunchConfig) -> None:
            assert sky_cfg.cmd is not None
            command_path = tmp_path / "worker-command.sh"
            command_path.write_text(sky_cfg.cmd)
            completed.append(
                subprocess.run(
                    ["/bin/bash", "worker-command.sh"],
                    cwd=tmp_path,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            )

        monkeypatch.setattr(skypilot_launch, "dispatch_via_skypilot", _execute_worker)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "synth-setter-generate-dataset",
                "experiment=generate_dataset/smoke-shard",
                f"render.plugin_path={_TEST_PLUGIN_VST3}",
                f"skypilot_launch.compute_template={compute_template}",
            ],
        )

        cast("Callable[[], None]", generate_dataset_cli.main)()

        assert len(completed) == 1
        return completed[0], compute_template

    return _dispatch


def test_main_remote_worker_command_repairs_stale_unpinned_checkout_then_executes(
    tmp_path: Path,
    remote_worker_dispatch: Callable[
        [Path, Path, Path], tuple[subprocess.CompletedProcess[str], Path]
    ],
) -> None:
    """An unpinned stale worker repairs Python before executing the entrypoint.

    :param tmp_path: Scratch worker checkout and safe worker-venv target.
    :param remote_worker_dispatch: Locally executes the generated worker command.
    """
    worker_root = _make_stale_worker_checkout(tmp_path)
    worker_venv = tmp_path / "worker-venv"
    stale_python = worker_venv / "bin/python"
    stale_python.parent.mkdir(parents=True)
    _write_executable(stale_python, "#!/bin/bash\nexit 1\n")
    trace = tmp_path / "worker-trace.log"
    fake_bin = _make_fake_worker_runtime(tmp_path, trace)
    result, compute_template = remote_worker_dispatch(worker_root, worker_venv, fake_bin)
    assert result.returncode == 0, result.stdout + result.stderr
    repaired_version = subprocess.run(  # noqa: S603 -- controlled executable in tmp_path
        [worker_venv / "bin/python", "-c", "import sys; print(sys.version_info[:3])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert repaired_version == "(3, 12, 13)"
    install, invocation = trace.read_text().splitlines()
    assert install == "install:pip install --group runtime -e ."
    assert invocation.startswith("exec:experiment=generate_dataset/smoke-shard ")
    assert f"render.plugin_path={_TEST_PLUGIN_VST3}" in invocation
    assert f"skypilot_launch.compute_template={compute_template}" in invocation
    assert "+created_at=" in invocation


def _r2_claims_steal_worker(
    uri: str,
    storage_options: dict[str, str],
    worker_index: int,
    out: multiprocessing.Queue[list[tuple[int, int]]],
) -> None:
    """Hammer claims on the real-R2 table under an always-expired lease.

    Module-level so ``multiprocessing``'s ``spawn`` context can pickle it.

    :param uri: ``s3://`` URI of the shared claims table in real R2.
    :param storage_options: Lance object-store credentials for the table.
    :param worker_index: Distinguishes this worker's owner identity.
    :param out: Receives this worker's ``[(shard_id, claim_gen), ...]`` wins.
    """
    from datetime import timedelta

    from synth_setter.pipeline.shard_claims import ShardClaims

    claims = ShardClaims(
        uri=uri,
        storage_options=storage_options,
        owner=f"proc-{worker_index}",
        lease=timedelta(seconds=-5),
    )
    wins = []
    for _ in range(4):
        claimed = claims.claim()
        if claimed is not None:
            wins.append((claimed.shard_id, claimed.claim_gen))
    out.put(wins)


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.slow
def test_shard_claims_contention_on_real_r2_grants_each_generation_once() -> None:
    """Concurrent workers stealing claims in real R2 never share a generation.

    The load-bearing assumption behind claims mode is that a Lance
    conditional ``update`` re-evaluates its predicate when its commit
    conflicts — on R2's conditional-put commit protocol, not just the local
    filesystem. Three real OS processes hammer two rows under an
    always-expired lease; if R2 commits ever let two workers win the same
    ``(shard_id, claim_gen)``, or dropped a committed win, this fails.
    Auto-skips without R2; a fresh-prefix guard catches leftovers from a
    crashed prior run before any worker starts, and the unique prefix is
    purged in ``finally``. Worker results carry a 600s budget — generous
    against R2 latency spikes for ~26 tiny commits.
    """
    from synth_setter.pipeline.r2_io import lance_target
    from synth_setter.pipeline.schemas.r2_location import R2Location
    from synth_setter.pipeline.shard_claims import ShardClaims

    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    unique_prefix = f"test-runs/test_shard_claims_contention/{uuid.uuid4().hex[:12]}/"
    location = R2Location(bucket="intermediate-data", prefix=unique_prefix)
    uri, storage_options = lance_target(location.shard_claims_uri())
    assert storage_options is not None, "contention run must target real R2, not local mode"
    assert not r2_io.list_entries(f"r2://{location.bucket}/{unique_prefix}", recursive=True), (
        "unique prefix must start empty; a leftover here means a prior run leaked"
    )
    try:
        ShardClaims(uri=uri, storage_options=storage_options, owner="operator").populate(range(2))

        ctx = multiprocessing.get_context("spawn")
        out: multiprocessing.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=_r2_claims_steal_worker, args=(uri, storage_options, index, out))
            for index in range(3)
        ]
        for proc in procs:
            proc.start()
        results = [out.get(timeout=600) for _ in procs]
        for proc in procs:
            proc.join(timeout=120)

        all_wins = [win for wins in results for win in wins]
        assert all_wins, "at least one steal must land for the invariant to be exercised"
        assert len(set(all_wins)) == len(all_wins), "two workers won the same generation"
        rows = (
            lance.dataset(uri, storage_options=storage_options)
            .to_table(columns=["shard_id", "claim_gen"])
            .to_pylist()
        )
        # >= not ==: a verify-read racing a steal burns an unrecorded generation.
        assert sum(row["claim_gen"] for row in rows) >= len(all_wins), (
            "more wins recorded than generations"
        )
    finally:
        r2_io.purge_prefix(location.bucket, location.prefix)


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_generate_dataset_renders_shards_to_r2(
    cfg_dataset: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hydra`` renders every shard in ``spec.shards`` and uploads to real R2.

    The unique-per-run ``r2.prefix`` override keeps concurrent runs isolated; a
    best-effort ``rclone purge`` in ``finally`` removes the prefix even on test
    failure so we don't leak shards. Auto-skips when ``rclone`` is missing or
    ``rclone lsd r2:`` fails (contributor laptops, fork PRs without secrets).

    :param cfg_dataset: Hydra DictConfig composed with the
        ``generate_dataset/smoke-shard`` experiment.
    :param monkeypatch: Pins the single-worker rank/world env contract.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    unique_prefix = (
        f"test-runs/test_generate_dataset_renders_shards_to_r2/{uuid.uuid4().hex[:12]}/"
    )
    with open_dict(cfg_dataset):
        cfg_dataset.r2.prefix = unique_prefix

    spec = spec_from_cfg(cfg_dataset)
    try:
        from_hydra(cfg_dataset)
        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id), (
                f"staged attempt missing in R2: {shard.filename}"
            )
    finally:
        r2_io.purge_prefix(spec.r2.bucket, spec.r2.prefix)


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_generate_dataset_renders_obxf_shards_to_r2(
    cfg_dataset_obxf: DictConfig,
) -> None:
    """``from_hydra`` renders every OB-Xf shard with the real plugin and uploads to R2.

    The second-synth counterpart to ``test_generate_dataset_renders_shards_to_r2``,
    under ``render=obxf`` so the real OB-Xf VST3 renders each shard (no stub) before the
    ``rclone copy`` upload. The unique-per-run ``r2.prefix`` keeps concurrent runs
    isolated; a best-effort ``rclone purge`` in ``finally`` removes the prefix even on
    failure so we don't leak shards. Auto-skips when ``rclone`` is missing or
    ``rclone lsd r2:`` fails (contributor laptops, fork PRs without secrets), and
    when the OB-Xf bundle is absent: ``requires_vst`` only gates the env-selected
    synth (``SYNTH_SETTER_PLUGIN_PATH``), so a Surge-only host would otherwise fail
    here rather than skip when ``render=obxf``'s ``plugin_path`` is missing.

    :param cfg_dataset_obxf: ``render=obxf`` cfg composed with the
        ``generate_dataset/smoke-shard`` experiment; carries the real OB-Xf bundle,
        preset, and pinned renderer version.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    unique_prefix = (
        f"test-runs/test_generate_dataset_renders_obxf_shards_to_r2/{uuid.uuid4().hex[:12]}/"
    )
    with open_dict(cfg_dataset_obxf):
        cfg_dataset_obxf.r2.prefix = unique_prefix

    spec = spec_from_cfg(cfg_dataset_obxf)
    assert spec.render.param_spec_name == "obxf"
    obxf_bundle = Path(spec.render.plugin_path)
    if not obxf_bundle.exists():
        pytest.skip(f"OB-Xf bundle not found at {obxf_bundle} (render=obxf plugin_path)")
    try:
        from_hydra(cfg_dataset_obxf)
        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id), (
                f"staged attempt missing in R2: {shard.filename}"
            )
    finally:
        r2_io.purge_prefix(spec.r2.bucket, spec.r2.prefix)


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_generate_dataset_shard_cadence_renders_one_identical_patch_per_shard(
    cfg_dataset: DictConfig,
) -> None:
    """``render.param_sample_cadence="shard"`` makes every sample in a shard share one patch.

    Drives the real ``generate_dataset`` entrypoint (``from_hydra``) end-to-end
    under shard cadence, then reads each shard's row range from its finalized
    split dataset in R2 (winner fragments commit in shard order, so a shard's
    rows sit at a spec-derived offset) and asserts its ``param_array`` rows
    are all identical — the one-patch-per-shard invariant the #489 variance
    probe relies on. Auto-skips without R2; purges the unique prefix in
    ``finally`` so a failure can't leak shards.

    :param cfg_dataset: Hydra DictConfig composed with the
        ``generate_dataset/smoke-shard`` experiment.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    unique_prefix = f"test-runs/test_generate_dataset_shard_cadence/{uuid.uuid4().hex[:12]}/"
    with open_dict(cfg_dataset):
        cfg_dataset.r2.prefix = unique_prefix
        cfg_dataset.render.param_sample_cadence = "shard"

    spec = spec_from_cfg(cfg_dataset)
    assert spec.render.param_sample_cadence == "shard"
    storage_options = r2_io.r2_storage_options()
    try:
        from_hydra(cfg_dataset)
        with tempfile.TemporaryDirectory() as raw_work_dir:
            finalize_lance(spec, Path(raw_work_dir))
        split_of: dict[int, Split] = {
            shard_id: split
            for split, (lo, hi) in spec.split_shard_ranges.items()
            for shard_id in range(lo, hi)
        }
        for shard in spec.shards:
            split = split_of[shard.shard_id]
            first_shard_in_split = spec.split_shard_ranges[split][0]
            offset = (shard.shard_id - first_shard_in_split) * spec.render.samples_per_shard
            s3_uri = r2_io.to_s3_uri(spec.r2.split_lance_uri(split))
            rows = (
                lance.dataset(s3_uri, storage_options=storage_options)
                .to_table(columns=["param_array"])
                .column("param_array")
                .to_numpy(zero_copy_only=False)
            )
            params = np.stack(rows[offset : offset + spec.render.samples_per_shard])
            assert params.shape[0] == spec.render.samples_per_shard
            assert np.array_equal(params, np.broadcast_to(params[0], params.shape)), (
                f"shard {shard.filename} has non-identical param rows under shard cadence"
            )
    finally:
        r2_io.purge_prefix(spec.r2.bucket, spec.r2.prefix)


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_oracle_eval_inline_writes_bounded_audio_metrics(
    cfg_dataset: DictConfig,
    tmp_path: Path,
) -> None:
    """The ``smoke-shard-with-oracle-eval`` CLI run writes bounded ``audio/*`` metrics.

    Only the launcher ``main`` runs the inline oracle eval (as its own
    ``synth_setter.cli.eval`` subprocess), so this drives the real CLI rather
    than ``from_hydra``. The eval writes ``metrics.json`` under
    ``oracle_eval/<run_id>/metrics/`` below the pinned ``hydra.run.dir``; we
    assert each ``audio/*`` aggregate is finite and the mean distances / ``rms``
    cosine stay within ``ORACLE_AUDIO_METRIC_BOUNDS`` — the envelope
    ``tests/test_train.py`` pins per-sample.

    :param cfg_dataset: Composed config; read only for ``r2.bucket`` (cleanup purge).
    :param tmp_path: Holds the Hydra run dir (hence the eval's ``metrics.json``)
        and the pinned operator workspace.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    # Override prefix_root (not prefix) so finalize_from_spec's assert_r2_prefix_matches
    # passes — the check validates prefix == make_r2_prefix(prefix_root, task_name, run_id).
    prefix_root = (
        f"test-runs/test_oracle_eval_inline_writes_bounded_audio_metrics/{uuid.uuid4().hex[:12]}"
    )
    run_dir = tmp_path / "hydra_run"
    worktree_src = Path(__file__).resolve().parents[1] / "src"
    # Prepend this worktree's src so the subprocess imports the same
    # synth_setter/configs as the collected test, not a sibling editable install.
    env = {
        **os.environ,
        "WANDB_MODE": "offline",
        "PYTHONPATH": f"{worktree_src}:{os.environ.get('PYTHONPATH', '')}",
        "SYNTH_SETTER_WORKSPACE": str(tmp_path),
    }
    try:
        result = subprocess.run(  # noqa: S603 — args are test-controlled literals
            [
                sys.executable,
                "-m",
                "synth_setter.cli.generate_dataset",
                "experiment=generate_dataset/smoke-shard-with-oracle-eval",
                f"r2.prefix_root={prefix_root}",
                f"hydra.run.dir={run_dir}",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"generate-dataset CLI exited {result.returncode}\n"
            f"--- STDOUT (tail) ---\n{result.stdout[-2000:]}\n"
            f"--- STDERR (tail) ---\n{result.stderr[-2000:]}"
        )

        # One metrics.json per split: oracle_eval/<split>/<run_id>/metrics/metrics.json.
        metrics_files = list(run_dir.glob("oracle_eval/*/*/metrics/metrics.json"))
        assert len(metrics_files) == 3, (
            f"expected three oracle-eval metrics.json files (one per split) under "
            f"{run_dir}/oracle_eval/; got {metrics_files}"
        )
        bounds = ORACLE_AUDIO_METRIC_BOUNDS
        for mf in metrics_files:
            metrics = json.loads(mf.read_text())
            # All splits resume one wandb run: test keeps the bare ``audio/*`` key
            # while train/val are namespaced ``<split>/audio/*`` so none overwrites.
            split = mf.parent.parent.parent.name
            metric_prefix = "" if split == "test" else f"{split}/"
            for name in _ORACLE_AUDIO_METRICS:
                for stat in ("mean", "std"):
                    key = f"{metric_prefix}audio/{name}_{stat}"
                    value = metrics.get(key)
                    assert isinstance(value, float) and math.isfinite(value), (
                        f"{key} is not a finite float: {value!r} (split={split}, metrics={metrics})"
                    )

            # fake_oracle returns params verbatim, so the re-rendered audio matches
            # the target up to Surge XT render jitter: mean distances stay under the
            # canonical envelope and the rms cosine stays above its floor.
            assert metrics[f"{metric_prefix}audio/mss_mean"] < bounds.mss_max, (split, metrics)
            assert metrics[f"{metric_prefix}audio/wmfcc_mean"] < bounds.wmfcc_max, (split, metrics)
            assert metrics[f"{metric_prefix}audio/sot_mean"] < bounds.sot_max, (split, metrics)
            assert metrics[f"{metric_prefix}audio/rms_mean"] > bounds.rms_min, (split, metrics)
    finally:
        r2_io.purge_prefix(cfg_dataset.r2.bucket, f"{prefix_root}/")


@pytest.mark.integration_r2
@pytest.mark.r2
@pytest.mark.requires_vst
@pytest.mark.slow
def test_oracle_eval_inline_writes_shuffled_audio_metrics_when_params_uniform(
    cfg_dataset: DictConfig,
    tmp_path: Path,
) -> None:
    """Oracle eval with ``param_sample_cadence=shard`` writes bounded ``shuffled_audio/*`` metrics.

    ``param_sample_cadence=shard`` gives every sample in the test shard the
    same ``params.csv``. The auto-shuffle probe (#489) in
    ``compute_audio_metrics`` detects uniform params and runs a second metrics
    pass with permuted ``pred.wav``, writing ``aggregated_metrics_shuffled.csv``;
    ``_load_audio_metrics`` then merges those values into ``metrics.json`` under
    ``shuffled_audio/<name>_{mean,std}``.

    Asserts each audio metric (mss, wmfcc, sot, rms) produces a finite, bounded
    value under both the ``audio/`` and ``shuffled_audio/`` prefixes.  Because
    all samples share one patch, shuffled predictions match the same target as
    the originals, so the shuffled means satisfy the same
    ``ORACLE_AUDIO_METRIC_BOUNDS`` envelope.

    :param cfg_dataset: Composed config; read only for ``r2.bucket`` (cleanup
        purge).
    :param tmp_path: Holds the Hydra run dir (hence the eval's
        ``metrics.json``) and the pinned operator workspace.
    """
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 not reachable (rclone not on PATH or `rclone lsd r2:` failed)")

    # Override prefix_root (not prefix) so finalize_from_spec's assert_r2_prefix_matches
    # passes — the check validates prefix == make_r2_prefix(prefix_root, task_name, run_id).
    r2_prefix_root = f"test-runs/test_oracle_eval_shuffled_audio_metrics/{uuid.uuid4().hex[:12]}"
    run_dir = tmp_path / "hydra_run"
    worktree_src = Path(__file__).resolve().parents[1] / "src"
    env = {
        **os.environ,
        "WANDB_MODE": "offline",
        "PYTHONPATH": f"{worktree_src}:{os.environ.get('PYTHONPATH', '')}",
        "SYNTH_SETTER_WORKSPACE": str(tmp_path),
    }
    try:
        result = subprocess.run(  # noqa: S603 — args are test-controlled literals
            [
                sys.executable,
                "-m",
                "synth_setter.cli.generate_dataset",
                "experiment=generate_dataset/smoke-shard-with-oracle-eval",
                f"r2.prefix_root={r2_prefix_root}",
                f"hydra.run.dir={run_dir}",
                # Uniform params within each shard so the auto-shuffle probe fires.
                "render.param_sample_cadence=shard",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"generate-dataset CLI exited {result.returncode}\n"
            f"--- STDOUT (tail) ---\n{result.stdout[-2000:]}\n"
            f"--- STDERR (tail) ---\n{result.stderr[-2000:]}"
        )

        # One metrics.json per split: oracle_eval/<split>/<run_id>/. Each split is
        # a single 4-sample shard, so cadence=shard makes every split's params
        # uniform and the shuffle probe fires for all three.
        metrics_files = list(run_dir.glob("oracle_eval/*/*/metrics/metrics.json"))
        assert len(metrics_files) == 3, (
            f"expected three oracle-eval metrics.json files (one per split) under "
            f"{run_dir}/oracle_eval/; got {metrics_files}"
        )

        bounds = ORACLE_AUDIO_METRIC_BOUNDS
        for mf in metrics_files:
            metrics = json.loads(mf.read_text())
            # test keeps bare keys; train/val are namespaced — the prefix applies
            # to both the audio/ and shuffled_audio/ groups.
            split = mf.parent.parent.parent.name
            metric_prefix = "" if split == "test" else f"{split}/"
            for name in _ORACLE_AUDIO_METRICS:
                for stat in ("mean", "std"):
                    for group in ("audio", "shuffled_audio"):
                        key = f"{metric_prefix}{group}/{name}_{stat}"
                        value = metrics.get(key)
                        assert isinstance(value, float) and math.isfinite(value), (
                            f"{key} is not a finite float: {value!r} "
                            f"(split={split}, metrics={metrics})"
                        )

            # Uniform params → shuffled pred matches the same target; means satisfy
            # the same oracle envelope as the non-shuffled pass.
            for group in ("audio", "shuffled_audio"):
                assert metrics[f"{metric_prefix}{group}/mss_mean"] < bounds.mss_max, (
                    split,
                    metrics,
                )
                assert metrics[f"{metric_prefix}{group}/wmfcc_mean"] < bounds.wmfcc_max, (
                    split,
                    metrics,
                )
                assert metrics[f"{metric_prefix}{group}/sot_mean"] < bounds.sot_max, (
                    split,
                    metrics,
                )
                assert metrics[f"{metric_prefix}{group}/rms_mean"] > bounds.rms_min, (
                    split,
                    metrics,
                )
    finally:
        r2_io.purge_prefix(cfg_dataset.r2.bucket, f"{r2_prefix_root}/")


def test_cfg_dataset_carries_ram_bounded_num_workers_for_oracle_eval(
    cfg_dataset: DictConfig,
) -> None:
    """Generate composes the VST datamodule's RAM-bounded worker default.

    Unlike ``train`` / ``evaluate``, generate never builds a datamodule: it
    forwards this value into the oracle-eval subprocess's argv, and the shard
    render pool sizes itself from ``available_cpus() // 2`` independently. The
    forwarding helper is private, which this module may not import, so the
    composed default is the consumable surface a test can pin here — the argv
    itself is covered by the oracle-eval inline tests above.

    Lance workers are ~1.4 GB each and the previous default of 11 exhausted a
    32 GB host (#1916).

    :param cfg_dataset: Composed config; read only for ``datamodule.num_workers``.
    """
    assert cfg_dataset.datamodule.num_workers == 4
