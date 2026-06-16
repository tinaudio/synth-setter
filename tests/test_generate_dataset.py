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
prefix) when all sample dirs share uniform ``params.csv`` (#489). The
integration tests auto-skip when ``rclone`` / R2 creds are absent.

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
import os
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import h5py
import lance
import numpy as np
import pytest
from omegaconf import DictConfig, open_dict

from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_shard import LANCE_DATA_STORAGE_VERSION
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.evaluation._oracle_helpers import ORACLE_AUDIO_METRIC_BOUNDS
from tests.helpers.dummy_shards import stub_renderer

# The predict-mode oracle eval (surge/fake_oracle) dumps one mean+std per audio
# metric; predict leaves ``trainer.callback_metrics`` empty, so these are the
# only keys in ``metrics.json`` (see ``synth_setter.evaluation.compute_audio_metrics``).
_ORACLE_AUDIO_METRICS = ("mss", "wmfcc", "sot", "rms")

_REPO_ROOT = Path(__file__).resolve().parents[1]

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


def test_cfg_dataset_without_copy_dataset_root_uri_composes_with_no_copy_source(
    cfg_dataset: DictConfig,
) -> None:
    """A default compose leaves ``spec.copy_dataset_root_uri`` unset (``dataset.yaml`` sets ``null``).

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    spec = spec_from_cfg(cfg_dataset)
    assert spec.copy_dataset_root_uri is None


def test_cfg_dataset_with_copy_dataset_root_uri_composes_copy_source_into_spec(
    cfg_dataset: DictConfig,
) -> None:
    """A ``copy_dataset_root_uri`` override flows through ``spec_from_cfg`` into ``DatasetSpec``.

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    with open_dict(cfg_dataset):
        cfg_dataset.copy_dataset_root_uri = "r2://bucket/prefix/task/run"

    spec = spec_from_cfg(cfg_dataset)
    assert spec.copy_dataset_root_uri == "r2://bucket/prefix/task/run"


def test_cfg_dataset_copy_dataset_root_uri_with_wds_output_is_rejected(
    cfg_dataset: DictConfig,
) -> None:
    """``spec_from_cfg`` rejects a ``copy_dataset_root_uri`` paired with ``output_format='wds'``.

    The copy path reads each source shard as an HDF5 ``param_array``, so the
    ``DatasetSpec`` validator fails the spec at construction when output is not hdf5.

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    with open_dict(cfg_dataset):
        cfg_dataset.copy_dataset_root_uri = "/data/source-dataset"
        cfg_dataset.output_format = "wds"

    with pytest.raises(ValueError, match="supports output_format='hdf5' only"):
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
@pytest.mark.parametrize(
    ("output_format", "shard_suffix"),
    [("hdf5", ".h5"), ("wds", ".tar"), ("lance", ".lance")],
)
def test_from_hydra_renders_every_shard_to_fake_r2_then_resume_skips(
    output_format: str,
    shard_suffix: str,
    cfg_dataset: DictConfig,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_hydra`` renders every owned shard to a fake R2, then skips them on resume.

    Drives the real worker entrypoint end-to-end on the CPU-fast loop: the Surge
    VST3 subprocess is replaced by ``stub_renderer`` (writes a deterministic
    validation-shaped shard) and ``r2:`` is a local-filesystem rclone remote via
    ``fake_r2_remote``, so the partition → render → ``rclone copy`` upload →
    skip-existing probe loop (#750) runs with no real plugin and no real R2.
    Parametrized over ``output_format`` so every format's config surface
    runs the same loop with its own shard suffix (#1600). Asserts
    (1) ``smoke-shard`` partitions into one shard per split, (2) every shard
    lands under its spec-derived R2 URI with the format's suffix, (3) the Lance
    leg writes at the pinned ``LANCE_DATA_STORAGE_VERSION`` (#1714), and (4) a
    second ``from_hydra`` pass renders nothing because the probe finds all
    shards already present.

    :param output_format: Dataset output format the run is pinned to.
    :param shard_suffix: File suffix the format's shards must carry.
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
        cfg_dataset.output_format = output_format
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        # Pin r2.prefix so the spec built here for assertions and the one
        # from_hydra rebuilds internally derive the same shard URIs — an unpinned
        # created_at would fire its default factory twice and diverge the run_id.
        cfg_dataset.r2.prefix = "fake-r2/test-run/"
        # Disable the default wandb logger: generate() would call wandb.init() and block.
        cfg_dataset.logger = None

    # Local rclone errors on a missing path; real R2 returns None. Wrap to bridge
    # that gap so the skip probe sees the absent-object contract unchanged.
    real_object_size = r2_io.object_size

    def _fake_r2_object_size(r2_uri: str) -> int | None:
        try:
            return real_object_size(r2_uri)
        except subprocess.CalledProcessError:
            return None

    monkeypatch.setattr(r2_io, "object_size", _fake_r2_object_size)

    # Same local-vs-real bridge for the directory (Lance) skip-probe: a local
    # rclone remote errors on an absent prefix where real R2 lists empty.
    real_directory_exists = r2_io.r2_directory_exists

    def _fake_r2_directory_exists(r2_uri: str) -> bool:
        try:
            return real_directory_exists(r2_uri)
        except subprocess.CalledProcessError:
            return False

    monkeypatch.setattr(r2_io, "r2_directory_exists", _fake_r2_directory_exists)

    spec = spec_from_cfg(cfg_dataset)
    # smoke-shard partitions into one shard per split, so the stub covers train→val→test.
    assert spec.split_shard_ranges == {"train": (0, 1), "val": (1, 2), "test": (2, 3)}

    render_shard = stub_renderer(spec)
    with patch(
        "synth_setter.cli.generate_dataset._check_call_streamed",
        side_effect=render_shard,
    ):
        from_hydra(cfg_dataset)

    for shard in spec.shards:
        assert shard.filename.endswith(shard_suffix)
        if spec.output_format.is_directory:
            # Probe the committed manifest, mirroring the production skip-probe:
            # asserts the shard landed AND was committed (not orphaned fragments).
            assert r2_io.r2_directory_exists(f"{spec.r2.shard_uri(shard)}/_versions"), (
                f"committed shard missing in fake R2: {shard.filename}"
            )
            if output_format == "lance":
                # fake_r2_remote materializes r2://<bucket>/<key> at <root>/<bucket>/<key>.
                shard_key = spec.r2.shard_uri(shard).removeprefix(r2_io.R2_URI_SCHEME)
                shard_dir = fake_r2_remote / shard_key
                assert (
                    lance.dataset(str(shard_dir)).data_storage_version
                    == LANCE_DATA_STORAGE_VERSION
                ), f"shard {shard.filename} not written at pinned storage version"
        else:
            size = r2_io.object_size(spec.r2.shard_uri(shard))
            assert size is not None and size > 0, f"shard missing in fake R2: {shard.filename}"

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
    :param monkeypatch: Pins the single-worker env, the moduleinfo-only plugin, and
        the local-vs-real R2 skip-probe bridge.
    """
    monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
    monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
    with open_dict(cfg_dataset):
        cfg_dataset.output_format = "hdf5"
        cfg_dataset.render.plugin_path = str(_TEST_PLUGIN_VST3)
        cfg_dataset.render.renderer_version = _TEST_PLUGIN_VERSION
        cfg_dataset.r2.prefix = "fake-r2/seed-run/"
        cfg_dataset.logger = None

    # Local rclone errors on a missing object where real R2 returns None; bridge it
    # so the skip-existing probe sees every shard as absent and renders all of them.
    real_object_size = r2_io.object_size

    def _fake_r2_object_size(r2_uri: str) -> int | None:
        try:
            return real_object_size(r2_uri)
        except subprocess.CalledProcessError:
            return None

    monkeypatch.setattr(r2_io, "object_size", _fake_r2_object_size)

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
            size = r2_io.object_size(spec.r2.shard_uri(shard))
            assert size is not None and size > 0, f"shard missing in R2: {shard.filename}"
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
            size = r2_io.object_size(spec.r2.shard_uri(shard))
            assert size is not None and size > 0, f"shard missing in R2: {shard.filename}"
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
    under shard cadence, then downloads each shard and asserts its ``param_array``
    rows are all identical — the one-patch-per-shard invariant the #489 variance
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
    try:
        from_hydra(cfg_dataset)
        for shard in spec.shards:
            with r2_io.downloaded_to_tempfile(spec.r2.shard_uri(shard)) as local:
                with h5py.File(local, "r") as f:
                    param_dataset = f["param_array"]
                    assert isinstance(param_dataset, h5py.Dataset)
                    params = param_dataset[:]
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
