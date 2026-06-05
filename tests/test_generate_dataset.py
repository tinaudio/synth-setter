"""Tests for the ``synth-setter-generate-dataset`` CLI entrypoint.

Covers tests that exercise ``from_hydra`` or the CLI subprocess end-to-end:
an ``integration_r2``-gated end-to-end render that drives ``from_hydra``
against ``cfg_dataset`` and asserts every shard lands at the spec-derived R2
URI in real Cloudflare R2; and an ``integration_r2`` subprocess run of the
``smoke-shard-with-oracle-eval`` experiment that asserts the inline oracle
eval's ``metrics.json`` holds bounded ``audio/*`` metrics. The integration
tests auto-skip when ``rclone`` / R2 creds are absent.

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

import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from synth_setter.cli.generate_dataset import from_hydra, spec_from_cfg
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.schemas.spec import DatasetSpec
from tests.evaluation._oracle_helpers import ORACLE_AUDIO_METRIC_BOUNDS

# The predict-mode oracle eval (surge/fake_oracle) dumps one mean+std per audio
# metric; predict leaves ``trainer.callback_metrics`` empty, so these are the
# only keys in ``metrics.json`` (see ``synth_setter.evaluation.compute_audio_metrics``).
_ORACLE_AUDIO_METRICS = ("mss", "wmfcc", "sot", "rms")

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_cfg_dataset_without_datasetsrc_composes_with_no_copy_source(
    cfg_dataset: DictConfig,
) -> None:
    """A default compose leaves ``spec.datasetsrc`` unset (``dataset.yaml`` sets ``null``).

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    spec = spec_from_cfg(cfg_dataset)
    assert spec.datasetsrc is None


def test_cfg_dataset_with_datasetsrc_composes_copy_source_into_spec(
    cfg_dataset: DictConfig,
) -> None:
    """A ``datasetsrc`` override flows through ``spec_from_cfg`` into ``DatasetSpec``.

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    with open_dict(cfg_dataset):
        cfg_dataset.datasetsrc = {"copy_dataset_root": "/data/source-dataset"}

    spec = spec_from_cfg(cfg_dataset)
    assert spec.datasetsrc is not None
    assert spec.datasetsrc.copy_dataset_root == "/data/source-dataset"


def test_cfg_dataset_datasetsrc_with_wds_output_is_rejected(
    cfg_dataset: DictConfig,
) -> None:
    """``spec_from_cfg`` rejects a ``datasetsrc`` paired with ``output_format='wds'``.

    The copy path reads each source shard as an HDF5 ``param_array``, so the
    ``DatasetSpec`` validator fails the spec at construction when output is not hdf5.

    :param cfg_dataset: Function-scoped fixture composing ``dataset.yaml`` with the
        ``generate_dataset/smoke-shard`` experiment and ``tmp_path``-pinned paths.
    """
    with open_dict(cfg_dataset):
        cfg_dataset.datasetsrc = {"copy_dataset_root": "/data/source-dataset"}
        cfg_dataset.output_format = "wds"

    with pytest.raises(ValueError, match="supports output_format='hdf5' only"):
        spec_from_cfg(cfg_dataset)


# Fields RenderConfig defaults in the model but ``render/surge_xt.yaml`` now also
# surfaces, so Hydra's struct mode accepts a plain ``render.<field>=...`` override
# (no ``+``) from any experiment, not only ones that pre-declare the key. Each value
# is an off-default override distinct from the surfaced default.
_SURFACED_RENDER_DEFAULTS = {
    "samples_per_render_batch": 16,
    "max_retries": 3,
    "parallel": True,
    "plugin_reload_cadence": "once",
    "gui_toggle_cadence": "once",
    "param_sample_cadence": "shard",
}

# An experiment that sets none of ``_SURFACED_RENDER_DEFAULTS``, so a successful
# plain override proves the key comes from the base render config, not the experiment.
_NO_CADENCE_EXPERIMENT = "experiment=generate_dataset/ci-materialize-test"


def _spec_from_dataset_overrides(overrides: list[str]) -> DatasetSpec:
    """Compose ``dataset.yaml`` with extra overrides and round-trip through ``DatasetSpec``.

    :param overrides: Hydra override strings appended after the experiment selector.
    :returns: The validated spec built from the composed cfg.
    """
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(config_name="dataset", overrides=[_NO_CADENCE_EXPERIMENT, *overrides])
        return spec_from_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize(("field", "override_value"), list(_SURFACED_RENDER_DEFAULTS.items()))
def test_base_render_config_accepts_plain_override_for_surfaced_default(
    field: str, override_value: object
) -> None:
    """A plain ``render.<field>=`` override composes against a no-cadence experiment.

    Struct mode rejects ``render.<field>=`` (without ``+``) when the key is absent
    from the composed tree, so this passing proves ``render/surge_xt.yaml`` surfaces
    the field's default; the override value round-trips onto the spec.

    :param field: RenderConfig field surfaced in the base render config.
    :param override_value: Off-default value passed on the Hydra CLI for that field.
    """
    spec = _spec_from_dataset_overrides([f"render.{field}={override_value}"])
    assert getattr(spec.render, field) == override_value


def test_base_render_config_defaults_match_render_config_model() -> None:
    """A no-override compose yields the surfaced YAML defaults on all platforms.

    ``gui_toggle_cadence`` is surfaced as ``"once"`` (safe on all platforms including
    Darwin, where ``"render"`` is rejected by the validator — #714).
    """
    spec = _spec_from_dataset_overrides([])
    assert spec.render.samples_per_render_batch == 32
    assert spec.render.max_retries == 0
    assert spec.render.parallel is False
    assert spec.render.plugin_reload_cadence == "render"
    assert spec.render.gui_toggle_cadence == "once"
    assert spec.render.param_sample_cadence == "sample"


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

    prefix = (
        f"test-runs/test_oracle_eval_inline_writes_bounded_audio_metrics/{uuid.uuid4().hex[:12]}/"
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
                f"+r2.prefix={prefix}",
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

        metrics_files = list(run_dir.glob("oracle_eval/*/metrics/metrics.json"))
        assert len(metrics_files) == 1, (
            f"expected one oracle-eval metrics.json under {run_dir}/oracle_eval/; "
            f"got {metrics_files}"
        )
        metrics = json.loads(metrics_files[0].read_text())

        for name in _ORACLE_AUDIO_METRICS:
            for stat in ("mean", "std"):
                key = f"audio/{name}_{stat}"
                value = metrics.get(key)
                assert isinstance(value, float) and math.isfinite(value), (
                    f"{key} is not a finite float: {value!r} (metrics={metrics})"
                )

        # fake_oracle returns params verbatim, so the re-rendered audio matches
        # the target up to Surge XT render jitter: mean distances stay under the
        # canonical envelope and the rms cosine stays above its floor.
        bounds = ORACLE_AUDIO_METRIC_BOUNDS
        assert metrics["audio/mss_mean"] < bounds.mss_max, metrics
        assert metrics["audio/wmfcc_mean"] < bounds.wmfcc_max, metrics
        assert metrics["audio/sot_mean"] < bounds.sot_max, metrics
        assert metrics["audio/rms_mean"] > bounds.rms_min, metrics
    finally:
        r2_io.purge_prefix(cfg_dataset.r2.bucket, prefix)
