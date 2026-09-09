"""Hydra composition and validation for inline oracle candidate renderers."""

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from synth_setter.cli.generate_dataset import _resolve_oracle_candidate
from synth_setter.pipeline.schemas.spec import DatasetSpec


@pytest.fixture()
def candidate_cfg() -> Iterator[DictConfig]:
    """Compose a source dataset with a fully selected candidate renderer.

    :yields DictConfig: Composed dataset configuration for candidate validation.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/ultramaster-kr106-lance-smoke",
                "synth@oracle_eval.candidate.synth=ultramaster_kr106",
                "render@oracle_eval.candidate.render=vst",
            ],
        )
    try:
        yield cfg
    finally:
        GlobalHydra.instance().clear()


def test_candidate_aliases_compose_existing_synth_and_render_groups(
    candidate_cfg: DictConfig,
) -> None:
    """Package aliases reuse synth and render groups without altering the source.

    :param candidate_cfg: Composed source and candidate configuration.
    """
    source = DatasetSpec.from_hydra_cfg(candidate_cfg).render

    candidate = _resolve_oracle_candidate(candidate_cfg, source)

    assert candidate is not None
    assert candidate.synth.name == "ultramaster_kr106"
    assert candidate.renderer_backend == "pedalboard"
    assert candidate.plugin_reload_cadence == "once"
    assert candidate.gui_toggle_cadence == "once"
    assert source.renderer_backend == "dawdreamer"


def test_candidate_absent_preserves_source_only_eval() -> None:
    """Keep legacy source-only evaluation when both aliases are absent."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/ultramaster-kr106-lance-smoke"],
        )
    try:
        source = DatasetSpec.from_hydra_cfg(cfg).render
        assert _resolve_oracle_candidate(cfg, source) is None
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    "selection",
    [
        "synth@oracle_eval.candidate.synth=ultramaster_kr106",
        "render@oracle_eval.candidate.render=vst",
    ],
)
def test_candidate_with_only_one_group_raises(selection: str) -> None:
    """Reject a partially selected candidate before dataset generation.

    :param selection: The sole candidate package alias to select.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/ultramaster-kr106-lance-smoke",
                selection,
            ],
        )
    try:
        source = DatasetSpec.from_hydra_cfg(cfg).render
        with pytest.raises(ValueError, match="requires both synth and render groups"):
            _resolve_oracle_candidate(cfg, source)
    finally:
        GlobalHydra.instance().clear()


def test_candidate_with_different_param_spec_raises(candidate_cfg: DictConfig) -> None:
    """Reject candidates whose parameter indices cannot represent source rows.

    :param candidate_cfg: Mutable source and candidate configuration.
    """
    candidate_cfg.oracle_eval.candidate.synth.name = "surge_xt"
    candidate_cfg.oracle_eval.candidate.synth.param_spec_name = "surge_xt"
    candidate_cfg.oracle_eval.candidate.synth.plugin_path = "plugins/Surge XT.vst3"
    candidate_cfg.oracle_eval.candidate.synth.plugin_state_path = "presets/surge-base.vstpreset"
    candidate_cfg.oracle_eval.candidate.synth.synth_version = "1.3.4"
    source = DatasetSpec.from_hydra_cfg(candidate_cfg).render

    with pytest.raises(ValueError, match="param_spec_name"):
        _resolve_oracle_candidate(candidate_cfg, source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 48000),
        ("channels", 1),
        ("velocity", 99),
        ("signal_duration_seconds", 3.0),
    ],
)
def test_candidate_with_different_audio_condition_raises(
    candidate_cfg: DictConfig,
    field: str,
    value: int | float,
) -> None:
    """Reject candidates that change a physical audio condition.

    :param candidate_cfg: Mutable source and candidate configuration.
    :param field: Audio-condition field to change.
    :param value: Candidate value that differs from the source.
    """
    candidate_cfg.oracle_eval.candidate.render[field] = value
    source = DatasetSpec.from_hydra_cfg(candidate_cfg).render

    with pytest.raises(ValueError, match=field):
        _resolve_oracle_candidate(candidate_cfg, source)


def test_oracle_subprocess_uses_candidate_group_and_strict_stored_target_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pin candidate composition and stored-target verification in eval argv.

    :param monkeypatch: Replaces subprocess execution with argument capture.
    :param tmp_path: Isolated finalized-dataset root.
    """
    import lance
    import pyarrow as pa

    import synth_setter.cli.generate_dataset as gd

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for name in ("train.lance", "val.lance"):
        (dataset_root / name).touch()
    (dataset_root / "stats.npz").touch()
    predict_file = dataset_root / "test.lance"
    lance.write_dataset(pa.table({"row": [0]}), predict_file)
    subprocess_call = MagicMock()
    monkeypatch.setattr(gd, "_check_call_streamed", subprocess_call)

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[
                "experiment=generate_dataset/smoke-shard",
                "synth@oracle_eval.candidate.synth=surge_simple_surgepy",
                "render@oracle_eval.candidate.render=surgepy",
            ],
        )
    try:
        source = DatasetSpec.from_hydra_cfg(cfg).render
        candidate = _resolve_oracle_candidate(cfg, source)
        assert candidate is not None
        gd._run_oracle_eval_subprocess(
            dataset_root,
            tmp_path / "run",
            "run-id",
            render=candidate,
            num_workers=0,
            predict_file=predict_file,
            render_group="surgepy",
        )
    finally:
        GlobalHydra.instance().clear()

    argv = subprocess_call.call_args.args[0]
    assert "render=surgepy" in argv
    assert "evaluation.require_exact_param_oracle=true" in argv
    assert "evaluation.rerender_target=false" in argv


def test_inline_candidate_runs_source_and_candidate_on_each_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Evaluate each split once per renderer role with isolated metric keys.

    :param monkeypatch: Replaces generation, storage, and evaluation side effects.
    :param tmp_path: Isolated operator workspace.
    """
    import synth_setter.cli.generate_dataset as gd

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            "oracle_eval.upload=false",
            "synth@oracle_eval.candidate.synth=surge_simple",
            "render@oracle_eval.candidate.render=vst",
            "oracle_eval.candidate.render.renderer_backend=dawdreamer",
            "oracle_eval.candidate.render.plugin_reload_cadence=render",
            "oracle_eval.candidate.render.gui_toggle_cadence=never",
        ],
    )
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(gd, "write_spec_locally", lambda _spec, out: Path(out) / "input.json")
    monkeypatch.setattr(gd, "upload_spec", lambda _spec: "r2://bucket/input.json")
    monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", lambda *_args: None)
    monkeypatch.setattr(gd.r2_io, "download_dir_no_overwrite", lambda *_args: None)
    monkeypatch.setattr(gd, "_loggers_pinned_to_spec", lambda *_args: [])
    monkeypatch.setattr(gd, "generate", lambda *_args: None)
    monkeypatch.setattr(gd, "finalize_from_spec", lambda *_args: None)
    runtime_check = MagicMock()
    monkeypatch.setattr(gd, "ensure_dawdreamer_runtime", runtime_check)
    oracle = MagicMock()
    monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle)

    gd.main()

    assert [call.args[0] for call in runtime_check.call_args_list] == [
        "pedalboard",
        "dawdreamer",
    ]
    assert oracle.call_count == 6
    calls = oracle.call_args_list
    assert [call.kwargs["metric_prefix"] for call in calls] == [
        "source/train/",
        "source/val/",
        "source/",
        "candidate/train/",
        "candidate/val/",
        "candidate/",
    ]
    assert [call.kwargs["predict_file"].name for call in calls] == [
        "train.lance",
        "val.lance",
        "test.lance",
        "train.lance",
        "val.lance",
        "test.lance",
    ]
    assert all(call.kwargs["render_group"] == "vst" for call in calls)


def test_inline_candidate_uploads_role_specific_render_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each paired upload identifies its renderer role and effective config.

    :param monkeypatch: Replaces generation, storage, evaluation, and upload boundaries.
    :param tmp_path: Isolated operator workspace.
    """
    import synth_setter.cli.generate_dataset as gd

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            "oracle_eval.upload=true",
            "synth@oracle_eval.candidate.synth=surge_simple",
            "render@oracle_eval.candidate.render=vst",
            "oracle_eval.candidate.render.renderer_backend=dawdreamer",
            "oracle_eval.candidate.render.plugin_reload_cadence=render",
            "oracle_eval.candidate.render.gui_toggle_cadence=never",
        ],
    )
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(gd, "write_spec_locally", lambda _spec, out: Path(out) / "input.json")
    monkeypatch.setattr(gd, "upload_spec", lambda _spec: "r2://bucket/input.json")
    monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", lambda *_args: None)
    monkeypatch.setattr(gd.r2_io, "download_dir_no_overwrite", lambda *_args: None)
    monkeypatch.setattr(gd, "_loggers_pinned_to_spec", lambda *_args: [])
    monkeypatch.setattr(gd, "generate", lambda *_args: None)
    monkeypatch.setattr(gd, "finalize_from_spec", lambda *_args: None)
    monkeypatch.setattr(gd, "ensure_dawdreamer_runtime", lambda *_args: None)
    monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gd, "new_oracle_probe_launch_id", lambda: "launch-paired")
    upload = MagicMock(return_value="r2://bucket/probe")
    monkeypatch.setattr(gd, "upload_oracle_probe", upload)

    gd.main()

    assert [call.kwargs["role"] for call in upload.call_args_list] == [
        "source",
        "source",
        "source",
        "candidate",
        "candidate",
        "candidate",
    ]
    source = upload.call_args_list[0].kwargs["provenance"]
    candidate = upload.call_args_list[3].kwargs["provenance"]
    assert source.candidate_render == source.source_render
    assert candidate.source_render.renderer_backend == "pedalboard"
    assert candidate.candidate_render.renderer_backend == "dawdreamer"
    assert all(call.kwargs["launch_id"] == "launch-paired" for call in upload.call_args_list)


def test_remote_dispatch_ignores_incomplete_inline_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A local-only candidate alias cannot block a remote dataset dispatch.

    :param monkeypatch: Replaces storage and remote dispatch boundaries.
    :param tmp_path: Isolated operator workspace.
    """
    import synth_setter.cli.generate_dataset as gd
    import synth_setter.pipeline.skypilot_launch as sl

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            "skypilot_launch/compute=runpod/smoke",
            "oracle_eval_inline=true",
            "synth@oracle_eval.candidate.synth=surge_simple",
        ],
    )
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(gd, "write_spec_locally", lambda _spec, out: Path(out) / "input.json")
    monkeypatch.setattr(gd, "upload_spec", lambda _spec: "r2://bucket/input.json")
    monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", lambda *_args: None)
    dispatch = MagicMock()
    monkeypatch.setattr(sl, "dispatch_via_skypilot", dispatch)

    gd.main()

    dispatch.assert_called_once()
