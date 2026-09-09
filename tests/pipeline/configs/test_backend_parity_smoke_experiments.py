"""Configuration contracts for local cross-backend oracle smoke runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("experiment", "source_synth", "candidate_synth", "candidate_backend", "render_batch"),
    [
        pytest.param(
            "surge-simple-pedalboard-to-dawdreamer-smoke",
            "surge_simple",
            "surge_simple",
            "dawdreamer",
            4,
            id="surge-dawdreamer",
        ),
        pytest.param(
            "surge-simple-pedalboard-to-surgepy-smoke",
            "surge_simple",
            "surge_simple_surgepy",
            "surgepy",
            4,
            id="surge-surgepy",
        ),
        pytest.param(
            "ultramaster-kr106-pedalboard-to-dawdreamer-smoke",
            "ultramaster_kr106",
            "ultramaster_kr106",
            "dawdreamer",
            1,
            id="kr106-dawdreamer",
        ),
    ],
)
def test_backend_parity_smoke_composes_source_and_candidate_contracts(
    experiment: str,
    source_synth: str,
    candidate_synth: str,
    candidate_backend: str,
    render_batch: int,
) -> None:
    """Each preset selects one Pedalboard source and its matching candidate host.

    :param experiment: Hydra experiment preset name.
    :param source_synth: Expected source synth config name.
    :param candidate_synth: Expected candidate synth config name.
    :param candidate_backend: Expected candidate renderer backend.
    :param render_batch: Expected source render batch size.
    """
    cfg = _compose_experiment(experiment)

    assert cfg.task_name == experiment
    assert cfg.train_val_test_sizes == [8, 4, 4]
    assert cfg.finalize_inline is True
    assert cfg.oracle_eval_inline is True
    assert cfg.oracle_eval.upload is True
    assert cfg.skypilot_launch.get("compute") is None

    assert cfg.synth.name == source_synth
    assert cfg.render.renderer_backend == "pedalboard"
    assert cfg.render.samples_per_render_batch == render_batch
    assert cfg.render.samples_per_shard == 4
    assert cfg.render.parallel is True

    candidate = cfg.oracle_eval.candidate
    assert candidate.synth.name == candidate_synth
    assert candidate.synth.param_spec_name == cfg.synth.param_spec_name
    assert candidate.render.renderer_backend == candidate_backend
    assert candidate.render.plugin_reload_cadence == "render"
    assert candidate.render.gui_toggle_cadence == "never"
    assert candidate.render.sample_rate == cfg.render.sample_rate
    assert candidate.render.channels == cfg.render.channels
    assert candidate.render.velocity == cfg.render.velocity
    assert candidate.render.signal_duration_seconds == cfg.render.signal_duration_seconds


def test_kr106_backend_parity_smoke_preserves_retry_budget() -> None:
    """KR-106 retains the retry budget established by its canonical smoke."""
    cfg = _compose_experiment("ultramaster-kr106-pedalboard-to-dawdreamer-smoke")

    assert cfg.render.max_retries == 20


def _compose_experiment(experiment: str) -> DictConfig:
    """Compose one backend-parity experiment with filesystem interpolations resolved.

    :param experiment: Hydra experiment preset name.
    :return: Composed dataset configuration.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=[f"experiment=generate_dataset/{experiment}"],
        )
    cfg.paths.root_dir = str(_REPO_ROOT)
    cfg.paths.output_dir = str(_REPO_ROOT)
    cfg.paths.work_dir = str(_REPO_ROOT)
    return cfg
