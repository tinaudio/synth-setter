"""Unit tests for the minimal #489 cadence-sweep runner's plan-building core.

These pin the pure ``sweeps(n)`` plan: every sweep is named for the synth
variant it actually renders, and the control sweep is a genuine no-copy control
rather than a duplicate of its copy probe. No real R2 or W&B is touched.
"""

from __future__ import annotations

import pytest

from synth_setter.tools import cadence_sweep_489 as sweep


def _config_by_label(configs: list[dict], label: str) -> dict:
    """Return the single sweep config named ``generate_dataset_<label>``.

    :param configs: The ``sweeps(n)`` output.
    :param label: Experiment label passed to ``_sweep``.
    :returns: The matching config.
    """
    wanted = f"generate_dataset_{label}"
    matches = [config for config in configs if config["name"] == wanted]
    assert len(matches) == 1, (
        f"expected exactly one {wanted!r}, got {[c['name'] for c in configs]}"
    )
    return matches[0]


def test_sweeps_rejects_size_below_one() -> None:
    """A sub-1 size leaves every split empty, so it is rejected up front."""
    with pytest.raises(ValueError, match="must be >= 1"):
        sweep.sweeps(0)


def test_sweep_name_carries_the_variant_it_renders_without_a_stale_suffix() -> None:
    """Each sweep is named for the synth it renders, with no hardcoded ``surge_xt`` tail.

    The runner builds both surge_xt and surge_simple sweeps; a fixed suffix would mislabel every
    surge_simple sweep as surge_xt (and double it on surge_xt).
    """
    assert {config["name"] for config in sweep.sweeps(2)} == {
        "generate_dataset_shuffle_cadence_probe_surge_xt",
        "generate_dataset_cadence_probe_surge_xt",
        "generate_dataset_control_cadence_probe_surge_xt",
        "generate_dataset_shuffle_cadence_probe_surge_simple",
        "generate_dataset_cadence_probe_surge_simple",
        "generate_dataset_control_cadence_probe_surge_simple",
        "generate_dataset_cadence_probe_surge_simple_xt_preset",
    }


def test_simple_control_omits_the_copy_uri_so_it_regenerates_fresh() -> None:
    """The surge_simple control omits the copy URI, mirroring the surge_xt control.

    Without this the control replays the copy source under the same cadence grid as
    ``cadence_probe_surge_simple`` (differing only in ``task_name``), so it is no control at all.
    """
    configs = sweep.sweeps(2)
    control = _config_by_label(configs, "control_cadence_probe_surge_simple")
    copy_probe = _config_by_label(configs, "cadence_probe_surge_simple")

    assert not any("copy_dataset_root_uri=" in t for t in control["command"])
    assert any("copy_dataset_root_uri=" in t for t in copy_probe["command"])


def test_swept_args_macro_is_last_so_grid_values_win_over_fixed_pins() -> None:
    """``${args_no_hyphens}`` must be the final command token in every sweep.

    Hydra applies later overrides last, so the swept grid macro has to follow the fixed pins;
    placing it earlier would let a fixed pin silently shadow an overlapping swept key.
    """
    for config in sweep.sweeps(2):
        assert config["command"][-1] == "${args_no_hyphens}"


def test_grid_cell_count_is_the_product_of_each_swept_knob() -> None:
    """The grid size is the product of every swept knob's value count (2 reload x 4 gui = 8)."""
    config = _config_by_label(sweep.sweeps(2), "cadence_probe_surge_xt")
    assert sweep._grid_cell_count(config) == 8


def test_run_agent_bounds_the_agent_to_the_grid_cell_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_agent`` passes ``--count <cells>`` so the agent exits when the grid is exhausted.

    Unbounded, wandb 0.26.1 keeps the agent alive after the grid is exhausted and re-dispatches an
    already-finished run, so the runner's sequential loop never reaches the next sweep (#489).

    :param monkeypatch: Captures the ``subprocess.run`` argv without spawning a real agent.
    """
    captured: list[list[str]] = []
    monkeypatch.setattr(sweep.subprocess, "run", lambda argv, **kwargs: captured.append(argv))

    sweep._run_agent("sid-123", 8)

    (argv,) = captured
    assert argv[:2] == ["wandb", "agent"]
    assert "--count" in argv
    assert argv[argv.index("--count") + 1] == "8"
    assert argv[-1].endswith("/sid-123")


def test_run_bounds_every_agent_by_its_own_grid_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run`` creates all sweeps, then drives each agent bounded by that sweep's grid size.

    :param monkeypatch: Stubs source generation and ``wandb.sweep``, and records the per-agent count.
    """
    monkeypatch.setattr(sweep, "_run_generate", lambda overrides: None)
    monkeypatch.setattr(sweep.wandb, "sweep", lambda config, **kwargs: "sid")
    counts: list[int] = []
    monkeypatch.setattr(sweep, "_run_agent", lambda sweep_id, count: counts.append(count))

    sweep.run(2)

    assert counts == [8] * 7


def test_run_rejects_size_below_one_before_generating_any_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run`` validates ``n`` up front, before launching any ``generate_dataset`` subprocess.

    The ``n >= 1`` guard lives in ``sweeps``; ``run`` must reach it before the expensive source
    generation, so an invalid size fails fast instead of burning two subprocess runs.

    :param monkeypatch: Replaces ``_run_generate`` with a recorder so the test can assert no
        source generation was attempted.
    """
    generated: list[list[str]] = []
    monkeypatch.setattr(sweep, "_run_generate", lambda overrides: generated.append(overrides))

    with pytest.raises(ValueError, match="must be >= 1"):
        sweep.run(0)

    assert generated == []
