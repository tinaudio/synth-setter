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
