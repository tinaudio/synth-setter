"""Unit tests for the #489 cadence-investigation orchestrator's functional core.

These pin the pure plan-building behavior — the derived copy-source URI, the
experiment set, the wandb sweep-config shape, per-cell override expansion, and
the source/copy match-set consistency that replaces the old hardcoded
``copy_dataset_root_uri``. The real generate -> copy -> oracle round trip is
covered end-to-end (real plugin + real R2) in
``tests/integration/test_cadence_investigation_489_e2e.py``.
"""

from __future__ import annotations

import pytest

from synth_setter.tools import cadence_investigation_489 as inv


def test_reference_copy_uri_default_prefix_root_matches_canonical_run_root() -> None:
    """The derived copy-source URI equals the canonical reference run root.

    This is the exact string the three paired-copy sweeps previously hardcoded, so deriving it
    (instead of hardcoding) cannot move where copies read from.
    """
    assert inv.reference_copy_uri() == "r2://intermediate-data/data/ref-surge-xt-489/paired-ref-v1"


def test_reference_copy_uri_custom_prefix_root_is_isolated_under_that_root() -> None:
    """A non-default prefix_root relocates the whole run root under it."""
    assert (
        inv.reference_copy_uri(prefix_root="test-runs/abc")
        == "r2://intermediate-data/test-runs/abc/ref-surge-xt-489/paired-ref-v1"
    )


def test_build_experiments_returns_two_within_run_and_three_copy_probes() -> None:
    """The investigation is exactly five experiments; three replay the copy source."""
    experiments = inv.build_experiments(inv.SMOKE)

    names = [e.name for e in experiments]
    assert names == [
        "shuffle_probe",
        "reuse_depth",
        "copy_reload",
        "copy_gui",
        "copy_repro",
    ]
    assert [e.name for e in experiments if e.needs_copy_source] == [
        "copy_reload",
        "copy_gui",
        "copy_repro",
    ]


def test_reuse_depth_grid_tracks_scale_reuse_depths() -> None:
    """Reuse depth is the swept knob, so its grid is the scale's reuse depths."""
    reuse = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "reuse_depth")
    assert reuse.grid == {"render.samples_per_shard": [1, 2]}


def test_build_sweep_config_for_copy_experiment_pins_program_and_injects_uri() -> None:
    """A copy sweep's command pins the program, injects the URI, and ends with args."""
    copy_reload = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "copy_reload")
    uri = "r2://intermediate-data/test-runs/x/ref-surge-xt-489/paired-ref-v1"

    cfg = inv.build_sweep_config(copy_reload, copy_uri=uri)

    assert cfg["program"] == "src/synth_setter/cli/generate_dataset.py"
    assert cfg["entity"] == "tinaudio"
    assert cfg["project"] == "synth-setter"
    assert cfg["method"] == "grid"
    assert cfg["command"][:2] == ["${interpreter}", "${program}"]
    assert cfg["command"][-1] == "${args_no_hyphens}"
    assert f"copy_dataset_root_uri={uri}" in cfg["command"]
    assert cfg["parameters"] == {"render.plugin_reload_cadence": {"values": ["once", "render"]}}


def test_build_sweep_config_for_within_run_experiment_omits_copy_uri() -> None:
    """A within-run sweep never carries a copy URI in its command."""
    shuffle = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "shuffle_probe")

    cfg = inv.build_sweep_config(shuffle, copy_uri=inv.reference_copy_uri())

    assert not any("copy_dataset_root_uri=" in tok for tok in cfg["command"])


def test_expand_cells_is_the_grid_product_with_fixed_overrides_in_every_cell() -> None:
    """Each cell is the fixed overrides plus one value per gridded knob."""
    copy_gui = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "copy_gui")
    uri = inv.reference_copy_uri(prefix_root="test-runs/x")

    cells = inv.expand_cells(copy_gui, copy_uri=uri)

    # grid is gui_toggle_cadence over three values -> three cells.
    assert len(cells) == 3
    gui_values = sorted(
        tok.split("=", 1)[1]
        for cell in cells
        for tok in cell
        if tok.startswith("render.gui_toggle_cadence=")
    )
    assert gui_values == ["never", "once", "render"]
    for cell in cells:
        assert f"copy_dataset_root_uri={uri}" in cell
        assert "render.param_spec_name=surge_xt" in cell


def test_source_and_copy_experiments_agree_on_the_copy_preflight_match_set() -> None:
    """Source-gen and every copy cell share param_spec / samples_per_shard / sizes.

    Copy reads same-named shards under the source run root, so a mismatch in the match set would
    misalign shard filenames or the param encoding. One Scale feeds both sides, so they cannot
    drift — this asserts that invariant.
    """
    scale = inv.SMOKE
    source = set(inv.reference_overrides(scale, prefix_root="data"))
    match_keys = (
        "render.param_spec_name",
        "render.samples_per_shard",
        "train_val_test_sizes",
    )

    def _pick(overrides: list[str]) -> dict[str, str]:
        return {
            tok.split("=", 1)[0]: tok.split("=", 1)[1]
            for tok in overrides
            if tok.split("=", 1)[0] in match_keys
        }

    source_match = _pick(list(source))
    uri = inv.reference_copy_uri(prefix_root="data")
    for exp in inv.build_experiments(scale):
        if not exp.needs_copy_source:
            continue
        for cell in inv.expand_cells(exp, copy_uri=uri):
            assert _pick(cell) == source_match, exp.name


def test_expand_cells_threads_prefix_root_into_every_cell() -> None:
    """A custom prefix_root lands in every cell so the whole run stays isolated."""
    copy_gui = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "copy_gui")

    cells = inv.expand_cells(
        copy_gui,
        copy_uri=inv.reference_copy_uri(prefix_root="test-runs/x"),
        prefix_root="test-runs/x",
    )

    assert all("r2.prefix_root=test-runs/x" in cell for cell in cells)


def test_build_sweep_config_threads_prefix_root_into_command() -> None:
    """A custom prefix_root reaches the wandb sweep command too."""
    copy_reload = next(e for e in inv.build_experiments(inv.SMOKE) if e.name == "copy_reload")

    cfg = inv.build_sweep_config(
        copy_reload, copy_uri=inv.reference_copy_uri(), prefix_root="test-runs/x"
    )

    assert "r2.prefix_root=test-runs/x" in cfg["command"]


def test_dry_run_executes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--dry-run`` plans only: no generate subprocess, no wandb calls.

    :param monkeypatch: Replaces the subprocess and wandb entry points with a
        sentinel that fails if dry-run ever calls them.
    """

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not execute side effects")

    monkeypatch.setattr(inv.subprocess, "run", _boom)
    monkeypatch.setattr(inv.wandb, "sweep", _boom)
    monkeypatch.setattr(inv.wandb, "agent", _boom)

    inv.main(["--dry-run", "--scale", "smoke", "--launcher", "wandb"])


def test_local_count_caps_cells_run_per_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--count`` caps how many cells each local experiment runs.

    Bounds the heavy real-render work in the e2e to one source run plus one copy
    cell, so the orchestration cannot fan out unbounded subprocesses.

    :param monkeypatch: Replaces ``_run_generate`` with a recorder so the cap is
        observed without launching any real generate subprocess.
    """
    runs: list[list[str]] = []
    monkeypatch.setattr(inv, "_run_generate", lambda overrides: runs.append(overrides))

    inv.main(["--launcher", "local", "--scale", "smoke", "--only", "copy_reload", "--count", "1"])

    # copy_reload grids two reload cadences, but --count 1 runs one cell; the
    # source generation is the other invocation.
    assert len(runs) == 2
