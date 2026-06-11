"""Unit tests for the minimal #489 cadence-sweep runner.

These pin the pure plan-building behavior — the derived copy-source URI, the five
sweep configs, the source overrides, and the source/copy match-set consistency
that keeps the derived ``copy_dataset_root_uri`` aligned across producer and
consumer — plus the create-before-run ordering of :func:`run`.
"""

from __future__ import annotations

import pytest
import wandb.env

from synth_setter.tools import cadence_sweep_489 as inv

# Match-set keys every copy cell must echo from the source so shards align.
_MATCH_KEYS = ("render.param_spec_name", "render.samples_per_shard", "train_val_test_sizes")


def _pick_match_set(tokens: list[str]) -> dict[str, str]:
    """Extract the copy-preflight match-set tokens from a token list.

    :param tokens: Hydra ``key=value`` tokens (sweep command or source overrides).
    :returns: The ``key -> value`` mapping restricted to the match-set keys.
    """
    pairs = (tok.split("=", 1) for tok in tokens if "=" in tok)
    return {key: value for key, value in pairs if key in _MATCH_KEYS}


def test_reference_copy_uri_default_is_canonical_run_root() -> None:
    """The derived copy-source URI equals the canonical reference run root."""
    assert inv.reference_copy_uri() == "r2://intermediate-data/data/ref-surge-xt-489/paired-ref-v1"


def test_reference_copy_uri_honors_prefix_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-default ``PREFIX_ROOT`` relocates the whole run root under it.

    :param monkeypatch: Repoints the module ``PREFIX_ROOT`` to a throwaway root.
    """
    monkeypatch.setattr(inv, "PREFIX_ROOT", "test-runs/abc")
    assert (
        inv.reference_copy_uri()
        == "r2://intermediate-data/test-runs/abc/ref-surge-xt-489/paired-ref-v1"
    )


def test_source_overrides_pin_reference_identity_and_sizes() -> None:
    """The source overrides pin the reference run identity and the ``[n,n,n]`` splits."""
    overrides = inv.source_overrides(3)
    assert f"experiment={inv._SOURCE_EXPERIMENT}" in overrides
    assert "task_name=ref-surge-xt-489" in overrides
    assert "run_id=paired-ref-v1" in overrides
    assert "train_val_test_sizes=[3,3,3]" in overrides
    assert "render.samples_per_shard=3" in overrides


def test_source_overrides_below_one_raises_value_error() -> None:
    """A size below 1 fails fast: no split could hold a sample."""
    with pytest.raises(ValueError, match="dataset size must be >= 1"):
        inv.source_overrides(0)


def test_source_skips_oracle_eval_while_probes_run_it() -> None:
    """The source uses the no-eval experiment; every probe runs the with-oracle-eval one."""
    assert inv._SOURCE_EXPERIMENT != inv._PROBE_EXPERIMENT
    assert f"experiment={inv._SOURCE_EXPERIMENT}" in inv.source_overrides(2)
    for cfg in inv.sweeps(2):
        assert f"experiment={inv._PROBE_EXPERIMENT}" in cfg["command"]


def test_sweeps_are_five_named_probes_in_run_order() -> None:
    """The investigation is exactly five sweeps: two within-run probes then three copy probes."""
    assert [cfg["name"] for cfg in inv.sweeps(2)] == [
        "generate_dataset_shuffle_probe_surge_xt",
        "generate_dataset_reuse_depth_surge_xt",
        "generate_dataset_copy_reload_surge_xt",
        "generate_dataset_copy_gui_surge_xt",
        "generate_dataset_copy_repro_surge_xt",
    ]


def test_sweep_config_pins_program_entity_project_and_brackets_command() -> None:
    """Each sweep pins the program/entity/project and brackets its command with placeholders."""
    cfg = inv.sweeps(2)[0]
    assert cfg["program"] == inv.PROGRAM
    assert cfg["entity"] == inv.ENTITY
    assert cfg["project"] == inv.PROJECT
    assert cfg["method"] == "grid"
    assert cfg["command"][:2] == ["${interpreter}", "${program}"]
    assert cfg["command"][-1] == "${args_no_hyphens}"


def test_copy_probes_inject_copy_uri_and_within_run_probes_omit_it() -> None:
    """The three copy probes carry the copy URI; the two within-run probes never do."""
    cfgs = inv.sweeps(2)
    uri = inv.reference_copy_uri()
    for cfg in cfgs[2:]:
        assert f"copy_dataset_root_uri={uri}" in cfg["command"]
    for cfg in cfgs[:2]:
        assert not any(tok.startswith("copy_dataset_root_uri=") for tok in cfg["command"])


def test_reuse_depth_grid_is_no_reuse_and_full_reuse() -> None:
    """Reuse depth is the swept knob: no-reuse (1) and full-reuse (n)."""
    reuse = inv.sweeps(5)[1]
    assert reuse["parameters"] == {"render.samples_per_shard": {"values": [1, 5]}}


def test_reuse_depth_grid_collapses_to_one_cell_at_size_one() -> None:
    """At ``n == 1`` the no-reuse and full-reuse endpoints collapse to one depth."""
    reuse = inv.sweeps(1)[1]
    assert reuse["parameters"] == {"render.samples_per_shard": {"values": [1]}}


def test_source_and_copy_probes_share_the_copy_preflight_match_set() -> None:
    """Source generation and every copy probe agree on param spec / samples_per_shard / sizes.

    Copy reads same-named shards under the source run root, so a mismatch would misalign shard
    filenames or the param encoding. One size feeds both sides, so they cannot drift.
    """
    source_match = _pick_match_set(inv.source_overrides(2))
    for cfg in inv.sweeps(2):
        if not any(tok.startswith("copy_dataset_root_uri=") for tok in cfg["command"]):
            continue
        assert _pick_match_set(cfg["command"]) == source_match


def test_prefix_root_threads_into_every_sweep_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom ``PREFIX_ROOT`` reaches every sweep command and relocates the copy URI.

    :param monkeypatch: Repoints the module ``PREFIX_ROOT`` to a throwaway root.
    """
    monkeypatch.setattr(inv, "PREFIX_ROOT", "test-runs/x")
    for cfg in inv.sweeps(2):
        assert "r2.prefix_root=test-runs/x" in cfg["command"]
    copy_reload = inv.sweeps(2)[2]
    assert f"copy_dataset_root_uri={inv.reference_copy_uri()}" in copy_reload["command"]


def test_run_generates_source_then_creates_every_sweep_before_any_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run`` generates the source, then creates all five sweeps before any agent runs.

    Creating every sweep up front guarantees all five appear in the W&B UI regardless of agent
    fate.

    :param monkeypatch: Stubs generate, sweep creation, and the per-agent runner to record the
        global order of generate vs. create vs. run events without side effects.
    """
    events: list[str] = []
    ids = iter(["s0", "s1", "s2", "s3", "s4"])
    monkeypatch.setattr(inv, "_run_generate", lambda overrides: events.append("generate"))
    monkeypatch.setattr(inv.wandb, "sweep", lambda *a, **k: events.append("create") or next(ids))
    monkeypatch.setattr(inv, "_run_agent", lambda sweep_id: events.append("run"))

    inv.run(2)

    assert events == ["generate", *["create"] * 5, *["run"] * 5]


def test_run_generate_invokes_the_generate_module_with_the_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_generate`` runs ``python -m <generate module>`` with the overrides appended.

    :param monkeypatch: Captures the argv the generate subprocess is launched with.
    """
    import sys

    captured: dict[str, object] = {}
    monkeypatch.setattr(inv.subprocess, "run", lambda argv, **kwargs: captured.update(argv=argv))

    inv._run_generate(["task_name=ref-surge-xt-489", "render.samples_per_shard=2"])

    assert captured["argv"] == [
        sys.executable,
        "-m",
        inv._GENERATE_MODULE,
        "task_name=ref-surge-xt-489",
        "render.samples_per_shard=2",
    ]


def test_run_agent_disables_flapping_and_targets_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent runs as a ``wandb agent`` subprocess with flapping disabled in its env.

    :param monkeypatch: Captures the argv and environment the agent subprocess is launched with.
    """
    captured: dict[str, object] = {}

    def _record(argv: list[str], **kwargs: object) -> None:
        captured["argv"] = argv
        captured["env"] = kwargs["env"]

    monkeypatch.setattr(inv.subprocess, "run", _record)

    inv._run_agent("abc123")

    assert captured["argv"] == ["wandb", "agent", f"{inv.ENTITY}/{inv.PROJECT}/abc123"]
    assert captured["env"][wandb.env.AGENT_DISABLE_FLAPPING] == "true"  # type: ignore[index]


def test_main_size_drives_run_and_defaults_to_the_full_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--size N`` drives ``run(N)``; absent, it defaults to the full #489 run.

    :param monkeypatch: Replaces ``run`` with a recorder so the resolved size is observed
        without launching any real run.
    """
    sizes: list[int] = []
    monkeypatch.setattr(inv, "run", lambda n: sizes.append(n))

    inv.main(["--size", "5"])
    inv.main([])

    assert sizes == [5, inv.DEFAULT_SIZE]


def test_main_size_below_one_raises_before_any_side_effect() -> None:
    """``--size 0`` fails fast in source-override building, before any subprocess."""
    with pytest.raises(ValueError, match="dataset size must be >= 1"):
        inv.main(["--size", "0"])
