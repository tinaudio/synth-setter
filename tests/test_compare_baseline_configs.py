"""Compare resolved Hydra configs across two repo roots via a python PATH shim.

Each script under test must invoke a `@hydra.main` python app exactly once.
A PATH shim intercepts that python call, appends `--cfg job --resolve` (so
Hydra dumps the fully-resolved config to stdout instead of running main()),
and captures the YAML for comparison.

Add cases by appending to ``EQUAL_CASES`` or ``DIFF_CASES`` below — each case
is a ``(baseline_repo, current_repo, script_rel)`` triple.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

FIXTURE_BASELINE_REPO = FIXTURES / "baseline_repo"
FIXTURE_CURRENT_REPO = FIXTURES / "current_repo"
FIXTURE_DIFF_REPO = FIXTURES / "current_diff_repo"
FIXTURE_SCRIPT_REL = "scripts/hydra_app.sh"
FIXTURE_TASKS = 4


@dataclass(frozen=True)
class CompareCase:
    """A single (baseline, current, script, task_id) tuple for parametrize."""

    baseline: Path
    current: Path
    script_rel: str
    task_id: int

    def id(self) -> str:
        """Filesystem-safe parametrize id derived from this case's fields."""
        return (
            f"{self.baseline.name}__vs__{self.current.name}__{self.script_rel}__task{self.task_id}"
        )


def test_worktree_for_ref_smoke(worktree_for_ref) -> None:
    """worktree_for_ref materializes a real worktree at HEAD."""
    head = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607 — git on PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    wt = worktree_for_ref(head)
    assert wt.is_dir()
    # Linked worktrees have a .git file (not a directory) pointing back to the main repo.
    assert (wt / ".git").exists()


@pytest.fixture
def real_python() -> str:
    """Resolve the system ``python`` interpreter the shim should forward to."""
    py = shutil.which("python")
    assert py, "python not on PATH"
    return py


ROLE_LABELS = {1: "base", 2: "curr"}
_NOOP_SHIMS = ("mamba", "module")


@pytest.fixture
def shim_factory(tmp_path: Path, real_python: str, request: pytest.FixtureRequest):
    """Yield a callable that builds a per-invocation `python` PATH shim.

    Each call returns ``(shim_dir, out_yaml)`` for an isolated shim that
    appends ``--cfg job --resolve`` to whatever python invocation the script
    under test makes, redirecting stdout to ``out_yaml``.

    When the ``--compare-baseline-configs-keep-yaml-dir`` CLI option is set,
    captured YAMLs are written to
    ``<keep-yaml-dir>/<sanitized-test-id>__<role>.yaml`` so they survive
    pytest's tmp cleanup and have descriptive filenames. Role is ``base`` for
    the first call in a test, ``curr`` for the second, and ``out{n}`` for any
    additional calls.
    """
    counter = {"n": 0}
    keep_dir_str = request.config.getoption("--compare-baseline-configs-keep-yaml-dir")
    # Resolve to an absolute path: the shim runs after the subprocess cd's into
    # the repo root, so a relative keep-yaml-dir would be interpreted there.
    keep_dir = Path(keep_dir_str).resolve() if keep_dir_str else None
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
    safe_node = re.sub(r"[^A-Za-z0-9._-]+", "_", request.node.name)

    def _make() -> tuple[Path, Path]:
        counter["n"] += 1
        idx = counter["n"]
        shim_dir = tmp_path / f"shim-{idx}"
        shim_dir.mkdir()
        if keep_dir is None:
            out_yaml = tmp_path / f"out-{idx}.yaml"
        else:
            role = ROLE_LABELS.get(idx, f"out{idx}")
            out_yaml = keep_dir / f"{safe_node}__{role}.yaml"
        shim = shim_dir / "python"
        shim.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                printf 'python args: %s\\n' "$*"
                "{real_python}" "$@" --cfg job --resolve > "{out_yaml}"
                """
            )
        )
        shim.chmod(0o755)
        for name in _NOOP_SHIMS:
            noop = shim_dir / name
            noop.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '{name} args: %s\\n' "$*" >&2
                    exit 0
                    """
                )
            )
            noop.chmod(0o755)
        return shim_dir, out_yaml

    return _make


def _run_under_shim(
    shim_dir: Path,
    repo: Path,
    script_rel: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``script_rel`` from ``repo`` with ``shim_dir`` prepended on PATH."""
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 — args are local Path/str values, not user-controlled
        ["/bin/bash", "-c", 'cd "$1" && bash "$2"', "_", str(repo), script_rel],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _resolve_pair(case: CompareCase, shim_factory) -> tuple[dict, dict]:
    """Run baseline and current scripts under shims and return loaded YAMLs."""
    assert (case.baseline / case.script_rel).is_file(), (
        f"missing: {case.baseline / case.script_rel}"
    )
    assert (case.current / case.script_rel).is_file(), f"missing: {case.current / case.script_rel}"

    base_dir, base_yaml = shim_factory()
    curr_dir, curr_yaml = shim_factory()
    case_env = {"SGE_TASK_ID": str(case.task_id)}

    base_proc = _run_under_shim(base_dir, case.baseline, case.script_rel, extra_env=case_env)
    assert base_proc.returncode == 0, base_proc.stderr
    assert "python args:" in base_proc.stdout

    curr_proc = _run_under_shim(curr_dir, case.current, case.script_rel, extra_env=case_env)
    assert curr_proc.returncode == 0, curr_proc.stderr
    assert "python args:" in curr_proc.stdout

    base_cfg = yaml.safe_load(base_yaml.read_text())
    curr_cfg = yaml.safe_load(curr_yaml.read_text())
    assert base_cfg, f"empty resolved YAML at {base_yaml}"
    assert curr_cfg, f"empty resolved YAML at {curr_yaml}"
    return base_cfg, curr_cfg


def get_num_experiments(path: Path) -> int:
    """Count the number of experiments a script will run by parsing its SGE_TASK_ID usage."""
    assert path.is_file(), f"missing: {path}"
    assert path.suffix == ".txt", f"unexpected experiment txt file type: {path}"
    with open(path) as f:
        n = sum(1 for line in f if line.strip())
    return n


def test_get_num_experiments() -> None:
    """Sanity-check that experiment counting matches the surge experiments.txt."""
    expected = 8
    actual = get_num_experiments(REPO_ROOT / "jobs" / "train" / "surge" / "experiments.txt")
    assert actual == expected, f"expected {expected} experiments, got {actual}"


EQUAL_CASES: list[CompareCase] = [
    CompareCase(FIXTURE_BASELINE_REPO, FIXTURE_CURRENT_REPO, FIXTURE_SCRIPT_REL, t)
    for t in range(1, FIXTURE_TASKS + 1)
]


@pytest.mark.parametrize("case", EQUAL_CASES, ids=[c.id() for c in EQUAL_CASES])
def test_baseline_and_current_resolved_hydra_configs_are_equal(
    shim_factory, case: CompareCase
) -> None:
    """Resolved Hydra config from baseline and current repos must match."""
    baseline_cfg, current_cfg = _resolve_pair(case, shim_factory)
    assert baseline_cfg == current_cfg, (baseline_cfg, current_cfg)


K_OSC_TRAIN_CASES: list[CompareCase] = [
    CompareCase(REPO_ROOT, REPO_ROOT, "jobs/train/kosc/train.sh", t)
    for t in range(
        1,
        get_num_experiments(REPO_ROOT / "jobs" / "train" / "kosc" / "experiments.txt") + 1,
    )
]


def test_k_osc_train_cases() -> None:
    """Sanity-check K-OSC train case fan-out matches experiments.txt line count."""
    expected_tasks = 44
    assert len(K_OSC_TRAIN_CASES) == expected_tasks
    for case in K_OSC_TRAIN_CASES:
        assert case.task_id <= expected_tasks, (
            f"unexpected task_id {case.task_id} in case {case.id()}"
        )


@pytest.mark.parametrize("case", K_OSC_TRAIN_CASES, ids=[c.id() for c in K_OSC_TRAIN_CASES])
def test_kosc_train_configs_are_equal(shim_factory, case: CompareCase) -> None:
    """Resolved K-OSC train Hydra config must be stable across repo states."""
    baseline_cfg, current_cfg = _resolve_pair(case, shim_factory)
    assert baseline_cfg == current_cfg, (baseline_cfg, current_cfg)


SURGE_TRAIN_CASES: list[CompareCase] = [
    CompareCase(REPO_ROOT, REPO_ROOT, "jobs/train/surge/train.sh", t)
    for t in range(
        1,
        get_num_experiments(REPO_ROOT / "jobs" / "train" / "surge" / "experiments.txt") + 1,
    )
]


def test_surge_train_cases() -> None:
    """Sanity-check SURGE train case fan-out matches experiments.txt line count."""
    expected_tasks = 8
    assert len(SURGE_TRAIN_CASES) == expected_tasks
    for case in SURGE_TRAIN_CASES:
        assert case.task_id <= expected_tasks, (
            f"unexpected task_id {case.task_id} in case {case.id()}"
        )


@pytest.mark.parametrize("case", SURGE_TRAIN_CASES, ids=[c.id() for c in SURGE_TRAIN_CASES])
def test_surge_train_configs_are_equal(shim_factory, case: CompareCase) -> None:
    """Resolved SURGE train Hydra config must be stable across repo states."""
    baseline_cfg, current_cfg = _resolve_pair(case, shim_factory)
    assert baseline_cfg == current_cfg, (baseline_cfg, current_cfg)


DIFF_CASES: list[CompareCase] = [
    CompareCase(FIXTURE_BASELINE_REPO, FIXTURE_DIFF_REPO, FIXTURE_SCRIPT_REL, t)
    for t in range(1, FIXTURE_TASKS + 1)
]


@pytest.mark.parametrize("case", DIFF_CASES, ids=[c.id() for c in DIFF_CASES])
def test_baseline_and_current_resolved_hydra_configs_differ(
    shim_factory, case: CompareCase
) -> None:
    """Inequality sanity-check: divergent fixture must produce a different config."""
    baseline_cfg, current_cfg = _resolve_pair(case, shim_factory)
    assert baseline_cfg != current_cfg, (baseline_cfg, current_cfg)


def test_resolve_pair_rejects_empty_yaml(shim_factory) -> None:
    """A no-op script produces no stdout, so the captured YAML is empty.

    `_resolve_pair` must surface that as an assertion failure rather than
    silently returning None and letting downstream comparisons compare
    None == None.
    """
    noop_repo = FIXTURES / "noop_repo"
    case = CompareCase(noop_repo, noop_repo, "scripts/hydra_app.sh", 1)
    with pytest.raises(AssertionError, match="empty resolved YAML"):
        _resolve_pair(case, shim_factory)


def test_injected_host_name_propagates_into_resolved_hydra_config(shim_factory) -> None:
    """INJECTED_HOST_NAME env var must reach the resolved Hydra config and its interpolations."""
    baseline = FIXTURE_BASELINE_REPO
    script_rel = FIXTURE_SCRIPT_REL
    assert (baseline / script_rel).is_file()

    shim_dir, out_yaml = shim_factory()
    expected = "pytest-injected.example.com"
    proc = _run_under_shim(
        shim_dir, baseline, script_rel, extra_env={"INJECTED_HOST_NAME": expected}
    )
    assert proc.returncode == 0, proc.stderr

    cfg = yaml.safe_load(out_yaml.read_text())
    assert cfg["host"] == expected, cfg
    assert cfg["url"] == f"{expected}:5432", cfg
