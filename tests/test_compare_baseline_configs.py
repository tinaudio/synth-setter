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
FIXTURE_SCRIPT_REL = "scripts/baseline_app.sh"
FIXTURE_TASKS = 4


@dataclass(frozen=True)
class RefCompareCase:
    """A single (baseline_ref, current_ref, scripts, task_id) tuple for parametrize.

    ``current_ref=None`` means "compare against the live working tree."
    Both ``baseline_script_rel`` and ``current_script_rel`` are always set;
    set them to the same path for equality cases, different paths for the
    inequality fixture pattern.
    """

    baseline_ref: str
    current_ref: str | None
    baseline_script_rel: str
    current_script_rel: str
    task_id: int = 0

    def id(self) -> str:
        """Filesystem-safe parametrize id derived from this case's fields."""
        cmp = (
            ""
            if self.current_script_rel == self.baseline_script_rel
            else f"__cmp_{Path(self.current_script_rel).parent.name}"
        )
        cur = self.current_ref[:7] if self.current_ref else "live"
        return (
            f"{self.baseline_ref[:7]}_vs_{cur}"
            f"__{Path(self.baseline_script_rel).parent.name}{cmp}"
            f"__task{self.task_id}"
        )


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


def test_ref_compare_case_id_renders_correctly() -> None:
    """RefCompareCase.id() collapses the cmp suffix when both script_rels match."""
    c1 = RefCompareCase("abc1234", None, "scripts/foo.sh", "scripts/foo.sh", task_id=1)
    assert "abc1234_vs_live" in c1.id()
    assert "cmp_" not in c1.id()
    assert "task1" in c1.id()

    c2 = RefCompareCase("abc1234", "def5678", "a/foo.sh", "b/bar.sh", task_id=2)
    assert "abc1234_vs_def5678" in c2.id()
    assert "cmp_b" in c2.id()
    assert "task2" in c2.id()


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


# Hydra config keys whose values are derived from the invocation cwd
# (rootutils.find_root + ${paths.root_dir} interpolations) and therefore
# always differ between the baseline worktree and the live REPO_ROOT.
# Stripped before equality comparison so the assertion catches schema/value
# drift, not mechanical path divergence.
INVOCATION_PATH_KEYS = ("root_dir", "data_dir", "log_dir", "work_dir")


def _strip_invocation_paths(cfg: dict) -> dict:
    """Return a copy of ``cfg`` with invocation-derived path keys removed from ``paths``."""
    result = dict(cfg)
    paths = result.get("paths")
    if isinstance(paths, dict):
        result["paths"] = {k: v for k, v in paths.items() if k not in INVOCATION_PATH_KEYS}
    return result


def _assert_resolved_configs_equal(baseline: dict, current: dict) -> None:
    """Assert the resolved configs match modulo invocation-derived path keys."""
    base = _strip_invocation_paths(baseline)
    cur = _strip_invocation_paths(current)
    assert base == cur, (base, cur)


def _resolve_pair(
    baseline_path: Path,
    current_path: Path,
    baseline_script_rel: str,
    current_script_rel: str,
    task_id: int,
    shim_factory,
) -> tuple[dict, dict]:
    """Run baseline and current scripts under shims and return loaded YAMLs."""
    assert (baseline_path / baseline_script_rel).is_file(), (
        f"missing: {baseline_path / baseline_script_rel}"
    )
    assert (current_path / current_script_rel).is_file(), (
        f"missing: {current_path / current_script_rel}"
    )

    base_dir, base_yaml = shim_factory()
    curr_dir, curr_yaml = shim_factory()
    case_env = {"SGE_TASK_ID": str(task_id)}

    base_proc = _run_under_shim(base_dir, baseline_path, baseline_script_rel, extra_env=case_env)
    assert base_proc.returncode == 0, base_proc.stderr
    assert "python args:" in base_proc.stdout

    curr_proc = _run_under_shim(curr_dir, current_path, current_script_rel, extra_env=case_env)
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


def _build_equal_cases(baseline_ref: str) -> list[RefCompareCase]:
    """Build the equality fixture's case list against ``baseline_ref``.

    The two sides reference the script under different basenames because the
    fixture was renamed (``hydra_app.sh`` → ``baseline_app.sh``) on this
    branch. The baseline ref still has the old name; the live tree has the
    new one. Once this PR merges and the default ref flips to the merge SHA,
    both fields collapse to the same path.
    """
    return [
        RefCompareCase(
            baseline_ref=baseline_ref,
            current_ref=None,
            baseline_script_rel="tests/fixtures/baseline_repo/scripts/hydra_app.sh",
            current_script_rel="tests/fixtures/baseline_repo/scripts/baseline_app.sh",
            task_id=t,
        )
        for t in range(1, FIXTURE_TASKS + 1)
    ]


def _build_diff_cases(baseline_ref: str) -> list[RefCompareCase]:
    """Build the inequality fixture's case list against ``baseline_ref``.

    Baseline side runs ``baseline_repo`` (port 5432); current side runs
    ``current_diff_repo`` (port 6543) so the resolved configs deterministically
    differ. The baseline-side path uses the pre-rename basename
    (``hydra_app.sh``) because that's what exists at the baseline ref; the
    current side uses the renamed ``diff_app.sh``. ``current_diff_repo`` is
    further renamed to ``diff_repo`` in Step 8.
    """
    return [
        RefCompareCase(
            baseline_ref=baseline_ref,
            current_ref=None,
            baseline_script_rel="tests/fixtures/baseline_repo/scripts/hydra_app.sh",
            current_script_rel="tests/fixtures/current_diff_repo/scripts/diff_app.sh",
            task_id=t,
        )
        for t in range(1, FIXTURE_TASKS + 1)
    ]


def _build_train_cases(
    baseline_ref: str, script_rel: str, experiments_path: Path
) -> list[RefCompareCase]:
    """Build a train-script case list with one case per line in ``experiments_path``."""
    return [
        RefCompareCase(
            baseline_ref=baseline_ref,
            current_ref=None,
            baseline_script_rel=script_rel,
            current_script_rel=script_rel,
            task_id=t,
        )
        for t in range(1, get_num_experiments(experiments_path) + 1)
    ]


def _build_kosc_train_cases(baseline_ref: str) -> list[RefCompareCase]:
    """Build the K-OSC train fixture's case list against ``baseline_ref``."""
    return _build_train_cases(
        baseline_ref,
        "jobs/train/kosc/train.sh",
        REPO_ROOT / "jobs" / "train" / "kosc" / "experiments.txt",
    )


def _build_surge_train_cases(baseline_ref: str) -> list[RefCompareCase]:
    """Build the SURGE train fixture's case list against ``baseline_ref``."""
    return _build_train_cases(
        baseline_ref,
        "jobs/train/surge/train.sh",
        REPO_ROOT / "jobs" / "train" / "surge" / "experiments.txt",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Build ``case`` parametrize lists for ref-based tests at collection time.

    Ref-comparison tests need ``baseline_ref`` baked into each parametrized
    case, but the ref comes from ``--compare-baseline-configs-baseline-ref``
    — a CLI option that isn't available at module-import time, so the usual
    ``@pytest.mark.parametrize(..., EQUAL_CASES)`` decorator can't construct
    the cases. Pytest's collection hook fires *after* CLI parsing and *before*
    any test runs, exposing ``metafunc.config.getoption(...)`` and
    ``metafunc.parametrize(...)`` — the exact bridge needed.

    The hook fires for every collected test, so it filters: tests without a
    ``case`` parameter are skipped, and tests with one are dispatched by
    function name. As more tests migrate from path-based ``CompareCase`` to
    ``RefCompareCase`` (Step 7 of the migration), they get added here.
    """
    if "case" not in metafunc.fixturenames:
        return
    baseline_ref = metafunc.config.getoption("--compare-baseline-configs-baseline-ref")
    name = metafunc.function.__name__
    if name == "test_baseline_and_current_resolved_hydra_configs_are_equal":
        cases = _build_equal_cases(baseline_ref)
        metafunc.parametrize("case", cases, ids=[c.id() for c in cases])
    elif name == "test_baseline_and_current_resolved_hydra_configs_differ":
        cases = _build_diff_cases(baseline_ref)
        metafunc.parametrize("case", cases, ids=[c.id() for c in cases])
    elif name == "test_kosc_train_configs_are_equal":
        cases = _build_kosc_train_cases(baseline_ref)
        metafunc.parametrize("case", cases, ids=[c.id() for c in cases])
    elif name == "test_surge_train_configs_are_equal":
        cases = _build_surge_train_cases(baseline_ref)
        metafunc.parametrize("case", cases, ids=[c.id() for c in cases])


def test_baseline_and_current_resolved_hydra_configs_are_equal(
    shim_factory, worktree_for_ref, case: RefCompareCase
) -> None:
    """Resolved Hydra config at ``baseline_ref`` must match the live working tree."""
    baseline_path = worktree_for_ref(case.baseline_ref)
    current_path = REPO_ROOT
    baseline_cfg, current_cfg = _resolve_pair(
        baseline_path,
        current_path,
        case.baseline_script_rel,
        case.current_script_rel,
        case.task_id,
        shim_factory,
    )
    _assert_resolved_configs_equal(baseline_cfg, current_cfg)


def test_k_osc_train_cases() -> None:
    """Sanity-check K-OSC train case fan-out matches experiments.txt line count."""
    cases = _build_kosc_train_cases("placeholder-ref")
    expected_tasks = 44
    assert len(cases) == expected_tasks
    for case in cases:
        assert case.task_id <= expected_tasks, (
            f"unexpected task_id {case.task_id} in case {case.id()}"
        )


def test_kosc_train_configs_are_equal(
    shim_factory, worktree_for_ref, case: RefCompareCase
) -> None:
    """Resolved K-OSC train Hydra config at ``baseline_ref`` must match the live tree."""
    baseline_path = worktree_for_ref(case.baseline_ref)
    baseline_cfg, current_cfg = _resolve_pair(
        baseline_path,
        REPO_ROOT,
        case.baseline_script_rel,
        case.current_script_rel,
        case.task_id,
        shim_factory,
    )
    _assert_resolved_configs_equal(baseline_cfg, current_cfg)


def test_surge_train_cases() -> None:
    """Sanity-check SURGE train case fan-out matches experiments.txt line count."""
    cases = _build_surge_train_cases("placeholder-ref")
    expected_tasks = 8
    assert len(cases) == expected_tasks
    for case in cases:
        assert case.task_id <= expected_tasks, (
            f"unexpected task_id {case.task_id} in case {case.id()}"
        )


def test_surge_train_configs_are_equal(
    shim_factory, worktree_for_ref, case: RefCompareCase
) -> None:
    """Resolved SURGE train Hydra config at ``baseline_ref`` must match the live tree."""
    baseline_path = worktree_for_ref(case.baseline_ref)
    baseline_cfg, current_cfg = _resolve_pair(
        baseline_path,
        REPO_ROOT,
        case.baseline_script_rel,
        case.current_script_rel,
        case.task_id,
        shim_factory,
    )
    _assert_resolved_configs_equal(baseline_cfg, current_cfg)


def test_baseline_and_current_resolved_hydra_configs_differ(
    shim_factory, worktree_for_ref, case: RefCompareCase
) -> None:
    """Inequality sanity-check: divergent fixture must produce a different config."""
    baseline_path = worktree_for_ref(case.baseline_ref)
    baseline_cfg, current_cfg = _resolve_pair(
        baseline_path,
        REPO_ROOT,
        case.baseline_script_rel,
        case.current_script_rel,
        case.task_id,
        shim_factory,
    )
    # Strip invocation paths so the inequality reflects real content drift,
    # not the unavoidable worktree-vs-live path divergence.
    base = _strip_invocation_paths(baseline_cfg)
    cur = _strip_invocation_paths(current_cfg)
    assert base != cur, (base, cur)


def test_resolve_pair_rejects_empty_yaml(shim_factory) -> None:
    """A no-op script produces no stdout, so the captured YAML is empty.

    `_resolve_pair` must surface that as an assertion failure rather than
    silently returning None and letting downstream comparisons compare
    None == None.
    """
    noop_repo = FIXTURES / "noop_repo"
    script_rel = "scripts/noop_app.sh"
    with pytest.raises(AssertionError, match="empty resolved YAML"):
        _resolve_pair(noop_repo, noop_repo, script_rel, script_rel, 1, shim_factory)


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
