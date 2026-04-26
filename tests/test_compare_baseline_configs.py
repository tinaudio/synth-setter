"""Sanity-check that current Hydra configs haven't silently drifted from a baseline ref.

Baseline results in this repo were produced by running the scripts under
``jobs/train/...``. Those scripts compose Hydra configs that determine model
architecture, optimizer settings, callbacks, logger setup, and so on. If a
config edit lands that silently changes what those scripts resolve to, future
runs will only match the baseline numbers by accident. This test is the
guardrail against that — when it fails, either the change was intentional
(bump the baseline) or it wasn't (investigate before merging).

Mechanism: for each pinned ref, materialize a detached worktree at that ref,
run the script there under a PATH shim that captures the resolved Hydra YAML
(``python ... --cfg job --resolve``), run the same script in the live tree,
and assert the two YAMLs match modulo invocation/deployment-volatile keys
(``INVOCATION_PATH_KEYS``, ``ACCEPTED_DIFFS``).

Two pinned constants below: ``FIXTURE_BASELINE`` covers the synthetic-fixture
sanity tests, and ``MODEL_BASELINE`` pins the ``jobs/train/{kosc,surge}/train.sh``
configs against a known-good model snapshot. When a config change is
intentional and is what we want future runs to inherit, bump the relevant
constant forward to a SHA on ``main`` that includes the change.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests._baseline_worktree import _git, _ref_exists, _try_fetch_ref

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

FIXTURE_BASELINE_REPO = FIXTURES / "baseline_repo"
FIXTURE_SCRIPT_REL = "scripts/baseline_app.sh"
FIXTURE_TASKS = 4

# Expected per-experiments-list line counts. These are the contract: changing
# experiments.txt without bumping these intentionally surfaces as a failed
# sanity test. Compare against the constant, not against `get_num_experiments`,
# so the assertion isn't tautological.
EXPECTED_KOSC_TASKS = 44
EXPECTED_SURGE_TASKS = 8

# Baseline refs for ref-comparison tests. Prefer tags over SHAs so CI can
# fetch the baseline cheaply via `fetch-tags: true` on actions/checkout.
# FIXTURE_BASELINE pins the synthetic-fixture equality + inequality tests.
# MODEL_BASELINE pins the K-OSC + SURGE train.sh tests against a known-good
# model-config snapshot (tag v0.0.0, == 79552d2).
# Update on PR merge: MODEL_BASELINE bumps when a published-results-relevant
# config change lands and a new release tag is cut.
FIXTURE_BASELINE = "624ea3c0d91698c53c7fad478294594f37854610"  # branch tip, not on main yet
MODEL_BASELINE = "v0.0.0"


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

    def slug(self) -> str:
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


def test_ref_compare_case_slug_renders_correctly() -> None:
    """RefCompareCase.slug() collapses the cmp suffix when both script_rels match."""
    c1 = RefCompareCase("abc1234", None, "scripts/foo.sh", "scripts/foo.sh", task_id=1)
    assert "abc1234_vs_live" in c1.slug()
    assert "cmp_" not in c1.slug()
    assert "task1" in c1.slug()

    c2 = RefCompareCase("abc1234", "def5678", "a/foo.sh", "b/bar.sh", task_id=2)
    assert "abc1234_vs_def5678" in c2.slug()
    assert "cmp_b" in c2.slug()
    assert "task2" in c2.slug()


def test_worktree_for_ref_smoke(worktree_for_ref: Callable[[str], Path]) -> None:
    """worktree_for_ref materializes a real worktree at HEAD."""
    head = _git("rev-parse", "HEAD", check=True).stdout.strip()
    wt = worktree_for_ref(head)
    assert wt.is_dir()
    # Linked worktrees have a .git file (not a directory) pointing back to the main repo.
    assert (wt / ".git").exists()


def test_pinned_baselines_resolve(worktree_for_ref: Callable[[str], Path]) -> None:
    """Both pinned BASELINE constants resolve to fetchable refs (collection-time guard).

    Without this, a stale FIXTURE_BASELINE or MODEL_BASELINE only surfaces deep
    inside a parametrized test failure. This test fails fast and loudly on the
    first cheap call to ``worktree_for_ref`` for each ref.
    """
    if not _ref_exists(FIXTURE_BASELINE):
        _try_fetch_ref(FIXTURE_BASELINE)
    assert _ref_exists(FIXTURE_BASELINE), f"FIXTURE_BASELINE {FIXTURE_BASELINE!r} unfetchable"
    if not _ref_exists(MODEL_BASELINE):
        _try_fetch_ref(MODEL_BASELINE)
    assert _ref_exists(MODEL_BASELINE), f"MODEL_BASELINE {MODEL_BASELINE!r} unfetchable"


@pytest.fixture
def real_python() -> str:
    """Return the interpreter the shim should forward to.

    Uses ``sys.executable`` so the shim runs the same Python that pytest is
    running under (with all the test deps installed). ``shutil.which("python")``
    would pick up whatever ``python`` is on PATH, which can be a different
    interpreter (system Python without hydra/yaml/etc.) under venv/uv/conda.
    """
    return sys.executable


ROLE_LABELS = {1: "base", 2: "curr"}
# Production train.sh scripts call `mamba activate` and `module load`; the shim
# directory ships no-op stubs for both so the script doesn't fail before reaching
# the python invocation that the harness wants to intercept.
_NOOP_SHIMS = ("mamba", "module")


@pytest.fixture
def shim_factory(
    tmp_path: Path, real_python: str, request: pytest.FixtureRequest
) -> Callable[[], tuple[Path, Path]]:
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
    """Run ``script_rel`` from ``repo`` with ``shim_dir`` prepended on PATH.

    Sandboxes ``HOME``/``XDG_CACHE_HOME``/``XDG_CONFIG_HOME`` to a subdir of
    ``shim_dir`` so any ``rm -rf ~/...`` in the script under test (e.g. the
    ``rm -rf ~/.triton/cache`` line in ``jobs/train/*/train.sh``) targets the
    sandbox, not the developer's real home.
    """
    home_sandbox = shim_dir / "home"
    home_sandbox.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    env["HOME"] = str(home_sandbox)
    env["XDG_CACHE_HOME"] = str(home_sandbox / ".cache")
    env["XDG_CONFIG_HOME"] = str(home_sandbox / ".config")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 — args are local Path/str values, not user-controlled
        ["/bin/bash", "-c", 'cd "$1" && bash "$2"', "_", str(repo), script_rel],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# Dotted-path keys stripped before the equality comparison so the assertion
# catches schema/value drift, not noise. Two named groups for self-documentation:

# Derived from rootutils + invocation cwd; always differ between the baseline
# worktree and the live REPO_ROOT.
INVOCATION_PATH_KEYS: tuple[str, ...] = (
    "paths.root_dir",
    "paths.data_dir",
    "paths.log_dir",
    "paths.work_dir",
)

# Each entry below is an explicit, audit-able claim that this divergence from
# the published-results resolved config doesn't change model behavior. Strip
# is asymmetric: `_strip_dotted_keys` pops the key where present and no-ops
# where absent, so a key that exists on current but not on baseline (e.g.
# `logger.tensorboard` — added to many_loggers.yaml after v0.0.0) becomes
# silently equal. Keep this list short and reviewable; bump the baseline
# rather than expanding it whenever possible.
ACCEPTED_DIFFS: tuple[str, ...] = (
    "logger.tensorboard",  # added post-v0.0.0; observability only, no model impact
    "logger.wandb.entity",  # env-derived (${oc.env:WANDB_ENTITY,null})
    "logger.wandb.log_model",  # changed `true` → "all" (artifact upload policy, not training)
    "logger.wandb.project",  # env-derived (${oc.env:WANDB_PROJECT,synth-setter})
)


def _strip_dotted_keys(cfg: dict, dotted_paths: tuple[str, ...]) -> dict:
    """Return a deep-copy of ``cfg`` with each dotted key path removed."""
    result = copy.deepcopy(cfg)
    for path in dotted_paths:
        parts = path.split(".")
        node = result
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                break
        else:
            # for-else: this branch runs only when the inner loop completed
            # without `break` — i.e., the full intermediate path was
            # traversable. Pop the leaf key off the resolved parent dict.
            node.pop(parts[-1], None)
    return result


def _assert_resolved_configs_equal(baseline: dict, current: dict) -> None:
    """Assert the resolved configs match modulo invocation/deployment-volatile keys."""
    keys = INVOCATION_PATH_KEYS + ACCEPTED_DIFFS
    base = _strip_dotted_keys(baseline, keys)
    cur = _strip_dotted_keys(current, keys)
    assert base == cur, (base, cur)


def _assert_resolved_configs_differ(baseline: dict, current: dict) -> None:
    """Assert the resolved configs differ modulo invocation/deployment-volatile keys.

    Mirror of ``_assert_resolved_configs_equal`` for the inequality fixture
    pattern. Strips the same volatile keys before comparing so the inequality
    reflects real content drift, not unavoidable worktree-vs-live noise.
    """
    keys = INVOCATION_PATH_KEYS + ACCEPTED_DIFFS
    base = _strip_dotted_keys(baseline, keys)
    cur = _strip_dotted_keys(current, keys)
    assert base != cur, (base, cur)


def _resolve_pair(
    baseline_path: Path,
    current_path: Path,
    baseline_script_rel: str,
    current_script_rel: str,
    task_id: int,
    shim_factory: Callable[[], tuple[Path, Path]],
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
    """Return the number of non-empty lines in an ``experiments.txt`` file.

    The train scripts use SGE array indexing (``SGE_TASK_ID``) to pick one
    line per task, so the line count = the number of experiments / parametrize
    cases the test should generate.
    """
    assert path.is_file(), f"missing: {path}"
    assert path.suffix == ".txt", f"unexpected experiment txt file type: {path}"
    with open(path, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    return n


def test_get_num_experiments() -> None:
    """Sanity-check that experiment counting matches the surge experiments.txt."""
    actual = get_num_experiments(REPO_ROOT / "jobs" / "train" / "surge" / "experiments.txt")
    assert actual == EXPECTED_SURGE_TASKS, (
        f"expected {EXPECTED_SURGE_TASKS} experiments, got {actual}"
    )


def _build_equal_cases(baseline_ref: str) -> list[RefCompareCase]:
    """Build the equality fixture's case list against ``baseline_ref``.

    The two sides reference the script under different basenames because the
    fixture was renamed (``hydra_app.sh`` → ``baseline_app.sh``) on this
    branch. The baseline ref still has the old name; the live tree has the
    new one. When ``FIXTURE_BASELINE`` is bumped to a SHA that has the
    rename, both fields collapse to the same path.
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
    ``diff_repo`` (port 6543) so the resolved configs deterministically differ.
    The baseline-side path uses the pre-rename basename (``hydra_app.sh``)
    because that's what exists at the baseline ref; the current side uses the
    renamed ``diff_app.sh`` under the renamed ``diff_repo``.
    """
    return [
        RefCompareCase(
            baseline_ref=baseline_ref,
            current_ref=None,
            baseline_script_rel="tests/fixtures/baseline_repo/scripts/hydra_app.sh",
            current_script_rel="tests/fixtures/diff_repo/scripts/diff_app.sh",
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


EQUAL_CASES = _build_equal_cases(FIXTURE_BASELINE)
DIFF_CASES = _build_diff_cases(FIXTURE_BASELINE)
KOSC_CASES = _build_kosc_train_cases(MODEL_BASELINE)
SURGE_CASES = _build_surge_train_cases(MODEL_BASELINE)


@pytest.mark.network
@pytest.mark.parametrize("case", EQUAL_CASES, ids=[c.slug() for c in EQUAL_CASES])
def test_baseline_and_current_resolved_hydra_configs_are_equal(
    shim_factory: Callable[[], tuple[Path, Path]],
    worktree_for_ref: Callable[[str], Path],
    case: RefCompareCase,
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
    assert len(KOSC_CASES) == EXPECTED_KOSC_TASKS


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("case", KOSC_CASES, ids=[c.slug() for c in KOSC_CASES])
def test_kosc_train_configs_are_equal(
    shim_factory: Callable[[], tuple[Path, Path]],
    worktree_for_ref: Callable[[str], Path],
    case: RefCompareCase,
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
    assert len(SURGE_CASES) == EXPECTED_SURGE_TASKS


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("case", SURGE_CASES, ids=[c.slug() for c in SURGE_CASES])
def test_surge_train_configs_are_equal(
    shim_factory: Callable[[], tuple[Path, Path]],
    worktree_for_ref: Callable[[str], Path],
    case: RefCompareCase,
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


@pytest.mark.network
@pytest.mark.parametrize("case", DIFF_CASES, ids=[c.slug() for c in DIFF_CASES])
def test_baseline_and_current_resolved_hydra_configs_differ(
    shim_factory: Callable[[], tuple[Path, Path]],
    worktree_for_ref: Callable[[str], Path],
    case: RefCompareCase,
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
    _assert_resolved_configs_differ(baseline_cfg, current_cfg)


def test_resolve_pair_rejects_empty_yaml(
    shim_factory: Callable[[], tuple[Path, Path]],
) -> None:
    """A no-op script produces no stdout, so the captured YAML is empty.

    `_resolve_pair` must surface that as an assertion failure rather than
    silently returning None and letting downstream comparisons compare
    None == None.
    """
    noop_repo = FIXTURES / "noop_repo"
    script_rel = "scripts/noop_app.sh"
    with pytest.raises(AssertionError, match="empty resolved YAML"):
        _resolve_pair(noop_repo, noop_repo, script_rel, script_rel, 1, shim_factory)


def test_injected_host_name_propagates_into_resolved_hydra_config(
    shim_factory: Callable[[], tuple[Path, Path]],
) -> None:
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
