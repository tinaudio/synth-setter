# Design Doc: CI-Lite Dependency Groups

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-06-03
> **Tracking**: #1139

______________________________________________________________________

### Index

| §   | Section                                               | What it covers                                            |
| --- | ----------------------------------------------------- | --------------------------------------------------------- |
| 1   | [Context & Problem](#1-context--problem)              | Why lightweight CI jobs break, and why the hack is forced |
| 2   | [Goals & Non-Goals](#2-goals--non-goals)              | What this change must and must not do                     |
| 3   | [Solution](#3-solution)                               | Light base + PEP 735 groups + `default-groups`            |
| 4   | [Torch Index Routing](#4-torch-index-routing)         | Migrating the `cpu`/`cu128` routing off `[project]`       |
| 5   | [Migration Plan](#5-migration-plan)                   | Per-callsite changes and ordering                         |
| 6   | [Drift Guard](#6-drift-guard)                         | The test that keeps the base honest                       |
| 7   | [Alternatives Considered](#7-alternatives-considered) | Why extras and `--only-group` don't work here             |
| 8   | [Risks & Rollout](#8-risks--rollout)                  | Failure modes and validation                              |

______________________________________________________________________

## 1. Context & Problem

A handful of CI jobs import `synth_setter.pipeline.*` to do cheap structural
work — chiefly `Validate spec structure`, which runs
`python -m synth_setter.pipeline.ci.validate_spec` on the runner host (outside
the Docker image). These jobs must **not** install the full runtime
(`torch`, `skypilot`, `librosa`, `h5py`, `dask`, `pedalboard`, …) — that is
minutes of wheel downloads for a job that only needs to parse a Pydantic model.

The current workaround (three workflows: `validate-dataset-shards.yaml`,
`spec-materialization.yml`, `test-spec-materialization.yml`):

```bash
pip install --no-deps -e .
pip install "pydantic>=2,<3" python-dotenv   # hand-picked, by hand
```

This is fragile. The hand-picked list is a guess at the import closure of
`validate_spec` + `r2_io`, maintained by humans in three places. **Any new
module-level import on that path breaks the job silently** until CI fails:

- #1120 added `from dotenv import dotenv_values` to `r2_io.py` → broke the job
  on #1129 / #1135.
- #1390 added `from omegaconf import DictConfig, OmegaConf` to
  `schemas/spec.py` → broke the scheduled *Test Dataset Generation* run with
  `ModuleNotFoundError: No module named 'omegaconf'`. (Unblocked immediately by
  #1395, which makes that import lazy — but that only kicks the can.)

**Why the `--no-deps` is currently forced.** `uv`/`pip` always install a
project's `[project.dependencies]` whenever they install the project itself.
Today those dependencies *are* the full heavy runtime, so the only way to get a
light env with `synth_setter` importable is `--no-deps` + a hand-picked add-back.
An extra (`pip install -e .[lite]`) cannot help: **PEP 621 extras are
additive** — `.[lite]` installs the full base *plus* the extra, never a subset.

The empirically-confirmed import closure of the lite entrypoints
(`validate_spec.main` + `r2_io.ensure_r2_env_loaded`) is exactly:

```
pydantic, python-dotenv     (+ their transitives: pydantic-core, annotated-types, typing-extensions)
```

Note `spec_uri` (the `synth-setter-spec-uri` entrypoint) imports `hydra`, so it
is **not** a lite entrypoint and stays on the full env.

## 2. Goals & Non-Goals

**Goals**

- Lite CI jobs install `synth_setter` + only its lite closure, with **no
  `--no-deps` hack** and **no hand-picked dep list**.
- A **single source of truth** for the lite base; impossible to drift silently.
- A new top-level import on the lite path **fails fast** (a guard step), not
  silently at the next unrelated CI run.
- `uv sync` with no flags still installs **everything** — zero change to local
  developer onboarding or to the ~15 full-runtime install sites.

**Non-Goals**

- Re-architecting the torch backend selection model (`cpu`/`cu128`); we migrate
  its *location* only, preserving behavior.
- Splitting dev tooling further (test/lint/typecheck groups). Out of scope;
  `dev` stays as-is.
- Changing what the Docker image or GPU/CPU test matrix installs.

## 3. Solution

Move the heavy runtime out of `[project.dependencies]` into PEP 735
`[dependency-groups]`, leaving the base as just the lite closure. Aggregate the
heavy groups under a `runtime` meta-group, fold `runtime` into `dev`, and point
`default-groups` at `dev` so a **bare `uv sync` still installs everything**.

```toml
[project]
dependencies = [
  "pydantic>=2",
  "python-dotenv",
]   # the lite closure — pinned by a test (§6)

[dependency-groups]
torch   = ["torch>=2.0.0", "torchvision>=0.15.0", "torchaudio>=2.0.0",
           "lightning>=2.6.0", "torchmetrics>=0.11.4"]
config  = ["hydra-core==1.3.2", "hydra-colorlog==1.2.0", "hydra-optuna-sweeper==1.2.0",
           "omegaconf", "pydantic-settings>=2", "pyyaml", "tomli; python_version < '3.11'"]
compute = ["skypilot[runpod,oci,kubernetes]==0.12.0", "runpod==1.8.1", "oci",
           "kubernetes==35.0.0"]
data    = ["dask[distributed]", "h5py", "hdf5plugin", "webdataset", "pandas", "numpy"]
audio   = ["librosa", "pedalboard", "pyloudnorm", "mido", "scipy>=1.14,<1.15"]
metrics = ["pesto-pitch==2.0.1", "dtw-python==1.7.4", "kymatio==0.3.0", "POT", "einops"]
util    = ["click<8.2", "structlog", "tenacity", "loguru==0.7.3", "rich", "sh",
           "threadpoolctl", "matplotlib", "tensorboard", "wandb"]
runtime = [
  {include-group = "torch"}, {include-group = "config"}, {include-group = "compute"},
  {include-group = "data"},  {include-group = "audio"},  {include-group = "metrics"},
  {include-group = "util"},
]
dev = [ "...existing dev tooling...", {include-group = "runtime"} ]

[tool.uv]
default-groups = ["dev"]   # bare `uv sync` => base + dev + runtime = everything
```

Resulting install matrix (all governed by one `uv.lock` — groups resolve
together, so no per-job pin drift):

| Caller                        | Command                                     | Gets                                  |
| ----------------------------- | ------------------------------------------- | ------------------------------------- |
| Developer / full CI / Docker  | `uv sync` (or `--group dev`, unchanged)     | project + base + everything           |
| **Lite CI** (`validate_spec`) | `uv sync --frozen --no-default-groups`      | project (importable) + lite base only |
| Torch-only job (hypothetical) | `uv sync --no-default-groups --group torch` | project + base + torch                |

The lite jobs switch from `setup-python` + `pip install --no-deps -e .` to
`astral-sh/setup-uv` + `uv sync --frozen --no-default-groups` — consistent with
every other workflow in the repo, which already uses `uv sync --frozen`.

The exact group partition above is a **first cut**; the only hard constraint is
that `[project.dependencies]` equals the lite-entrypoint closure (§6). Grouping
of the *heavy* deps is for readability and future per-job slicing — it does not
affect correctness as long as `runtime` aggregates all of them.

## 4. Torch Index Routing

This is the one non-mechanical part. Today torch is index-routed per platform:

```toml
[project.optional-dependencies]
cpu   = ["torch>=2.0.0", "torchvision...", "torchaudio..."]   # extras that
cu128 = ["torch>=2.0.0", "torchvision...", "torchaudio..."]   # trigger routing

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "cpu",   marker = "sys_platform == 'linux' or ..." },
  { index = "pytorch-cu128", extra = "cu128", marker = "sys_platform == 'linux' or ..." },
]
```

The `cpu`/`cu128` **extras stay** — they are the legitimate, mutually-exclusive
backend *switches* (`[tool.uv].conflicts` requires ≥2). What changes is that
the bare `torch>=2.0.0` requirement leaves `[project.dependencies]` and lands in
the `torch` **group**. The macOS path — which today relies on torch being in
`[project.dependencies]` so bare `uv sync` pulls the PyPI MPS wheel — keeps
working because `default-groups = ["dev"]` ⊇ `runtime` ⊇ `torch`, so bare
`uv sync` still requires torch and resolves it from PyPI when neither backend
extra is active.

`[tool.uv.sources]` supports conditioning a source on a **group** as well as an
extra (verified: `{ index = "...", group = "torch" }` resolves and routes
correctly on uv 0.11). The migration keeps the `extra = "cpu"` / `extra = "cu128"` source rows unchanged — they continue to fire whenever the backend
extra is passed, regardless of whether torch is required via the base or the
`torch` group. No source-row change is strictly required; this is called out so
the implementer validates the lockfile resolves identically for all three
platforms (Linux-cpu, Linux-cu128, macOS) before/after.

## 5. Migration Plan

1. **`pyproject.toml`**: move heavy deps from `[project.dependencies]` into the
   groups in §3; trim the base to the lite closure; add `default-groups = ["dev"]`; fold `runtime` into `dev`. Drop the now-redundant
   `[project.optional-dependencies].torch` aggregation if nothing else consumes
   it (it is referenced by `all`, Makefile, `environment.yaml`,
   `sync_worker_checkout.sh` — audit each; keep or repoint).
2. **`uv lock`**: regenerate; diff the lockfile to confirm the resolved set is
   unchanged for the full install on all three platforms.
3. **Full-install sites (~15)**: those already pass `--no-default-groups --group dev`; since `dev ⊇ runtime`, they are **unchanged**. Verify by grep, don't
   assume.
4. **Lite sites (3 workflows)**: replace the `pip install --no-deps -e .` +
   hand-pick steps with `astral-sh/setup-uv` + `uv sync --frozen --no-default-groups`, and keep the import smoke-guard step.
5. **Non-uv callsites** (Makefile, `environment.yaml`,
   `scripts/sync_worker_checkout.sh` — the `pip install` paths that can't honor
   `[tool.uv.sources]`): repoint from `.[torch,dev]` to whatever still resolves
   the heavy set; these are the trickiest and must be checked individually.

## 6. Drift Guard

A test pins `[project.dependencies]` to the empirically-determined lite closure
and asserts it stays light, so the base can't silently grow back into a heavy
install (and the lite job can't silently start needing a group):

- `set(project.dependencies names) == {"pydantic", "python-dotenv"}` — the
  closure of `validate_spec.main` + `r2_io.ensure_r2_env_loaded`.
- base excludes `{torch, hydra, skypilot, librosa, h5py, omegaconf, …}`.
- each lite workflow runs the import smoke-guard
  `python -c "from synth_setter.pipeline.ci.validate_spec import main; from synth_setter.pipeline.r2_io import ensure_r2_env_loaded"` after install.

The guard belongs in `tests/infra/`. Unlike the superseded approach, there is
**no workflow-list-vs-extra mirror to police** — `uv sync --no-default-groups`
derives the base from `pyproject.toml` directly, so the only invariant is
"base == closure," checked in one place.

## 7. Alternatives Considered

| Approach                                                                                                       | Why rejected                                                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pip install -e .[pipeline-ci]` extra (the original #1139 AC)                                                  | PEP 621 extras are **additive** — installs the full heavy base *plus* the extra. Cannot produce a subset. Mechanically impossible.                                                    |
| Keep `--no-deps`, add a canonical extra + test-pin the three workflows' hand-picked lists to it + smoke guards | Works, but **retains the `--no-deps` hack** the issue title calls out, and the mirror (extra ↔ three pip lines) is a smell needing its own test. Strictly worse than moving the deps. |
| `uv sync --only-group lite`                                                                                    | `--only-group` **omits the project package** (verified: `ModuleNotFoundError`). `validate_spec` can't import `synth_setter`.                                                          |
| Put `src/` on `PYTHONPATH` without installing                                                                  | Phase 3 made `synth_setter` import-only-when-installed (src layout); fragile and re-introduces a bespoke path hack.                                                                   |

The chosen design is the only one that satisfies all four goals: no `--no-deps`,
single source of truth, fail-fast, and bare `uv sync` unchanged.

## 8. Risks & Rollout

- **A full-install site silently goes light.** Mitigation: step 3 verifies each
  site keeps `--group dev` (⊇ runtime); the import smoke-guard only protects the
  lite path, so a too-light *full* env would surface as a normal import error in
  that job's real work. Acceptable — it fails in-job, not silently.
- **Lockfile churn / per-platform torch.** Mitigation: step 2 diffs the resolved
  set for Linux-cpu, Linux-cu128, and macOS before merging; CI's existing
  `test`, `cpu-slow`, `test-mps`, and `docker-build-validation` jobs are the
  acceptance gate.
- **Non-uv `pip install .[torch,dev]` callsites.** Highest-touch item (step 5);
  each is audited individually rather than blanket-edited.

**Rollout**: single PR (pyproject + lockfile + 3 lite workflows + drift test +
any non-uv callsite repoints), gated on the full existing CI matrix going green.
The interim lazy-import fix (#1395) is already merged/independent and needs no
revert.
