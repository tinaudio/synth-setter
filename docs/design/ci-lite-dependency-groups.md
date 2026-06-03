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
| 2   | [Lite Entrypoints & Closure](#2-lite-entrypoints--closure) | The 5 lite sites and their union closure             |
| 3   | [Goals & Non-Goals](#3-goals--non-goals)              | What this change must and must not do                     |
| 4   | [Solution](#4-solution)                               | Light base + PEP 735 groups + `default-groups`            |
| 5   | [Torch Index Routing](#5-torch-index-routing)         | Migrating the `cpu`/`cu128` routing off `[project]`       |
| 6   | [Migration Plan](#6-migration-plan)                   | Per-callsite changes and ordering                         |
| 7   | [Drift Guard](#7-drift-guard)                         | The test that keeps the base honest                       |
| 8   | [Alternatives Considered](#8-alternatives-considered) | Why extras and `--only-group` don't work here             |
| 9   | [Risks & Rollout](#9-risks--rollout)                  | Failure modes and validation                              |

______________________________________________________________________

## 1. Context & Problem

A handful of CI jobs import `synth_setter.pipeline.*` to do cheap structural
work — chiefly `Validate spec structure`, which runs
`python -m synth_setter.pipeline.ci.validate_spec` on the runner host (outside
the Docker image). These jobs must **not** install the full runtime
(`torch`, `skypilot`, `librosa`, `h5py`, `dask`, `pedalboard`, …) — that is
minutes of wheel downloads for a job that only needs to parse a Pydantic model.

The current workaround installs the project `--no-deps` and hand-picks the
import closure back on top, e.g. `validate-dataset-shards.yaml`:

```bash
pip install --no-deps -e .
pip install "pydantic>=2,<3" python-dotenv   # hand-picked, per workflow
```

This is fragile. The hand-picked list is a per-workflow guess at an entrypoint's
import closure. **Any new module-level import on that path breaks the job
silently** until CI fails:

- #1120 added `from dotenv import dotenv_values` to `r2_io.py` → broke the job
  on #1129 / #1135.
- #1390 added `from omegaconf import DictConfig, OmegaConf` to
  `schemas/spec.py` → broke the scheduled *Test Dataset Generation* run with
  `ModuleNotFoundError: No module named 'omegaconf'`. (Unblocked by #1395, which
  makes that import lazy — but that only kicks the can.)

The lists have also **drifted under-spec**: `spec-materialization.yml` and
`test-spec-materialization.yml` install only `pydantic`, omitting the
`python-dotenv` that `validate_spec`'s `r2_io` import actually requires — a
latent break waiting on a code path that loads it.

**Why the `--no-deps` is currently forced.** `pip`/`uv` always install a
project's `[project.dependencies]` whenever they install the project itself.
Today those dependencies *are* the full heavy runtime, so the only way to get a
light env with `synth_setter` importable is `--no-deps` + a hand-picked add-back.
An extra (`pip install -e .[lite]`) cannot help: **PEP 621 extras are
additive** — `.[lite]` installs the full base *plus* the extra, never a subset.

## 2. Lite Entrypoints & Closure

There are **five** lite CI sites, exercising **three** distinct import closures
(all empirically confirmed against the tree, post-#1395):

| Site (workflow)                  | Entrypoint                               | Import closure          |
| -------------------------------- | ---------------------------------------- | ----------------------- |
| `validate-dataset-shards.yaml`   | `pipeline.ci.validate_spec`              | `pydantic`, `python-dotenv` |
| `spec-materialization.yml`       | `pipeline.ci.validate_spec`              | `pydantic`, `python-dotenv` |
| `test-spec-materialization.yml`  | `pipeline.ci.validate_spec`              | `pydantic`, `python-dotenv` |
| `r2-auth-probe.yaml`             | `pipeline.r2_io.ensure_r2_env_loaded`    | `python-dotenv`         |
| `docker-build-validation.yml`    | `pipeline.ci.load_image_config`          | `pydantic`, `pyyaml`    |

`validate_spec` reaches `python-dotenv` transitively (`validate_spec → spec_io →
r2_io`, which does `from dotenv import dotenv_values`). `load_image_config`
parses YAML via `pyyaml`. The **union closure** — the single light base that
serves all five sites — is therefore:

```
pydantic, python-dotenv, pyyaml   (+ transitives: pydantic-core, annotated-types, typing-extensions)
```

Note `spec_uri` (the `synth-setter-spec-uri` entrypoint, used by
`test-dataset-generation.yml`) imports `hydra`, so it is **not** a lite
entrypoint and stays on the full env.

## 3. Goals & Non-Goals

**Goals**

- Lite CI jobs install `synth_setter` + only the union light base, with **no
  `--no-deps` hack** and **no hand-picked dep list**.
- A **single source of truth** for the lite base; impossible to drift silently
  (and fixes the existing under-spec in two workflows).
- A new top-level import on a lite path **fails fast** (a guard step), not
  silently at the next unrelated CI run.
- `uv sync` with no flags still installs **everything** — zero change to local
  developer onboarding or to the ~15 full-runtime install sites.

**Non-Goals**

- Re-architecting the torch backend selection model (`cpu`/`cu128`); we migrate
  its *location* only, preserving behavior.
- Splitting dev tooling further (test/lint/typecheck groups). Out of scope.
- Changing what the Docker image or GPU/CPU test matrix installs.

## 4. Solution

Move the heavy runtime out of `[project.dependencies]` into PEP 735
`[dependency-groups]`, leaving the base as just the union light closure.
Aggregate the heavy groups under a `runtime` meta-group, fold `runtime` into
`dev`, and point `default-groups` at `dev` so a **bare `uv sync` still installs
everything**.

```toml
[project]
dependencies = [
  "pydantic>=2",
  "python-dotenv",
  "pyyaml",
]   # the union lite closure — pinned by a test (§7)

[dependency-groups]
torch   = ["torch>=2.0.0", "torchvision>=0.15.0", "torchaudio>=2.0.0",
           "lightning>=2.6.0", "torchmetrics>=0.11.4"]
config  = ["hydra-core==1.3.2", "hydra-colorlog==1.2.0", "hydra-optuna-sweeper==1.2.0",
           "omegaconf", "pydantic-settings>=2", "tomli; python_version < '3.11'"]
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

**Key mechanic — groups are not pip-installable.** PEP 735 dependency-groups are
*not* part of the package's distribution metadata, so `pip install -e .` /
`uv pip install -e .` install **only** `[project.dependencies]` (the light
base), never a group. That means lite sites don't need a new install model —
once the base is light, they simply **drop `--no-deps` and the hand-picked
line**:

| Caller                          | Command (after)                       | Gets                                  |
| ------------------------------- | ------------------------------------- | ------------------------------------- |
| Developer / full CI / Docker    | `uv sync` (or `--group dev`, unchanged) | project + base + everything         |
| **Lite CI** (any of the 5)      | `pip install -e .` / `uv sync --no-default-groups` | project (importable) + light base only |
| Torch-only job (hypothetical)   | `uv sync --no-default-groups --group torch` | project + base + torch              |

All installs are governed by one `uv.lock` (groups resolve together, so no
per-job pin drift). The exact group partition above is a **first cut**; the only
hard constraint is that `[project.dependencies]` equals the union lite closure
(§7). Grouping of the *heavy* deps is for readability and future per-job
slicing — it does not affect correctness as long as `runtime` aggregates all of
them.

## 5. Torch Index Routing

This is the one non-mechanical part. Today torch is index-routed per platform:

```toml
[project.optional-dependencies]
cpu   = ["torch>=2.0.0", "torchvision...", "torchaudio..."]   # extras that
cu128 = ["torch>=2.0.0", "torchvision...", "torchaudio..."]   # trigger routing

[tool.uv.sources]
# three identical blocks — one each for torch, torchvision, torchaudio
torch = [
  { index = "pytorch-cpu",   extra = "cpu",   marker = "sys_platform == 'linux' or ..." },
  { index = "pytorch-cu128", extra = "cu128", marker = "sys_platform == 'linux' or ..." },
]
torchvision = [ ...same... ]
torchaudio  = [ ...same... ]
```

The `cpu`/`cu128` **extras stay** — they are the legitimate, mutually-exclusive
backend *switches* (`[tool.uv].conflicts` requires ≥2). What changes is that
the bare `torch`/`torchvision`/`torchaudio` requirements leave
`[project.dependencies]` and land in the `torch` **group**. The macOS path —
which today relies on torch being in `[project.dependencies]` so bare `uv sync`
pulls the PyPI MPS wheel — keeps working because `default-groups = ["dev"]` ⊇
`runtime` ⊇ `torch`, so bare `uv sync` still requires torch and resolves it from
PyPI when neither backend extra is active.

`[tool.uv.sources]` supports conditioning a source on a **group** as well as an
extra (verified: `{ index = "...", group = "torch" }` resolves and routes
correctly on uv 0.11.2). The migration keeps the `extra = "cpu"` / `extra =
"cu128"` source rows unchanged — they continue to fire whenever the backend
extra is passed, regardless of whether torch is required via the base or the
`torch` group. **All three source blocks (torch, torchvision, torchaudio) move
together; editing only `torch` would split the lockfile.** No source-row change
is strictly required; this is called out so the implementer validates the
lockfile resolves identically for all three platforms before/after.

**The backend extra flag is load-bearing and unchanged.** Every full-install
site invokes `uv sync --frozen --extra cpu --no-default-groups --group dev`
(or `--extra cu128`). After this change those sites are unchanged *because* (a)
`dev ⊇ runtime ⊇ torch` now supplies the torch requirement, and (b) `--extra
cpu` still selects the CPU index via the unchanged source rows. Both halves must
remain — the migration must not drop `--extra cpu`/`--extra cu128` from any site.

## 6. Migration Plan

1. **`pyproject.toml`**: move heavy deps from `[project.dependencies]` into the
   groups in §4; trim the base to the union closure
   (`pydantic`, `python-dotenv`, `pyyaml`); add `default-groups = ["dev"]`; fold
   `runtime` into `dev`. Audit `[project.optional-dependencies].torch` /
   `all` — keep or repoint per their consumers (see step 5).
2. **`uv lock`**: regenerate; diff the lockfile to confirm the resolved set is
   unchanged for the full install on **all three platforms** (Linux-cpu,
   Linux-cu128, macOS).
3. **Full-install sites (~15)**: already pass `--extra <backend>
   --no-default-groups --group dev`; since `dev ⊇ runtime`, they are
   **unchanged**. Verify by grep that none drop the backend extra — don't assume.
4. **Lite sites (5)**: `validate-dataset-shards.yaml`, `spec-materialization.yml`,
   `test-spec-materialization.yml`, `r2-auth-probe.yaml`, and
   `docker-build-validation.yml`. Replace each `pip install --no-deps -e .` +
   hand-pick pair with a single `pip install -e .` (or
   `uv pip install --system -e .` where the workflow already uses `uv pip`;
   `docker-build-validation.yml` uses `--system`). Keep/add the import
   smoke-guard step. The light base now covers all three closures uniformly,
   eliminating the existing under-spec.
5. **Non-uv `pip install .[…]` callsites** (cannot honor `[tool.uv.sources]`):
   `Makefile` (`.[torch,dev]`), `scripts/sync_worker_checkout.sh` (`.[torch]`),
   `environment.yaml`. Audit each individually and repoint to whatever still
   resolves the heavy set after the move.

## 7. Drift Guard

A test in `tests/infra/` pins `[project.dependencies]` to the empirically-derived
union closure and asserts it stays light, so the base can't silently grow back
into a heavy install (and a lite job can't silently start needing a group):

- `set(project.dependencies names) == {"pydantic", "python-dotenv", "pyyaml"}` —
  the union closure of `validate_spec.main`, `r2_io.ensure_r2_env_loaded`, and
  `load_image_config.main`.
- base excludes `{torch, hydra, skypilot, librosa, h5py, omegaconf, …}`.
- each lite workflow runs an import smoke-guard for *its* entrypoint after
  install, e.g.
  `python -c "from synth_setter.pipeline.ci.validate_spec import main"` /
  `... import load_image_config` / `... r2_io import ensure_r2_env_loaded`.

Unlike the superseded extra-mirror approach, there is **no
workflow-list-vs-extra mirror to police** — `pip install -e .` derives the base
from `pyproject.toml` directly, so the only invariant is "base == union
closure," checked in one place.

## 8. Alternatives Considered

| Approach                                            | Why rejected                                                                                                                    |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `pip install -e .[pipeline-ci]` extra (original #1139 AC) | PEP 621 extras are **additive** — installs the full heavy base *plus* the extra. Cannot produce a subset. Mechanically impossible. |
| Keep `--no-deps`, add a canonical extra + test-pin each workflow's hand-picked list to it + smoke guards | Works, but **retains the `--no-deps` hack** the issue title calls out, and the mirror (extra ↔ N pip lines) is a smell needing its own test. Strictly worse than moving the deps. |
| `uv sync --only-group lite`                         | `--only-group` **omits the project package** (verified: `ModuleNotFoundError`). Lite entrypoints can't import `synth_setter`.    |
| Put `src/` on `PYTHONPATH` without installing       | Phase 3 made `synth_setter` import-only-when-installed (src layout); fragile and re-introduces a bespoke path hack.              |

The chosen design is the only one that satisfies all four goals: no `--no-deps`,
single source of truth, fail-fast, and bare `uv sync` unchanged.

## 9. Risks & Rollout

- **A full-install site silently goes light.** Mitigation: step 3 verifies each
  site keeps `--group dev` (⊇ runtime) *and* its backend extra; a too-light
  *full* env surfaces as a normal import error in that job's real work, not
  silently. Acceptable — it fails in-job.
- **Lockfile churn / per-platform torch.** Mitigation: step 2 diffs the resolved
  set for Linux-cpu, Linux-cu128, and macOS before merging; the existing `test`,
  `cpu-slow`, `test-mps`, and `docker-build-validation` jobs are the acceptance
  gate.
- **`load_image_config` needs `pyyaml`, not just the `validate_spec` closure.**
  Resolved by basing on the **union** closure (§2), so `pyyaml` is in the light
  base and `docker-build-validation.yml` is covered without a bespoke install.
- **A lite site is missed** (e.g. `r2-auth-probe.yaml`, which the first scope
  pass undercounted). Mitigation: §2 enumerates all five from a tree-wide grep of
  the `--no-deps` + hand-pick pattern; the migration PR must convert all five or
  it leaves the hack in place.
- **Non-uv `pip install .[…]` callsites.** Highest-touch item (step 5); each is
  audited individually rather than blanket-edited.

**Rollout**: single PR (pyproject + lockfile + 5 lite workflows + drift test +
any non-uv callsite repoints), gated on the full existing CI matrix going green.
The interim lazy-import fix (#1395) is already independent and needs no revert.
