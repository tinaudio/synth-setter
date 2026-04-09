# CHANGELOG


## v0.2.0 (2026-04-09)

### Chores

- Add .devcontainer config for GitHub Codespaces
  ([#498](https://github.com/tinaudio/synth-setter/pull/498),
  [`baa51de`](https://github.com/tinaudio/synth-setter/commit/baa51defaa0cad36d75da462e1a61380ad389ff5))

* chore: add .devcontainer config for GitHub Codespaces

Reuses the existing tinaudio/perm:dev-snapshot image (same as RunPod) so VST-dependent tests,
  generate_dataset -> R2 uploads, and CPU training work identically in a Codespace. GPU training
  still runs on RunPod.

The post-create.sh installs only the editable workspace (--no-deps) since deps are already baked
  into the image, then initializes submodules and wires up pre-commit hooks.

Uses the image's MODE=idle entrypoint path instead of overrideCommand -- the image's own API, not a
  bypass.

Closes #186

* chore: mark workspace as safe.directory in post-create

Codespaces runs post-create.sh as root against a workspace whose files may be owned by another UID,
  which trips git's safe.directory protection (CVE-2022-24765 mitigation) and blocks git submodule
  update and pre-commit install.

Addresses Copilot review comment on #498.

- Add authors to pyproject.toml ([#483](https://github.com/tinaudio/synth-setter/pull/483),
  [`7219f83`](https://github.com/tinaudio/synth-setter/commit/7219f8319ffd2734cb38f09a2148684207d03fc4))

- Add GPLv3 LICENSE file ([#470](https://github.com/tinaudio/synth-setter/pull/470),
  [`2178c90`](https://github.com/tinaudio/synth-setter/commit/2178c90d50787bdce4a7a5086977c07cae211af2))

- Rewrite .env.example with complete variable inventory
  ([#471](https://github.com/tinaudio/synth-setter/pull/471),
  [`5ec6527`](https://github.com/tinaudio/synth-setter/commit/5ec652780e6825e85d4a531a2d19cc636c138139))

* chore: rewrite .env.example with complete variable inventory

* address review feedback on PR #471

* chore: address second round of review feedback on PR #471

- Add optional logger tokens (COMET_API_TOKEN, NEPTUNE_API_TOKEN) in a dedicated section, noting
  they are template-provided and not actively used - Document that PROJECT_ROOT is auto-set by
  rootutils (not a .env variable) - Flip AWS_ENDPOINT_URL to commented-out with a warning about
  global scope; promote WANDB_S3_ENDPOINT_URL as the preferred override

* chore: clarify RUNPOD_API_KEY is planned, comment it out

Mark the RunPod section as planned (#71) and comment out the variable since no code paths currently
  reference it. Addresses review feedback.

- Switch skills submodule URL from SSH to HTTPS
  ([#474](https://github.com/tinaudio/synth-setter/pull/474),
  [`9976cde`](https://github.com/tinaudio/synth-setter/commit/9976cde3a280a7c6b7c12638d6bda60b3118990e))

- **code-health**: Label good-first-issue starter set and link from CONTRIBUTING
  ([#511](https://github.com/tinaudio/synth-setter/pull/511),
  [`f701439`](https://github.com/tinaudio/synth-setter/commit/f7014393f277182b5a87a1b4bae3218069be0316))

Label three curated starter issues and add a "Good first issues" section to CONTRIBUTING.md linking
  to the label filter, so newcomers have a clear entry point after running through Getting started.

Labeled: - #33 (docs/logging: add debug log of resolved config in train()) - #38 (tests: replace
  deprecated pkg_resources with importlib.metadata) - #51 (typing: resolve inconsistencies flagged
  by Copilot in PR #49)

Closes #464 Part of #457

- **docker**: Migrate from pip to uv with --torch-backend
  ([#484](https://github.com/tinaudio/synth-setter/pull/484),
  [`a691b33`](https://github.com/tinaudio/synth-setter/commit/a691b335e2cc80046e4ea11fa38a3e2dcc83226c))

* chore(docker): replace pip with uv pip in Dockerfile and remaining refs

Migrate all pip install commands to uv pip install across Dockerfile, CI workflows, Makefile, and
  docs. Keeps pip wheel for building wheels (uv doesn't support pip wheel yet). Installs uv via COPY
  from the official ghcr.io/astral-sh/uv image and uses uv venv instead of pip+virtualenv for venv
  creation.

Closes #424

* Update docker/ubuntu22_04/Dockerfile

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* fix(make): revert uv install to pip install uv

The curl | sh approach installs uv to ~/.local/bin which may not be on PATH in a non-interactive
  Make recipe, causing uv: command not found on the next line. pip install uv avoids this entirely
  since it installs into the active environment's bin/ already on PATH.

Refs #484

* fix(docker): use --no-deps for offline wheel install to bypass uv local version bug

uv's strict PEP 440 resolver rejects torch wheels with +cu128 local version suffixes (torchvision
  requires torch==2.11.0 but the wheel is torch-2.11.0+cu128). Since pip wheel already resolved all
  dependencies correctly, install every built wheel directly with --no-deps to skip uv's resolver
  entirely.

* chore(docker): replace pip wheel stages with single uv pip install

Delete the wheels-torch and wheels stages entirely. uv resolves and installs all deps in one pass
  from PyPI + the PyTorch CUDA index, using its built-in cache (mounted at /root/.cache/uv) for
  cross-build persistence. This eliminates both the duplicate transitive dep problem (fsspec) and
  the +cu128 local version mismatch.

* fix(docker): add --index-strategy unsafe-best-match for PyTorch index

The PyTorch CUDA index carries stale copies of common packages (e.g. requests 2.28.1). uv's default
  index strategy only considers the first index that has a package, blocking deps that need newer
  versions from PyPI. unsafe-best-match picks the best version across all indexes.

Verified via dry-run: uv resolves torch==2.11.0+cu128 + all transitive deps cleanly with this flag.

* chore(docker): use uv --torch-backend, bump uv to 0.11.2, drop pip

Replace --extra-index-url + --index-strategy unsafe-best-match with uv's native --torch-backend
  cu128 flag. uv routes only PyTorch ecosystem packages to the CUDA index automatically — no index
  mixing, no unsafe flags.

Changes across the config/CI/Makefile/docs chain: - TORCH_INDEX_URL build arg → TORCH_BACKEND
  (default: cu128) - DOCKER_TORCH_IDX Makefile var → DOCKER_TORCH_BACKEND - torch_index_url in image
  config → torch_backend - Bump uv from 0.4.29 to 0.11.2 (--torch-backend requires ≥0.5) - Drop
  python3-pip from Dockerfile (nothing uses pip anymore) - Drop PIP_DISABLE_PIP_VERSION_CHECK env
  var

* fix(test): update newline injection fixtures to use torch_backend

The TestNewlineInjection tests added on main use inline YAML fixtures with the old torch_index_url
  field. Update to torch_backend to match the renamed ImageConfig schema.

* chore(docker): update stale comment referencing removed wheel stages

---------

- **plumb**: Dedicated spec file with 124 requirements across 15 sections
  ([#468](https://github.com/tinaudio/synth-setter/pull/468),
  [`ea082b9`](https://github.com/tinaudio/synth-setter/commit/ea082b9807dc6383b190596152a4e3a84e289029))

* chore: switch plumb spec to dedicated file and seed cross-cutting invariants

- Bump plumb-dev pin to f4160bb (tip of dev branch) - Change spec_paths from design docs to
  plumb_spec.md - Seed spec with 10 cross-cutting invariants from design docs

Closes #467

* chore(plumb): expand spec with domain requirements and move to docs/

Move plumb_spec.md to docs/plumb_spec.md and expand from 10 cross-cutting invariants to 124
  requirements across 15 sections.

New sections: Reconciliation and Resumability, Model Promotion, Artifact Provenance, Concurrency and
  Crash Resilience, CLI Interface.

Expanded sections: Audio Dataset Generation (lifecycle markers, quarantine, HDF5 contents), Shard
  Validation (content hashing, completeness checks), Dataset Finalization (normalization stats,
  virtual datasets, WebDataset archives, dataset card), Model Training (checkpoint intervals,
  provenance), Model Evaluation (specific metrics, denormalization, conditional R2 download),
  Container Environment (MODE specifics, credential fallback), Storage Layout (R2 path conventions,
  S3-compatible references).

All requirements are atomic, testable, active voice, and abstracted from implementation details per
  plumb_spec formatting rules.

Part of #466 Closes #467

### Continuous Integration

- Add experiments domain to taxonomy ([#493](https://github.com/tinaudio/synth-setter/pull/493),
  [`3729c32`](https://github.com/tinaudio/synth-setter/commit/3729c3250c7b2742ae7d12c1ceb2eebb80abc275))

* feat(ci): add experiments domain to taxonomy

Register a new `experiments` domain for one-off validation experiments (result replication, baseline
  parity, render variability benchmarks). Distinct from `evaluation` (production eval pipeline) and
  `testing` (unit/integration test infra).

Updates all 8 synth-setter locations that enumerate domains: - CI gate DOMAIN_LABELS
  (pr-metadata-gate.yaml) - PostToolUse hook DOMAIN_LABELS (verify-gh-taxonomy.sh) - All 5 issue
  template domain+milestone dropdowns - docs/design/github-taxonomy.md §6 labels, §7 milestones, §8
  views

Fixes pre-existing drift found while touching these files: the `testing v1.0.0` milestone was
  missing from all 5 template milestone dropdowns and §7 milestones table (the milestone existed on
  GitHub but couldn't be selected from templates); §8 views was missing rows for `monitoring` and
  `testing`. All 8 files now list the same 10 domains.

The companion skill update (SKILL.md in tinaudio/skills) is in tinaudio/skills#53. Submodule pointer
  bump will follow that PR's merge.

Refs #492

* chore(skills): bump submodule to pick up experiments domain

Picks up tinaudio/skills#53 which adds `experiments` to the github-taxonomy skill's 3 domain
  enumerations and fixes pre-existing skill drift (previously missing `documentation` and
  `monitoring`).

After this bump, the skill's domain list matches the 10-domain set enforced by pr-metadata-gate.yaml
  and verify-gh-taxonomy.sh in this same PR.

- Validate GITHUB_OUTPUT values against newline injection
  ([#473](https://github.com/tinaudio/synth-setter/pull/473),
  [`593c615`](https://github.com/tinaudio/synth-setter/commit/593c615c95cb4868cb23a99cfd884c8e2eff9111))

* fix(ci): validate GITHUB_OUTPUT values against newline injection

Values written to GITHUB_OUTPUT are now checked for newline (\n) and carriage-return (\r) characters
  before writing. A config value containing either character would previously inject arbitrary
  key-value pairs into the Actions output file. The fix raises ValueError with a clear message when
  a newline is detected.

Fixes #333

* fix: clarify error message for newline and carriage-return injection

- **devcontainer**: Derive workspaceFolder from host directory name
  ([#502](https://github.com/tinaudio/synth-setter/pull/502),
  [`02bb634`](https://github.com/tinaudio/synth-setter/commit/02bb63483b3e2dfdf8c99b1e55455980f419460b))

* fix(devcontainer): derive workspaceFolder from host directory name

The devcontainer config hardcoded workspaceFolder to /workspaces/synth-setter and post-create.sh
  hardcoded cd to the same path. This only worked in GitHub Codespaces (repo always cloned to
  /workspaces/<repo-name>) and in local clones literally named "synth-setter". Forks cloned to
  custom directory names and git worktrees failed with chdir exit 127 in postCreateCommand.

Changes:

- devcontainer.json: workspaceFolder uses ${localWorkspaceFolderBasename} substitution, matching the
  devcontainer CLI's mount-target derivation. - post-create.sh: walks up from the script's location
  to find the .project-root anchor (the project's existing rootutils convention), instead of
  hardcoding the path. - post-create.sh: defensively unsets core.hooksPath before pre-commit
  install, stripping any absolute host-path that may leak from the host .git/config (harmless in
  Codespaces; breaks local devcontainer users who ran pre-commit install on the host).

Also documents the supported local-devcontainer workflow in docs/getting-started.md §2g: open the
  container on the main working tree and create git worktrees *inside* the container. Mounting a
  worktree directly from the host does not work because the worktree's .git file points to a host
  path outside the container's bind mount.

Closes #186

* style(docs): add blank line before HR in credential rotation guide

mdformat requires a blank line between a heading and a horizontal rule separator. Without it, the
  rendered output collapses the `______` line visually and mdformat rejects the file. Failing the
  `Code Quality Main` workflow on main since baa51de (#498 merge).

Found by pre-commit run --all-files.

* address review feedback on PR #502

Copilot review round:

- post-create.sh header: reframe as "Dev container first-run setup for both Codespaces and local
  devcontainers" so the file header matches reality after this PR (comment #3037279752). -
  post-create.sh final echo: change "Codespace ready" to "Dev container ready" so local users aren't
  misled (comment #3037279754). - post-create.sh hooksPath unset: add explicit --local scope so the
  command can never touch the global git config even if cwd drifts (comment #3037291219). -
  post-create.sh .project-root error: include the search-start path and a remediation hint so users
  who opened the container on a subdirectory can self-diagnose (comment #3037291225). -
  docs/getting-started.md §2g caveat: clarify that exporting GITHUB_TOKEN alone is not sufficient —
  git needs a credential helper configured (e.g., gh auth login && gh auth setup-git, or a
  PAT-backed credential store) (comment #3037279746).

### Documentation

- Add CITATION.cff for project citation metadata
  ([#475](https://github.com/tinaudio/synth-setter/pull/475),
  [`7c5e439`](https://github.com/tinaudio/synth-setter/commit/7c5e4393449684b581f0b7043727cf4662f7cd96))

* docs: add CITATION.cff for project citation metadata

* docs: add missing affiliation for Khaled Tinubu in CITATION.cff

Add Google affiliation to both the top-level authors list and the preferred-citation authors block,
  matching the existing affiliation pattern used for the first author.

* docs: simplify CITATION.cff to self-citation only

- Add CONTRIBUTING.md with contributor onboarding guide
  ([#477](https://github.com/tinaudio/synth-setter/pull/477),
  [`7c7c252`](https://github.com/tinaudio/synth-setter/commit/7c7c2523e2d72858ab611d74de68dab905eb8926))

* docs: add CONTRIBUTING.md with contributor onboarding guide

* docs: address review feedback on CONTRIBUTING.md

- Add HTTPS override instructions for SSH submodule URL (comment 3025900308) - Document where
  PLUMB_SKIP is implemented (comment 3025900314) - Remove broken CODE_OF_CONDUCT.md and LICENSE
  links (comment 3025900327)

* docs: clarify structlog as intended standard for new pipeline code

* docs: add bats to prerequisites and note requirement for test-bats

* Update CONTRIBUTING.md

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- Add credential rotation operational runbook
  ([#478](https://github.com/tinaudio/synth-setter/pull/478),
  [`5e9d20d`](https://github.com/tinaudio/synth-setter/commit/5e9d20d2c7d5598aafc41cfbf789fd025019c612))

* docs: add credential rotation operational runbook

* docs: address PR #478 review feedback on credential rotation guide

- Fix R2_ENDPOINT storage location: sourced from image config YAML (configs/image/dev-snapshot.yaml)
  via pipeline.ci.load_image_config, not from GitHub Secrets - Simplify R2 endpoint/bucket note per
  reviewer suggestion: remove undefined R2_BUCKET reference, clarify both are non-secret config

* docs: address second round of review feedback on PR #478

- Fix claude-review.yml trigger: runs on needs-claude-review label, not PR open - Replace hardcoded
  RunPod API key with $RUNPOD_API_KEY env var reference - Replace hardcoded Docker Hub token with
  $DOCKERHUB_TOKEN env var reference

* docs: address round 3 review feedback on PR #478

- Clarify r2_endpoint YAML key name in inventory table and R2 Endpoint section - Replace inline
  WANDB_API_KEY placeholder with env var reference - Replace inline GIT_PAT values with env var
  references in verification and rebuild examples

* Revise credential rotation steps and add issue creation

Updated credential rotation guide to simplify instructions and add a step for creating a GitHub
  issue.

- Add getting-started tutorial for new contributors
  ([#481](https://github.com/tinaudio/synth-setter/pull/481),
  [`864ac9f`](https://github.com/tinaudio/synth-setter/commit/864ac9f2364047448b5829ea529aa76d3a7da32d))

* docs: add getting-started tutorial for new contributors

* docs: fix Hydra override syntax and trainer config in tutorial

- Use experiment= (not +experiment=) since train.yaml already defines experiment: null - Use
  trainer.max_steps (not trainer.max_epochs) matching actual trainer config - Use model.optimizer.lr
  (not model.lr) matching nested optimizer config - mdformat table alignment and ordered list
  normalization

* docs: address review feedback on getting-started tutorial

- Add trainer.min_steps=null override to quickstart command so the run actually stops at 5,000 steps
  (default min_steps is 400,000) - Fix checkpoint path to match Hydra output dir pattern:
  logs/{task_name}/{experiment_name}/{run_name}-{timestamp}/checkpoints/ - Fix rclone verification:
  use lsd (list directories) instead of ls, and correct top-level dirs to data/, train/, eval/ - Fix
  eval section: ckpt_path is required, not optional - Add DOCKER_BUILD_FLAGS=--load to Docker build
  example

* docs: fix submodule SSH note, rclone env vars, and Docker secret handling

- Add note about SSH-based submodule URL and HTTPS workaround for contributors who clone via HTTPS
  without SSH keys configured - Fix rclone env var names to use RCLONE_CONFIG_R2_* prefix for local
  rclone auto-configuration, and clarify that R2_* names are for Docker BuildKit secrets - Replace
  inline Docker build credentials with set -a/source .env pattern to avoid leaking secrets in shell
  history

* docs: add CPU/MPS trainer note and Docker credential warning

Address round 3 review feedback on PR #481: - Add trainer=cpu/mps guidance for the k-osc quickstart
  command - Add warning that Docker images contain credentials in the filesystem

* Update docs/getting-started.md

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- Add GitHub Actions workflow reference ([#504](https://github.com/tinaudio/synth-setter/pull/504),
  [`8906d0e`](https://github.com/tinaudio/synth-setter/commit/8906d0e9e81faa3a07ca7ea727b887d5be784c06))

* docs: add GitHub Actions workflow reference

Adds docs/reference/github-actions.md documenting the 20 workflows in .github/workflows/. Captures
  intent, secret purposes, cross-workflow dependencies, and non-obvious gotchas — not literal YAML
  transcription.

Closes #503

* docs: correct gpu-x64 runner classification

The gpu-x64 runner is a GitHub-hosted larger runner (per the YAML comment "GitHub GPU runner ships
  NVIDIA driver 12080"), not self-hosted. Fix the overview line, catalog row, gotcha section, and
  anchor link.

Refs #503

* docs: note that paths filters affect which workflows run

The original Skip CI line overstated CI coverage: 4 workflows (test, test-dataset-generation,
  bats-tests, docker-build-validation) have paths: filters, so doc-only changes don't trigger the
  full matrix.

- Add glossary and architecture overview ([#482](https://github.com/tinaudio/synth-setter/pull/482),
  [`bae4184`](https://github.com/tinaudio/synth-setter/commit/bae4184028427c8b02141a594dc1e571c9f331ac))

* docs: add glossary and architecture overview

* docs: apply mdformat table formatting

* docs: fix feed-forward model class names in glossary

- Add module docstrings and scripts inventory
  ([#476](https://github.com/tinaudio/synth-setter/pull/476),
  [`7c15ef2`](https://github.com/tinaudio/synth-setter/commit/7c15ef2a31adaf6775f9ae1304484aceef886494))

* docs: add module docstrings and scripts inventory

Add one-line module docstrings to src/, src/data/, src/models/, and pipeline/ __init__.py files.
  Create scripts/README.md with an inventory table covering all 18 scripts and 2 data directories.

Refs #463

* docs: address review feedback on PR #476

- Fix MFCD typo to MFCC in compute_audio_metrics.py description - List actual docker_entrypoint.sh
  modes (idle, passthrough, generate_dataset)

* docs: quote Surge XT path instead of escaping space in scripts README

* docs: exclude pipeline from find_packages in setup.py

The pipeline/ directory has its own __init__.py (predating this PR), so find_packages() was silently
  including it in the installed distribution. Add an explicit exclude so only src/ is packaged.

Refs #476

- Expand README with badges, install guide, and project overview
  ([#479](https://github.com/tinaudio/synth-setter/pull/479),
  [`3debf52`](https://github.com/tinaudio/synth-setter/commit/3debf52cfd897e1a042e1f5645e0dbfee65671a8))

* docs: expand README with badges, install guide, and project overview

* address review feedback on PR #479

* docs: remove non-existent pipeline dirs from project structure

Remove pipeline/stages/ and pipeline/backends/ from the README project structure section -- these
  directories do not exist yet (planned in #72 and #71). Also update PR description to remove
  incorrect license badge claim.

Refs #460

* docs: add Acknowledgments section, fold Publication, update license to GPL-3.0

* docs: add conda activate command and env name hint to README

Refs #479

* Update README.md

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- **ci-automation**: Document 40-char SHA requirement for Docker workflow git_ref
  ([#509](https://github.com/tinaudio/synth-setter/pull/509),
  [`55b0c10`](https://github.com/tinaudio/synth-setter/commit/55b0c107f5fcc46a4ead3ed0041f74f2890cd203))

* docs(ci-automation): document git_ref SHA resolution in Docker workflow

Clarify that the Docker build workflow's git_ref input accepts any ref form but the build always
  pins to the commit SHA resolved by `git rev-parse HEAD` after checkout. Strengthen the Makefile
  GIT_REF doc to note that Makefile targets pass GIT_REF verbatim to `git checkout --detach` after
  only `git fetch origin` and require a full SHA for reliable resolution, unlike the CI workflow
  which resolves any ref form to a SHA before the build.

Workflow-level SHA validation is already provided by load_image_config's Pydantic validator
  (pipeline/schemas/image_config.py), so no new workflow check is added.

Closes #332

* docs: describe both SYNTH_PERMUTATIONS_GIT_REF paths in Makefile

The Dockerfile uses SYNTH_PERMUTATIONS_GIT_REF in two stages: the `synth-setter-src` stage downloads
  a GitHub tarball at that ref, and the `dev-snapshot` stage runs `git checkout --detach` after `git
  fetch origin`. The previous GIT_REF block mentioned only the git-checkout path. Also drops the
  too-strong "branch/tag names may not be fetched" claim — `git fetch origin` fetches all branch
  heads and tags by default; the real constraint is that a full SHA reachable from a pushed
  branch/tag is the only ref form that reliably satisfies both paths.

Addresses Copilot review comment on PR #509.

### Features

- **evaluation**: Add interactive Surge XT preview script
  ([#531](https://github.com/tinaudio/synth-setter/pull/531),
  [`46caaf0`](https://github.com/tinaudio/synth-setter/commit/46caaf0c797602fab1dd9990c8cc1432bdf4d508))

opens Surge XT GUI via pedalboard with real-time audio streaming.

### Testing

- Shrink test_train_ddp_sim to fix limit_val_batches under DDP sharding
  ([#515](https://github.com/tinaudio/synth-setter/pull/515),
  [`bd794a3`](https://github.com/tinaudio/synth-setter/commit/bd794a3982f7784cbff25c1e1e6f54293bd2c7fd))

The fixture's limit_val_batches=0.1 yielded 0 val batches per rank under ddp_spawn with devices=2
  (10 val batches → 5/rank → 0.1 × 5 = 0.5, which Lightning rejects). Override with integer limits
  and shrink model/data/batch to match the tiny-model tests so DDP-on-CPU finishes in ~13s instead
  of consuming several minutes on the full ksin model and dataset.

Closes #46

- **benchmarks**: Initialize HydraConfig before resolving cfg_train
  ([#501](https://github.com/tinaudio/synth-setter/pull/501),
  [`20587b8`](https://github.com/tinaudio/synth-setter/commit/20587b881119e09361db7374ef43bd95cd30c776))

* test(benchmarks): initialize HydraConfig before resolving cfg_train

The test_config_resolution_speed benchmark walks the full cfg_train tree with
  OmegaConf.to_container(resolve=True). The tree contains ${hydra:runtime.cwd} from
  configs/paths/default.yaml (work_dir), whose resolver only works when HydraConfig has been set for
  the config. The cfg_train fixture does not call HydraConfig().set_config(), so the benchmark
  raises during resolution — mirroring test_train_config, call set_config() inside the test.

Refs #500

* test(benchmarks): strip hydra section before resolving cfg_train

Setting HydraConfig makes ${hydra:runtime.*} resolvers work, but the hydra subtree itself contains
  hydra.run.dir = ${run_name}-... which references a key only defined by experiment configs —
  resolve=True on the full tree still fails with InterpolationKeyError. In production Hydra strips
  its own hydra section before handing the config to the user task, so the benchmark now mirrors
  that: pop 'hydra' from a copy and resolve the user-facing subtree.

---------

Co-authored-by: a <a@as-mac-mini.taile31224.ts.net>

- **testing**: Move heavy training tests to GPU runner
  ([#506](https://github.com/tinaudio/synth-setter/pull/506),
  [`c596bea`](https://github.com/tinaudio/synth-setter/commit/c596bea76d6ad3f2d3f19f1cf0fcea221e6c1d74))

The nightly-full-suite runner (ubuntu-latest: 2 vCPU / 7 GB, no GPU) has been killed by GitHub infra
  ("runner lost communication") 17/17 times since the workflow was added 2026-03-21 — the CPU
  train+eval loop in test_train_eval exhausts the runner. Move the 4 heavy slow CPU training tests
  to the GPU runner (test-expensive.yml, twice-weekly):

- test_eval.py::test_train_eval - test_train.py::test_train_epoch_double_val_loop -
  test_train.py::test_train_resume - test_train.py::test_train_fast_dev_run (deleted — identical to
  test_train_fast_dev_run_gpu_compile after GPU migration)

Keep test_train_ddp_sim on CPU (its purpose is to verify ddp_spawn on CPU), and
  test_train_fast_dev_run_tiny_model_tiny_data remains the one CPU training test on every PR.

After this change, nightly-full-suite's CPU training load drops from 6 tests to 2, which should
  allow the runner to complete without being killed.

Refs #505

Co-authored-by: a <a@as-mac-mini.taile31224.ts.net>

- **testing**: Rename pytest.fail msg= kwarg to reason= for pytest 9 compat
  ([#508](https://github.com/tinaudio/synth-setter/pull/508),
  [`d29e250`](https://github.com/tinaudio/synth-setter/commit/d29e250ae73d94c7890a90100595f48b3c531e76))

pytest 9 removed the `msg=` keyword alias from `pytest.fail` (deprecated since pytest 7, renamed to
  `reason=`). run_sh_command.py still used the removed kwarg, so every call raised:

TypeError: _Fail.__call__() got an unexpected keyword argument 'msg'

This broke all 5 tests in tests/test_sweeps.py whenever the `sh` package was installed (nightly.yml,
  test-expensive.yml). The failures were masked until #506 unblocked the nightly runner hang.

Refs #507

Co-authored-by: a <a@as-mac-mini.taile31224.ts.net>


## v0.1.4 (2026-04-02)

### Bug Fixes

- **ci**: Add fork guard to auto-approve workflow
  ([#449](https://github.com/tinaudio/synth-setter/pull/449),
  [`dd6a411`](https://github.com/tinaudio/synth-setter/commit/dd6a411621b5f80bf952b591eb7c41f3b0ec4b26))

The comment said "not a draft or fork" but only checked draft. Compare head repo against base repo
  to reject fork PRs.

### Chores

- **ci**: Enable uv caching and migrate remaining pip references
  ([#431](https://github.com/tinaudio/synth-setter/pull/431),
  [`53f0290`](https://github.com/tinaudio/synth-setter/commit/53f0290e4e9dc7fbce223ee4c88cc2b98f8ceefa))

* chore(ci): enable uv caching and migrate remaining pip references

Enable `enable-cache: true` on all `astral-sh/setup-uv@v6` steps for warm-cache CI speedups. Migrate
  Makefile install/coverage targets and docs from pip to uv pip.

Closes #423

* docs(readme): add uv install reference and bootstrap Makefile

Add `pip install uv` bootstrap to Makefile install/coverage targets so they work without uv
  pre-installed. Add uv install comment to README quick-start.

Addresses review feedback on PR #431.

- **ci**: Restrict auto-approve to ktinubu PRs only
  ([#447](https://github.com/tinaudio/synth-setter/pull/447),
  [`6a122d9`](https://github.com/tinaudio/synth-setter/commit/6a122d94e98934f76d180eb7686c010514d3d73e))

Add author check before evaluating CI/review conditions so the approval bot only acts on PRs
  authored by ktinubu.

- **plumb**: Expand .plumbignore to reduce false-positive triggers
  ([#445](https://github.com/tinaudio/synth-setter/pull/445),
  [`71d0758`](https://github.com/tinaudio/synth-setter/commit/71d0758ffd11dc2408e70ac0b8b06e82830bb704))

### Documentation

- **ci**: Add phase-parenting rule and goal milestones to taxonomy
  ([#444](https://github.com/tinaudio/synth-setter/pull/444),
  [`e065319`](https://github.com/tinaudio/synth-setter/commit/e06531955321a4cc4965643b6da105ecce63da73))

* docs(ci): add phase-parenting rule and goal milestones to taxonomy

* chore: bump skills submodule to include phase-parenting enforcement

* fix(ci): upgrade phase-parenting hook from WARN to BLOCK

* fix(ci): update header comment to match BLOCK behavior


## v0.1.3 (2026-04-01)

### Bug Fixes

- **ci**: Align taxonomy hook issue-ref fallback with CI gate
  ([#419](https://github.com/tinaudio/synth-setter/pull/419),
  [`4a2bf35`](https://github.com/tinaudio/synth-setter/commit/4a2bf35c703dc0d922cbd4013fabfe4aaebf5a04))

* fix(ci): align taxonomy hook issue-ref fallback with CI gate

The local verify-gh-taxonomy.sh hook only matched keyword-prefixed issue references
  (Fixes/Closes/Refs #N). The CI gate also accepts bare #N references and markdown hyperlinks. Add a
  fallback grep to match any #N in the PR body when keywords aren't found.

Fixes #418

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

### Chores

- Remove prod and dev-live Docker targets
  ([#417](https://github.com/tinaudio/synth-setter/pull/417),
  [`af0bb0f`](https://github.com/tinaudio/synth-setter/commit/af0bb0fc073431ab6802ab6e91e4403669374e4a))

- **ci**: Replace pip with uv in all CI workflows
  ([#426](https://github.com/tinaudio/synth-setter/pull/426),
  [`4edf13e`](https://github.com/tinaudio/synth-setter/commit/4edf13e070d1a443e7e853937ff416d1a2436dd9))

Refs #423

- **plumb**: Add coverage mapping and test annotations
  ([#428](https://github.com/tinaudio/synth-setter/pull/428),
  [`6f0f05f`](https://github.com/tinaudio/synth-setter/commit/6f0f05f9387f10b8422c2a23d52917e43a675395))

* chore(plumb): run plumb coverage

* chore: add plumb decisions and model budget

* chore(plumb): plumb coverage and test mapping

* chore: run make format on plumb coverage files

### Documentation

- Consolidate Claude workflow rules into CLAUDE.md
  ([#413](https://github.com/tinaudio/synth-setter/pull/413),
  [`a9d48f7`](https://github.com/tinaudio/synth-setter/commit/a9d48f75d4f24b9ae8375123c22d80665c8d565e))

* docs: consolidate Claude workflow rules into CLAUDE.md

Add a Workflow Rules section covering commit conventions, PR/issue linking, verification format,
  review comment etiquette, and GitHub project selection. These rules were previously stored as
  local Claude memory files and are now version-controlled and shared.

Fixes #412

* docs: narrow hyperlink rule to chat responses only

Markdown hyperlinks for issue/PR refs should only be used in chat responses (for IDE clickability).
  PR bodies should use bare Fixes #N so GitHub auto-close keywords work correctly.

Refs #412

- Fix doc drift in CLAUDE.md and doc-map.yaml (10 findings)
  ([#396](https://github.com/tinaudio/synth-setter/pull/396),
  [`7d77e98`](https://github.com/tinaudio/synth-setter/commit/7d77e98f49ee72124bfe9b7b397842e2767df0bf))

* docs: fix doc drift in CLAUDE.md and doc-map.yaml (10 findings)

Update pipeline architecture section to reflect actual directory structure (entrypoints/, ci/,
  constants.py instead of stages/, backends/). Fix doc-map.yaml: remove duplicate docker-spec.md
  entry, remove stale wandb placeholder, fix incorrect metric names, update rclone attribution, add
  missing source patterns.

Refs #392

* docs: restore planned pipeline architecture in CLAUDE.md

Previous commit silently removed planned directories (stages/, backends/) and schemas (report, card,
  sample) that have open tracking issues. Restore as planned items with issue references (#71, #72,
  #74).

- Fix doc drift in design docs (21 findings)
  ([#393](https://github.com/tinaudio/synth-setter/pull/393),
  [`f6ba0cf`](https://github.com/tinaudio/synth-setter/commit/f6ba0cfb06ac5e7a33257494bdcf4f8394ee0dff))

* docs: fix doc drift in design docs (21 findings)

Update data-pipeline.md: R2 bucket synth-data → intermediate-data, add implementation status notes,
  fix schema fields, mark planned CLI. Update training-pipeline.md: fix log_model claims (true not
  "all"), mark resolved W&B identity gap. Update eval-pipeline.md: fix renderscript.sh description,
  mark wandb resolver as not implemented, R2 bucket rename.

Refs #392

* docs: restore design targets and add issue refs in design docs

Restore 4-check validation as design target (3-check is current partial impl, #103). Add issue refs
  to renderscript auto-detect (#86), make targets (#72), and log_model design target. Design intent
  must be preserved alongside current state.

* docs: add entrypoint mode names to design docs

Name the Docker entrypoint modes in design docs: generate stage = MODE=generate-shards (#407),
  finalize = MODE=finalize-shards (#408), training = MODE=train (#409), eval = MODE=eval (#410).
  Note generate_dataset as legacy MVP pending deprecation (#411).

* docs: address review comments on PR #393

Fix 3-check/4-check inconsistencies in data-pipeline.md: align validation descriptions with actual
  validate_shard.py checks, note 4-check as design target (#103). Fix eval-pipeline.md log_model
  table cell.

* docs: address round 2 review comments on PR #393

Fix MODE=train wording (experiment branch, not main). Fix log_model claims (true = best+last, not
  every checkpoint). Fix checkpoint policy table (intermediates need "all"). Fix renderscript
  Decision header (design target). Fix generate_dataset invocation (env var, not positional arg).

- Fix doc drift in docker-spec.md and docker.md (12 findings)
  ([#397](https://github.com/tinaudio/synth-setter/pull/397),
  [`89336cb`](https://github.com/tinaudio/synth-setter/commit/89336cb25ed26a97bd51648ef3dd4cc00aed70a1))

* docs: fix doc drift in docker-spec.md and docker.md (12 findings)

Rewrite stale "Current vs. Planned" section — MODE dispatch is fully implemented. Fix W&B auth claim
  (baked, not runtime-only). Fix broken links (rclone.md, test_image_config.py path). Correct
  BuildKit secrets table (r2_endpoint is build-arg). Add missing R2_BUCKET documentation. Update
  YAML snippet and test count.

Refs #392

* docs: add tracking issue refs to docker docs planned items

Add #310 ref to rclone.md planned notes. Add #265 ref to docker-spec MODE dispatch section for
  traceability to the original tracking issue.

* docs: replace pipeline-worker with entrypoint mode inventory

Replace pipeline-worker (wrong abstraction) with generate-shards (#407). Add full mode inventory:
  generate-shards, finalize-shards (#408), train (#409) as scoped; eval (#410) as planned. Add
  generate_dataset deprecation note (#411).

* docs: address review comment on PR #397

Fix prod target build instructions — DOCKER_BUILD_FLAGS cannot override IMAGE arg because Makefile
  appends it after.

* docs: address round 2 review comments on PR #397

Fix BUILD_MODE "always" to "default". Fix GIT_REF "requires" to "should set" (has default=main).

- Fix doc drift in unmapped docs (17 findings)
  ([#394](https://github.com/tinaudio/synth-setter/pull/394),
  [`19212b5`](https://github.com/tinaudio/synth-setter/commit/19212b5692d99531706d06bfed21bbaa5b5d890c))

* docs: fix doc drift in unmapped docs (17 findings)

Add status banners to completed/unimplemented docs (org-migration-checklist,
  promotion-pipeline-reference). Update README beyond ISMIR 2025 submission. Add implementation
  status notes to implementation plans. Fix old repo URL in lint-cleanup agent.

Refs #392

* docs: restore ISMIR 2025 citation and code map in README

Previous commit deleted academic provenance (ISMIR 2025 reference, online supplement link) and
  detailed code navigation map. Restore both as dedicated sections alongside the updated project
  description.

* docs: replace pipeline-worker with generate-shards in impl plan

The pipeline-worker abstraction was wrong — the entrypoint mode IS the worker. Replace all
  references with MODE=generate-shards (#407). Add tracking issue refs. Note experiment branch as
  prior art.

* docs: address review comments on PR #394

Fix README H1 rendering inside HTML div. Mark already-existing files as completed in data-pipeline
  implementation plan file lists.

* docs: address round 2 review comments on PR #394

Fix README pipeline description (no __main__.py). Fix training implementation plan status from NOT
  STARTED to INCOMPLETE.

- Fix doc drift in wandb-integration.md and storage-provenance-spec.md (13 findings)
  ([#398](https://github.com/tinaudio/synth-setter/pull/398),
  [`6d60923`](https://github.com/tinaudio/synth-setter/commit/6d60923baa90edd41c529e899fde1ee61c971eff))

* docs: fix doc drift in wandb-integration.md and storage-provenance-spec.md (13 findings)

Update wandb-integration.md: entity/project now env-var driven, mark resolved Known Gaps (#1, #3),
  document log_wandb_provenance(), fix stale code version hash. Update storage-provenance-spec.md:
  R2 bucket synth-data → intermediate-data, mark unimplemented CLI references, update workflow
  table, add implementation status note.

Refs #392

* docs: mark pipeline.cli finalize as planned in table cells

Add (planned) annotation to pipeline.cli finalize references in table cells for consistency with the
  callout notes and promote.yml pattern.

* docs: add MODE=finalize-shards ref to storage provenance spec

Note that the finalize step runs as MODE=finalize-shards in Docker (#408), alongside the planned
  pipeline.cli finalize CLI interface.

* docs: address review comment on PR #398

Fix command field type in provenance table — it's a joined string, not a list.

* docs: address round 2 review comments on PR #398

Fix workflow table: Full Tests trigger is schedule+dispatch (not push), Data Generation is
  workflow_call on ubuntu-latest-4core with image_tag/config_path inputs.

- Remove prod/dev-live Docker targets and keep log_model=true
  ([#416](https://github.com/tinaudio/synth-setter/pull/416),
  [`c1860dc`](https://github.com/tinaudio/synth-setter/commit/c1860dcd183a1e53796b382ab5474bfa24403909))

- **ci**: Taxonomy cleanup — naming conventions, epics table, standalone tasks
  ([#429](https://github.com/tinaudio/synth-setter/pull/429),
  [`2a51eaa`](https://github.com/tinaudio/synth-setter/commit/2a51eaa6e66853a908d1243455be97ca8052ca03))

* docs(ci): update taxonomy — naming conventions, epics table, standalone tasks, plumb pin

- Add Epic:/Feature: naming convention to match existing Phase:/Task: pattern - Complete the current
  epics table with all 10 active epics (#114, #148, #149, #264, #321 were missing) - Remove
  standalone task concept — all issues must trace to an epic - Update CLAUDE.md to reflect mandatory
  epic lineage - Bump plumb-dev pin to a0dd821 (strips GIT_* env vars in worktree hooks)

Fixes #427

* fix(ci): sync DOMAIN_LABELS across hook, CI gate, and taxonomy doc

Hook was missing documentation, CI gate was missing monitoring, taxonomy doc §6/§7 was missing
  monitoring. All three now list the same 9 domain labels.

Refs #427

* fix(ci): address review feedback on PR #429

- Use full 40-char SHA for plumb-dev pin (comment #3019650141) - Broaden §3 epic lineage rule to all
  work types, not just Tasks (comment #3019650163) - Update §2 examples to use new naming
  conventions (comment #3019650165) - Remove standalone task language from task.yml template
  (comment #3019650172) - Add documentation and monitoring to all issue template dropdowns


## v0.1.2 (2026-03-31)

### Bug Fixes

- **ci**: Narrow test.yml path filter and fix taxonomy hook regex
  ([#400](https://github.com/tinaudio/synth-setter/pull/400),
  [`dd5c443`](https://github.com/tinaudio/synth-setter/commit/dd5c443eba70a84f060768c2e786203abad840f0))

* fix(ci): narrow test.yml path filter and fix taxonomy hook regex

test.yml: Switch both push and pull_request triggers from paths-ignore to explicit paths so only
  changes to src/, pipeline/, tests/, configs/, scripts/, requirements, pyproject, setup.py, and the
  workflow itself trigger the test suite. Previously, changes to docker/, notebooks/, jobs/,
  .github/ templates, and other non-test files would trigger unnecessary runs.

verify-gh-taxonomy.sh: Add fallback #N regex so the hook recognizes issue references inside markdown
  hyperlinks like [#399](url), matching the same pattern pr-metadata-gate.yaml uses.

Fixes #399

* fix(ci): dedup and exclude self-reference in taxonomy hook fallback

The fallback #N regex didn't deduplicate or exclude the PR's own number, which could cause the hook
  to validate the PR itself as a linked issue. Add sort -un and grep -v to match
  pr-metadata-gate.yaml behavior.

Refs #399

### Build System

- Bump skills submodule (pr-checkbox v3 + description)
  ([#362](https://github.com/tinaudio/synth-setter/pull/362),
  [`007f958`](https://github.com/tinaudio/synth-setter/commit/007f9581d45c77e37612a509efde84abacdcbd32))

- Replace local skills with tinaudio/skills submodule
  ([#331](https://github.com/tinaudio/synth-setter/pull/331),
  [`666c861`](https://github.com/tinaudio/synth-setter/commit/666c861c3dc4942a3d2e26f3338b963bce3a6a59))

* build: replace local skills with tinaudio/skills submodule

Move all 13 skills to tinaudio/skills repo for cross-project reuse. Skills are mounted via git
  submodule at .claude/skills/ — same path as before, so all skill references in CLAUDE.md and hooks
  continue to work unchanged.

Add submodule note to CLAUDE.md Git Workflow section.

Refs #330

* chore: retrigger copilot review

- **ci**: Generalize BATS workflow to auto-discover tests
  ([#325](https://github.com/tinaudio/synth-setter/pull/325),
  [`c3681d9`](https://github.com/tinaudio/synth-setter/commit/c3681d9d78ee8f1b9bb7017ebfc8780b50b5b7f2))

Replace hardcoded entrypoint-tests workflow with a catch-all bats-tests workflow that uses glob path
  triggers and `bats --recursive tests/` for automatic test discovery.

- **docker**: Add image build-and-push workflow
  ([#313](https://github.com/tinaudio/synth-setter/pull/313),
  [`c7d4584`](https://github.com/tinaudio/synth-setter/commit/c7d4584bc74e908ab4599a1f21f0628a68456c78))

* build(docker): add image build-and-push workflow with metadata and DockerHub push

Evolve docker-build-validation.yml from a local-only dev-live build into a full image creation
  workflow that builds dev-snapshot, tags via docker/metadata-action, and pushes to Docker Hub via
  docker/build-push-action.

- Upgrade runner to ubuntu-latest-4core (16 GiB RAM) to fix OOM - Load build args from
  configs/image/dev-snapshot.yaml via image_config.py (reuses tested Pydantic schema instead of raw
  yq) - Add docker/login-action for Docker Hub authentication - Add docker/metadata-action for
  OCI-standard tags and labels - Replace make docker-build-dev-live with docker/build-push-action -
  Pass all BuildKit secrets (GIT_PAT, R2, W&B) - Update smoke tests to pull pushed image from Docker
  Hub

Refs #311

* fix(docker): use SHA-pinned tag for smoke tests to avoid race condition

The mutable dev-snapshot tag could be overwritten by a concurrent workflow run between push and
  pull. Smoke tests now pull the immutable dev-snapshot-<full-sha> tag instead.

Also switch from type=sha (7-char short SHA) to type=raw with full github.sha so the tag and smoke
  test reference are guaranteed to match.

* fix(docker): remove global TARGETARCH ARG and wire R2 config to workflow

Global-scope `ARG TARGETARCH` / `ARG TARGETPLATFORM` shadowed the automatic platform args that
  buildx sets via --platform, causing TARGETARCH to be empty in every build stage.

The workflow also referenced a nonexistent R2_ENDPOINT GitHub secret. Now extracts r2_endpoint and
  r2_bucket from the image config YAML and passes them as a Docker secret and build-arg
  respectively.

* fix(docker): pass R2_ENDPOINT as build-arg, not secret

d17dc09 changed R2_ENDPOINT from a Docker secret mount to a build ARG. The workflow still passed it
  as a secret, so the ARG was unset and bash's set -u caused "unbound variable" before the
  empty-check ran.

* fix(docker): use git init+fetch in dev-snapshot for non-empty WORKDIR

The parent stage (builder-install-synth-setter-deps) creates a plugins/ symlink in
  /home/build/synth-setter/, so git clone into '.' fails with "destination path already exists and
  is not an empty directory".

Replace with git init + git fetch + git checkout FETCH_HEAD, which works in non-empty directories
  and fetches only the needed commit.

* fix(docker): fetch all refs before checkout (SHA fetch unsupported)

git fetch origin <sha> fails — GitHub doesn't expose raw SHAs as fetchable refs. Use git fetch
  origin (all refs) then checkout the SHA, matching the original git clone behavior.

* fix(docker): use checked-out SHA instead of github.sha

In workflow_dispatch, github.sha is the tip of the dispatching branch, not the checked-out ref. When
  git_ref differs from the dispatch branch (e.g. scheduled runs defaulting to main), the image was
  tagged, built, and tested against the wrong commit. Capture git rev-parse HEAD after checkout and
  use it consistently.

* chore(docker): rename SHA step to source, bump setup-python to v6

Rename step id from 'ref' to 'source' for clarity — steps.source.outputs.sha reads as "the SHA of
  the source we're building". Bump actions/setup-python from v5 to v6 for consistency with other
  workflows.

* refactor(docker): extract config loader to scripts/ci/load_image_config.py

Replace inline python -c block with a standalone script that takes --config, --github-sha, and
  --issue-number args. Cleaner, testable locally, and avoids YAML/Python quoting gymnastics.

Restore pip install step (pyyaml/pydantic not available on bare runner) and set PYTHONPATH=. so the
  scripts package is importable.

* fix(docker): use pip install --no-deps -e . instead of PYTHONPATH hack

Registers the project package so cross-package imports work without PYTHONPATH=. — the standard
  approach for CI scripts that need project modules without pulling heavy ML dependencies.

* fix(docker): add ARG TARGETPLATFORM to arm64-vars, revert to PYTHONPATH

Add ARG TARGETPLATFORM to arm64-vars stage so the diagnostic echo receives the automatic buildx
  value. Without the declaration, the variable was always <unset>.

Revert pip install --no-deps -e . back to PYTHONPATH=. — the editable install registers src/
  (synth-permutations) not scripts/, so the cross-directory import still failed. PYTHONPATH=. is the
  correct approach for non-package directories like scripts/.

* chore(docker): add TODO comment for PYTHONPATH workaround (#323)

* fix(docker): add early Docker Hub push-scope verification

Request a token with pull+push scope from the Docker Hub auth endpoint before the expensive build.
  Fails fast with a clear error message if the token lacks write permissions, instead of building
  for 30+ minutes and failing at push time.

* test(docker): add pytest smoke tests for Docker image validation

Replace inline python -c commands (which had IndentationError from YAML-indented Python) with proper
  pytest tests in tests/docker/.

- test_pedalboard_importable: verifies pedalboard package installs - test_surge_xt_loads: verifies
  VST plugin loads under headless X11 - Add docker_smoke marker to pyproject.toml - Workflow calls
  pytest by test ID inside the container

* fix(test): add skip guards to Docker smoke tests for host CI

The smoke tests are designed to run inside the Docker image but pytest also collects them on the
  host CI runner. Add skipif guards so they skip gracefully when pedalboard/VST aren't available.

* test(ci): add unit tests for scripts/ci/load_image_config.py

Cover GITHUB_OUTPUT file writing, stdout fallback, append mode, missing required args, and invalid
  SHA validation. 7 tests.

* build(docker): trigger build validation on Docker-related PR changes

Add pull_request trigger with path filters for docker/, configs/image/, scripts/image_config.py,
  requirements*.txt, and tests/docker/.

PRs get build-only validation (no push, no smoke tests). Dispatch and schedule runs still do full
  build + push + smoke tests.

Also fix docstring in test_smoke.py to clarify CI vs manual usage.

* fix(docker): checkout PR head SHA for pull_request trigger

The checkout ref defaulted to 'main' for pull_request events because github.event.inputs is
  undefined (inputs are workflow_dispatch only). Use github.event.pull_request.head.sha for PRs,
  which is undefined for other events and falls through to the dispatch/schedule defaults.

* fix(docker): move Docker Hub login before buildx setup

setup-buildx-action pulls moby/buildkit from Docker Hub. Without authentication, this fails when the
  runner IP is rate-limited (triggered by earlier failed login attempts with a bad token). Moving
  login before buildx and making it unconditional ensures authenticated pulls.

* fix(docker): pass Docker Hub creds via env vars, dynamic issue_number

- Move Docker Hub credentials from curl -u (visible in process argv) to env vars (DH_USER, DH_TOKEN)
  for the push-scope check - Add issue_number workflow_dispatch input (default: 311) - For
  pull_request events, use github.event.pull_request.number - For schedule, fall back to 311
  (tracking issue)

- **docker**: Remove ImageConfig defaults, add R2 config fields
  ([#318](https://github.com/tinaudio/synth-setter/pull/318),
  [`d17dc09`](https://github.com/tinaudio/synth-setter/commit/d17dc09375204907e0c48324f364034942c62f5f))

* internal-feat(docker): remove ImageConfig defaults, add r2_endpoint and r2_bucket

Refs #311

* test(docker): update image_config tests for required fields and R2 config

* build(docker): add r2_endpoint and r2_bucket to image config YAML

* build(docker): use R2_ENDPOINT as build-arg instead of secret

R2_ENDPOINT is not sensitive (it's a well-known Cloudflare URL), so pass it as a plain ARG instead
  of a BuildKit secret. This simplifies the build and aligns with the image config schema which
  treats it as a non-secret field.

r2_access_key_id and r2_secret_access_key remain as BuildKit secrets.

* build(docker): pass R2_ENDPOINT as build-arg in Makefile

Update DOCKER_SECRETS block to pass R2_ENDPOINT via --build-arg instead of --secret, matching the
  Dockerfile change.

- **docker**: Split wheels stage into torch and app layers
  ([#346](https://github.com/tinaudio/synth-setter/pull/346),
  [`7e89340`](https://github.com/tinaudio/synth-setter/commit/7e8934033b525f7c85ffc3af1d60294c590d77db))

* build(docker): split wheels stage into torch and app layers

* build(docker): use --find-links and requirements-app.txt in wheels stage

- Build only app wheels (not full requirements.txt) to avoid re-resolving torch from PyPI without
  the CUDA index URL - Add --find-links /wheels so transitive torch deps are satisfied from the
  existing CUDA wheels built in the wheels-torch stage

- **pre-commit**: Exclude CHANGELOG.md from mdformat and codespell
  ([#340](https://github.com/tinaudio/synth-setter/pull/340),
  [`0580555`](https://github.com/tinaudio/synth-setter/commit/05805555cdc08169b158963758a6260d2d605638))

Machine-generated CHANGELOG.md trips two pre-commit hooks: - mdformat: mixed bullet markers, line
  wrapping, thematic breaks - codespell: typos baked in from original commit messages

Exclude it from both hooks. README.md and .claude/* are already excluded from mdformat for similar
  reasons.

### Chores

- Add plumb coverage caches and bump plumb ref
  ([#391](https://github.com/tinaudio/synth-setter/pull/391),
  [`f5bf838`](https://github.com/tinaudio/synth-setter/commit/f5bf8384bbde539e13e47142974d0e32ba7b8b0f))

* chore: add plumb coverage caches and bump plumb ref

Track code_coverage_map.json and coverage.json (force-added past gitignore) so plumb coverage
  results persist across sessions. Update config.json with program_models assignments. Bump
  plumb-dev ref to feat/claude-code-cli-backend branch.

Refs #388

* chore: gitignore .plumb dir, fix trailing newline, add force-add rule

- Add Plumb tooling, fix coverage perf, bump skills
  ([#389](https://github.com/tinaudio/synth-setter/pull/389),
  [`bd0a084`](https://github.com/tinaudio/synth-setter/commit/bd0a08475eaa58f2d6535f9fb56281a73dee7110))

* chore: add Plumb spec/test/code sync tooling

Initialize Plumb to keep specs (docs/), tests (tests/), and code in sync. Adds config, 778 extracted
  requirements from existing docs, ignore patterns, and CLAUDE.md workflow instructions.

Refs #388

* chore: pin plumb-dev fork and extend .plumbignore

Pin plumb-dev to ktinubu/plumb fork which fixes coverage_reporter to respect .plumbignore patterns
  during source scanning. Add .venv*/, .virtualenv*/, .git/, .claude/, __pycache__/, and build
  artifact dirs to .plumbignore so coverage skips them (51k → 57 files).

* chore: gitignore plumb coverage caches

These are regenerated by plumb coverage and would cause noisy diffs and merge conflicts if tracked.
  The incremental cache (code_coverage_map.json) is rebuilt per-developer on demand.

* chore: bump skills submodule (plumb skill, github-taxonomy fix)

* chore: fix mdformat lint in CLAUDE.md plumb section

* chore: add hatchling to requirements for plumb-dev Docker build

plumb-dev uses hatchling as its build backend. The Docker install stage runs with --no-index, so
  hatchling must be pre-built in the wheels cache.

* chore: address review feedback on PR #389

- Remove docs/ from .plumbignore to avoid confusion with spec_paths - Pin plumb-dev to immutable
  commit SHA instead of branch name - Add Setup subsection documenting plumb init and hook conflict

* docs: remove reference docs from plumb spec_paths

* Apply plumb spec files in claude.md

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- Add pre-commit branch-echo hook ([#368](https://github.com/tinaudio/synth-setter/pull/368),
  [`b3c86c1`](https://github.com/tinaudio/synth-setter/commit/b3c86c1e5e764f554c40131cccef4cb2dc179565))

* chore: add pre-commit branch-echo hook to project settings

Add a PreToolUse hook that echoes the current branch name to stderr before any git commit command.
  Acts as a safety net against committing to the wrong branch.

Also unignore .claude/settings.json so project-wide hooks are tracked.

* fix(chore): revert .gitignore change — force-add suffices

The settings.json file was force-added to the index, so the .gitignore exception is unnecessary. Git
  tracks indexed files regardless of ignore patterns.

* fix(hook): handle detached HEAD and add description field

Address PR review: git branch --show-current prints empty on detached HEAD, so fall back to
  "DETACHED HEAD". Add description field for consistency with other hooks in the file.

- Bump skills submodule (4 new skills) ([#370](https://github.com/tinaudio/synth-setter/pull/370),
  [`422bb02`](https://github.com/tinaudio/synth-setter/commit/422bb02cc48dea384f634f19be9da125065b2003))

* chore: bump skills submodule (4 new skills)

Updates .claude/skills to include: - pr-preflight - gha-workflow-validator - tdd-refactor -
  multi-repo-pr

* chore: update skills submodule to merged main

Points at 81e10b5 (skills#45 merged) instead of the feature branch.

- **ci**: Add epic-lineage enforcement to taxonomy skill, CI gate, and CLAUDE.md
  ([#374](https://github.com/tinaudio/synth-setter/pull/374),
  [`402b4fd`](https://github.com/tinaudio/synth-setter/commit/402b4fd75c2ca8249fe0e4154935095a758e4a91))

Adds Step 2.5 to the github-taxonomy skill to verify that linked issues trace back to an Epic via
  the sub-issue hierarchy. Adds a hard-failure epic lineage check to pr-metadata-gate.yaml that
  walks up to 4 levels of parents via GraphQL. Documents the epic traceability requirement in
  CLAUDE.md with standalone-task exceptions per the taxonomy doc.

Fixes #373

- **ci**: Add pr-review-resolver skill and enforce epic lineage in hook
  ([#377](https://github.com/tinaudio/synth-setter/pull/377),
  [`e422224`](https://github.com/tinaudio/synth-setter/commit/e422224b5c97b93ab1f74c293fa9ac6af3925480))

* chore: bump skills submodule to include pr-review-resolver

* fix(ci): add epic lineage and hierarchy blocks to taxonomy hook

The verify-gh-taxonomy.sh hook checked CI minimum three (type, label, milestone) but never verified
  epic lineage — letting PRs through that the pr-metadata-gate CI workflow would reject. Also, issue
  creation only warned about missing hierarchy instead of blocking.

Changes: - Add check_epic_lineage helper (walks parent chain via GraphQL) - PR mode: hard BLOCK if
  linked issue has no Epic ancestor - Issue creation mode: hard BLOCK to force hierarchy, project
  board, and priority setup before proceeding

### Continuous Integration

- **docker**: Switch to registry cache and load image locally for smoke tests
  ([#347](https://github.com/tinaudio/synth-setter/pull/347),
  [`017cbdf`](https://github.com/tinaudio/synth-setter/commit/017cbdf9dd519fc08d6da16d3f9989766ec73e55))

* ci(docker): switch to registry cache and load image locally for smoke tests

* ci(docker): guard cache-to for fork PRs and clarify cache docs

- Disable registry cache-to on pull_request events so fork PRs without Docker Hub secrets don't fail
  - Clarify docs: buildx prune clears local cache only; add instructions for clearing the remote
  registry cache tag

- **gpu**: Move GPU tests to twice-weekly schedule
  ([#335](https://github.com/tinaudio/synth-setter/pull/335),
  [`3977c03`](https://github.com/tinaudio/synth-setter/commit/3977c031a5efcbbd7bf26ef13e8c64b05a252f71))

Replace per-push trigger with cron schedule (Mon + Thu 06:00 UTC). Keeps workflow_dispatch for
  on-demand runs.

Refs #334

### Documentation

- Add project-specific doc-map.yaml ([#348](https://github.com/tinaudio/synth-setter/pull/348),
  [`d644834`](https://github.com/tinaudio/synth-setter/commit/d6448346f3c3e9007f936c84b4a5c343cd5c1b65))

* docs: add project-specific doc-map.yaml and bump skills submodule

* docs: fix dead source patterns in doc-map.yaml

- scripts/image_config.py → pipeline/schemas/image_config.py - scripts/generate_shards.py →
  scripts/entrypoint_generate_shards.py - Remove deleted files: src/data/uploader.py,
  scripts/finalize_shards.py, scripts/setup-rclone.sh - Comment out rclone section
  (docs/reference/rclone.md doesn't exist yet)

- Add project-specific doc-map.yaml ([#352](https://github.com/tinaudio/synth-setter/pull/352),
  [`098c725`](https://github.com/tinaudio/synth-setter/commit/098c725f26fd8279547f408e6ccb0c71dfa0b14c))

* docs: add project-specific doc-map.yaml and bump skills submodule

* docs: add documentation domain to github-taxonomy

* ci: add documentation domain to pr-metadata-gate

* docs: overhaul doc-map.yaml with full project coverage

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- Add refactoring, design principles, and git push verification guidelines
  ([#366](https://github.com/tinaudio/synth-setter/pull/366),
  [`d09dd8a`](https://github.com/tinaudio/synth-setter/commit/d09dd8a1909a0be001d7a91bd9943ede943d4868))

* docs: add refactoring, design principles, and git push verification guidelines

Refs #N/A

* docs: list explicit file extensions in refactoring guideline

Refs #367

* docs: add implementation approach guidelines to CLAUDE.md

- **docker**: Add Docker usage reference ([#315](https://github.com/tinaudio/synth-setter/pull/315),
  [`c1de1d2`](https://github.com/tinaudio/synth-setter/commit/c1de1d20fa664ec922de75443a2a30b8218510e3))

* docs(docker): add Docker usage reference

Practical guide covering building, running, and debugging Docker images. Complements docker-spec.md
  (contract/spec) with how-to content:

- Build targets (dev-live, dev-snapshot) via Make and raw buildx - Running containers with MODE
  dispatch (idle, passthrough) - CI workflow: image_config.py validation, DockerHub push, SHA-pinned
  tags - Debugging: OOM, headless VST, entrypoint errors

Refs #311

* docs(docker): incorporate review feedback on Docker reference

Address reviewer feedback: - Add intro paragraph defining audience and purpose - Clarify secret
  lifecycle: BuildKit mounts vs persisted config files (rclone.conf, .netrc) with explicit
  "Persisted to" column - Reorder sections: move BuildKit secrets into Setup (before Building) so
  the mental model is built before it's needed - Simplify Make target table: drop redundant Command
  column - Add --load/--push explanation and DOCKER_TORCH_IDX override note - Add smoke test example
  after first build - Add docker run --env-file .env recommendation - Note why MODE has no default
  (avoid silent misconfiguration) - Simplify CI steps from 7-item numbered list to 4-item summary -
  Add comment to manual trigger command - Use GitHub alert syntax for security and OOM warnings -
  Standardize debugging subsection format (consistent numbered lists) - Make all cross-references
  clickable markdown links

Refs #314

* docs(docker): fix dev-live entrypoint examples and add future plans

Address remaining Copilot review comments: - Add entrypoint differences table showing which targets
  support MODE dispatch (dev-snapshot, prod) vs fallback (dev-live) - Fix MODE=idle example: use
  dev-snapshot instead of dev-live - Fix volume mount example: use /home/build/synth-setter (not
  /workspace), override entrypoint with --entrypoint bash - Fix smoke test: use --entrypoint bash
  for dev-live - Note PR-triggered build-only mode in CI section - Note Makefile local tag format
  differs from CI tags - Add Section 6 (Future plans): dev-live MODE support, MODE=train,
  MODE=pipeline-worker

* docs(docker): fix debug session example to use single-terminal flow

The previous example started a foreground container without -d or --name, then suggested docker exec
  from another terminal — confusing. Simplified to a single docker run with bash appended.

- **pipeline**: Align design docs and CLAUDE.md with PR #305
  ([#338](https://github.com/tinaudio/synth-setter/pull/338),
  [`fd60408`](https://github.com/tinaudio/synth-setter/commit/fd60408047333cf9f182a5d5128ed8a1b9990873))

* docs(pipeline): align design docs with PR #305 implementation choices

- Rename RunConfig → DatasetConfig (more descriptive, matches implementation) - Replace splits
  dict[str, int] with SplitsConfig Pydantic model - Replace flat schemas.py with schemas/ package
  directory

Refs #337

* docs: update CLAUDE.md architecture to include pipeline/ package

- **training**: Add configuration reference doc
  ([#384](https://github.com/tinaudio/synth-setter/pull/384),
  [`3e60c47`](https://github.com/tinaudio/synth-setter/commit/3e60c47c6131a0ffc944c9100d91c298142ddad2))

* docs(training): add configuration reference doc

Refs #383

* docs(training): note input_spec.json path divergence (#385)

Refs #385

* docs(training): fix config drift wording, link #386 and #387

Config drift protection is planned (per design doc) but not yet enforced in the current
  implementation. Updated wording to reflect this.

Refs #386, #387

* chore: address review feedback on PR #384

- Fix R2 path in §1 table: metadata/input_spec.json → {r2_prefix}/input_spec.json (aligns with §2.1
  diagram and actual code in generate_dataset.py) - Narrow doc-map.yaml configs/** to specific
  subdirectories (configs/dataset/**, configs/experiment/**, configs/train.yaml, configs/eval.yaml)
  to prevent false-positive drift alerts on routine config changes

### Internal-Feat

- **pipeline**: Add dataset generation workflow
  ([#344](https://github.com/tinaudio/synth-setter/pull/344),
  [`cf74d5c`](https://github.com/tinaudio/synth-setter/commit/cf74d5c0c3fba86799145fa5bd5b5e24da592712))

* internal-feat(pipeline): add dataset generation workflow and entrypoint mode

Add a GitHub Actions workflow that generates VST datasets inside the Docker container, a CI config
  loader for dataset YAML configs, and a generate_shards entrypoint mode for the Docker image.

- scripts/ci/load_dataset_config.py: emits dataset config fields to GITHUB_OUTPUT, following the
  image config loader pattern - scripts/entrypoint_generate_shards.py: reads env vars, parses
  config, invokes generate_vst_dataset.py as subprocess - scripts/docker_entrypoint.sh: new
  MODE=generate_shards dispatches to headless X11 wrapper + generate helper -
  .github/workflows/dataset-generation.yml: manual dispatch with configurable samples/R2 upload, PR
  validation with 10 samples

Refs #277, refs #267

* refactor(pipeline): replace shell echo block with testable Python param resolver

All run parameters are now derived from the dataset config with no hardcoded magic numbers. PR mode
  uses sample_batch_size for num_samples (one batch = minimum smoke test). Dispatch mode uses
  provided values with config-derived fallbacks.

* docs(docker): document MODE=generate_shards entrypoint mode

Add generate_shards to the MODE table, document DATASET_CONFIG, NUM_SAMPLES, and OUTPUT_DIR env
  vars, and add usage examples.

* fix(pipeline): remove num_samples override, address PR review feedback

Remove --num-samples-override from all scripts — num_samples is always derived from shard_size *
  num_shards in the config. The resolve script uses sample_batch_size for PR smoke tests, shard_size
  * num_shards for dispatch.

Also addresses PR review comments: - Validate output_format == 'hdf5' in entrypoint helper (#6) -
  Validate upload_to_r2 input in param resolver (#3) - Add set -o pipefail to docker run step (#5) -
  Fix docker.md X11 attribution wording (#4) - Remove NUM_SAMPLES env var from docs and entrypoint

* docs: overhaul doc-map.yaml with verified sources and full coverage

* docs(docker): document dataset generation workflow artifacts

Describe the run manifest artifact bundle contents, how to download and inspect it, and retention
  period.

* internal-feat(pipeline): add CI smoke-test dataset config

32 samples, single shard. Used by the dataset-generation workflow on pull_request events instead of
  branching on event name.

* refactor(pipeline): rename generate_shards→generate_dataset, enforce single-shard MVP

Each invocation now generates one shard (shard_size samples). Multi-shard raises
  NotImplementedError. Output file is shard-000000.hdf5 instead of {config_id}.hdf5 — aligns with
  design doc shard naming convention.

* refactor(pipeline): remove num_samples from CI plumbing, simplify resolver

- resolve_dataset_run_params.py: remove event-name branching and num_samples. Just fills empty
  inputs with defaults. - load_dataset_config.py: emit shard_size and num_shards as separate fields
  instead of derived num_samples. - Config YAML is the single source of truth for generation
  parameters. PR mode uses ci-smoke-test.yaml instead of event-name branching.

* refactor(pipeline): update workflow for single-shard MVP and renamed mode

Use ci-smoke-test.yaml for PR trigger, MODE=generate_dataset, shard-000000.hdf5 output,
  shard_size/num_shards in manifest. Remove num_samples from all plumbing.

* docs(docker): rename generate_shards→generate_dataset, update for single-shard MVP

Remove NUM_SAMPLES env var, document shard_size-based generation, update manifest fields to
  shard_size/num_shards.

* docs: update doc-map.yaml for generate_dataset rename

Update entrypoint pattern references and MODE value list.

* refactor(pipeline): rewrite for DataPipelineSpec, delete CI scripts

Container does everything: materialize spec, upload spec to R2, generate shard, upload shard to R2.
  No CI Python scripts needed.

- Rewrite entrypoint to use materialize_spec() from DataPipelineSpec - Delete load_dataset_config.py
  and resolve_dataset_run_params.py - Simplify workflow to: pull image, one docker run, upload
  artifact - Replace OUTPUT_DIR with RUN_METADATA_DIR (spec.json only) - spec.json IS the manifest —
  no separate manifest construction

Tests will fail at import until #354 (DataPipelineSpec) merges.

Refs #354, refs #277, refs #267

* refactor(pipeline): rename DataPipelineSpec→DatasetPipelineSpec

Align with #354 naming convention.

Refs #354, refs #267

* test(docker): add BATS tests for MODE=generate_dataset

Test that missing DATASET_CONFIG env var exits nonzero with a clear error message. Full generation
  testing requires headless X11 + VST which is only available inside the Docker container.

* fix(pipeline): move plugin_path validation from model to materialize_spec

The model_validator checked plugin_path exists on disk at construction time, which breaks
  deserialization on machines without the VST plugin (e.g., finalize-only, CI validation). Move the
  check to materialize_spec() where it belongs — only materialization needs the plugin on disk.

Refs #354

* fix: restore doc-map.yaml sections lost during rebase

The rebase conflict resolution incorrectly dropped eval pipeline, github taxonomy, docker-spec, and
  other sections added by #348/#352. Restored main's version and applied only our 3 targeted
  changes.

* refactor(ci): split dataset workflow into reusable + test pattern

Split dataset-generation.yml into two files matching the pattern from spec-materialization.yml /
  test-spec-materialization.yml:

- dataset-generation.yml: reusable workflow_call building block (inputs: image_tag, config_path,
  artifact_name) - test-dataset-generation.yml: test workflow with dispatch + PR triggers, calls the
  reusable workflow with ci-smoke-test.yaml for PRs

* fix(pipeline): add channels to DatasetPipelineSpec, fix entrypoint arg building

- Add channels field to DatasetPipelineSpec and _build_pipeline_spec - Fix _build_generate_args: use
  dict for options, take ShardSpec + output_dir (shard owns filename, builder owns path
  construction) - Use spec.shard_size instead of nonexistent shard.row_count - Use spec.channels
  instead of nonexistent shard.audio_shape - Fix pyright: cast spec.run_id to DatasetRunId for
  make_r2_prefix

Refs #354, refs #277

* fix(pipeline): type run_id as DatasetRunId on DatasetPipelineSpec

run_id is always a DatasetRunId (from make_dataset_wandb_run_id). Using the NewType annotation
  removes the need for explicit casts at call sites.

* refactor(pipeline): improve entrypoint tests — state over interaction testing

Replace mock spec factory with real DatasetPipelineSpec fixture. Convert interaction tests to state
  assertions where possible. Make build_generate_args public. Add subprocess/rclone failure
  propagation tests. Simplify change-detector test to structural assertions.

Refs #344

* fix(pipeline): use canonical input_spec.json filename from constants

Replace hardcoded spec.json with INPUT_SPEC_FILENAME constant in the entrypoint. Update workflow
  YAML, tests, docs, and entrypoint comments to use the canonical input_spec.json name from
  pipeline.constants.

Refs #354, refs #344

* fix(ci): use ci-smoke-test.yaml as default in test workflow

The test workflow should default to the smoke test config, not the production 480k config. The
  production config is for dispatch runs from the reusable workflow, not the test workflow.

* internal-feat(pipeline): add shard validation for CI dataset generation tests

Validates HDF5 shard against DatasetPipelineSpec: checks expected datasets exist (audio, mel_spec,
  param_array) and row counts match shard_size. Used by test-dataset-generation.yml to verify
  generation output after R2 upload.

Refs #344, refs #267

* ci(pipeline): add shard validation step to test workflow

After generation, download the shard from R2 via Docker and validate it against the spec using
  pipeline.ci.validate_shard. Checks HDF5 structure, expected datasets, and row count against
  shard_size.

validate-spec and validate-shard run in parallel after generate.

* refactor(pipeline): add r2_prefix to DatasetPipelineSpec, simplify workflow

Add r2_prefix field to DatasetPipelineSpec, computed during materialization from config_id + run_id.
  Replaces brittle regex parsing of run_id in the test workflow. The entrypoint now reads
  spec.r2_prefix instead of computing it independently.

Refs #344, refs #354

* refactor(pipeline): extract R2_BUCKET constant, remove hardcoded bucket name

Add R2_BUCKET to pipeline.constants. Entrypoint and test workflow now read the bucket name from the
  constant instead of hardcoding 'intermediate-data'. The workflow reads it via Docker to avoid
  duplicating the value in YAML.

* fix(ci): use volume-mount pattern for dataset generation workflow

The published Docker image doesn't have MODE=generate_dataset yet (it's being added in this PR).
  Mount the PR's code into the container and run the entrypoint directly, matching the pattern from
  spec-materialization.yml. This tests the PR's code against the image's environment (Surge XT,
  Python, rclone).

* fix(ci): add PYTHONPATH for volume-mounted code in Docker workflows

The Docker image's editable install was built before pipeline/constants.py existed. Setting
  PYTHONPATH ensures Python finds all modules from the mounted code regardless of the stale editable
  install.

* fix(pipeline): address PR review comments — validate_spec fields, doc accuracy

- Add channels and r2_prefix to _REQUIRED_FIELDS in validate_spec.py - Update docker.md artifact
  section: correct artifact name (test-run-metadata), remove nonexistent config YAML from bundle,
  fix two files not three - Fix docker run examples: use ci-smoke-test.yaml (not 480k config which
  raises NotImplementedError with num_shards > 1)

* fix(ci): read R2_BUCKET from checkout, not Docker image

The Docker image doesn't have pipeline.constants yet. Read it from the checked-out code on the
  runner instead (PYTHONPATH=.).

* fix(ci): mount code at editable install path for validate_shard

The Docker image's editable install resolves pipeline.* from /home/build/synth-setter. Mount the PR
  code there (not /code) so new modules like pipeline.ci.validate_shard are discoverable.

* refactor(pipeline): move entrypoint from scripts/ to pipeline/entrypoints/

Resolves recurring Docker module import issues — the entrypoint now lives in the pipeline package
  and is discoverable via the editable install. Run as python -m
  pipeline.entrypoints.generate_dataset.

Fixes #361 Refs #344

* docs(docker): fix stale entrypoint path in docker-spec.md

Update MODE table to reference pipeline.entrypoints.generate_dataset instead of the old scripts/
  path.

- **pipeline**: Add DatasetConfig, YAML loader, and R2 prefix generation
  ([#305](https://github.com/tinaudio/synth-setter/pull/305),
  [`dd577e3`](https://github.com/tinaudio/synth-setter/commit/dd577e35c41b38fce209a5f1feeaf4e17bf8bf22))

* internal-feat(pipeline): add DatasetConfig model and YAML loader

Refs #275

* internal-feat(pipeline): add R2 prefix generation

Refs #276

* internal-fix(pipeline): harden config loader and fix test issues from review

Address Copilot review feedback on PR #305: - load_dataset_config: .exists() → .is_file() so
  directories raise FileNotFoundError - load_dataset_config: guard against empty/non-mapping YAML
  with clear TypeError - conftest: shallow .copy() → copy.deepcopy() to isolate nested dict
  mutations - conftest: yaml.dump → yaml.safe_dump for safer serialization - test_prefix: patch
  datetime.now to eliminate flaky midnight-rollover race

Refs #275, Refs #276

* refactor(pipeline): add NewType wrappers for DatasetConfigId and DatasetRunId

Prevent silent argument swaps in functions like make_r2_prefix(config_id, run_id) where both params
  are str. NewType gives pyright visibility with zero runtime cost.

* refactor(pipeline): add R2Prefix NewType for make_r2_prefix return type

Completes the NewType coverage so all three pipeline identifiers (DatasetConfigId, DatasetRunId,
  R2Prefix) are type-distinct.

- **pipeline**: Add DatasetPipelineSpec, ShardSpec, and materialize_spec
  ([#356](https://github.com/tinaudio/synth-setter/pull/356),
  [`6b26bd9`](https://github.com/tinaudio/synth-setter/commit/6b26bd924706262fcdf7321a44c906e285dc1ee8))

* internal-feat(pipeline): add PipelineSpec, ShardSpec, and materialize_spec

Frozen runtime specification materialized from DatasetConfig. Contains per-shard seeds, shapes,
  filenames, and row ranges. Same config + same code version = same spec (deterministic).

Also adds extract_renderer_version() for platform-specific VST3 plugin version extraction (Linux
  moduleinfo.json, macOS Info.plist).

Includes dedicated CI workflow for spec materialization tests on Linux.

Refs #354 Refs #267

* fix(pipeline): address PR review feedback on PipelineSpec

- Change created_at field from str to datetime for type-level validation - Use tuples instead of
  lists for immutable collections (expected_datasets, shards) to enforce deep immutability - Add
  NotImplementedError guard for unsupported WDS output format - Expand CI workflow push.paths to
  match pull_request.paths

Refs #354

* internal-feat(pipeline): rename to DatasetPipelineSpec, add generation fields

Rename PipelineSpec → DatasetPipelineSpec for consistency with DatasetConfig naming. Add 6
  generation parameters (plugin_path, preset_path, velocity, signal_duration_seconds, min_loudness,
  sample_batch_size) so workers have all rendering config in the spec.

Add model_validator to check plugin_path exists at construction time.

* docs(pipeline): document parse error exceptions in extract_renderer_version

Add json.JSONDecodeError and plistlib.InvalidFileException to the docstring Raises section. These
  can occur if version files are present but malformed.

* refactor(pipeline): remove ShardSpec, add num_params to DatasetPipelineSpec

ShardSpec expanded per-shard values (seeds, shapes, filenames) that are trivially derivable from
  shard index + top-level fields. Shape metadata (audio_shape, mel_shape, param_shape) is output
  metadata, not generation input — belongs on a dataset card, not the generation spec.

Workers derive per-shard values at runtime: seed = base_seed + shard_id filename =
  f"shard-{shard_id:06d}.h5" row_start = shard_id * shard_size

num_params is captured from the param_spec registry at materialization time since workers need it to
  allocate HDF5 datasets.

* refactor(pipeline): add lean ShardSpec, replace num_shards with property

Re-add ShardSpec with only per-shard values (shard_id, filename, seed). Remove num_shards field —
  len(spec.shards) is the single source of truth, exposed via a @property for convenience.

* docs(pipeline): update design docs for DatasetPipelineSpec rename

Update §14.1 schema to match implementation: PipelineSpec → DatasetPipelineSpec, lean ShardSpec
  (shard_id, filename, seed only), num_shards as @property, generation fields on spec. Remove shape
  metadata (audio_shape, mel_shape, param_shape) from ShardSpec — these are output metadata, not
  generation inputs.

* docs(pipeline): add inline comments to all DatasetPipelineSpec fields

* fix(pipeline): add plugin_path guard in extract_renderer_version, bump CI timeout

extract_renderer_version now checks plugin_path.exists() first and raises a clear FileNotFoundError
  instead of a misleading "no version files in Contents/" message when the plugin itself doesn't
  exist.

Bump spec CI workflow timeout from 10 to 30 minutes to account for cold-runner pip install times
  (torch, pedalboard, etc.).

* ci(pipeline): rewrite spec workflow as reusable Docker integration test

Replace the redundant pytest-based workflow with a Docker integration test that materializes a real
  spec inside the production container.

- Add reusable workflow_call interface (accepts image_tag input) - Add workflow_dispatch for manual
  trigger - Pull image from Docker Hub, mount PR code, run materialize_spec - Inspect output spec
  (validate code_version SHA, shard seeds, renderer_version) - Upload spec.json as GitHub artifact
  (30-day retention) - Add ci-materialize-test.yaml config (3 shards for multi-shard verification) -
  Add scripts/ci/materialize_spec_smoke.py for in-container execution

* ci(pipeline): split spec workflow into reusable step and test

spec-materialization.yml: reusable building block (workflow_call). Takes image_tag + config_path,
  materializes spec in Docker, validates structure (required fields, valid SHA, non-empty
  renderer_version), uploads artifact. No value assertions — generic for any config.

test-spec-materialization.yml: test workflow (workflow_dispatch). Calls the reusable workflow with
  ci-materialize-test.yaml, downloads artifact, asserts test-specific values (3 shards, seeds
  [42,43,44], config passthrough fields).

* refactor(ci): extract inline Python from workflows into scripts

Move structural validation to scripts/ci/validate_spec_structure.py and test assertions to
  scripts/ci/validate_spec_test_values.py. Both are now linted by ruff/pyright and maintainable
  outside YAML.

* refactor(ci): move CI scripts to pipeline/ci, combine validators, add tests

Move materialize_spec_smoke → pipeline/ci/materialize_spec. Combine validate_spec_structure +
  validate_spec_test_values → pipeline/ci/validate_spec with --test-values flag.

Add tests for both validation functions (plain dict in, errors out). Delete scripts/ci/ originals —
  all logic now in pipeline/ci/.

* fix(pipeline): use input_spec.json filename from design doc

Add pipeline/constants.py with INPUT_SPEC_FILENAME — canonical name from
  docs/design/data-pipeline.md §7.1 storage layout. Was incorrectly using "spec.json" instead of
  "input_spec.json".

* fix(ci): add git safe.directory for volume-mounted repo in Docker

git rev-parse HEAD fails inside Docker when the repo is volume-mounted from the GitHub runner
  (different UID). Add safe.directory config before running materialize_spec.

* docs(ci): add comments explaining Docker mount pattern in spec workflow

Explains why we volume-mount, recreate the plugin symlink, add safe.directory, and use headless X11.

* fix(pipeline): add pedalboard fallback for renderer version extraction

The prebuilt Surge XT .deb doesn't include Contents/moduleinfo.json — only the .so binary. Fall back
  to pedalboard.VST3Plugin.version which reads the version from the VST3 factory info embedded in
  the binary.

Static file checks (moduleinfo.json, Info.plist) remain as fast paths for plugins that include
  metadata files.

* docs(pipeline): fix stale materialize_spec signatures in implementation plan

Update function signature to match implementation (2-arg, no optional overrides). Update reference
  test to use patch_materialize_io fixture. Mark resolved schema gaps as fixed.

### Refactoring

- **pipeline**: Freeze SplitsConfig for immutability
  ([#357](https://github.com/tinaudio/synth-setter/pull/357),
  [`3c0eaf2`](https://github.com/tinaudio/synth-setter/commit/3c0eaf208fd1a1f2acf635b1c374c4045880f216))

SplitsConfig fields (train, val, test) should not be mutated after construction. Adding frozen=True
  enforces this at the Pydantic level. Prepares for PipelineSpec deep immutability (#354).

Refs #354

- **pipeline**: Move CI script from scripts/ci/ to pipeline/ci/
  ([#359](https://github.com/tinaudio/synth-setter/pull/359),
  [`a824c56`](https://github.com/tinaudio/synth-setter/commit/a824c56ab7c2c8854fe3bb0484ec58cc277df4cc))

The load_image_config CLI wrapper now lives at pipeline/ci/ and is invocable as `python -m
  pipeline.ci.load_image_config`, eliminating the PYTHONPATH=. hack in the Docker build workflow.

Refs #323

- **pipeline**: Move config schemas to pipeline/schemas/
  ([#343](https://github.com/tinaudio/synth-setter/pull/343),
  [`3f21db6`](https://github.com/tinaudio/synth-setter/commit/3f21db61acb22ac82cae6dca8b330cfb33317ab5))

* refactor(pipeline): move config and prefix to pipeline/schemas/

Aligns with the directory layout in docs/design/data-pipeline.md §14 which specifies
  pipeline/schemas/ for Pydantic models and ID helpers.

Refs #267

* refactor(pipeline): move image config schema to pipeline/schemas/

Co-locates ImageConfig with DatasetConfig under pipeline/schemas/, aligning all Pydantic config
  schemas in one place.

* fix(pipeline): update stale path references after schema move

Update workflow PR trigger, doc links, config comment, and test docstring to reference the new
  pipeline/schemas/ locations.

Refs #342

### Testing

- **wandb**: Add env var resolution tests to test_configs.py
  ([#376](https://github.com/tinaudio/synth-setter/pull/376),
  [`f97fc7e`](https://github.com/tinaudio/synth-setter/commit/f97fc7e9b7df6acd413406d06831b8f993118160))

* test(wandb): add OmegaConf env var resolution integration tests

Closes the only test coverage gap found during Phase 1-3 pr-checkbox verification: no test verified
  that configs/logger/wandb.yaml resolves WANDB_ENTITY and WANDB_PROJECT from environment variables.

Three tests added: - entity resolves from WANDB_ENTITY env var - project resolves from WANDB_PROJECT
  env var - defaults to tinaudio/synth-setter when env vars unset

Refs #265, refs #375

* refactor(test): merge wandb config tests into test_configs.py

Move 3 wandb env var resolution tests from standalone test_wandb_integration.py into test_configs.py
  where config tests belong. Fix docstring to link issue #265 instead of opaque "Task 1.2".


## v0.1.1 (2026-03-26)

### Bug Fixes

- **tests**: Replace pkg_resources with importlib.metadata
  ([#298](https://github.com/tinaudio/synth-setter/pull/298),
  [`eda524b`](https://github.com/tinaudio/synth-setter/commit/eda524b363a1520b88877886bdc960568809f89d))

setuptools 82.0.0 removed pkg_resources, breaking test collection for all files that import from
  tests/helpers/. Port the fix from the experiment branch.

Refs #265

### Build System

- **ci**: Cap torch<2.7.0 on GPU runner to match driver
  ([#259](https://github.com/tinaudio/synth-setter/pull/259),
  [`5e9d191`](https://github.com/tinaudio/synth-setter/commit/5e9d19175efdaadb8d7fe5b8deca4fedc245ebce))

* fix(ci): cap torch<2.7.0 on GPU runner to match NVIDIA driver 12080

GitHub's gpu-x64 runner ships driver 12080 (CUDA 12.0). torch>=2.7.0 bundles CUDA 13.x and fails at
  runtime with "NVIDIA driver too old". Use PIP_CONSTRAINT to cap torch in CI without changing
  requirements.

* fix(ci): correct CUDA version comment (12.0 → 12.8)

Driver version 12080 maps to CUDA 12.8, not 12.0.

Refs #259

- **ci-automation**: Add --warn-undefined-variables to makefile, add Verified and Won't Fix statuses
  to taxonomy
  ([`82446c1`](https://github.com/tinaudio/synth-setter/commit/82446c1642e5920ef909147a03142046c42a0b4b))

* chore(build): warn on undefined Makefile variables to catch typos

* docs(ci-automation): add Verified and Won't Fix statuses to taxonomy

Closes #260

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* docs(ci-automation): address review comments on status lifecycle

- Fix Won't Fix reachability description to match any-status rule - Clarify issue close timing
  (close at Verified, not Done) - Leading-zero on step 09 is enforced by mdformat (aligns with 10,
  11)

Refs #261

---------

- **docker**: Add Docker build infrastructure for main
  ([#229](https://github.com/tinaudio/synth-setter/pull/229),
  [`3a54beb`](https://github.com/tinaudio/synth-setter/commit/3a54bebbfec29131ea45f9bc05039484a04ec9fd))

* feat(notebooks): add flush+reset investigation notebook and CI workflow

Adds a Jupyter notebook that systematically compares audio output across four flush strategies
  (none, post_param, post_render, all) using MFCC and multi-resolution spectral loss. Includes
  self-consistency, shared vs isolated instance, and ordering dependence tests.

CI workflow runs the notebook weekly in Docker, converts to HTML with embedded audio, and uploads as
  an artifact with provenance metadata.

Refs #227

* feat(docker): add Docker build infrastructure and entrypoint for main

Add the Dockerfile, Makefile docker targets, headless VST bootstrap script, and a minimal
  passthrough entrypoint so that Docker images (dev-live, dev-snapshot, prod) can build successfully
  from main.

SYNTH_PERMUTATIONS_GIT_REF defaults to main instead of experiment.

Closes #228

* fix(docker): address PR review comments for Docker build infra

- Guard entrypoint against empty args with clear error message - Fix typo: "pluginsn" → "plugins" in
  headless script - Guard cleanup trap PIDs individually to avoid unbound variable errors - Use full
  SHA (git rev-parse HEAD) instead of short for reproducible builds - Remove stray `--` after build
  context that breaks docker buildx - Remove pipe to `tee` that masks docker build exit codes - Use
  surge-package-filepath.txt variable for install path consistency - Rename workflow to
  docker-build-validation.yml (notebook moves to PR B) - Replace notebook workflow with barebones
  Docker build + smoke test - Upgrade actions/checkout to v6 per repo convention

Refs #228

- **docker**: Add idle and passthrough Docker entrypoint modes
  ([#290](https://github.com/tinaudio/synth-setter/pull/290),
  [`13d8f55`](https://github.com/tinaudio/synth-setter/commit/13d8f559c8e95f20571e595a1163bdba02e547e7))

* internal-feat(docker): add idle and passthrough entrypoint modes

Add MODE dispatch to docker_entrypoint.sh: - MODE=idle: sleep infinity for interactive debugging -
  MODE=passthrough: exec args or exit 0 (no-op for CI) - MODE required: error if unset (footgun
  prevention) - Unknown MODE: error with available modes listed

Add BATS tests (15 cases) and make test-entrypoint target.

Refs #272, #273

* internal-fix(docker): use portable trap loop for idle mode

Replace exec sleep infinity (GNU-only) with a portable signal-aware wait loop that works on macOS
  and Linux.

Rewrite idle tests to exercise real behavior (no mocking needed). Use self-documenting arg names
  (SHOULD_BE_IGNORED, SHOULD_NOT_RUN). Add SIGTERM clean exit test.

* internal-fix(docker): eliminate sleeps from idle tests

Move trap before echo in entrypoint so echo acts as a readiness signal. Poll for echo in tests
  instead of sleeping — zero timing dependencies, runs in milliseconds.

* internal-fix(docker): revert to exec sleep infinity, skip idle tests on macOS

The trap loop was over-engineering — sleep infinity is a Docker entrypoint that only runs inside
  Linux containers. Idle tests skip on macOS with a clear reason; they run for real in Linux CI.

No mocks, no sleeps, no helpers. 15 tests (3 skip on macOS).

* docs(docker): add usage documentation to entrypoint script

Add header block with mode descriptions, usage examples, and cross-references so the entrypoint is
  self-documenting.

* internal-fix(docker): fix idle message test race condition on Linux

Poll for echo output instead of killing immediately — the assertion IS the synchronization. Caught
  by running BATS in Docker (Linux).

* docs(docker): add concurrency semantics to idle test comments

Document why each idle test is deterministic (or timing-invariant) despite using background
  processes: poll-as-assertion for message test, negative assertion for args test,
  fork-before-resume for liveness test.

* ci(docker): add BATS entrypoint test workflow

Runs entrypoint BATS tests on Ubuntu (Linux) where idle mode's sleep infinity works natively.
  Triggers on changes to the entrypoint script or test file.

Refs #288

- **docker**: Add image creation config schema and loader
  ([#297](https://github.com/tinaudio/synth-setter/pull/297),
  [`49cc078`](https://github.com/tinaudio/synth-setter/commit/49cc078ee3dd0477d00655c38a69d3450db6f973))

* internal-feat(ci): add image creation config schema and loader (#274)

Pydantic-validated ImageConfig with github_sha (40-char hex), issue_number (positive int), and
  image_config_id (derived from config filename stem). load_image_config() merges static YAML with
  runtime inputs at the trust boundary.

* fix(docker): use is_file() instead of exists() for config path check

exists() returns true for directories/symlinks, causing confusing IsADirectoryError instead of the
  documented FileNotFoundError.

Refs #303

* internal-feat(docker): add static image config fields and YAML merge

Add dockerfile, image, base_image, base_image_tag, build_mode, target_platform, and torch_index_url
  as static fields in ImageConfig. load_image_config now merges YAML content with runtime inputs
  instead of discarding it. Unknown keys are rejected by Pydantic strict mode.

Refs #303, #304

- **docker**: Correct Surge install path and workflow SHA
  ([#232](https://github.com/tinaudio/synth-setter/pull/232),
  [`db66ab2`](https://github.com/tinaudio/synth-setter/commit/db66ab2b54c15c3fec498ab1fd24c073f473b2fc))

* fix(docker): correct surge-package-filepath.txt path and use checked-out SHA

- Fix Surge install path: file is at /surge-package-filepath.txt (COPY'd from arch-vars), not
  /tmp-artifacts/surge-package-filepath.txt - Use git rev-parse HEAD instead of github.sha for
  Docker build so workflow_dispatch with custom git_ref builds the correct commit

Refs #228

* fix(ci): remove unused GIT_REF and meta step from dev-live workflow

docker-build-dev-live uses CURRENT_LOCAL_GIT_REF (git rev-parse HEAD) internally in the Makefile.
  Passing GIT_REF on the command line has no effect for this target. Remove the dead code.

### Chores

- **build**: Warn on undefined Makefile variables to catch typos
  ([#245](https://github.com/tinaudio/synth-setter/pull/245),
  [`0b4fe6c`](https://github.com/tinaudio/synth-setter/commit/0b4fe6c30c4247bac8c641f0e1ae72e59e6f6b81))

- **ci**: Set up mutmut mutation testing ([#302](https://github.com/tinaudio/synth-setter/pull/302),
  [`6b70380`](https://github.com/tinaudio/synth-setter/commit/6b7038027467066335b13b39c0248abfa7c0ab7f))

* chore(ci): set up mutmut mutation testing

Configure mutmut v3.5 for mutation testing: - Add [tool.mutmut] config to pyproject.toml (scripts/
  only — src/ is excluded due to mutmut v3 asserting module names don't start with "src.") - Add
  `make mutmut` target to Makefile - Add mutants/ and .mutmut-cache/ to .gitignore

Refs #296

* chore(test): udpate pytest_add_cli_args

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- **claude**: Add shared project hooks to .claude/settings.json
  ([#292](https://github.com/tinaudio/synth-setter/pull/292),
  [`32b7883`](https://github.com/tinaudio/synth-setter/commit/32b78832e586c2305be755ae85985091c84b0ca3))

Move hook configurations to project-level settings.json (committed via git add -f since .claude/ is
  gitignored). Includes credential protection, ruff auto-format, auto-test runner, taxonomy
  verification, and PR checkbox trigger. Each hook has a description field documenting its purpose
  and trigger conditions.

Refs #265

- **code-health**: Add conventional commit guidance and gitlint enforcement
  ([#214](https://github.com/tinaudio/synth-setter/pull/214),
  [`e2fec3d`](https://github.com/tinaudio/synth-setter/commit/e2fec3d10cd73889a7aca96f34883f53ae0dedcd))

* chore: add conventional commit guidance to CLAUDE.md

* chore: add .gitlint config and document major version bumps

- Add .gitlint with contrib-title-conventional-commits rule so gitlint actually enforces
  conventional commit format (was running with defaults) - Add major version bump guidance (feat!: /
  BREAKING CHANGE:) - Reference .gitlint config file in CLAUDE.md text

- **code-health**: Add internal-feat and internal-fix commit prefixes
  ([#222](https://github.com/tinaudio/synth-setter/pull/222),
  [`afa99b6`](https://github.com/tinaudio/synth-setter/commit/afa99b67eca7ee9d4bc29d98dda64e21a79ff3a2))

* chore(code-health): add internal-feat and internal-fix commit prefixes

New conventional commit types for building features across multiple PRs:

- internal-feat: new code building toward a feature not yet user-facing (new internal API, module,
  config schema). Tested, valid, but not exposed. No version bump. - internal-fix: fix to internal
  code not yet exposed. No version bump.

Updated in: - CLAUDE.md: commit message guidance with when-to-use section - .gitlint: allowed types
  for gitlint enforcement - pyproject.toml: semantic-release allowed_tags (no bump configured)

* fix(code-health): add revert to allowed_tags and patch_tags

Copilot review caught that revert: was in .gitlint but missing from semantic-release config and
  CLAUDE.md. On an append-only main, a revert produces a novel codebase state that users haven't
  seen, so a patch bump is appropriate.

- **deps**: Pin pydantic>=2 and add mutmut
  ([#307](https://github.com/tinaudio/synth-setter/pull/307),
  [`c56ce2f`](https://github.com/tinaudio/synth-setter/commit/c56ce2fd64d6e212668f384d0969b9b7a9833db8))

pydantic>=2 makes the v2 dependency explicit — PRs #297 and #305 use v2-only APIs (field_validator,
  model_validator, ConfigDict).

mutmut==3.5.* for mutation testing setup in PR #302.

Fixes #303

- **skill**: Update pr-checkbox with scope-matching hierarchy
  ([#309](https://github.com/tinaudio/synth-setter/pull/309),
  [`06ff928`](https://github.com/tinaudio/synth-setter/commit/06ff9281fd817dd3b419f59f39edd079e5650860))

* chore(skill): update pr-checkbox with scope-matching and integration level

Replace the one-directional "always use highest level" escalation rule with bidirectional
  scope-matching: pick the narrowest level that fully exercises the promise. Adds Level 1 (Full
  integration) as a distinct tier above "run the tool," recognizes over-specification as a failure
  mode alongside under-specification, and adds promise-matching table with concrete examples.

Refs #308

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* Update scope-matching rule in SKILL.md

Clarified the scope-matching rule for skill levels.

---------

### Continuous Integration

- **automation**: Add snooze-issue GitHub Action
  ([#250](https://github.com/tinaudio/synth-setter/pull/250),
  [`0c31118`](https://github.com/tinaudio/synth-setter/commit/0c31118319628a55b1a1e25dcf51147bb8d3ab6f))

* ci(automation): add snooze-issue GitHub Action for temporary issue deferral

Adds two workflows: - snooze-issue.yml: listens for /snooze comments to close issues temporarily -
  unsnooze-issues.yml: daily cron to reopen issues whose snooze timer expired

* ci(automation): bump actions/checkout to v6 for consistency

* ci(automation): fix permissions and PR-comment filter for snooze workflows

- Add contents: read permission for actions/checkout in both workflows - Filter out PR comments in
  snooze-issue to avoid unintended triggers

Refs #250

### Documentation

- Add monitoring prefix and git worktree guidance
  ([#239](https://github.com/tinaudio/synth-setter/pull/239),
  [`fbd6b65`](https://github.com/tinaudio/synth-setter/commit/fbd6b657f7dfade4925ceb84d480b6f809129b7d))

* docs: add monitoring prefix and git worktree guidance to CLAUDE.md

* docs: fix monitoring prefix config and worktree cleanup wording

- **docker**: Add Docker specification reference
  ([#291](https://github.com/tinaudio/synth-setter/pull/291),
  [`cffed86`](https://github.com/tinaudio/synth-setter/commit/cffed86afa9355a6acfb6c52084102492580c37d))

* docs(docker): add Docker specification reference

Succinct reference for entrypoint MODE dispatch, image targets, baked env vars, and known design
  issues.

Refs #265

* docs(docker): clarify spec vs current behavior, fix credential description

- **monitoring**: Add a reference guide for the current state of the wandb integration
  ([#283](https://github.com/tinaudio/synth-setter/pull/283),
  [`2a3a020`](https://github.com/tinaudio/synth-setter/commit/2a3a020e2ff4df213ef0832f8f750ce2aca3ba37))

* docs(monitoring): add a reference guide for the current state of the wandb integration

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

### Monitoring

- **data-pipeline**: Add R2 shard report script
  ([#238](https://github.com/tinaudio/synth-setter/pull/238),
  [`9f82ba0`](https://github.com/tinaudio/synth-setter/commit/9f82ba0a06e1ad541bf9f414d8e698dc0b0d33ff))

* feat(data-pipeline): add R2 shard report script

CLI script to analyze R2 shard prefixes — counts h5/metadata files, logical shards, total sizes, and
  flags corrupt/empty-shell files under a configurable size threshold.

Typed API: analyze_shards() returns ShardReport TypedDict, format_report() renders to plain text.
  Uses RcloneFile NamedTuple for self-documenting file entries.

Tests assert on typed dict fields (not string matching), with an R2 integration test that auto-skips
  when rclone is unavailable.

Refs #236

* fix(data-pipeline): address PR review comments on r2 shard report

* style(data-pipeline): add docstrings to test functions, revert tests/ interrogate exclude

Interrogate requires 80% docstring coverage. Instead of excluding tests/ from interrogate, add
  one-line docstrings to each test method.

* style(data-pipeline): document ruff per-file-ignores for r2 shard report

Add inline comments explaining why S603, S607, and T201 are suppressed for the shard report script
  and its test file.

* internal-fix: include stdout/stderr in failure path

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* raise error when unexpeected rclone output

---------

- **wandb**: Wandb provenance helper and config cleanup
  ([#285](https://github.com/tinaudio/synth-setter/pull/285),
  [`1e9cd88`](https://github.com/tinaudio/synth-setter/commit/1e9cd88e6d01ba3710981341fdfb7d5faf2e4655))

* internal-feat(monitoring): add wandb provenance config helper

Log github_sha, image_tag, and command to wandb.config at run start.

Refs #270

* internal-fix(monitoring): replace hardcoded wandb entity/project

Use env-var-backed defaults (tinaudio/synth-setter) per storage-provenance-spec §10. Legacy runs
  under benhayes/synth-permutations remain read-only.

Refs #271

* refactor(monitoring): rewrite provenance tests with fakes and state assertions

Replace MagicMock-heavy interaction tests with FakeWandbConfig fake and state assertions. Use real
  subprocess for git SHA (higher fidelity), mock only at the wandb boundary. Parametrize git failure
  test across FileNotFoundError and CalledProcessError.

* docs(monitoring): document hardcoded wandb entity/project in sweep configs

Sweep YAML files use hardcoded values because wandb sweep CLI does not support OmegaConf resolvers.
  WANDB_ENTITY/WANDB_PROJECT env vars override at runtime.

Refs #286

### Testing

- **ci**: Add flush investigation notebook and CI workflow
  ([#231](https://github.com/tinaudio/synth-setter/pull/231),
  [`d8b849a`](https://github.com/tinaudio/synth-setter/commit/d8b849a82627d81cd15b42ecacb8cb115312aa13))

* test(ci): add flush investigation notebook and CI workflow

Adds a Jupyter notebook that systematically compares audio output across four flush strategies
  (none, post_param, post_render, all) using MFCC and multi-resolution spectral loss. Includes
  self-consistency, shared vs isolated instance, and ordering dependence tests with heatmap
  visualizations.

CI workflow runs the notebook weekly in Docker (with headless X11 for VST loading), converts to HTML
  with embedded audio, and uploads as an artifact with provenance metadata.

Depends on Docker build infra from #229.

Closes #230

* fix(ci): address PR review comments on flush investigation

- Use git rev-parse HEAD instead of github.sha for Docker build + provenance - Mount only notebooks/
  dir, not full workspace (avoids hiding image plugins) - Pass SURGE_VST3_PATH env var for CI plugin
  path portability - Use load_plugin() instead of raw VST3Plugin() to match production init - Define
  STRATEGY_DESC once in helpers cell, remove duplicate - Clarify random configs are hardcoded
  constants, not runtime-generated - Fix incomplete markdown sentence in §4

Refs #230

- **ci**: Cache MNIST dataset to prevent flaky macOS test failures
  ([#241](https://github.com/tinaudio/synth-setter/pull/241),
  [`8d3b69b`](https://github.com/tinaudio/synth-setter/commit/8d3b69bc73c6befecf0d46c8c804da513f6e69c6))

* fix(ci): cache MNIST dataset to prevent flaky macOS test failures

* fix(ci): temporarily skip MNIST test on macOS due to download outage

The MNIST dataset host is currently unreliable from macOS CI runners, causing consistent
  test_mnist_datamodule failures. Skip this test on macOS until the cache is populated from a
  successful download.

Refs #240

- **testing**: Add requires_vst marker and preset-dependent param regression tests
  ([#233](https://github.com/tinaudio/synth-setter/pull/233),
  [`61e392b`](https://github.com/tinaudio/synth-setter/commit/61e392b6fbf37b2e930c145eaabd09fe1d1922d8))

Adds @pytest.mark.requires_vst for tests needing the Surge XT VST plugin. These tests are skipped
  locally (no plugin) but run in Docker builds where the plugin is installed.

Regression tests verify that preset-dependent parameters (e.g. a_osc_1_sawtooth) are accessible
  after load_preset() + flush+reset. This prevents future changes from accidentally gating the
  post-load flush.

Refs #225


## v0.1.0 (2026-03-21)

### Bug Fixes

- **ci**: Deduplicate check runs in auto-approve to ignore stale failures
  ([#175](https://github.com/tinaudio/synth-setter/pull/175),
  [`2596532`](https://github.com/tinaudio/synth-setter/commit/2596532e06cd1ce706f4856ee680f51bcff2bcc5))

* fix(ci): deduplicate check runs in auto-approve to ignore stale failures

The check-runs API returns ALL runs for a commit, including superseded re-runs of the same check. If
  check-pr-metadata failed on the first trigger but passed on re-run, the old failure still appeared
  in the results, permanently blocking auto-approve.

Fix: group check runs by name, sort by ID, take only the latest run per name before evaluating
  pass/fail/pending.

Fixes #174

* fix(ci): drop --paginate from check-runs query

The check-runs API paginates at 100 results. With --paginate, jq runs per-page rather than on the
  combined result, so group_by would only deduplicate within each page. Since this repo has ~5
  workflows and will never hit 100 check runs per commit, dropping --paginate is the simplest fix.
  If scale changes, switch to --paginate --slurp with flatten.

Refs #174

- **ci**: Enable Claude inline comments and switch to label-only trigger
  ([#115](https://github.com/tinaudio/synth-setter/pull/115),
  [`bb896de`](https://github.com/tinaudio/synth-setter/commit/bb896dea81d2a8731564ba21765f51444f8d0983))

- Add mcp__github_inline_comment__create_inline_comment to allowed tools (was denied, causing "No
  buffered inline comments") - Change trigger from opened/labeled to labeled-only with
  needs-claude-review label for on-demand reviews - Update prompt to instruct Claude to use the
  inline comment tool

- **ci**: Replace archived trilom/file-changes-action with tj-actions/changed-files
  ([#131](https://github.com/tinaudio/synth-setter/pull/131),
  [`697b025`](https://github.com/tinaudio/synth-setter/commit/697b02513a9365c637d0179cb1ce478719660603))

The trilom action fails with 403 after org transfer and is unmaintained.

Refs #25

- **ci**: Set codecov threshold to 1% — was allowing 100% coverage decrease
  ([#191](https://github.com/tinaudio/synth-setter/pull/191),
  [`36a45c0`](https://github.com/tinaudio/synth-setter/commit/36a45c0786206e4615ebc48adf45745462062188))

- **ci**: Skip no-commit-to-branch hook in main branch CI
  ([`628fa8e`](https://github.com/tinaudio/synth-setter/commit/628fa8e7aeded2a4d89227d68c92c839f28080db))

The no-commit-to-branch hook prevents commits to main locally, but always fails in CI because the
  checkout is on main. Use the standard SKIP env var to bypass it in the Code Quality Main workflow.

- **ci**: Switch auto-approve from check_suite to workflow_run trigger
  ([#64](https://github.com/tinaudio/synth-setter/pull/64),
  [`7c1991e`](https://github.com/tinaudio/synth-setter/commit/7c1991e51fcfdc2526861f3a798a467ac5f10de5))

check_suite events from GitHub Actions don't trigger other workflows. Switch to workflow_run which
  fires when Tests, Code Quality PR, or Claude Code Review complete. Also add workflow_dispatch for
  manual runs.

- **ci**: Update release-drafter comment from "master" to "main"
  ([`a77f8a6`](https://github.com/tinaudio/synth-setter/commit/a77f8a64477c5e9785a50981493772fc9655c128))

- **ci**: Use -m "not slow" marker filter instead of -k substring match
  ([`bdb194d`](https://github.com/tinaudio/synth-setter/commit/bdb194d27ec93bce1082f2bfcc0c18c9ab5ce748))

-k "not slow" filters by test name substring, not by pytest markers, so @pytest.mark.slow tests were
  silently running in every CI job.

- **config**: Update default model from ksin_ff to ffn
  ([`0c64016`](https://github.com/tinaudio/synth-setter/commit/0c6401640283ffd19ea7470cac7034fc1086b881))

- **datamodule**: Add configurable pin_memory to KSinDataModule
  ([`73fd6a5`](https://github.com/tinaudio/synth-setter/commit/73fd6a53378e8daa68a8696d1efe4c9ac069d39f))

- **datamodule**: Add configurable pin_memory to SurgeDataModule
  ([`12ab977`](https://github.com/tinaudio/synth-setter/commit/12ab97717109f4b790c77bb701cc4de87efea7f8))

- **pre-commit**: Remove duplicate --force-exclude from ruff hooks
  ([`d08a6ca`](https://github.com/tinaudio/synth-setter/commit/d08a6ca5c39a88b06df2f9f242aa3c83dc9815e3))

ruff-pre-commit already passes --force-exclude in its entry point, so adding it in args causes
  "cannot be used multiple times" error.

- **pre-commit**: Replace directory excludes with individual file excludes
  ([`f7d5196`](https://github.com/tinaudio/synth-setter/commit/f7d5196228df0bca615c8394b8198b0d91ca4ffb))

Expanded all directory-based exclude patterns (jobs/, scripts/, configs/, sweeps/, notebooks/) to
  list each failing file individually. This makes excludes explicit and allows a bot to fix files
  one at a time. Also removes duplicate --force-exclude from ruff hooks (already passed by
  ruff-pre-commit's entry point).

- **pre-commit**: Switch docformatter to local hook
  ([`300e66f`](https://github.com/tinaudio/synth-setter/commit/300e66f8ab21ce0a353a0214f878d3780224ed0e))

- **tests**: Disable torch.compile in test config fixture
  ([`bd3751d`](https://github.com/tinaudio/synth-setter/commit/bd3751d702ce06e6921e4108cff9284b51ba9344))

- **tests**: Reduce batch size in GPU test to prevent OOM
  ([`42514de`](https://github.com/tinaudio/synth-setter/commit/42514deb83c42c5060e7d1659a11462dcd94c25e))

- **tests**: Register OmegaConf resolvers at conftest module level
  ([`2ca7915`](https://github.com/tinaudio/synth-setter/commit/2ca7915b8d1bd250e19664ba55c11ca74d575efe))

- **tests**: Remove lr_monitor callback from test fixtures
  ([`48ba9ce`](https://github.com/tinaudio/synth-setter/commit/48ba9ce4358f1ecd60edb773bdc6674a8c3fcb44))

- **utils**: Guard OmegaConf resolver registration against double-registration
  ([`6e41ca6`](https://github.com/tinaudio/synth-setter/commit/6e41ca677a116dbe9ba10c312b56baff491cac43))

### Chores

- Add check-github-workflows, check-json, and validate-pyproject hooks
  ([`e900c6d`](https://github.com/tinaudio/synth-setter/commit/e900c6d69dbe020290f833b672a1b10c894096b2))

Replace generic check-jsonschema with check-github-workflows for schema validation of
  .github/workflows/*.yml. Add check-json for basic JSON syntax checking. Add validate-pyproject for
  pyproject.toml schema validation.

- Add check-jsonschema for future JSON/YAML validation
  ([`a0bb801`](https://github.com/tinaudio/synth-setter/commit/a0bb801762828c192553f81ceb10a1bdad1484d1))

Adds check-jsonschema hook for validating JSON and YAML files against JSON schemas. Currently
  configured as a pass-through for future schema validation rules.

- Add pyright type checker with legacy file exclusions
  ([`5be69cd`](https://github.com/tinaudio/synth-setter/commit/5be69cdde409d61ea5e544d2f080ce47cbcbc925))

Adds pyright (static type checker) as a local pre-commit hook. Configured to exclude all files
  currently failing type checks, allowing the hook to pass while preventing regressions in typed
  code going forward.

- Add ruff configuration to pyproject.toml
  ([`c8d909e`](https://github.com/tinaudio/synth-setter/commit/c8d909e3fb61824686d9e5c3ac162ceeb709cdc1))

- Fix prettier formatting in configs and remove exclusions
  ([`a3968c7`](https://github.com/tinaudio/synth-setter/commit/a3968c7856debee51751f80aa116c179bc088122))

- Fix trailing newlines in configs and remove pre-commit exclusions
  ([`9fbaae2`](https://github.com/tinaudio/synth-setter/commit/9fbaae2f41b3b66814bed8bd8e2496379357a3f1))

- Fix trailing whitespace and remove pre-commit exclusions
  ([`59fc1de`](https://github.com/tinaudio/synth-setter/commit/59fc1defab4b93f805730e89b24558a78e1b7294))

- Gitignore .claude/ except skills directory (will be used later to add claudeskills to repo for
  agents)
  ([`9a6130c`](https://github.com/tinaudio/synth-setter/commit/9a6130c0a3d13370b7ce7a1af703e459e3dde2a6))

- Reformat docstrings and remove docformatter exclusions
  ([`88e6a3f`](https://github.com/tinaudio/synth-setter/commit/88e6a3fe0ad8a0d551faa64126ae90f8ede06127))

- Reformat source files to pass ruff and docformatter
  ([`af72a57`](https://github.com/tinaudio/synth-setter/commit/af72a57b7d6759bfe5ee0ee96bd333863622c4ae))

Replaces the broad extend-exclude list with per-file-ignores targeting only the legacy files with
  known linting issues. This enables ruff to check more files while suppressing specific rules in
  files that aren't ready for cleanup.

- **ci**: Add PR metadata gate compliance to github-taxonomy skill
  ([#197](https://github.com/tinaudio/synth-setter/pull/197),
  [`005a62c`](https://github.com/tinaudio/synth-setter/commit/005a62cd7297337079250fd3b681da84c8755290))

* chore(ci): add PR metadata gate compliance to github-taxonomy skill

Add a required "Ensuring PRs Pass the Metadata Gate" section that runs before every gh pr create.
  The section:

- Verifies linked issues have all 3 required fields (issue type, domain label, milestone) with exact
  gh/GraphQL commands - Provides a quick-create one-shot command for new compliant issues - Includes
  domain-to-milestone mapping table - Updates the PR checklist to include linked-issue compliance

This ensures the pr-metadata-gate CI check passes automatically when Claude creates PRs.

Closes #196

* fix(skill): address Copilot review on github-taxonomy skill

- Parameterize issue type in Step 3 GraphQL snippet (was hardcoded to "Task", now says "ISSUE_TYPE"
  with comment to replace) - Add bare #N as a valid linking option in Step 4 (passes gate, no
  auto-close) - Make PR checklist consistent: Fixes/Closes vs Refs vs bare #N

Refs #196

- **deps**: Add h5py, hdf5plugin, dask[distributed] to requirements
  ([`bf864fb`](https://github.com/tinaudio/synth-setter/commit/bf864fb09ec053dd7496a79069cc07fcda51c364))

- **docs**: Align doc work items with GitHub issue metadata
  ([#130](https://github.com/tinaudio/synth-setter/pull/130),
  [`6b5ed2a`](https://github.com/tinaudio/synth-setter/commit/6b5ed2acb0238a5f029e74d76d426c72db4ad3e4))

* chore(docs): align doc work items with GitHub issue metadata

Audit both implementation docs against GitHub issues and fix metadata gaps: priorities, labels,
  blocking relationships, and missing references.

GitHub changes: - Create `storage` domain label for cross-cutting R2/rclone work - Set priorities on
  22 issues (data pipeline + eval pipeline) - Add 13 missing blocking relationships from eval doc §8
  matrix - Fix titles on #3, #7, #120, #121 to conventional commit format - Remove non-domain labels
  from #76, #89, #97 - Add `storage` label to #90, #91, #92, #93, #99 - Create #128 (W&B checkpoint
  resolver) as sub-issue of #98

Doc changes: - data-pipeline-implementation-plan.md: add #120, #121, #122 to §11 - eval-pipeline.md:
  add #92, #95, #96, #128 to §8 blocking matrix; add #128 to §12 PR#2 and Phase 7 -
  github-taxonomy.md: add `storage` label, multi-label policy, view - github-taxonomy SKILL.md: add
  `storage` to domain label list

* fix(docs): address PR review — remove stale ASCII graph, add #92 blocker

- Remove ASCII dependency diagram from eval doc §8 (goes stale; table is the authoritative source) -
  Add #90 as blocker for #92 in the blocking matrix (rclone wrapper is a prerequisite for R2
  checkpoint sync)

* ci: retrigger code-quality

- **gitignore**: Exclude plugins/ for VST3 build artifacts
  ([`6dfde31`](https://github.com/tinaudio/synth-setter/commit/6dfde31aa2ba2bc8ae035134012c827314e92679))

- **lint**: Exclude notebooks/, scripts/, and src/utils/__init__.py from ruff
  ([`67b69af`](https://github.com/tinaudio/synth-setter/commit/67b69af0b3eb14ab5e6b018eca7f520bc9780577))

These files have pre-existing lint issues tracked in #25. Also includes ruff auto-fixes in
  tests/helpers/ (modernize typing imports to PEP 604).

- **pre-commit**: Add "ot" to codespell ignore list
  ([`5b8b518`](https://github.com/tinaudio/synth-setter/commit/5b8b518bc60527ad778300c9098bce69b73e7c97))

- **pre-commit**: Add checkmake hook
  ([`7cfbd11`](https://github.com/tinaudio/synth-setter/commit/7cfbd11ce8c6118011e3309f09d2e4157fc14ff3))

- **pre-commit**: Add no-commit-to-branch, gitlint, debug-statements, and C++ linters
  ([`3df9cca`](https://github.com/tinaudio/synth-setter/commit/3df9cca91f44f7c0b09a1da8ce3df8add6ba29fb))

Add no-commit-to-branch to prevent accidental commits to main. Add gitlint for conventional commit
  message validation. Add debug-statements to catch pdb/breakpoint() left in code. Add clang-format
  (Google style) and cpplint for future C++ files; both are no-ops until .cpp/.h files exist. Add
  missing shebangs to 4 shell scripts.

- **pre-commit**: Add pbr dependency to bandit, add legacy excludes
  ([`73e442f`](https://github.com/tinaudio/synth-setter/commit/73e442fc8faf4fb6d724dfa5808fa5612c9b148a))

- **pre-commit**: Bump interrogate to 1.7.0, add legacy excludes
  ([`974a389`](https://github.com/tinaudio/synth-setter/commit/974a3896a2714ef34d497c4a0713037eabdf66c4))

- **pre-commit**: Bump mdformat to 1.0.0, pin dependencies
  ([`fa4b4f8`](https://github.com/tinaudio/synth-setter/commit/fa4b4f8e7fbf0d0263ab12e44754e345dc7bccc6))

- **pre-commit**: Narrow interrogate exclusions to files below 80% coverage
  ([`e001261`](https://github.com/tinaudio/synth-setter/commit/e001261c47479d2a250b2e4bb3d0375f2808a4f1))

Remove 21 files that already meet the 80% docstring threshold from the interrogate exclude list.
  Remaining 27 exclusions have genuine gaps.

- **pre-commit**: Remove nbqa hooks
  ([`b901ddf`](https://github.com/tinaudio/synth-setter/commit/b901ddfe8ba879484ddb6ee37126b19450cd6528))

- **pre-commit**: Switch pyright to language: system and narrow exclusions
  ([`8188d52`](https://github.com/tinaudio/synth-setter/commit/8188d52a885d9150e918901c3a3bcc1add3f9437))

Use active environment instead of isolated venv so pyright can resolve project deps. Reduces
  exclusion list from 57 broad patterns to 34 specific files that have real type errors.

- **review**: Add code review infrastructure and lint cleanup agent
  ([#31](https://github.com/tinaudio/synth-setter/pull/31),
  [`46ac2f4`](https://github.com/tinaudio/synth-setter/commit/46ac2f4cf1e187519db71513203759a321176041))

* Add lint cleanup agent documentation

Document the lint cleanup agent's goal, scope, workflow, rules, and completion criteria.

* Revise lint cleanup process in documentation

Updated lint cleanup instructions to reflect changes in tools and processes.

* fix(review): correct tool references and harden CI workflows

- Replace Black/flake8/bandit references with ruff-format/ruff in CLAUDE.md, lint-cleanup agent, and
  python-style skill to match actual pre-commit config - Fix CLAUDE.md ruff rule list to match
  pyproject.toml (E,F,I,S,T,UP,W) - Add explicit error handling in pr-metadata-gate.yaml for gh API
  failures instead of silently defaulting to 0 - Consolidate redundant claude-review.yml jobs, add
  API key validation, remove over-scoped id-token:write, add timeout-minutes - Add fork PR
  skip-notice job and SKIP env var comment

* fix(docs): fix broken backtick formatting in lint-cleanup agent

* fix(docs): fix docformatter config reference in lint-cleanup agent

* fix(docs): clarify auto-fix command flags in lint-cleanup agent

* style: apply prettier formatting to claude-review workflow

- **skills**: Add design-doc skill to project
  ([#113](https://github.com/tinaudio/synth-setter/pull/113),
  [`28c852d`](https://github.com/tinaudio/synth-setter/commit/28c852d6a1b04e31ddc0c4818634c93669b877e8))

- **tests**: Add TODO comments for issues #39 and #40
  ([`25b70f9`](https://github.com/tinaudio/synth-setter/commit/25b70f992afae9de8b7e60b03aed22e9dc073df8))

### Code Style

- **tests**: Modernize type hints to PEP 604/585 builtins
  ([`d303b41`](https://github.com/tinaudio/synth-setter/commit/d303b418bae3dd6b6a732e670c008f4b6451e3f0))

### Continuous Integration

- Add GitHub App auto-approve bot ([#57](https://github.com/tinaudio/synth-setter/pull/57),
  [`c828c7a`](https://github.com/tinaudio/synth-setter/commit/c828c7a4ace84486887fab4247422bd0d5f15092))

* ci: add GitHub App auto-approve bot

Approves PRs when all conditions are met: - PR is not a draft - All CI checks pass (tests,
  code-quality, coverage) - Claude review verdict is APPROVE (no BLOCK issues) - No unresolved
  Copilot review threads

Requires APPROVAL_BOT_APP_ID and APPROVAL_BOT_PRIVATE_KEY secrets from a GitHub App with
  pull-request write permission.

* fix(ci): address Copilot review feedback on auto-approve bot

- Move draft/fork checks into steps (fix check_suite trigger skipping) - Add base branch check (only
  approve PRs targeting main) - Add issues:read and checks:read permissions - Require explicit
  VERDICT: APPROVE (fail closed on unknown verdicts) - Match specific app slug for duplicate check
  instead of any [bot]

* fix(ci): remove Claude dependency from approval bot

Bot now approves based on: 1. Not a draft/fork, targets main 2. All CI checks pass 3. No unresolved
  Copilot review threads

Claude review is independent — it leaves comments but doesn't gate approval.

* fix(ci): require Copilot review before auto-approve

Prevents instant approval when CI passes before Copilot has reviewed. Bot now requires at least one
  Copilot review to exist AND all threads resolved.

* fix(ci): fail auto-approve on check failures, simplify pending logic

- Failed checks now fail the workflow (exit 1) instead of silently skipping - Pending checks skip
  without promising re-evaluation - Simplified check-runs query into a single API call

* fix(ci): add --repo flag to gh pr review to fix missing git context

- Add HYDRA_FULL_ERROR, skip slow tests, increase verbosity
  ([`c54cecd`](https://github.com/tinaudio/synth-setter/commit/c54cecd411a246768ec525af7994babcb973bed9))

- Add test-expensive.yml for full test suite including slow tests
  ([`bea71dc`](https://github.com/tinaudio/synth-setter/commit/bea71dc181311da28a0de7f1d1278aa33b021ee8))

Runs the complete test suite (no marker filtering) on manual dispatch and post-merge to main.
  GPU-requiring tests are auto-skipped on CPU runners via @RunIf(min_gpus=1).

- Add workflow_dispatch trigger to test.yml
  ([`0c18b88`](https://github.com/tinaudio/synth-setter/commit/0c18b8838c94e85eba867aed25461380eda058b8))

- Bump actions in code-quality workflows to latest versions
  ([`41df01c`](https://github.com/tinaudio/synth-setter/commit/41df01c7982570d3fd2051d3ddd234ca2d1a5715))

- Bump actions in test.yml to latest versions
  ([`e2c5dd1`](https://github.com/tinaudio/synth-setter/commit/e2c5dd16765bc3c583169d70622364fe4ac61a35))

- Bump release-drafter to v6
  ([`3a0bd2c`](https://github.com/tinaudio/synth-setter/commit/3a0bd2cb4d5e693396704b90b4ec14f7aa0b2d26))

- Drop Python 3.8/3.9, drop Windows, simplify macOS matrix
  ([`ba1189d`](https://github.com/tinaudio/synth-setter/commit/ba1189d4b2c91905f5c16e64f4274a32a35e364b))

- Improve CI workflow efficiency, auto-approve visibility, and Claude review UX
  ([#109](https://github.com/tinaudio/synth-setter/pull/109),
  [`54841e7`](https://github.com/tinaudio/synth-setter/commit/54841e7bf0eba7e6029d8e22347f616725f6651b))

* ci: skip test suite for docs-only changes, make Claude review manual

- Add paths-ignore for **/*.md and docs/** to test.yml and test-expensive.yml - Switch
  claude-review.yml from pull_request trigger to workflow_dispatch with pr_number input for
  on-demand reviews - Remove fork skip-notice job (no longer relevant without PR trigger)

* ci(claude-review): add fork guard and persist-credentials: false

- Reject fork PRs before checkout to prevent secrets exposure - Set persist-credentials: false to
  avoid token leakage via git config

Addresses review comments on #109.

* ci(auto-approve): fail workflow when conditions not met instead of silent success

Only show a green checkmark when the bot actually approves the PR. All other cases (checks pending,
  draft, fork, no Copilot review) now exit 1 with a descriptive error message.

* ci(auto-approve): use exit 78 (neutral) for not-ready-yet states

- Checks pending, Copilot not reviewed, unresolved threads → exit 78 (grey skipped icon, not red X)
  - Actual failures (wrong base, draft, fork, failed checks) → exit 1 (red X) - Approval → exit 0
  (green tick)

* ci(auto-approve): use Checks API for honest neutral/success/failure status

GitHub Actions doesn't support neutral exit codes from shell steps (exit 78 removed in v1→v2;
  actions/runner#662). Use the Checks API via actions/github-script to create a separate
  "Auto-approve status" check run with the correct conclusion: - success (green): PR was approved -
  neutral (grey): not ready yet (checks pending, Copilot not reviewed) - failure (red): ineligible
  (draft, fork, wrong base, failed checks)

The workflow itself always succeeds — it's plumbing. The check run is what shows the real status on
  the PR.

* ci: remove fork guard, align checkout v6, fix Copilot login match

- Remove fork guard from claude-review.yml (workflow_dispatch is only triggerable by collaborators
  with write access) - Align actions/checkout@v4 → v6 to match other workflows - Fix GraphQL filter
  to use startswith("copilot-pull-request-reviewer") instead of exact match, since bot logins may
  include [bot] suffix

* fix(auto-approve): exclude Auto-approve status check from CI query

The Checks API check run we create ("Auto-approve status") appears in subsequent runs' check-runs
  query. A previous failure conclusion would count toward FAILED and block approval even after
  conditions are resolved. Exclude it alongside auto-approve and claude-review.

* ci(auto-approve): remove redundant fork check

Fork PRs can't pass the other conditions anyway — CI won't run (no secrets), Copilot won't review —
  so auto-approve will never approve a fork PR. The existing conditions are the block.

* ci(claude-review): switch to pull_request trigger with label-based re-review

workflow_dispatch can't post inline PR comments — the MCP server that handles them requires
  pull_request context (claude-code-action#635).

Switch to pull_request trigger with: - Automatic review on PR open - On-demand re-review via
  "needs-claude-review" label - paths-ignore for docs-only PRs - Fork PRs skipped (no secrets
  available)

* ci(claude-review): allow review on docs/markdown PRs

Claude reviewing docs is useful (catches inconsistencies, stale references). Cost is controlled by
  only auto-running on PR open, with label-based re-review after.

- Increase test timeouts
  ([`c5d48e4`](https://github.com/tinaudio/synth-setter/commit/c5d48e4d22e8543db9a9482a4c50fb735f058f54))

- Pin code-coverage runner to ubuntu-22.04
  ([`02e7068`](https://github.com/tinaudio/synth-setter/commit/02e7068cdcb96723f347d8a26bd8922a60447a2e))

- Remove redundant pip install pytest
  ([`fdaa6e9`](https://github.com/tinaudio/synth-setter/commit/fdaa6e9856ad480ccbb996447344e2c13c43cc54))

- **claude**: Add code review automation and coding standards
  ([`39ca79c`](https://github.com/tinaudio/synth-setter/commit/39ca79c41a5f5901794b782da39b5b47db1336e6))

Add CLAUDE.md with project conventions and seven review skills (tdd-implementation, code-health,
  ml-data-pipeline, project-standards, python-style, shell-style, ml-test), a review orchestrator
  skill, and a GitHub Action for automated PR reviews with dual checklist + deep review jobs. Also
  gitignore .claude/settings.local.json and plugins/.

- **github**: Add issue templates matching github-taxonomy conventions
  ([#170](https://github.com/tinaudio/synth-setter/pull/170),
  [`85691db`](https://github.com/tinaudio/synth-setter/commit/85691db5e31411bd2c03b4091b881ede42f3fb2f))

Templates enforce required metadata (domain label, milestone, parent issue) at creation time for all
  five issue types: Epic, Phase, Task, Bug, Feature. Blank issues disabled.

Refs #148, #149, #114

- **github**: Enforce taxonomy metadata in PR gate
  ([#171](https://github.com/tinaudio/synth-setter/pull/171),
  [`f45f868`](https://github.com/tinaudio/synth-setter/commit/f45f8689c4b958f5dc49f557eb191a4218e96a60))

* ci(github): enforce taxonomy metadata on linked issues in PR gate

PR metadata gate now checks that linked issues have an issue type, at least one domain label, and a
  milestone assigned. Fails with a clear message listing what's missing.

Refs #148, #149, #114

* fix(ci): address review feedback on metadata gate

- GraphQL failure: fail with error instead of silently producing false "no issue type" failure (was
  flaky) - REST failure: fail with error instead of silently skipping validation (was a free pass on
  API errors) - printf format string: use '%b' to avoid corrupted output if issue metadata contains
  % sequences - Skip PR numbers in body refs (PRs share issue number space)

* fix(ci): move REST fetch before GraphQL to detect PRs early

GraphQL issue() query fails for pull requests since PRs aren't issues in GraphQL. Move the REST call
  first so we can detect and skip PRs before the GraphQL type check runs.

* fix(ci): add taxonomy sync comment for hardcoded types/labels

- **review**: Add PR metadata gate and refine review skills
  ([`c94e015`](https://github.com/tinaudio/synth-setter/commit/c94e0155caf10f317f2ad982c9a6c046f3283c2c))

Add hard-fail CI check requiring every PR to have a milestone, linked issue, and GitHub project.
  Update code-health skill with H1-H3 gate rows. Fix python-style section numbering and review skill
  references.

- **review**: Enable full claude review output logging, relax metadata gate
  ([#56](https://github.com/tinaudio/synth-setter/pull/56),
  [`7e2fec9`](https://github.com/tinaudio/synth-setter/commit/7e2fec9b8fb422aa51ddd435fd02fc79c9a07677))

* ci(review): add auto-approve when no blocking issues found

* fix(ci): accept non-closing issue references in PR metadata gate

The gate now matches 'Part of #N', 'Related to #N', 'Ref #N', and 'See #N' in addition to closing
  keywords.

* fix(ci): address Copilot review feedback on auto-approve and metadata gate

- Clarify prompt so VERDICT comment is not prefixed with BLOCK:/WARN: - Filter auto-approve by
  github-actions[bot] author to prevent spoofing - Fix error message to match actual #N check

* ci(review): gate auto-approve on resolved Copilot threads

Auto-approve now checks that all Copilot review threads are resolved before approving. If any are
  unresolved, a warning is logged and approval is skipped.

* fix(ci): remove approval logic from claude-review, enable full output logging

Approval is handled by auto-approve.yml (PR #57). Enable show_full_output to debug why the review
  action ran but posted nothing.

- **review**: Grant tool permissions so review output posts to PR
  ([#65](https://github.com/tinaudio/synth-setter/pull/65),
  [`0957ac3`](https://github.com/tinaudio/synth-setter/commit/0957ac3fd1cd989c1e13e09ae238b85e448e5813))

* ci(review): grant tool permissions so review output posts to PR

Claude's Bash calls (git log, git show, gh pr) were denied during review runs, preventing inline
  comments from being posted. Add a settings block that pre-allows the read-only tools the reviewer
  needs.

* ci(approve): exclude claude-review from required CI checks

Claude review is advisory — if it fails (e.g. API credits exhausted) it should not block
  auto-approve. Exclude it from the check-runs gate alongside auto-approve itself.

- **test**: Use gpu-x64 runner for expensive tests
  ([#125](https://github.com/tinaudio/synth-setter/pull/125),
  [`5114b81`](https://github.com/tinaudio/synth-setter/commit/5114b816cc12b509697f308647024b37388b2ab1))

* ci(test): use gpu-x64 runner for expensive test workflow

GPU runner enables @RunIf(min_gpus=1) tests to execute instead of being skipped.

* ci(test): add gpu marker and run only GPU tests on gpu-x64

- Add @pytest.mark.gpu to all @RunIf(min_gpus=1) tests - Register gpu marker in pyproject.toml -
  Rename workflow to "GPU Tests", run pytest -m gpu only

### Documentation

- Add distributed data pipeline design doc ([#63](https://github.com/tinaudio/synth-setter/pull/63),
  [`e9246c0`](https://github.com/tinaudio/synth-setter/commit/e9246c0a4119970c14d4ccc630f20431313f73ca))

* docs: add distributed pipeline design document

* docs: update context to include project title

* docs: rename data pipeline deasign doc and align diagrams

- **design**: Add drafts for eval, train pipeline design + implementations docs
  ([#101](https://github.com/tinaudio/synth-setter/pull/101),
  [`437a050`](https://github.com/tinaudio/synth-setter/commit/437a0507aba6f35c0f73bcb970fd315adc7258c9))

- **design**: Add org migration checklist
  ([#116](https://github.com/tinaudio/synth-setter/pull/116),
  [`292c95a`](https://github.com/tinaudio/synth-setter/commit/292c95a90afa00ad2eda124695bae4374bb5d17c))

* docs(design): add org migration checklist

Actionable checklist for transferring synth-permutations from the ktinubu personal account to a
  GitHub organization. Covers pre-migration prep, the transfer process, post-migration org features
  (Issue Types, native blocking, Issue Fields), and verification steps.

* docs: move org migration checklist to docs/ (not a design doc)

- **design**: Add storage v1.0.0 milestone to github taxonomy
  ([#142](https://github.com/tinaudio/synth-setter/pull/142),
  [`09d6951`](https://github.com/tinaudio/synth-setter/commit/09d695192629de210e647af1c15e76f1b3b773d2))

- **design**: Align design doc, implementation plan, and GitHub issues
  ([#84](https://github.com/tinaudio/synth-setter/pull/84),
  [`bb0d526`](https://github.com/tinaudio/synth-setter/commit/bb0d526e9fe932fbc687bf5ca8b8cec56b4e75be))

* docs(design): add mel_shape to ShardSpec schema

§7.5 lists mel_spec as a validated HDF5 dataset but §14.1 ShardSpec only had audio_shape and
  param_shape. Add mel_shape: tuple[int, int] for explicit shape validation rather than deriving
  from audio_shape and sample rate.

Refs: #69

* docs(design): add implementation plan aligned with GitHub issues

Port the implementation plan to docs/design/ next to the design doc. All 14 steps cross-reference
  their GitHub issues (#68-#73, #76-#82). Review findings (M1-M5, G1-G10, N1-N8, B1-B9, R1-R14)
  folded into relevant steps. Reference test snippets preserved as specs.

New gaps found during port and review (GP1-GP10): - GP1: generate --dry-run not tested - GP2: status
  --json output not specified - GP3: no auth validation failure test - GP4: plugin_path validation
  before materialization - GP5: no --log-level CLI flag - GP6: worker quarantine path not in Step 10
  - GP7: skip-if-valid optimization missing - GP8: storage layer missing path helpers - GP9: status
  should overlay worker errors - GP10: design doc schema gaps to fix

Refs: #74

* docs(design): fold GP1-GP10 gaps into relevant implementation steps

Move gap findings from appendix-only into the steps they affect: - GP1/GP3/GP4/GP5 → Step 11
  (generate CLI): dry-run test, auth failure test, plugin_path validation, --log-level flag -
  GP2/GP9 → Step 12 (status CLI): --json flag, worker error overlay - GP6/GP7 → Step 10 (worker):
  quarantine path, skip-if-valid - GP8 → Step 6 (storage): quarantine/attempts/finalize path helpers
  - GP10 → Step 5 (schemas): design doc schema gaps to fix

* docs(design): add per-shard process isolation via multiprocessing spawn

Workers render each shard in a separate OS process using
  multiprocessing.Process(start_method="spawn"). A SIGSEGV or OOM kill in the VST plugin terminates
  only that child — the parent catches the exit code, quarantines the shard, and continues. spawn
  starts a fresh interpreter per child: no inherited plugin state, no shared globals.

Design doc: new §7.8.1 with trade-off table (direct call vs fork vs spawn vs subprocess). Updated
  §7.8 Layer 1 from try/except to process isolation. Implementation plan: updated Step 10 worker
  behaviors, Assumption 7, and test list.

Refs: #71, #74

* docs(design): align Phase/Step terminology across docs and GitHub issues

Rename "PR #1-6" → "Phase 1-6" and "Step 1-14" → "Step N.M" in both the implementation plan and
  design doc. Updates Appendix D to link the current implementation plan and GitHub issue hierarchy.

* docs(design): address Copilot review comments on PR #83

- Fix multiprocessing API: use get_context("spawn").Process() not Process(start_method="spawn") (4
  locations across both docs) - Fix SIGSEGV test: os.kill(os.getpid(), signal.SIGSEGV) not
  SystemExit(-11) - Add missing import random to design doc code example - Document EXIT trap
  SIGKILL limitation with mitigations - Add mel_shape inline comment clarifying (mels, frames) shape
  - Fix shard_id type in reference tests: int everywhere, format at path layer - Add Step 1.5
  (.env.example) section to implementation plan - Clarify input_spec.json (frozen spec) vs
  config.yaml (provenance) artifacts

* docs(design): address Copilot review round 2, simplify process isolation

- Fix TDD priority wording: scope test-first to implementation steps only - Add Step 1.5 to
  infrastructure convention list - Fix shard_id in status reference test (use
  spec.shards[i].shard_id) - Fix Appendix D Phase 1 step range to 1.1-1.5 - Fix OOM-killed wording
  to distinguish worker vs child processes - Simplify _render_shard: direct import instead of
  importlib dispatch - LocalBackend stays in-process for tests (closures OK, no spawn) - Dual-RNG
  seeding deferred to post-launch P3 (#100)

* docs(design): add issue refs for steps 2.2, 2.3, 4.1, 4.2, 6.1

Created sub-issues #102-#106 for previously untracked steps and added blocking relationships across
  all pipeline issues.

Refs #69, #71, #73, #74

- Add missing `import numpy as np` to _render_shard snippet (§7.8.1) - Comment out seeding lines in
  snippet to match P3 deferral note - Remove generate_fn from production worker interface — child
  imports make_dataset directly; LocalBackend accepts generate_fn for tests only - Rewrite
  Assumption 7 to match direct-import decision

Refs #84

* docs(design): address Copilot review round 3

- Remove stale generate_fn from run_worker signature (line 508) - Fix SIGSEGV test: must use spawn
  path, not LocalBackend in-process - Align _write_valid_shard helpers on shard_id: int signature -
  Add mutmut to Step 1.1 dev dependencies for verification strategy

* docs(design): address Copilot review round 4

- Fix step count: 14 → 15 (Step 1.5 was added but count not updated) - Add `import numpy as np` to
  P3 seeding snippet for self-containedness - Fix LocalBackend description in §7.9: runs in-process,
  not Docker - Fix trade-off table: spawn testability via LocalBackend in-process inject - Step 3.1
  now references both #70 (phase) and #7 (sub-issue)

* docs(design): add prior work section crediting benhayes@, update scale to 15M

- Add §1 "Prior Work" subsection crediting benhayes@'s generation infrastructure: VST rendering,
  param specs, resharding, Docker, upload, DataModule, and orchestration - Describe what the old
  pipeline did well at prototype scale - Explain why it breaks at research scale (500k-15M): local
  storage bottleneck, no crash resilience, no validation, no distributed coord - Update scale target
  from "500k-1M+" to "500k-15M" - Add "Builds on" credit line to implementation plan header

* docs(design): fix prior work scale — works up to hundreds of thousands

* docs(design): fix prior work attribution — credit benhayes@ accurately

benhayes@ built: VST rendering, param specs (~1300 lines), plugin interface, resharding, DataModule,
  40+ Hydra experiment configs, multi-logger support, Optuna integration, SGE job scripts, CI, docs.

ktinubu@ added: orchestration, R2 upload, Docker entrypoint, parallel shards, RunPod scaling,
  finalization.

* docs(design): rename implementation-plan.md to data-pipeline-implementation-plan.md

Matches the data-pipeline.md naming convention for discoverability. Updated Appendix D link and 7
  GitHub issue bodies (#74, #84, #102-#106).

- **design**: Align eval pipeline doc with GitHub taxonomy
  ([#136](https://github.com/tinaudio/synth-setter/pull/136),
  [`34d99b0`](https://github.com/tinaudio/synth-setter/commit/34d99b09299cb83392fc44de345eee66b67c41d6))

* docs(design): align eval pipeline doc with GitHub taxonomy hierarchy

Restructure the eval pipeline design doc to conform to the Epic → Phase → Task hierarchy defined in
  github-taxonomy.md. Replaces PR-group organization with 5 phases, adds issue mapping table,
  per-phase metadata, and canonical dependency tracking notes.

Refs #98, #99

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs(design): add phase issue numbers and review fixes

- Replace TBD phase entries with real issue numbers (#137–#141) - Add priority note to §8 Phase Plan
  header - Move #92 (R2 checkpoint sync) into Phase 2 as Task 2.4 - Add milestone/label columns to
  standalone tasks table (#95) - Add Phase issue numbers to section headings and per-phase metadata
  - Update task parent references to use issue numbers

Refs #98, #99, #137, #138, #139, #140, #141

* docs(design): move #95 into Phase 5, remove standalone tasks section

Refs #95, #141

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- **design**: Align eval pipeline with storage-provenance-spec
  ([#145](https://github.com/tinaudio/synth-setter/pull/145),
  [`0659037`](https://github.com/tinaudio/synth-setter/commit/0659037eff5b591a927e8f1ead9b807b838605b2))

* docs(design): align eval pipeline with storage-provenance-spec

Align eval pipeline design doc with the authoritative storage-provenance-spec: - R2 paths: 3-segment
  → 6-segment eval paths, add run ID to data paths - W&B: synth-permutations →
  tinaudio/synth-setter, fix artifact naming (model-flow-simple, data-surge-simple,
  eval-surge-simple), type eval-results - Fix job_type (evaluation), add_reference protocol (s3://),
  github_sha - Add cross-references to storage-provenance-spec and promotion-pipeline-reference -
  Adopt spec §1 ID terminology in wandb.config keys

Refs #98, #99, #122

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs(design): add explicit R2 path pattern ref, clarify dataset_root

- Add storage-provenance-spec §2 path pattern in §6.3 before upload example - Use spec variable
  names ({dataset_config_id}/{dataset_wandb_run_id}) in dataset_root config comments and Task 1.1
  description

Refs #98, #122

* docs(design): use underscore config IDs, fix PR review comments

- Config IDs use underscores to match config filenames: surge_simple, flow_simple (not hyphens) per
  storage-provenance-spec §1 - Reword resolver as proposed implementation (Task 3.1, #128) - Fix
  link anchor to storage-provenance-spec §9 - Fix relative path to promotion-pipeline-reference.md
  (../reference/)

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- **design**: Centralized provenance & storage design doc
  ([#127](https://github.com/tinaudio/synth-setter/pull/127),
  [`98f56b9`](https://github.com/tinaudio/synth-setter/commit/98f56b97b2c2cfcd58399a388522b800a6c73d97))

* docs(design): add model promotion pipeline reference

Add reference doc for the W&B → GitHub Release promotion workflow including promote script, GitHub
  Actions workflow, and usage examples.

Exclude from mdformat due to nested code fences in Python f-strings.

Refs #122

* docs(design): add storage & provenance spec

Centralized spec for R2 paths, W&B artifacts, IDs, GitHub Actions, and secrets. Authoritative source
  — other design docs will point here.

* fix(docs): use 4-backtick fences for nested code blocks

Replace mdformat exclusion with proper 4-backtick fences in promotion-pipeline-reference.md. Fixes
  nested triple-backtick f-strings in the promote script.

* docs(design): address PR review feedback

- Add standard headers (Status/Last Updated) to both docs - Fix GH_TOKEN vs GITHUB_TOKEN
  inconsistency in promote script - Update actions versions to v6, Python to 3.10 - Update defaults
  to tinaudio/synth-setter - Add TODO for unused `previous` param in format_release_body - Mark
  artifact-provenance-reference.md as TBD - Note W&B identity values are target pending migration

* docs(design): pare down promotion pipeline reference

Remove: - Architecture Overview (restates section headers) - Training Script Requirements (belongs
  in training doc) - What Gets Created (redundant with script/workflow) - Downstream CLI Model
  Download (speculative future) - Secrets Required (now in storage-provenance-spec §9) - What You
  Don't Need (noise)

~430 lines → ~220 lines.

* docs: move promotion-pipeline-reference to docs/reference/

It's a reference doc, not a design doc. Update link in storage-provenance-spec.md.

* docs(design): simplify artifact naming and drop milliseconds

- Artifact names use config_id only (e.g., diva-v1), W&B auto-versions - Drop milliseconds from
  timestamps: YYYYMMDDTHHMMSSZ - Add note about W&B 64-char run ID limit

* fix(docs): fix broken cross-ref link and iterator exhaustion bug

- Fix relative link to storage-provenance-spec.md (../design/) - Materialize used_artifacts()
  iterator before iterating to avoid exhaustion bug in format_release_body

* docs(design): add invariants, artifact-storage mapping, and review fixes

- Add config ID global uniqueness constraint - Add dataset immutability rule - Add GITHUB_SHA
  enforcement for CI workflows - Add secret scoping guidance - Replace migration note with concrete
  rule (legacy read-only) - Add artifact → storage mapping section - Add invariants section (5
  rules)

* docs(design): fix lineage DAG to show eval's dual inputs

Eval run consumes both the model artifact and a dataset artifact (which may differ from the training
  dataset for cross-eval).

* docs(design): clarify promotion inputs, release schema, alias strategy

- Specify promote input is train_run_id (not ambiguous run_id) - Add GitHub Release body schema -
  Define alias strategy: :latest (auto), :best (training), :production (promote)

* docs(design): rename to wandb_run_id, add eval to promotion, fix paths

- Rename *_run_id → *_wandb_run_id (path format agnostic to ID generation) - Add
  dataset_wandb_run_id to train/ and eval/ R2 paths for full lineage - Promote workflow now requires
  both train_wandb_run_id and eval_wandb_run_id - Release body schema includes train metrics, eval
  metrics, and both datasets - Promote sets :production alias on model artifact

* docs(design): fix R2 secrets scope and S3-compatible URI

- Add training to R2 secret users (needs read for shards, write for checkpoints) - Clarify artifact
  references use s3:// (R2 is S3-compatible), not r2://

* docs(design): add R2 endpoint env var and completion marker invariant

- Document AWS_ENDPOINT_URL / WANDB_S3_ENDPOINT_URL requirement for R2 - Add invariant: runs must
  not consume data lacking completion markers

* docs(design): prefix eval-results artifact names with eval-

Avoids W&B name collision when eval_config_id matches a dataset_config_id (e.g., diva-v1 as both
  dataset and eval-results).

* docs(design): prefix all artifact names by type

data-{config}, model-{config}, eval-{config} — prevents W&B name collisions across artifact types
  without relying on global uniqueness.

* docs(design): specify eval workflow inputs

Eval needs train_wandb_run_id (to find the model) and eval_config_id (which dataset to evaluate on).
  Generic 'experiment' was ambiguous.

* docs(design): remove config filename uniqueness constraint

No longer needed — type-prefixed artifact names (data-, model-, eval-) prevent collisions even if
  config filenames overlap across directories.

- **design**: Github metadata taxonomy, conventions & skill
  ([#108](https://github.com/tinaudio/synth-setter/pull/108),
  [`19ed75d`](https://github.com/tinaudio/synth-setter/commit/19ed75ddc97c9a3af308d7356b18d6cc7203d31e))

* docs(design): add GitHub metadata taxonomy with gap analysis

Document the full GitHub metadata taxonomy — 5 Projects V2, 21 labels, 3 milestones, 3 epics,
  parent-child relationships, blocking conventions, and priority tiers. Includes 7 mermaid diagrams
  (ER model, hierarchy trees, dependency DAGs, lifecycle states) and a gap analysis with migration
  recommendations.

Also creates training pipeline skeleton metadata: - training label - Training project (#5), linked
  to repo - training v1.0.0 milestone - Epic issue #107 - Links Code Health and Evaluation projects
  to repo

* docs(design): fix gaps G1-G11 — labels, milestones, templates, blocked issues

- G1: Delete ci label, migrate 8 issues to ci-automation - G2: Create ci-automation v1.0.0 and
  code-health v1.0.0 milestones - G5: Remove body-text Parent references from 7 issue bodies - G8:
  Add Start/Target Date fields to CI & Code Health projects - G9: Add issue templates (epic, phase,
  step, bug) - G10: Add Priority field to all 5 projects with values set - G11: Add blocked label to
  8 eval pipeline issues

* chore: remove blocked-by field from issue templates, make step ID optional

Blocked-by as free text duplicates the body-text convention we retired in G5. The blocked label is
  the single convention for tracking blockers. Step ID made optional since it may not be known at
  creation time.

* docs(design): remove completed gaps from taxonomy, keep only open items

* docs(design): unify Phase/Step convention, resolve G6

Replace dual planning conventions (Phase/Step vs PR grouping) with a single model: Phase = large
  feature, Step = testable unit, PR = shipping unit orthogonal to the hierarchy. Rename Evaluation
  project's PR Group field to Phase. Remove G6 from open items.

* docs(design): rewrite §7 body to unified Phase/Step convention

The previous commit updated the index row but the section body still described the old
  dual-convention model. This rewrites the full §7 body to document the unified Phase/Step/PR model
  and removes G6 from open items.

* docs(design): address Copilot review comments on taxonomy PR

- Fix mermaid state labels: InProgress → aliased "In Progress" - Fix label category count: 5 → 4 -
  Update projects table with Phase, Priority, date fields - Note pending PRs for eval-pipeline.md
  (#101) and braindump (#84) - Align step/phase templates with §7 PR-decoupled model - Update
  project field comparison diagram

* chore: soften epic template design doc wording

Not all epics start with a design doc — training started as a brain dump. Change "should have" to
  "link if one exists".

* docs(design): simplify taxonomy doc per review feedback

- Remove scrapped-decision history (retired body-text convention) - Replace mermaid mindmap with
  grouped table (fixes dark theme readability) - Replace exhaustive hierarchy diagrams with generic
  pattern example - Replace detailed critical path diagrams with text summaries - Replace project
  field comparison diagram with table (fixes truncation) - Remove eval blocking matrix (belongs in
  eval design doc) - Add file-overlap sequencing guidance to blocking conventions

* docs(design): remove hardcoded counts from taxonomy doc

Counts of labels, projects, milestones, etc. go stale as soon as something is added or removed. The
  tables are self-counting.

* docs(design): remove dates, issue counts, item counts from taxonomy

Due dates, issue counts per milestone, item counts per project, and priority distribution tables all
  go stale. The tables and GitHub itself are the source of truth for these numbers.

* docs(design): fix diagram truncation — replace mermaid with text

GitHub renders mermaid state diagrams with fixed-width boxes that truncate labels. Replace status
  workflow and issue lifecycle with text representations. Strip field details from ER diagram
  (already covered by tables).

* docs(design): rewrite taxonomy for native GitHub features

Assumes org migration complete. Major changes: - Issue Types (Epic, Phase, Step, Bug, Task) replace
  naming conventions - Native blocking replaces blocked label + body-text convention - Issue Fields
  replace priority labels - Hierarchy view documented for Projects - Labels simplified to
  domain-only (type/priority/blocking now native) - Removed open items section (all resolved by
  native features) - Added ISSUE_TYPE to ER diagram

* docs(design): delete issue templates, replace diagrams with text schemas

Issue templates only work via GitHub web UI, not via gh CLI / Claude. Replace all mermaid and ASCII
  diagrams with plain text schemas that render reliably everywhere. Style can be added later.

* chore: delete v 1.0.0 custom field from Data Pipeline project

Redundant with the data-pipeline v1.0.0 milestone. Milestones are the right tool for release
  tracking.

* docs(design): streamline taxonomy for native GitHub features

- Remove Phase/Priority custom project fields (replaced by Issue Types + Issue Fields) - Consolidate
  sections: hierarchy, parent-child, phase/step into one - Add §12 Changes Required with migration
  steps, labels/fields to retire - Project table simplified to Start/Target Date only

* docs(design): add cleanup commands to §12 Changes Required

Copy-pasteable gh commands for deleting retired labels, project fields, and migrating blocking
  relationships to native dependencies.

* docs: rename synth-permutations to synth-setter

* docs(design): replace Step type with Task, add Feature as orthogonal

Step was redundant with Task — a task under a phase *is* a step. Feature is orthogonal to the
  hierarchy (can be any scope).

Types split into hierarchy types (Epic, Phase) and work types (Task, Bug, Feature). Only Epic and
  Phase need to be created in org settings — Task, Bug, Feature are GitHub defaults.

* docs(design): keep Priority project field until Issue Fields is available

Issue Fields (org-level) is in public preview and not yet accessible. Priority stays as a
  per-project single-select field for now. Doc notes the migration path for when Issue Fields
  becomes available.

* docs(design): fix remaining Step references, replace with Task/Feature

The Step→Task rename from the earlier commit was overwritten by the full rewrite. This catches all
  remaining references in the index, overview, types table, and hierarchy view description.

* docs(design): remove Issue Fields references from taxonomy

Issue Fields is in public preview and not available to us. Priority uses project fields for now.
  We'll redesign when it lands.

* docs(design): reorder cleanup steps, add issue type assignment step

Migration should: set up native blocking first, then delete labels, then delete project fields, then
  assign issue types to existing issues.

* docs(design): single project model, fresh project for org migration

Replace 5 per-domain projects with a single org-level project. Domain labels + saved views replace
  separate projects. Migration creates a new project from scratch instead of transferring old ones.

* docs(taxonomy): update priority defs, milestone inheritance, PR linking

- Replace work-specific priority examples with abstract definitions - Correct milestone inheritance
  claim (GitHub doesn't auto-inherit) - Add PR-to-issue linking conventions table to issue lifecycle

* chore(skills): add github-taxonomy skill for metadata operations

Rigid process skill that enforces the project's GitHub metadata conventions (issue types, labels,
  milestones, priority, hierarchy, PR linking) by pointing to docs/design/github-taxonomy.md.

- **lint**: Clarify comments in pre-commit and pyproject.toml
  ([`d88cabc`](https://github.com/tinaudio/synth-setter/commit/d88cabc9c1965a6bbf5476bd27213947fb2da122))

Reword section comments to explain what each tool does and where its config lives. Remove stale
  references to replaced tools (flake8, isort, etc.), deduplicate interrogate exclude rationale,
  explain environment.yaml and codespell "ot" exclusions, annotate ruff rule codes and ignores, and
  drop commented-out dead code.

- **pipeline**: Align data pipeline docs with GitHub taxonomy
  ([#132](https://github.com/tinaudio/synth-setter/pull/132),
  [`ffb3e6d`](https://github.com/tinaudio/synth-setter/commit/ffb3e6dd945c6f3732bb7ed39e85ed22b40944a3))

* docs(pipeline): align data pipeline docs with GitHub taxonomy

Rename Step→Task, update Phase naming to "Phase N: Name" convention, add Tracking headers, and
  update repo URLs from ktinubu/synth-permutations to tinaudio/synth-setter per github-taxonomy.md
  conventions.

GitHub issue titles also renamed (6 Phases + 10 Tasks) to match.

Refs #74

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs(pipeline): fix straggler "steps" reference, add taxonomy cross-ref

* docs(pipeline): fix header casing: Last updated → Last Updated

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- **pipeline**: Align data pipeline docs with storage-provenance-spec
  ([#143](https://github.com/tinaudio/synth-setter/pull/143),
  [`8966227`](https://github.com/tinaudio/synth-setter/commit/896622760b8a9c67b87e5f6f7c22a3584ce87ea7))

* docs(pipeline): align data pipeline docs with storage-provenance-spec

Align both data pipeline design doc and implementation plan with the golden
  storage-provenance-spec.md conventions:

- R2 layout: adopt data/{dataset_config_id}/{dataset_wandb_run_id}/ - IDs: map run_id →
  dataset_wandb_run_id, experiment_name → dataset_config_id - W&B: project=synth-setter,
  job_type=data-generation, artifact=data-{config_id} - Timestamp: YYYYMMDDTHHMMSSZ format - Config
  path: configs/dataset/ (filename = dataset_config_id) - Add spec attribution lines to R2, W&B, and
  ID sections - Add dataset immutability rule and promotion pipeline reference - Update repo URLs to
  tinaudio/synth-setter

Refs #74, #122

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs(pipeline): fix r2:// → s3://, remove stale assumption 13

- Appendix E.3: r2:// → s3://synth-data/ (R2 is S3-compatible, W&B resolves via S3 API per
  storage-provenance-spec §11) - Remove stale assumption 13 (run_id format resolved by spec
  alignment)

* docs(pipeline): use plain issue refs, deep-link spec section anchors

Address PR review comments: - Tracking headers use plain #74 (not full URL) per taxonomy §9 - Spec
  references deep-link to section anchors (#1-ids, #2-r2-bucket-layout, etc.)

* docs(pipeline): fix CLI invocation, endpoint env vars, diagram R2 path

- Standardize CLI to python -m pipeline (not pipeline.cli) - Add WANDB_S3_ENDPOINT_URL alongside
  AWS_ENDPOINT_URL per spec §11 - Update reconciliation diagram R2 prefix to data/{cfg}/{id}/

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- **training**: Promote brain dump to design doc + implementation plan
  ([#147](https://github.com/tinaudio/synth-setter/pull/147),
  [`3df963a`](https://github.com/tinaudio/synth-setter/commit/3df963adb3f52780cee9617d52ee6da787596c87))

* docs(training): promote brain dump to design doc + implementation plan

Restructure training-pipeline.md from brain dump into a proper design doc and add a separate
  implementation plan, both aligned with storage-provenance-spec and github-taxonomy conventions:

Design doc: - Standard header (Tracking, Storage conventions, Issue tracking) - Context, Workflow,
  Goals, System Overview, Stages, R2/W&B Integration, Design Decisions, Phase Plan, Dependencies,
  Alternatives, Open Questions - Phase Plan with Task N.M numbering, all under Epic #107 - Spec
  attribution with deep-linked section anchors - Implementation recipes (R2 callback, RunPod
  launcher) in Appendix D - make resume target for R2 path UX

Implementation plan: - Branch strategy, per-phase file lists, completion criteria - Mirrors eval
  pipeline implementation plan structure

Refs #107

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs(training): adopt W&B-only checkpoint durability, defer R2 upload

Design decision: use Lightning's WandbLogger with log_model="all" for checkpoint durability instead
  of a custom R2 upload callback. This eliminates ~100 lines of custom code, removes rclone from the
  training path, and closes 3 open design questions (R2 mirror policy, GC, dual-copy cost). R2
  checkpoint upload deferred as a future optimization.

Changes: - §5.2: Durable Checkpoint Upload → W&B Checkpoint Durability - §6: R2 & W&B Integration →
  W&B Integration (W&B-only) - §7.2, §7.5: updated design decisions - §8: Task 2.1 → Enable
  log_model="all" (was R2 uploader) - §8: Task 2.2 → Resume from W&B artifact (was resume from R2) -
  §10: R2+W&B dual strategy moved to alternatives considered - §11: closed questions 2, 4, 5 (all
  R2-specific) - Appendix C: removed R2 column - Appendix D.1: removed R2 uploader callback recipe -
  Fix Copilot review: cross-ref impl plan, glossary default format, TOC - Implementation plan: Phase
  2 updated to match

Refs: #107

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Features

- **ci**: Add automated semantic versioning with python-semantic-release
  ([#194](https://github.com/tinaudio/synth-setter/pull/194),
  [`6aae1c6`](https://github.com/tinaudio/synth-setter/commit/6aae1c681b26b855a903abcf58b2f23f83011961))

* feat(ci): add automated semantic versioning with python-semantic-release

* fix(ci): address review feedback — commit loop, concurrency, checkout version

- **ci**: Add CODEOWNERS for automatic review routing
  ([#190](https://github.com/tinaudio/synth-setter/pull/190),
  [`4993ef7`](https://github.com/tinaudio/synth-setter/commit/4993ef7a4cf45b67eef5e1451adfc91805c06d36))

- **ci**: Add scheduled nightly test workflow
  ([#193](https://github.com/tinaudio/synth-setter/pull/193),
  [`4aab0eb`](https://github.com/tinaudio/synth-setter/commit/4aab0eb0947e69bce6047fdb4b4c0ce2ab4ada68))

* feat(ci): add scheduled nightly test workflow

* fix(ci): add sh package and fix notification wording in nightly workflow

- **ci**: Add stale issue/PR bot with 120-day threshold
  ([#201](https://github.com/tinaudio/synth-setter/pull/201),
  [`54f6f14`](https://github.com/tinaudio/synth-setter/commit/54f6f1456317686b540ee86b31c229ac2cc0a76f))

* feat(ci): add stale issue/PR bot with 120-day threshold

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* Remove exempt labels from stale workflow

Removed exempt labels for issues and PRs in stale.yml

---------

- **code-health**: Add install, coverage, ci-local Makefile targets
  ([#202](https://github.com/tinaudio/synth-setter/pull/202),
  [`a8cb5cd`](https://github.com/tinaudio/synth-setter/commit/a8cb5cdd9fbeffc763bf920c8589101109195fde))

* feat(code-health): add install, coverage, ci-local Makefile targets

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* docs(make): add comment explaining why coverage runs serially

GPU tests require exclusive device access — xdist parallelism causes flaky failures from VRAM
  contention when multiple workers hit the GPU simultaneously.

Refs #202

---------

- **code-health**: Add pr-checkbox verification skill
  ([#199](https://github.com/tinaudio/synth-setter/pull/199),
  [`28d4065`](https://github.com/tinaudio/synth-setter/commit/28d4065febd79fcbaa69196d2d0308033c500bd2))

* feat(code-health): add pr-checkbox verification skill

* fix(code-health): emphasize verify behavior not implementation in pr-checkbox skill

Add "Verify behavior, not implementation" as the primary principle. Grepping a diff proves someone
  typed a line — it does not prove the system works. The skill now enforces a hierarchy: run the
  tool > query live state > parse the file > grep the diff (weakest, metadata only).

* fix(code-health): rewrite pr-checkbox to enforce behavioral verification

Major rewrite: verification must exercise actual code paths, not grep diffs or parse files when the
  tool can be run directly.

Key additions: - Black box / restaurant / contract metaphors for behavioral testing - DO/DON'T table
  with concrete examples - Mandate to rewrite implementation-focused checks (with before→after) -
  "Spend time understanding what each check proves" section - Hierarchy from strongest (exercise
  code path) to weakest (grep diff)

* fix(code-health): enforce escalation rule and parsing-is-not-exercising in pr-checkbox

Major additions: - Escalation rule: MUST use highest hierarchy level, state reason for descent -
  "Parsing Is Not Exercising" section: yaml.safe_load is NOT behavioral - Self-audit gate: mandatory
  pre-run check, hard gate if >50% parse files - Rewriting examples: YAML parse→tool invoke,
  dry-run→actual run, YAML parse→workflow trigger - Anti-pattern #8: calling yaml.safe_load
  "behavioral" - Quick reference card

- **code-health**: Add pyrightconfig.json for explicit type checking
  ([#195](https://github.com/tinaudio/synth-setter/pull/195),
  [`1f7ba43`](https://github.com/tinaudio/synth-setter/commit/1f7ba43e47fb562cfcc67a65c001e2932b006158))

- **code-health**: Add wave-orchestration skill for parallel PR workflows
  ([#207](https://github.com/tinaudio/synth-setter/pull/207),
  [`1dc7430`](https://github.com/tinaudio/synth-setter/commit/1dc743077c29291f5170dbd16a1b41f41377c86f))

* feat(code-health): add wave-orchestration skill for parallel PR workflows

Closes #205

* feat(skill): add file-overlap detection to wave dependency analysis

Waves must check that no two PRs in the same wave modify the same file. If overlap is detected, bump
  one PR to the next wave to prevent merge conflicts between parallel worktrees.

Refs #114

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- **pipeline**: Add pipeline deps and split requirements files
  ([#75](https://github.com/tinaudio/synth-setter/pull/75),
  [`c8becdf`](https://github.com/tinaudio/synth-setter/commit/c8becdf7b290790979c6d50db5729351d7d96ad9))

Split flat requirements.txt into requirements-torch.txt (torch stack) and requirements-app.txt
  (everything else) for better Docker layer caching. Add pipeline dependencies: click, numpy,
  pydantic, pyyaml, structlog, tenacity, webdataset, runpod. Add checkmake.ini config and pipeline
  pytest marker.

Refs: #68

- **testing**: Add hypothesis property-based testing
  ([#206](https://github.com/tinaudio/synth-setter/pull/206),
  [`ad03670`](https://github.com/tinaudio/synth-setter/commit/ad0367047a91dea5e56d897258c1b73e00b1760e))

* feat(testing): add hypothesis property-based testing with OmegaConf example

* fix(testing): narrow exception handling and add type annotations in property test

- Catch OmegaConfBaseException instead of broad Exception to avoid swallowing real failures -
  Separate create/to_container so conversion failures propagate - Add parameter and return type
  annotations for test consistency

- **testing**: Add pytest-benchmark with config resolution benchmark
  ([#203](https://github.com/tinaudio/synth-setter/pull/203),
  [`89c239d`](https://github.com/tinaudio/synth-setter/commit/89c239d86026d219c26f8c7764806ebed05c4ffe))

- **testing**: Add pytest-xdist for parallel test execution
  ([#192](https://github.com/tinaudio/synth-setter/pull/192),
  [`fa10cab`](https://github.com/tinaudio/synth-setter/commit/fa10cab01eb3d941bbf70d8206e3cd409f7eefc1))

* feat(testing): add pytest-xdist for parallel test execution

* fix(testing): add CI parallelism, keep test-full serial for GPU safety

- **tests**: Add GPU compile test variant
  ([`23aab2f`](https://github.com/tinaudio/synth-setter/commit/23aab2f32e5dbc9cca606db4147228f72fb9c888))

- **tests**: Add lightweight tiny-model fast-dev-run test
  ([`9f140c9`](https://github.com/tinaudio/synth-setter/commit/9f140c99f1eb51fc454bddfef7a5185ef3b3816d))

### Refactoring

- **lint**: Move docformatter config to pyproject.toml
  ([`6ca4252`](https://github.com/tinaudio/synth-setter/commit/6ca42528aaef07d0171d6a25f8ba97b6c5f7ba26))

Centralizes wrap-summaries, wrap-descriptions, style, and black settings under [tool.docformatter].
  Pre-commit hook only passes --in-place.

- **lint**: Move interrogate config to pyproject.toml
  ([`a993e45`](https://github.com/tinaudio/synth-setter/commit/a993e4522633d995fcb53f67b4ace6c6ac2153ed))

Centralizes verbose, fail-under, and ignore flags under [tool.interrogate]. Pre-commit hook only
  keeps the exclude list since interrogate can't share ruff's extend-exclude.

- **lint**: Single-source ruff legacy excludes in pyproject.toml
  ([`f107bcf`](https://github.com/tinaudio/synth-setter/commit/f107bcfd87e8c41a4b38c48ea88d8df28e9d006e))

Move extend-exclude to pyproject.toml, remove duplicate regex from ruff hooks (--force-exclude
  already reads it). Interrogate keeps its own inline exclude since it can't read pyproject.toml
  ruff config.

- **pre-commit**: Deduplicate legacy exclude list with YAML anchor
  ([`9d78c4e`](https://github.com/tinaudio/synth-setter/commit/9d78c4e06b4ff720859b3708e3aee0131165f54b))

Define the shared exclude pattern once as &legacy_excludes on the ruff hook and reference it via
  *legacy_excludes on black, interrogate, and bandit.

- **pre-commit**: Replace black and bandit with ruff
  ([`bfce1f9`](https://github.com/tinaudio/synth-setter/commit/bfce1f9225ec916a1b94fa5cfc4b22d1f5180767))

Consolidate Python toolchain — ruff now handles linting, formatting, import sorting, syntax
  upgrades, and security checks.

- Replace black hook with ruff-format (black-compatible style) - Remove bandit hook, add S rules to
  ruff lint select (S101 ignored) - Add --force-exclude and types_or: [python, jupyter] to both ruff
  hooks - Fix ruff exclude globs (bare dir names instead of single-level globs) - Ignore E501 in
  ruff (ruff-format handles line length)

- **pre-commit**: Replace debug-statements hook with ruff T10 rules
  ([`2be7124`](https://github.com/tinaudio/synth-setter/commit/2be712420b1be422a37393e14742b53fc86f23c0))

All current print() hits are in legacy-excluded files so this is a no-diff change. Catches future
  print()/breakpoint() in new files.

- **tests**: Mark test_train_fast_dev_run as slow with compile
  ([`1c2691e`](https://github.com/tinaudio/synth-setter/commit/1c2691ebbc55bd477ee2c60170a8544cf1e5a49a))


## v0.0.0 (2025-06-07)
