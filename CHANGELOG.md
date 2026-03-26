# CHANGELOG


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
