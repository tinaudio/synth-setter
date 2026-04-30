# CHANGELOG


## v0.6.1 (2026-04-30)

### Bug Fixes

- **vst**: Reload plugin per render to eliminate every-other junk audio
  ([#713](https://github.com/tinaudio/synth-setter/pull/713),
  [`ceaf0fc`](https://github.com/tinaudio/synth-setter/commit/ceaf0fc54f29e875edba3e60a7b575b39d8ec41c))

* fix(vst): reload plugin per render to eliminate every-other junk audio

render_params now takes a plugin_path and reloads the VST3 plugin on every call, working around a
  stale-state bug where alternating renders produced silent or repeated audio. load_plugin's
  editor-pump uses a threading.Event + show_editor(stop_event) pattern (replacing the prior
  _thread.interrupt KeyboardInterrupt hack), which is what makes a per-call reload safe and fast
  enough to be the default.

generate_sample, make_dataset, and scripts/predict_vst_audio.py are updated to pass plugin_path
  through to render_params instead of pre-loading the plugin.

The xfail decorator on test_datasets_from_hardcoded_params_are_identical is removed: with this fix
  in place, the test no longer xpasses.

Closes #489 Refs #705 Refs #702

* docs(eval): update audio-similarity-benchmarks for #489 closure

The dashboard's framing described #489 as an open bug and called the all-pairs series its
  "regression signal". With #713 closing #489 via per-render plugin reload, the framing inverts: the
  all-pairs series is now the regression guard against the fix.

Also fixes the stale module path `src/data/vst/render_params` → `src/data/vst/core.py §
  render_params()`.

Refs #489 Refs #713

* test(vst): characterize that show_editor warm-up does not change rendered audio

Adds test_show_editor_warmup_does_not_change_rendered_audio: renders the hardcoded #489 patch N
  times each with the show_editor warm-up enabled and disabled (by swapping VST3Plugin.show_editor
  to a no-op around the second batch), then asserts every cross-path pair is within the same
  audio-similarity thresholds the round-trip tests use.

This is the empirical justification for the macOS fix in #714 — if the warm-up is not load-bearing
  for the per-render reload path, it can be dropped without changing output, which avoids the
  AppKit/CGS SIGTRAP that show_editor accumulation triggers in unbundled python on macOS.

Refs #489 Refs #714

* fix(vst): make load_plugin helper thread daemon + warn on stuck cleanup

If show_editor hangs past the join timeout, mark the helper thread daemon so it can't block process
  exit, and log a warning so the condition is visible. Cosmetic comment trim on test_preset_params
  explaining the post-call parameter readback inversion.

Refs #489

* refactor(vst): use threading.Timer for show_editor close timing

threading.Timer is the right primitive for 'fire X after N seconds'; hand-rolling it via Thread +
  time.sleep was reinventing it. Drops the _prepare_plugin helper and
  _PREPARE_PLUGIN_JOIN_TIMEOUT_SECONDS constant. timer.cancel() + close_editor.set() in the finally
  block is defensive against show_editor returning early for any reason.

Refs #489 #714

### Chores

- **testing**: Remove MNIST datamodule, model, configs, and tests
  ([#689](https://github.com/tinaudio/synth-setter/pull/689),
  [`feaa935`](https://github.com/tinaudio/synth-setter/commit/feaa935d70a19a7401d9749434d15b3009f2c2ac))

The Lightning-Hydra-Template MNIST scaffolding has been dead weight: test_mnist_datamodule is
  permanently skipped (#243) and three sweep tests were skipped because example.yaml referenced a
  missing model=mnist config (#514). Workflows still cached data/MNIST and pyright/pre-commit
  configs still pinned MNIST file paths even though no MNIST code path runs.

Delete: - src/data/mnist_datamodule.py - src/models/mnist_module.py -
  src/models/components/simple_dense_net.py (only consumer was MNIST) - configs/data/mnist.yaml -
  configs/hparams_search/mnist_optuna.yaml - configs/experiment/example.yaml -
  tests/test_datamodules.py - test_experiments + test_optuna_sweep{,_ddp_sim_wandb} from
  tests/test_sweeps.py (all skipped on #514)

Update workflows (test, test-expensive, test-conda) to drop the MNIST cache step and the macOS
  test_mnist_datamodule exclusion. Drop MNIST file paths from pyrightconfig.json and
  .pre-commit-config.yaml. Fix stale mnist_optuna reference in
  configs/hparams_search/ksin_optuna.yaml header comment, and update src/data/__init__.py docstring
  and the W&B / GitHub Actions / configuration / eval-pipeline reference docs.

Closes #688

### Continuous Integration

- Add test-expensive workflow for slow (non-GPU) tests on main
  ([#681](https://github.com/tinaudio/synth-setter/pull/681),
  [`11fad5f`](https://github.com/tinaudio/synth-setter/commit/11fad5fe552976e448f1f81df0d2d39abd858c6c))

* feat(ci): add test-expensive workflow for slow (non-GPU) tests on main

Repurpose test-expensive.yml as a post-merge runner for the @pytest.mark.slow suite (excluding
  GPU-marked tests) on ubuntu-latest, triggered on push to main and via workflow_dispatch. Slow
  regressions now surface on the integration branch close to the offending commit without slowing
  down PR feedback (PRs still skip slow).

Move the previous GPU-only workflow to test-gpu.yml so its filename matches its purpose. Update
  docs/reference and docs/design tables to reflect the split.

Closes #680

* fix(ci): add permissions to test-gpu.yml; sync github-actions.md

CodeQL flagged test-gpu.yml as missing an explicit permissions block. Add `contents: read` to scope
  the GITHUB_TOKEN minimally (the workflow only checks out code).

Update docs/reference/github-actions.md to reflect the post-split state: - concurrency: now lists
  both `release` and `test-expensive`. - caching: MNIST cache key is shared with test-expensive. -
  workflow table: fill in the test-expensive Gotcha cell with the shared cache note for symmetry
  with `test`.

Refs #680

* fix(ci): address Copilot review on test-expensive workflow

- test-expensive.yml: add `paths-ignore` (`docs/**`, `**/*.md`) so docs-only merges to main don't
  trigger a 90-minute slow-test run (Copilot, .github/workflows/test-expensive.yml:8). -
  docs/reference/github-actions.md: workflow table cell for `test-expensive` now mentions
  concurrency and paths-ignore alongside the shared MNIST cache (Copilot, line 17). -
  docs/design/storage-provenance-spec.md: trigger column for the Slow Tests row now includes manual
  dispatch (Copilot, line 180).

* docs: include `dispatch` in test.yml trigger column

`test.yml` declares `workflow_dispatch:` but the storage-provenance workflow table omits it. Update
  the cell for consistency with the other rows that already list `dispatch` (Copilot, line 177).

- Bump test-expensive runner to ubuntu-latest-4core
  ([#699](https://github.com/tinaudio/synth-setter/pull/699),
  [`13e031c`](https://github.com/tinaudio/synth-setter/commit/13e031c14c36fb4f665e861e56951f870bbfdcc6))

* fix(ci): bump test-expensive runner to ubuntu-latest-4core

The standard 2-core / 7 GB `ubuntu-latest` GitHub-hosted runner OOMs during the PyTorch CPU forward
  passes that `test_train_surge_xt[cpu]` and `test_train_eval_surge_xt[cpu]` exercise. Since this
  lane is the post-merge gate for `[cpu]` accelerator coverage of slow tests, an OOM here means
  regressions land on main with no signal.

Bump to `ubuntu-latest-4core` (4 vCPU / 16 GB) — the smallest GitHub-hosted label that fits the
  workload. Going wider buys nothing for this CPU-bound, single-process workload.

Closes #698

* docs(storage-provenance): sync Slow Tests runner row with ubuntu-latest-4core

The runner-bump in the previous commit makes the workflow table in storage-provenance-spec.md stale
  — it still listed `ubuntu-latest` as the Slow Tests runner. Update the row to match the workflow.

Refs #698

- Rename test-expensive → cpu-slow and auto-file failure tickets
  ([#708](https://github.com/tinaudio/synth-setter/pull/708),
  [`0884b26`](https://github.com/tinaudio/synth-setter/commit/0884b2646a5f6ec07ee8b45dd6671a52ce8cd11e))

Renames .github/workflows/test-expensive.yml to .github/workflows/cpu-slow.yml (workflow name "Slow
  Tests" → "CPU Slow Tests") and strips the Surge XT VST3 + headless X11 install / smoke-test steps
  now that VST slow tests live in test-vst-slow.yml. Tightens the pytest marker filter to also
  exclude requires_vst so any leftover VST-marked tests are skipped at collection time.

Adds a final post-merge-only step that auto-opens a ci-automation Bug assigned to ktinubu (milestone
  ci-automation v1.0.0, parented under Phase 1: Core CI #150) when the workflow fails on a push to
  main, with title-based dedupe that comments on the existing open issue instead of stacking
  duplicates. Uses gh api graphql best-effort to set issue type=Bug and parent sub-issue link; if
  either GraphQL call fails the issue still ships with label + milestone + assignee, which is enough
  for triage. Job-level permissions widen to issues: write so the workflow default (contents: read)
  isn't relaxed globally.

Updates references in docs/reference/github-actions.md, docs/reference/testing.md,
  docs/design/storage-provenance-spec.md, and docs/doc-map.yaml.

Refs #707

### Documentation

- Add comment-hygiene rule to CLAUDE.md ([#711](https://github.com/tinaudio/synth-setter/pull/711),
  [`fb06a81`](https://github.com/tinaudio/synth-setter/commit/fb06a81863b90b38e98a86a26a9ce9270fa1495a))

Future Claude sessions keep producing comments that restate constants, bake in counts, or enumerate
  list contents — all of which go stale the moment the code changes. Add a "Comment Hygiene"
  subsection under Writing Code with concrete bad examples and the rule (the code is the source of
  truth; a comment names the category, not its contents). Existing good-comment categories (WHY,
  invariants, workarounds, surprises) are preserved explicitly.

Refs #710

### Refactoring

- **configs**: Require explicit data/model; split trainer presets
  ([#687](https://github.com/tinaudio/synth-setter/pull/687),
  [`86d1aca`](https://github.com/tinaudio/synth-setter/commit/86d1acae360630df722ce314a49b307e64fb0217))

* configs: udpate default hydra configs

Co-authored-by: Copilot <copilot@github.com>

* continue

* docs(testing): align testing primer with mandatory data/model defaults

Three sections in the testing primer described state that this branch's config refactor invalidates:

- §4 claimed `cfg_train_global` and `cfg_eval_global` had asymmetric `limit_*_batches` presets and
  divergent `data`/`model`/`callbacks` defaults. The new conftest composes both with the same
  `data=ksin model=ffn trainer=cpu` overrides and pins dataset shape via integer
  `train_val_test_sizes` instead of fractional `limit_*_batches`. - §5 template's "align cfg_eval
  with cfg_train" step (copying data/model/ callbacks/limit_val_batches) is no longer needed. The
  template now has three phases instead of four. - Gotchas #2 and #3 (alignment + limit_*_batches
  asymmetry) no longer apply. Replaced with a single gotcha covering the new "explicit data/model
  required" behavior.

* internal-fix(configs): address Copilot review feedback on PR #687

- Rename configs/trainer/mps-32-true-non-determnistic.yaml to mps_32_true_non_deterministic.yaml:
  fix typo ("determnistic" → "deterministic") and switch to underscore convention used by other
  trainer presets (gpu, gpu_400k_steps, ddp_sim, cpu). - Fix dead-code metric_dict_1 binding in
  tests/test_train.py:: test_train_resume — assertion using it was removed earlier in this branch
  but the binding wasn't. - Drop logger=wandb override from cfg_eval_global; restore cfg.logger =
  None to keep tests offline-safe (no WANDB_API_KEY / network needed in CI). - Drop duplicate
  cfg.data.pin_memory = False assignment in both fixture blocks.

* internal-fix(configs): address second round of Copilot review on PR #687

- configs/eval.yaml: switch default callbacks from `default` to `none`. The `default` set includes
  `lr_monitor`, which Lightning's LearningRateMonitor hard-requires an active logger for — and
  eval.yaml has `logger: null`. Eval runs predict/test/validate, not training, so the
  training-oriented callbacks (model_checkpoint, lr_monitor, rich_progress_bar, plot_*) were never
  appropriate defaults here. Tracks the broader lr_monitor issue separately at #517.

- tests/conftest.py: drop the cfg_eval_global model_checkpoint overrides that no longer apply now
  that eval composes with `callbacks: none`. Eval doesn't write checkpoints, so save_top_k /
  save_last are dead config. Also drop the matching lr_monitor cleanup loop — there's no callbacks
  tree to clean up anymore on the eval fixture.

- tests/test_train.py: rewrite the docstring + drop the misleading "Prevent CPU unittest OOM"
  comment on test_train_fast_dev_run_tiny_model_tiny_data. The model/batch/dataset shrinks the
  comment described moved to the shared cfg_train fixture earlier on this branch; the test itself
  now only adds `fast_dev_run=True` on top.

* internal-fix(testing): update test_configs guards for new trainer defaults

The two test_configs.py tests still encoded the pre-PR trainer schema:

- test_cfg_train_t_max_interpolation_resolves composed train.yaml without data= (now mandatory) and
  set trainer.min_steps / trainer.max_steps / trainer.val_check_interval inside open_dict — those
  keys no longer exist in the composed trainer cfg, so the writes failed in struct mode. Replaced
  with data=surge + +trainer.max_steps=-1 to add the key for the interpolation guard, dropping the
  obsolete writes.

- test_cfg_train_trainer_keys_coherent_with_test_mode asserted
  min_steps/max_steps/val_check_interval take specific values from the fixture; with this branch's
  defaults stripped from trainer/default.yaml, those keys are no longer in the struct. The bug it
  guarded against (#625) is now structurally impossible. Reframed the test to assert the new
  structural invariant: step-based keys must NOT be present, and fixture sets max_epochs=1 /
  val_check_interval=1 / check_val_every_n_epoch=1.

* internal-fix(configs): restore val_check_interval=10_000 in trainer/default

The val_check_interval drop from 10_000 to 1000 in trainer/default.yaml was unintentional — it would
  have 10x'd validation cadence for every preset that inherits default (gpu, cpu, mps_*) without
  setting its own value, materially slowing long runs and increasing checkpoint I/O for any caller
  not pinned to gpu_400k_steps or an experiment override. Restore the prior default; experiments
  that want tighter validation keep overriding explicitly (kosc -> gpu_400k_steps already sets
  10_000; surge/base sets 10_000 directly).

* docs(configs): document surge max_steps requirement and point to experiments

- configs/experiment/surge/base.yaml: comment explaining that surge model configs interpolate
  ${trainer.max_steps} into the CosineAnnealingLR scheduler's T_max, so max_steps must be set; the
  values right below are the surge default. - docs/getting-started.md §5a: after the config tree,
  point readers to configs/experiment/{kosc,surge}/base.yaml as the canonical starting points
  showing how each model family is meant to be trained, including required values like
  trainer.max_steps for the surge LR scheduler.

---------

### Testing

- **configs**: Bump FIXTURE_BASELINE + cover jobs/predict scripts
  ([#684](https://github.com/tinaudio/synth-setter/pull/684),
  [`9a92d0a`](https://github.com/tinaudio/synth-setter/commit/9a92d0a075bb51928eecfabf35843a72fd18f797))

* fix(testing): bump FIXTURE_BASELINE to PR #679 merge SHA

PR #679 is now on main at 1bfa7ea, so the pre-merge workaround in _build_equal_cases /
  _build_diff_cases (different script_rel paths for baseline vs current to bridge the hydra_app.sh →
  baseline_app.sh rename) is obsolete. Bump the constant and collapse both pairs to the renamed
  path.

* test(testing): cover jobs/predict scripts in config-drift harness

Adds 18 parametrized cases (one per script under jobs/predict/, excluding the helper
  get-ckpt-from-wandb.sh) that compare resolved Hydra configs between MODEL_BASELINE (v0.0.0) and
  the live tree.

The predict scripts source get-ckpt-from-wandb.sh, which exits 1 when no checkpoint can be located
  via `find logs/train ...`. Pre-set CKPT_PATH to a real (empty) sandbox file so the sourced
  script's `[ -f $CKPT_PATH ]` guard passes without any real wandb resolution; the path appears
  verbatim in the resolved config, so a single per-test value keeps the two sides comparable.

Also widens RefCompareCase.slug() to include the script stem so the 18 predict cases (all under
  parent dir "predict", task_id 0) get distinct parametrize ids instead of pytest's _0..._17
  fallback.

- **data-pipeline**: Reproduce round-trip reproducibility failure for VST dataset generation
  ([#706](https://github.com/tinaudio/synth-setter/pull/706),
  [`9a33ed1`](https://github.com/tinaudio/synth-setter/commit/9a33ed197268d916af8d7c3a83b96bc29b319da3))

* test(data-pipeline): add xfail round-trip reproducibility tests for VST dataset generation

Two new e2e tests in tests/data/vst/test_generate_vst_dataset.py that exercise make_dataset
  round-trip reproducibility via _patched_sample, plus a third random-sampling sanity test.

The two round-trip tests are marked @pytest.mark.xfail(strict=True, reason="bug #489") because main
  does not yet carry the per-render plugin-reload workaround on
  feat/surge-xt-interactive-load-prediction (commits 086d80f / 9ff7f16). Without that workaround,
  ~50% of every-other render produces junk audio, and audio-metric assertions fail. strict=True
  ensures that an unexpected pass surfaces as a test failure so the bug gets revisited.

Refs #489

* feat(ci-automation): track VST audio-similarity test metrics over time

Implements #703.

Test-side: ``_emit_benchmark_metrics`` writes the five summary metrics to ``$BENCHMARK_OUTPUT_PATH``
  when set (no-op locally). ``_assert_audio_metrics_within_thresholds`` returns the metrics tuple so
  ``_assert_round_trip_matches`` can accumulate per-pair values, and emits the worst-case (mss-max,
  wmfcc-max, sot-max, rms-distance-max, mel-mean-abs) under ``vst-fixed-replay/`` when
  ``benchmark_name_prefix`` is passed. ``test_datasets_from_hardcoded_params_are_identical`` opts
  in.

Workflow: ``.github/workflows/test-expensive.yml`` sets ``BENCHMARK_OUTPUT_PATH`` on the pytest step
  and adds a ``benchmark-action/github-action-benchmark@v1`` publish step gated to ``push`` on
  ``refs/heads/main`` with ``hashFiles('bench.json') != ''``. ``contents: write`` is granted at the
  *job* (not workflow) level so only ``run_slow_tests`` can push to ``gh-pages``.

Also re-applies ``@pytest.mark.xfail(strict=True, reason="bug #489")`` to the two round-trip tests
  after the rename, and picks up the all-pairs worst-case check from the feature branch — the
  assertion that makes the xfail premise empirically true on main today.

Refs #489 Refs #703

* ci(test-expensive): allow workflow_dispatch to publish benchmark history

Adds a ``publish_metrics`` boolean input on the manual-dispatch trigger (default false) so a
  maintainer can bootstrap the ``gh-pages`` chart from a feature branch before main has merged the
  workflow. Push-to-main still always publishes; the new input is an explicit opt-in escape hatch.

Usage:

gh workflow run test-expensive.yml \ --ref test/vst-roundtrip-xfail-tests \ -f publish_metrics=true

Refs #703

* ci(test-vst-slow): move VST slow tests + benchmark publish into Docker

Bare ``ubuntu-latest`` runners hit "Timeout waiting for Xvfb to start" in ``test-expensive.yml``'s
  smoke-test step (https://github.com/tinaudio/synth-setter/actions/runs/25026506440), so the slow
  VST tests never reach pytest there. The benchmark publish step in ``test-expensive.yml`` was
  therefore unreachable too.

Add a separate ``test-vst-slow.yml`` workflow that runs
  ``tests/data/vst/test_generate_vst_dataset.py`` inside the ``tinaudio/synth-setter:dev-snapshot``
  Docker image, mirroring the working docker-pull pattern in ``dataset-generation.yml``.
  ``BENCHMARK_OUTPUT_PATH`` is set on the container; ``bench.json`` is mounted out via ``-v
  /tmp/bench`` and copied to the runner workspace for the
  ``benchmark-action/github-action-benchmark@v1`` publish step.

Triggers: push-to-main on relevant paths, plus ``workflow_dispatch`` with ``image_tag`` and
  ``publish_metrics`` inputs. The ``publish_metrics`` opt-in lets a maintainer bootstrap the
  ``gh-pages`` chart from a feature branch.

Reverts the benchmark instrumentation out of ``test-expensive.yml``: the ``BENCHMARK_OUTPUT_PATH``
  env var, the publish step, the dispatch input, and the job-level ``contents: write`` grant.
  ``test-expensive.yml`` goes back to its pre-#703 shape — its non-VST slow tests can remain there.

* ci(test-vst-slow): TEMPORARY bootstrap push-trigger from PR branch

Adds ``test/vst-roundtrip-xfail-tests`` to the push-trigger branch list and widens the publish
  step's ``if:`` to accept that ref. Lets us bootstrap the gh-pages benchmark chart from this PR
  branch before main has the workflow.

REVERT-ME: Roll back to ``branches: [main]`` and the main-only ``if:`` gate once the chart exists.
  See follow-up revert commit.

* fix(test-vst): drop xfail from sampled-params test (not a #489 reproducer)

``test_datasets_from_sampled_params_are_identical`` does NOT reproduce #489. Its rows use
  *different* random params per row (Stage 1 picks 5 random samples), so it has no all-pairs
  cross-comparison — only per-row ``expected[i]`` vs ``actual[i]`` checks. Per-row checks alone
  don't expose every-other-render junk because they only ever compare a row to itself across stages,
  not row-vs-row within a stage.

CI confirmed this on c69f985: the hardcoded test correctly XFAIL'd (all-pairs check caught the bug),
  the smoke test passed, but the sampled test XPASS'd against the strict marker.

The hardcoded test is the canonical #489 reproducer; the sampled test is a regression net for the
  round-trip API and should pass as-is.

* fix(test-vst): skip-fetch-gh-pages on first bootstrap

The benchmark action defaults to ``skip-fetch-gh-pages: false`` and runs ``git fetch ...
  gh-pages:gh-pages`` before any other step. On a first bootstrap where the ``gh-pages`` branch
  doesn't exist yet, that fetch fails with "couldn't find remote ref gh-pages" instead of letting
  the action create the branch.

Run 25138635107 (commit e0e191d) hit this — tests passed, publish step crashed at the fetch.

Setting ``skip-fetch-gh-pages: true`` lets the action take its local-only path: it generates
  ``data.js`` + ``index.html`` from ``bench.json``, commits them on a fresh ``gh-pages`` worktree,
  and ``auto-push`` creates the remote branch.

* ci: re-trigger after gh-pages bootstrap

* ci(test-vst): drop in-container symlink + add VST smoke + dummy fast-path

Three changes to ``.github/workflows/test-vst-slow.yml``:

1. Drop the ``mkdir -p plugins; ln -sf`` lines from the docker run. The base image already places
  the VST3 at ``/usr/lib/vst3/Surge XT.vst3``, and the bind mount over ``/home/build/synth-setter``
  hides the image-side symlink that the Dockerfile creates. Set
  ``SYNTH_SETTER_PLUGIN_PATH=/usr/lib/vst3/Surge XT.vst3`` so the test uses the absolute path the
  .deb installs to.

2. Add a ``Smoke-test Surge XT plugin load`` step before the test step, mirroring the local-runner
  smoke check in ``test-expensive.yml``. Fails fast if the plugin / image / mount layout is broken
  before committing to the much-longer pytest run.

3. Add a ``dummy_only`` workflow_dispatch input + a ``Write hardcoded dummy bench.json`` step gated
  on it. When set, the pull / smoke / test / surface steps are skipped and a hand-crafted
  ``bench.json`` is written directly to the workspace. Lets a maintainer iterate on the publish-step
  gating in ~10 seconds instead of ~5 minutes per cycle. Implies ``publish_metrics``.

Also revert the ``skip-fetch-gh-pages: true`` flag now that the ``gh-pages`` branch exists on the
  remote — the action's default fetch path now resolves it cleanly.

* ci(test-vst): rename benchmark bucket + use full metric names

Bucket: ``VST fixed-params replay`` → ``VST noise floor``. Reflects what the test actually measures
  — the floor of how well two render passes of identical params reproduce each other under the
  docker mitigation stack — rather than the now-misnamed historical reference to the
  ``fixed_*_params_list`` API the test no longer uses.

Metric series: drop project-internal abbreviations in favor of full names so the chart's left-hand
  legend is self-explanatory.

mss-max → multi-scale-spectral-loss-max wmfcc-max → dtw-aligned-mfcc-distance-max sot-max →
  spectral-optimal-transport-max (unit: W → Wasserstein) rms-distance-max →
  rms-envelope-cosine-distance-max mel-mean-abs → mel-spectrogram-mean-absolute-error

Also rename the ``benchmark_name_prefix`` argument from ``vst-fixed-replay`` to ``vst-noise-floor``
  so the on-chart series strings are consistent with the bucket.

The single existing bootstrap data point on ``gh-pages`` will be orphaned under the old bucket name
  — left for now since deleting it would mean a force-push to ``gh-pages`` and the noise-floor chart
  only becomes meaningful once a few runs land anyway.

* feat(ci-automation): split benchmark dashboards + timing metrics + docs

Splits the single benchmark dashboard into two
  (``test_datasets_from_hardcoded_params_are_identical`` → ``VST noise floor (1 preset N renders)``,
  ``test_datasets_from_sampled_params_are_identical`` → ``VST noise floor (random preset replay)``),
  since the action keys all entries from one bench JSON under one chart bucket so multi-dashboard
  needs separate files. ``_emit_benchmark_metrics`` now takes a ``bench_filename`` arg and reads
  ``BENCHMARK_OUTPUT_DIR``; each test passes its prefix as the filename; the workflow's Surface step
  copies both files; Publish is duplicated, one per bucket.

Adds two new metrics per bucket:

num-samples sentinel for fixture-size regressions wall-clock-seconds-per-render renderer perf drift

Each test brackets its ``make_dataset`` calls with ``time.perf_counter()`` and passes the elapsed
  total as ``total_render_seconds``.

New doc ``docs/reference/audio-similarity-benchmarks.md`` covers purpose, where to find the live
  charts + raw data, the two dashboard semantics, the seven metric series, threshold/alerting,
  workflow wiring, and operations (bootstrapping, pre-merge publishing, adding new dashboards,
  pruning history).

* fix(test-vst): address PR #706 review feedback

- Reword `render_params` reload references to present-tense bug-#489 descriptions; drop
  forward-references to the unmerged per-render reload workaround (commits 086d80f / 9ff7f16, PR
  #702). - Sync hardcoded-params docstring `num_samples` and test-name references to the actual
  `test_datasets_from_hardcoded_params_are_identical` body (num_samples=6, all-pairs check
  rationale). - Sync sampled-params docstring rationale to match issue #489 framing (drop the
  workaround commit citations). - Cache `mel[...]` and `params[...]` reads in
  `_assert_h5_structure_is_valid` to avoid double materialization. - Handle JSONDecodeError in
  `_emit_benchmark_metrics` by treating a truncated bench file as an empty list. - Pin
  `benchmark-action/github-action-benchmark@v1` -> the v1.22.0 commit SHA in `test-vst-slow.yml` for
  supply-chain hygiene. - Update `docs/reference/audio-similarity-benchmarks.md` to drop the
  forward-reference to the unmerged per-render reload workaround.

* fix(test-vst): address PR #706 review feedback (round 2)

Doc/wording fixes only — no behavior change:

- _assert_round_trip_matches docstring: ``BENCHMARK_OUTPUT_PATH`` → ``BENCHMARK_OUTPUT_DIR``
  (matches the actual env var read by _emit_benchmark_metrics and set by test-vst-slow.yml). Comment
  3164945781. - docs/reference/audio-similarity-benchmarks.md: "six series" → "seven series" with
  explicit call-out of the two non-distance sentinels (num-samples, wall-clock-seconds-per-render);
  the metric table already listed seven rows. Comment 3164945796. - test-vst-slow.yml dummy_only
  fast-path: include num-samples and wall-clock-seconds-per-render in the hardcoded bench JSON so
  the debug-only payload mirrors what _assert_round_trip_matches actually emits. Comment 3164945820.

Comment 3164945810 (temp branch in push.branches) is a duplicate of the round-1 thread already
  justified at 3164936475 / 3164936515 — kept intentionally and gated by an in-file removal note;
  will be reverted in a follow-up before merge once the gh-pages chart is bootstrapped.

xfail decorators, _HARDCODED_*_PARAMS, and gh-pages branch are not touched.

* chore(test-vst): remove dummy fast-path debug code from workflow

The ``dummy_only`` workflow_dispatch input + ``Write hardcoded dummy bench JSON files (debug-only
  fast path)`` step + all ``inputs.dummy_only`` references were scaffolding for iterating on the
  publish-step gating during the gh-pages bootstrap. The chart is live and the publish path is
  verified, so the dummy code is no longer load-bearing — it just adds noise to the workflow and
  gives operators a footgun (publishing junk to gh-pages by accident).

Reverts: - ``dummy_only`` dispatch input - "Write hardcoded dummy bench JSON files" step - ``if:
  inputs.dummy_only != true`` gates on Pull image, Smoke-test, Run VST tests, Surface -
  ``inputs.dummy_only == true`` clauses in both publish steps' ``if:``

* refactor(test-vst): factor benchmark emission out of round-trip helper

Per PR review feedback (r3165027905): the published "1 preset N renders" chart was wired to per-pair
  metrics, but the #489 reproducer is the all-pairs worst-case across the union of renders. The
  chart could look flat while the test xfails on the all-pairs assertion.

Refactor: - New ``RoundTripMetrics`` and ``AllPairsMetrics`` frozen dataclasses hold the four audio
  metrics + their respective extras (mel diff + num_samples for round-trip; pair count for
  all-pairs). - ``_assert_round_trip_matches`` returns ``RoundTripMetrics`` and no longer has any
  benchmark-emit logic. Drops ``benchmark_name_prefix`` and ``total_render_seconds`` params. -
  ``_assert_all_pairs_audio_metrics_within_thresholds`` returns ``AllPairsMetrics``. - New
  ``_emit_audio_similarity_benchmark_metrics(prefix, round_trip, all_pairs, total_render_seconds)``
  consumes either or both structs and writes the bench JSON. Round-trip series go under
  ``<prefix>/``; all-pairs series go under ``<prefix>/all-pairs-`` so both can coexist on the same
  chart bucket without name collisions. - Hardcoded test now emits BOTH structs — round-trip for
  context, all-pairs as the primary regression signal for #489. - Sampled test still emits only
  round-trip (cross-row pairs differ legitimately, no all-pairs check applies).

Adds six unit tests for ``_emit_audio_similarity_benchmark_metrics`` covering: env-unset no-op,
  round-trip-only schema, all-pairs-only schema, both-structs namespace separation, no-args
  no-write, and append-on-second-call. All run in <1s without the VST.

Updates ``docs/reference/audio-similarity-benchmarks.md`` to document the new ``all-pairs-*`` series
  + their role as the primary #489 signal on the hardcoded bucket.

* docs(test-vst): make hardcoded-test docstring self-contained

Drops the 'Variant of test_datasets_from_sampled_params_are_identical' framing and rewrites as a
  standalone description of what the test actually does.

- **surge**: Parametrized Surge XT train+eval e2e (cpu/mps/gpu)
  ([#674](https://github.com/tinaudio/synth-setter/pull/674),
  [`0d055b3`](https://github.com/tinaudio/synth-setter/commit/0d055b35ac85932c7584d67e57087c7432b35476))

* test(surge): add one-step train and end-to-end eval smoke tests

Add two GPU-gated tests covering the Surge XT flow-matching model: - `test_train_surge_xt_one_step`:
  trains `experiment=surge/flow_full` for exactly one step on the 5-sample fixture, asserts
  `global_step == 1`. - `test_train_eval_surge_xt`: trains, then chains predict -> VST audio render
  -> audio-metrics CSV, asserting on `predictions/`, `audio/sample_*/`, `metrics/metrics.csv`, and
  `metrics/aggregated_metrics.csv`.

Add three supporting fixtures in `tests/conftest.py`: `cfg_surge_xt_global` (package-scoped compose
  of `train.yaml` with `experiment=surge/flow_full` and the 5-sample fixture defaults),
  `cfg_surge_xt` (function-scoped `tmp_path` wrapper), and `cfg_surge_xt_eval` (function-scoped eval
  config composed from `eval.yaml` with `data`/`model` copied from the train config to match the
  checkpoint shape).

Closes #673 Refs #672

* fix(test): lazy-import VST scripts in test_train_eval_surge_xt

`scripts.compute_audio_metrics` transitively loads `torchaudio`, which fails binary load in the
  conda CI env. Module-level imports were breaking collection for the whole `tests/test_train.py`
  file on that runner. Move the VST-script imports inside the GPU-gated test body — the test is
  skipped in envs without a working torchaudio anyway.

Refs #673

* docs(testing): document cfg_surge_xt fixture group

Add a paragraph to the testing primer's §4 noting the new cfg_surge_xt_global / cfg_surge_xt /
  cfg_surge_xt_eval fixtures and what they parallel vs. extend from the existing cfg_train /
  cfg_eval pair.

* continue

* run tests on presubmit

* temp macos vst smoketest for ci

* test surge train test in ci

* add dataset validation in e2e test

* udpate docstrings

* add support for pytorch test fanout across gpu, mps, cpu

* -

* fixes

- **testing**: Config-drift harness comparing live configs vs pinned baseline ref
  ([#679](https://github.com/tinaudio/synth-setter/pull/679),
  [`1bfa7ea`](https://github.com/tinaudio/synth-setter/commit/1bfa7ea9c4b237a4561a9ac546a3e241ecff5951))

* tests: compare experiment configs with baseline

Co-authored-by: Copilot <copilot@github.com>

* udpate test

* update test

* continue

* fix(testing): address PR #679 review — sandbox HOME, harden worktree cleanup

- Sandbox HOME, XDG_CACHE_HOME, XDG_CONFIG_HOME under shim_dir in _run_under_shim. The real
  jobs/train/{kosc,surge}/train.sh runs `rm -rf ~/.triton/cache` before the python shim
  short-circuits anything, which would silently wipe the developer's Triton cache on every test run.
  (Copilot review comment #3142753423 / PR #679)

- Defensive shutil.rmtree fallback after `git worktree remove --force` in _baseline_worktree.py. If
  the path exists but isn't a registered worktree (interrupted prior run, manually-edited
  .git/worktrees), the remove is a silent no-op and the next `git worktree add` errors with "already
  exists". (Copilot review comment #3142753431 / PR #679)

* fix(testing): auto-fetch missing baseline refs; tag ref tests `network`

Replaces the pre-skip + workflow-fetch dance with self-contained ref acquisition inside the worktree
  fixture:

- worktree_for_ref now calls `git fetch --depth=1 origin <ref>` when the ref isn't locally known,
  then re-checks. Works for tags, branch tips, and arbitrary SHAs on remotes with
  uploadpack.allowAnySHA1InWant=true (GitHub default). RuntimeError only surfaces if both the local
  check and the fetch attempt fail. - The four ref-based tests (equality, diff, kosc, surge) now
  carry `@pytest.mark.network` so they can be deselected on offline runs via `-m "not network"`.
  Marker registered in pyproject.toml under strict-markers. - Removed _FIXTURE_BASELINE_SKIP
  machinery — fixture's auto-fetch makes the pre-skip check redundant; FIXTURE_BASELINE SHA fetches
  just like any other ref now. - Reverted the `with: fetch-tags: true` blocks added to test.yml,
  test-conda.yml, test-expensive.yml, nightly.yml — workflow no longer needs to know about baseline
  refs.

Addresses Copilot review comment #3142753429 on PR #679.

* fix(testing): two-step git fetch so tag baselines resolve in CI shallow clones

CI was failing on `worktree_for_ref("v0.0.0")`: `git fetch --depth=1 origin v0.0.0` puts the commit
  object in the local store and writes FETCH_HEAD, but does NOT create a local `refs/tags/v0.0.0`.
  Subsequent `git rev-parse --verify v0.0.0^{commit}` (which looks up the tag *name*, not the SHA)
  then fails and the harness raises RuntimeError.

Fix: after the bare fetch, if the ref still isn't resolvable, fall back to an

explicit tag refspec `+refs/tags/<ref>:refs/tags/<ref>` which does create the local tag. Two-step
  covers both SHAs (step 1) and tag names (step 2).

* fix(testing): address PR #679 review round 2 — xdist, sys.executable, slow marker, docstrings

- _baseline_worktree.py: suffix sanitized-ref slug with PYTEST_XDIST_WORKER so each xdist worker has
  a unique worktree name. Without this, multiple workers running `pytest -n auto -m "not slow"`
  would collide on the basename registered under .git/worktrees/<name>/. (comment #3142785710) -
  test_compare_baseline_configs.py: real_python returns sys.executable so the shim runs the same
  interpreter pytest is running under (with deps), instead of shutil.which("python") which can pick
  up a system Python missing deps. (comment #3142785713) - test_compare_baseline_configs.py: rewrite
  ACCEPTED_DIFFS comment to honestly describe each entry (including the asymmetric
  tensorboard-subtree strip — added post-v0.0.0, observability only). Behavior unchanged; comment
  was misleading. (comment #3142785721) - test_compare_baseline_configs.py: rewrite
  get_num_experiments docstring — it counts non-empty lines, doesn't parse SGE_TASK_ID. (comment
  #3142785727) - test_compare_baseline_configs.py: add @pytest.mark.slow to the kosc + surge
  model-baseline tests (44 + 8 cases, ~7 min). They now run in test-expensive.yml (`-m "slow and not
  gpu"`) instead of bloating the fast suite (`-n auto -m "not slow"`). (comments #3142785731,
  #3142800321)

* fix(testing): address PR #679 review round 3 — code-health pass

_baseline_worktree.py: - Extract `_git(*argv, check=False)` helper (centralizes the noqa rationale
  that was duplicated across ~9 subprocess.run sites). - `_try_fetch_ref` now returns per-attempt
  stderr; `worktree_for_ref` includes it in the RuntimeError so CI failures don't surface as "did
  not resolve it" with zero context. (comment #3142785710) - Session-end cleanup loop emits a
  `warnings.warn` on non-zero `git worktree remove` exit instead of swallowing stderr silently.
  (comment #3142785710)

test_compare_baseline_configs.py: - Add `test_pinned_baselines_resolve` collection-time guard so a
  stale FIXTURE_BASELINE / MODEL_BASELINE surfaces fast rather than deep inside a parametrized
  failure. (comment #3142800321) - Promote magic counts (8, 44) to module-level
  `EXPECTED_KOSC_TASKS` / `EXPECTED_SURGE_TASKS` constants. The sanity tests assert against the
  constant (not a tautological recomputation). (comment #3142785721) - Drop tautological `assert
  case.task_id <= expected_tasks` loops in the sanity tests — task_ids are constructed as range(1,
  N+1). (comment #3142785727) - Rename `RefCompareCase.id()` → `slug()` to stop shadowing the
  builtin. Update all four parametrize call sites + the unit test. (comment #3142785731) - Annotate
  `shim_factory` and `worktree_for_ref` fixture parameters across all six test signatures (Callable
  types). (comment #3142800321) - Replace `open(path)` with `open(path, encoding="utf-8")`. (comment
  #3142785727) - Add comment to `_NOOP_SHIMS` explaining mamba/module are env-activation tools the
  production train scripts source. (comment #3142785721) - Add inline comment to
  `_strip_dotted_keys` explaining the for-else branch runs only on no-break (full path traversable).
  (comment #3142785721) - Extract `_assert_resolved_configs_differ` for symmetry with the existing
  `_assert_resolved_configs_equal`; inequality test now reads as a single semantic line. (comment
  #3142785731) - Smoke test uses `_git("rev-parse", "HEAD", check=True)` instead of an inline
  subprocess.run with duplicated noqa pair. - Drop stale "Once this PR merges and the default ref
  flips" paragraph from `_build_equal_cases` docstring (defaults are hardcoded constants now).
  (comment #3142785731)

tests/fixtures/{baseline_repo,diff_repo}/scripts/*.py: - Add module docstrings to satisfy PY7.
  (comment #3142800321)

* fix(testing): address PR #679 review round 4 — locks, markers, dict-diff

_baseline_worktree.py: - Add `_git_lock(lock_path)` context manager (fcntl.flock on
  .git/baseline_worktree.lock) wrapping the fetch + prune + worktree-add block, plus the session-end
  cleanup loop. Worker-id-suffixed paths solved the worktree-name collision but not the per-repo
  locks git itself takes (.git/config.lock, FETCH_HEAD.lock) — this serializes shared-state ops
  across xdist workers. Verified with `pytest -n 4 -m network`. (comment #3142816350)

test_compare_baseline_configs.py: - test_pinned_baselines_resolve: add @pytest.mark.network — it
  triggers outbound git fetch via _try_fetch_ref. Without the marker, `pytest -m "not network"`
  silently runs it and fails on offline machines. (comment #3142832748) -
  test_pinned_baselines_resolve: actually call worktree_for_ref(...) on both pinned refs and assert
  the worktree materializes. The previous version took the fixture but only checked _ref_exists,
  leaving the fixture's worktree-creation path untested for the constants the test is supposed to
  validate. (comment #3142832755) - shim_factory: use request.node.nodeid (not .name) when
  sanitizing filenames for --keep-yaml-dir. nodeid includes the module path so files from different
  test modules with the same parametrize id can't overwrite each other in a shared keep directory.
  (comment #3142832759) - _assert_resolved_configs_{equal,differ}: drop the `, (base, cur)` message
  from the `assert base ==/!= cur` lines. The custom message defeats pytest's structured dict-diff
  output; without it pytest renders a readable per-key diff for the ~150-line config dicts. (comment
  #3142832760)

* fix(testing): address PR #679 review round 5 — public API + warn on cleanup

_baseline_worktree.py: - Drop leading underscore from `_git`, `_ref_exists`, `_try_fetch_ref` →
  `git`, `ref_exists`, `try_fetch_ref`. These were always meant to be used by the test module too;
  the underscore was a holdover. Module docstring now names them as the public API. (comment #1) -
  Add `git_or_warn(*argv, context)` helper — runs `git(*argv)` and emits `warnings.warn` on non-zero
  exit. Apply to the in-flight `worktree prune` and `worktree remove --force` calls (which
  previously swallowed stderr silently) and the session-end cleanup loop (which had the manual warn
  block). All best-effort cleanup steps now consistently surface failures through pytest's warnings
  summary. (comment #8)

test_compare_baseline_configs.py: - Update import + 7 call sites for the rename.

* fix(testing): address PR #679 review round 6 — git-common-dir, stale comments

_baseline_worktree.py: - Add `_git_common_dir()` helper using `git rev-parse --git-common-dir`.
  Replaces the hardcoded `REPO_ROOT / ".git" / ...` lock path, which would fail in a linked git
  worktree (where REPO_ROOT/.git is a *file* pointing at the main repo's `.git/worktrees/<name>/`,
  not a directory). Bonus: all linked worktrees of the same repo now resolve to the same lock path,
  so the lock serializes across worktrees too — not just across xdist workers in one checkout.
  (comment #3142850584)

test_compare_baseline_configs.py: - Rewrite the comment block above FIXTURE_BASELINE/MODEL_BASELINE:
  the old "Prefer tags ... so CI can fetch via fetch-tags: true on actions/checkout" was stale
  (workflows no longer set fetch-tags; the harness auto-fetches via try_fetch_ref). New text says
  tags are preferred for stability / discoverability. (comment #3142850588) - Add an IMPORTANT note
  next to FIXTURE_BASELINE warning that the current value is a branch-tip SHA on PR #679, not
  reachable from main, and must be bumped post-merge (or GitHub may eventually GC the orphan commit
  if the branch is deleted). Points readers at the merge-followup PR comment for the step-by-step
  procedure. (comment #3142850590)

* fix(testing): drop --depth=1 from try_fetch_ref to avoid missing-tree CI flake

CI failure on test_baseline_and_current_resolved_hydra_configs_are_equal:

git worktree add failed for ref '624ea3c0...': fatal: unable to read tree (6ecf2143...)

Tree 6ecf2143 is `docs/reference/` at commit 624ea3c. The fetch succeeded at returning the commit
  object — `ref_exists` saw it via `git rev-parse --verify <sha>^{commit}` and we proceeded — but a
  *subtree* referenced by the commit was silently dropped from the pack.

Root cause: shallow-fetch-by-SHA pack-negotiation bug. CI starts with a depth-1 clone of HEAD
  (`actions/checkout@v4` default). When we then run `git fetch --depth=1 origin <sha>`, the server's
  pack-objects looks at the client's "have" set (just HEAD), assumes the client probably has many
  subtrees that overlap with <sha>'s tree, and omits some of them. The depth-1 client doesn't
  actually have the *specific* subtree SHAs from <sha>'s revision (e.g., docs/reference/ was edited
  between 624ea3c and HEAD, so the SHAs differ). `git worktree add` then can't reconstruct <sha>'s
  working tree.

Fix: drop `--depth=1` from both fetch attempts. Without the depth constraint, git negotiates a
  complete pack — still incremental (only sends objects the client doesn't have), just no longer
  artificially shallow.

* fix(testing): address PR #679 review round 7 — empty keep-dir + relative git-common-dir

_baseline_worktree.py:

- worktree_for_ref: treat empty-string `--compare-baseline-configs-keep-yaml-dir` as unset. Argparse
  passes "" when the user writes the flag with no value
  (`--compare-baseline-configs-keep-yaml-dir=`), and `Path("").resolve()` then silently expands to
  the current working directory — would have spawned a `worktrees/` subdir wherever pytest was
  invoked. Switched to `or None` so empty strings collapse to None and fall through to the
  tmp_path_factory branch. (comment #3142862689)

- _git_common_dir: explicitly anchor against REPO_ROOT when `git rev-parse --git-common-dir` returns
  a relative path (which it does in the main repo — typically just `.git`). `Path(".git")` is
  interpreted against the process cwd, so the lock file would land in the wrong place when pytest is
  invoked from a directory other than REPO_ROOT, breaking the inter-process serialization. Now:
  absolute → return as-is; relative → REPO_ROOT / common_dir. (comment #3142862697)

---------


## v0.6.0 (2026-04-25)

### Features

- **monitoring**: Enable W&B logger by default in many_loggers compose
  ([#677](https://github.com/tinaudio/synth-setter/pull/677),
  [`9072185`](https://github.com/tinaudio/synth-setter/commit/907218574eca9fe950da09e7e2b89ef37b7f818c))

* feat(monitoring): enable W&B logger by default in many_loggers compose

Re-enable W&B in the default `many_loggers` compose so fresh installs log to W&B + CSV + TensorBoard
  out of the box. Reverses #612's opt-in switch.

Users without a W&B account can drop the `- wandb` line from `configs/logger/many_loggers.yaml` or
  override per run with `logger=csv` or `logger=tensorboard`.

Doc updates (via /doc-drift): - README.md: drop "opt-in" framing from features bullet and tracking
  note - docs/getting-started.md §4c: rewrite as "enabled by default" + how to disable -
  docs/reference/wandb-integration.md: update default-compose table rows and callback dispatch
  description - docs/doc-map.yaml: add `configs/logger/many_loggers.yaml` to the wandb integration
  mapping so future drift catches default-compose changes

Closes #676.

* docs(monitoring): address doc-drift report on wandb-default PR

Two findings from the post-PR /doc-drift review:

1. wandb-integration.md: bump stale "Code version" stamp from `0b55a9e`
  (`feat/wandb-optional-by-default`) to this PR's SHA/branch — the body was rewritten for
  W&B-by-default but the version pointer still named the opposite-direction branch.

2. doc-map.yaml: add `configs/logger/many_loggers.yaml` under the getting-started entry so future
  drift in the default compose is caught against §4c (which has detailed enable/disable
  instructions), not only against the wandb-integration reference.

Refs #676.

* docs: fix inverted disable-wandb instruction in README

Copilot caught that the README said to "drop W&B from the default compose by uncommenting `- wandb`"
  — but with wandb now enabled by default, the disable action is to comment it OUT (or remove the
  entry), not uncomment.

The corresponding instruction in docs/getting-started.md §4c was already correct ("comment out `-
  wandb`"); only README needed the fix.


## v0.5.0 (2026-04-21)

### Features

- **dataset**: Add generate_dataset to python docker entrypoint
  ([#667](https://github.com/tinaudio/synth-setter/pull/667),
  [`c759aed`](https://github.com/tinaudio/synth-setter/commit/c759aed92351f789a206cb5b18e2505935721c24))

* feat(docker): swap ENTRYPOINT to docker_entrypoint.py and de-override CI

Hooks up the click-based Python entrypoint shipped in #645 as the image's live ENTRYPOINT and
  retires the bash entrypoint + BATS suite. Workflows that previously bypassed the entrypoint via
  `--entrypoint bash` now go through the real ENTRYPOINT (`generate_dataset` / `passthrough`).

- docker/ubuntu22_04/Dockerfile: ENTRYPOINT = ["python", "/usr/local/bin/entrypoint.py"] - Delete
  scripts/docker_entrypoint.sh, tests/test_entrypoint.bats, .github/workflows/bats-tests.yml. -
  pipeline/entrypoints/generate_dataset.py: wrap the generate_vst_dataset.py subprocess with
  run-linux-vst-headless.sh at the audio-rendering boundary so the click CLI stays X11-agnostic and
  idle/passthrough don't pay the Xvfb startup cost. - Workflows: - dataset-generation.yml: split the
  old `--entrypoint bash -c` block into a `passthrough bash -c` bootstrap step (git safe.directory +
  plugin symlink + materialize_spec) and a `generate_dataset --spec` dispatch step through the
  default ENTRYPOINT. CONFIG_PATH is forwarded via `-e` instead of being spliced into the shell
  command text (removes a quoting hazard flagged on #645). - docker-build-validation.yml: convert
  the two smoke-test invocations from `--entrypoint bash -c` to `passthrough ...`. -
  spec-materialization.yml, flush-investigation.yml: same treatment. - test-dataset-generation.yml:
  drop `-e MODE=passthrough` and replace with the positional `passthrough` subcommand; drop the
  `scripts/docker_entrypoint.sh` trigger path. - Docs: scripts/README.md, docs/reference/docker.md,
  docs/reference/docker-spec.md, docs/doc-map.yaml, docs/design/data-pipeline-implementation-plan.md
  all drop the "bash live, python planned" framing, drop MODE/DATASET_CONFIG/
  RUN_METADATA_DIR/R2_BUCKET env-var references, and describe the click CLI as the live ENTRYPOINT.
  .env.example drops the same vars from the Docker runtime block.

Also picks up three unresolved #645 review carryovers: - Module docstring in
  scripts/docker_entrypoint.py said render_eval/train raise NotImplementedError; they raise
  click.ClickException — updated. - docs/reference/docker-spec.md R2-bucket paragraph updated to
  reflect the spec-driven flow (bucket lives in DatasetPipelineSpec.r2_bucket). - The
  command-injection/quoting hazard around `$CONFIG_PATH` in dataset-generation.yml is eliminated
  naturally by removing the nested bash -c expansion.

Tests: make test → 182 passed, 2 skipped; `make format` clean. End-to-end smoke test against the new
  ENTRYPOINT will exercise the built image once this PR's workflow run lands.

Closes #647 · Part of #265

* fix(docker): bake PYTHONPATH into image so default ENTRYPOINT finds pipeline

pipeline is excluded from find_packages in setup.py, so the editable install doesn't expose it.
  Running python /usr/local/bin/entrypoint.py only puts /usr/local/bin on sys.path, not the repo
  root — so 'from pipeline.entrypoints.generate_dataset import run' fails with ModuleNotFoundError
  at container start.

The old bash entrypoint path worked because dataset-generation.yml exported
  PYTHONPATH=/home/build/synth-setter before invoking the Python entrypoint. Migrating to the
  default ENTRYPOINT dropped that export; bake it into the image instead so every caller gets the
  same import surface without having to remember it.

Refs #647

* fix(packaging): include pipeline in editable install

find_packages() previously excluded pipeline/ on the grounds that it has its own __init__.py and
  setuptools was otherwise bundling it into an installed distribution. That was defensive against a
  PyPI-publication scenario this repo doesn't have (setup.py has placeholder name/url/author and
  there's no release workflow). The cost of the exclusion is that anything trying to import pipeline
  without CWD on sys.path has to hand-roll a PYTHONPATH — which is why the old dataset-generation
  workflow did 'export PYTHONPATH=/home/build/synth-setter' before every Python invocation.

With the Dockerfile now invoking the entrypoint by absolute path (python
  /usr/local/bin/entrypoint.py), that PYTHONPATH hack would need to move into the image (ENV
  PYTHONPATH=...). Dropping the exclude is strictly simpler: 'uv pip install -e .' now exposes
  pipeline through the venv the same way it exposes src, and the absolute-path entrypoint call just
  works.

Follow-ups if this repo ever does get published: - re-scope the exclude then, or switch to an
  explicit include=['src*', 'pipeline*'] - the pre-existing tests.* inclusion in the installed
  distribution is a separate hygiene issue, left untouched here

Reverts the 719c5d6 ENV PYTHONPATH workaround.

* ci: pass PYTHONPATH to docker runs until dev-snapshot is rebuilt

The dev-snapshot image on Docker Hub was built against setup.py's old
  find_packages(exclude=['pipeline', 'pipeline.*']) surface. Its editable install has a strict-mode
  PEP 660 finder with a static MAPPING dict that lists only {configs, src, tests} — no pipeline.
  Bind-mounting the PR's updated setup.py doesn't re-run the finder's installer, so 'from
  pipeline.entrypoints.generate_dataset import run' fails at import time in every docker run that
  goes through the click ENTRYPOINT.

The setup.py fix in 8b45db7 is durable but only takes effect once the image is rebuilt and
  re-pushed. Until then, override sys.path at runtime via -e PYTHONPATH — the same trick the old
  workflow used before #645. Harmless once the image is fresh.

Added to every docker run in: - dataset-generation.yml (bootstrap + generate) -
  spec-materialization.yml (materialize) - flush-investigation.yml (notebook run) -
  test-dataset-generation.yml (shard download + validate) - docker-build-validation.yml (both smoke
  tests)

Follow-up: once docker-build-validation pushes a fresh dev-snapshot on main, strip these PYTHONPATH
  overrides in a separate PR.


## v0.4.0 (2026-04-21)

### Documentation

- Add contributor-facing testing primer ([#651](https://github.com/tinaudio/synth-setter/pull/651),
  [`18f45d2`](https://github.com/tinaudio/synth-setter/commit/18f45d25e724c304ef3b0f544130ccac060152d1))

* docs: add contributor-facing testing primer

Add docs/reference/testing.md covering the bare minimum contributors need to read a test in this
  repo and write a new one — layout, conftest fixtures, pytest markers, the train→eval e2e pattern,
  and six gotchas distilled from recent review cycles (DataModule setup(stage) semantics, cfg_eval
  diverging from cfg_train, limit_val_batches parity, GPU marker stack, weights_only=False,
  mode→stage audit when extending eval.py).

Link from docs/getting-started.md §2f so contributors discover the primer right after they verify
  the install.

Scope is deliberately small (≤1 page, ~5 min read) to keep the doc from rotting. Per-test docs and
  ML-test theory are intentionally out of scope.

Closes #650.

* docs: rewrite testing primer to reference sources instead of echoing values

Two corrections surfaced in PR review:

1. Drift risk — the primer hardcoded marker lists, Makefile flags, fixture preset values, and CI
  selector strings. Copilot caught four cases where those values didn't match reality
  (tests/test_instantiators.py doesn't exist, make test uses -n auto -m 'not slow and not
  requires_vst', cfg_eval has no limit_train_batches preset either, GPU CI uses -m gpu not -m 'slow
  and gpu'). The root cause is the pattern: if the doc echoes code, it has to be re-synced on every
  code change.

2. Category bias — the previous version treated the repo as Hydra+Lightning e2e tests plus a few
  sibling files. Actually tests/ has pipeline tests (own conftest), property-based tests,
  benchmarks, sweep tests, script tests, docker smoke tests, and VST-gated integration tests. The
  primer didn't mention them, so a reader got a warped picture of the suite.

Rewrite: - §1 now catalogs all 10 test categories with links to a representative file for each (no
  file enumeration in prose). - §2 (invocation) defers to the Makefile and .github/workflows/ rather
  than copying their flags. Explicitly warns that CI and make selectors aren't identical. - §3 is
  new — a 'which shape fits your test' table keyed on goal, so a reader picks the right template
  before writing. - §4 (fixtures) describes the cfg_train/cfg_eval asymmetry structurally ('cfg_eval
  presets a subset') without naming specific keys. - §5 keeps the E2E Python template (the one
  hardcoded value that's genuinely the point of the primer — the shape), but explicitly notes that
  non-E2E categories should NOT copy it. - §6 gotchas reworded to point at source files; gotcha #4
  no longer asserts the CI selector string. - §7 pointers expanded with pipeline conftest, Makefile,
  CI workflow directory, and the pyproject.toml marker registry.

Addresses Copilot comments r3113275967, r3113276012, r3113276033, r3113276057.

Refs #650.

* docs: teach math.isfinite in the canonical E2E template

The train→eval E2E template in the testing primer had `assert eval_metric_dict["val/loss"] <
  float("inf")`. That idiom silently accepts `-inf` (since `-inf < +inf` is True). The template is
  explicitly the pattern new contributors copy-paste, so teaching `< float("inf")` here would
  propagate the subtle correctness gap to every future E2E test in this repo.

Swap to `math.isfinite(metric.item())` to match the tightened assertions in #655, which refits both
  `test_train_eval` and `test_train_validate` on main. Also adds an inline comment explaining the
  gotcha, since it's non-obvious why the stricter form matters for MSE-style losses.

Refs #650, refs #655.

* docs: address round 2 review on testing primer

- §1: correct the pipeline conftest claim — parent conftests ARE resolved by pytest, so
  cfg_train/cfg_eval are reachable under tests/pipeline/; pipeline tests just don't lean on them. -
  §2: fix CI workflow bullet descriptions — test-conda.yml is a single micromamba env run (not a
  matrix); nightly.yml is a full pytest run on CPU (not an expensive-only suite). - §5: replace
  invalid placeholder test_train_<what>(...) with a valid identifier test_train_e2e and a rename
  hint; snippet now copy-pastes without a SyntaxError.

### Features

- **scripts**: Click-based docker entrypoint with per-mode spec parsing
  ([#645](https://github.com/tinaudio/synth-setter/pull/645),
  [`6402042`](https://github.com/tinaudio/synth-setter/commit/6402042a8f35304fd314aec56f152acf4b4ac040))

Rewrites scripts/docker_entrypoint.py as a click group with five subcommands (idle, passthrough,
  generate_dataset, render_eval, train). Each spec-taking subcommand deserializes its --spec into a
  mode-specific pydantic model at the container boundary (parse-don't-validate) before handing off
  to the downstream.

pipeline/entrypoints/generate_dataset.py collapses to a single run(spec: DatasetPipelineSpec)
  function. Env-var reads (DATASET_CONFIG, R2_BUCKET, RUN_METADATA_DIR) are deleted; the __main__
  block fails loudly pointing callers at the new entrypoint.

DatasetConfig gains a required r2_bucket field (mirrored into DatasetPipelineSpec via
  materialize_spec). run_metadata_dir is dropped everywhere — in the new flow the host materializes
  the spec and passes it via --spec, so the container never needs to write the spec out via bind
  mount.

This breaks MODE=generate_dataset callers of the bash entrypoint. #647 will swap the Dockerfile
  ENTRYPOINT to the Python CLI and update workflows; deploy #645 and #647 back-to-back.

Deferred follow-ups (filed alongside this PR): - Spec content-addressing / hashing
  (s3://bucket/specs/<sha256>.json) - Structured error output for pydantic ValidationError in logs -
  Exit-code retry/don't-retry contract for orchestrator consumers - URI-based spec resolution
  (--spec s3://...) - Rename #410 eval → render_eval and propagate through docs

### Monitoring

- Route plot callbacks through Lightning loggers
  ([#646](https://github.com/tinaudio/synth-setter/pull/646),
  [`f830781`](https://github.com/tinaudio/synth-setter/commit/f830781148cfc89bc9f2a9c8b08a4bb2340ce8eb))

* fix(monitoring): route plot callbacks through Lightning loggers

Replace direct wandb.log / wandb.Image calls in the three plot callbacks (PlotLossPerTimestep,
  PlotPositionalEncodingSimilarity, PlotLearntProjection) with a small _log_figure helper that
  dispatches per-logger across trainer.loggers. The helper calls log_image on WandbLogger and
  experiment.add_figure on TensorBoardLogger, and silently skips loggers with no image API (e.g.
  CSVLogger). This removes the W&B auth-prompt bypass that survived #612's opt-in switch and lets
  plots land on whichever logger the user actually selected.

Closes #614

* docs(monitoring): update wandb-integration.md for Lightning logger dispatch

The callback dispatch refactor replaces direct wandb.log() / wandb.Image() calls with a per-logger
  dispatcher. Update §Overview, §2c, the PredictionWriter row in §2d, and mark Known Gap #6 as
  resolved (tracked by #614).

Refs #614

* fix(monitoring): guard _log_figure with rank-zero check for DDP safety

TensorBoard's SummaryWriter is not rank-safe and W&B's log_image from non-zero ranks can duplicate
  figures. Early-return from _log_figure unless trainer.is_global_zero. Added
  test_log_figure_is_noop_on_non_zero_rank.

* docs(monitoring): clarify _log_figure silent-skip intent

Expand the docstring to explain that skipping non-image-capable loggers (CSVLogger today) is
  intentional, and direct future contributors to add an isinstance branch here if an
  MLflow/Comet/Neptune logger is introduced.

* docs(monitoring): replace brittle line numbers with symbol refs in callbacks table

Line-number references in §2c and §2d of wandb-integration.md shifted twice during this PR (after
  the rank-zero guard and after the docstring expansion). Switch to stable Python-symbol references
  (\`src/utils/callbacks.py::Class.method\`) so future callback refactors don't drift these rows.


## v0.3.0 (2026-04-20)

### Documentation

- Add CLAUDE.md GPU verification rule to prevent false skips
  ([#640](https://github.com/tinaudio/synth-setter/pull/640),
  [`7ad9afb`](https://github.com/tinaudio/synth-setter/commit/7ad9afbee99039180d87e60590b5756802117ec5))

* docs: add CLAUDE.md GPU verification rule to prevent false skips

Fixes #639

* Apply suggestions from code review

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

- **documentation**: Clarify W&B opt-in in README and getting-started
  ([#632](https://github.com/tinaudio/synth-setter/pull/632),
  [`e8cfab5`](https://github.com/tinaudio/synth-setter/commit/e8cfab58fa506e99fd0dd020d72283780f345f2b))

* docs: clarify W&B opt-in in README and getting-started

PR #612 makes W&B opt-in (CSV + TensorBoard is the new default logger compose). Two reader-facing
  docs still framed W&B as always-on:

- README Features line + post-install blockquote now state W&B is opt-in and point to the
  enable/disable workflow in getting-started §4c. - getting-started §3a / §4c / §5b re-framed around
  the new default. §4c now has explicit Disabled / Enabled-per-run / Enabled-by-default paths; §5b's
  stale `logger=tensorboard` override example replaced with `logger=wandb` (TensorBoard is already
  in the default compose). - Removed the hard-coded `WANDB_ENTITY=tinaudio` example so external
  users don't route runs to the tinaudio org by default.

Refs #602, refs #598.

* docs: preserve §4c anchor — move opt-in flag from heading to body

The '— opt-in' suffix in the § heading changed the GitHub slug from '#4c-weights--biases-wb' to
  '#4c-weights--biases-wb--opt-in', breaking three existing in-repo links. Keeping the original
  heading preserves the anchor; the opt-in framing is bolded in the first sentence of the body
  instead.

Refs #602.

* docs: keep README W&B blockquote inline code on one line

Inline code span broke across a newline (`python src/train.py` opened on one line, `logger=wandb`
  closed on the next), so GFM rendered the closing backtick as a literal character. Reshaped the
  sentence so the `logger=wandb` reference sits on a single line.

### Features

- **data**: Enable standalone mode=validate for KSinDataModule + coverage test
  ([#636](https://github.com/tinaudio/synth-setter/pull/636),
  [`31aa1a4`](https://github.com/tinaudio/synth-setter/commit/31aa1a4782695e41220a86240023789cc7648ae0))

* test(testing): add test_train_validate to cover eval.py mode=validate path

Mirrors test_train_eval but sets cfg_eval.mode = "validate" so evaluate() exercises the
  trainer.validate(...) branch at src/eval.py:91, which previously had zero coverage. Asserts
  val/acc parity between the training run and the checkpoint re-validation.

Fixes #635

* fix(testing): address Copilot review on test_train_validate

Three fixes in tests/test_eval.py::test_train_validate:

- Remove `cfg_train.test = True` — unused post-train test phase wastes work for a validate-path test
  (comment 3112319113). - Align eval ckpt load: set `cfg_eval.model = cfg_train.model` and
  `cfg_eval.callbacks = cfg_train.callbacks` so the trained `ffn` checkpoint loads (comment
  3112319151). - Assert on `val/loss` instead of `val/acc`; `ksin_ff_module` only logs `val/loss`.
  Mirrors the `test/loss` pattern from test_train_eval (comment 3112319183).

* fix(testing): disable post-train test phase in test_train_validate

The prior round removed cfg_train.test = True but did not set it to False. Since configs/train.yaml
  defaults test: True, train() still ran the post-training trainer.test(...) phase, defeating the
  GPU-time reduction.

Explicitly set cfg_train.test = False so the post-train test phase is skipped under validate-mode
  coverage.

* fix(testing): mirror limit_val_batches in test_train_validate cfg_eval

The parity assertion abs(train[val/loss] - val[val/loss]) < 0.001 in test_train_validate cannot hold
  unless the standalone evaluate() run uses the same val-set subset as the training run.
  cfg_train_global sets trainer.limit_val_batches=0.1, but cfg_eval_global only sets
  trainer.limit_test_batches=0.1, so without mirroring the flag the val set runs unrestricted
  (default 1.0) inside evaluate() and produces a different val/loss than training reports.

Align limit_val_batches alongside the existing data/model/callbacks mirror in the
  open_dict(cfg_eval) block.

Refs #635

* fix(testing): handle stage=validate in KSinDataModule.setup

`src/data/ksin_datamodule.py`'s previous `if stage == "fit": ... else: self.test = ...` gate never
  built `self.val` when Lightning called `setup(stage="validate")` during `trainer.validate()`, so
  `val_dataloader()` returned a missing attribute and `tests/test_eval.py::test_train_validate`
  crashed with AttributeError.

Restructured to three independent stage-indexed branches (fit -> train, {fit, validate} -> val,
  {test, predict} -> test), matching Lightning's canonical DataModule pattern. See
  https://lightning.ai/docs/pytorch/stable/data/datamodule.html.

With this fix, `pytest tests/test_eval.py::test_train_validate -m "slow and gpu" -v` passes (7:47 on
  RTX 5060 Ti). The parity assertion `abs(train_val_loss - eval_val_loss) < 0.001` holds because
  `cfg_eval.trainer.limit_val_batches` was already pinned to `cfg_train.trainer.limit_val_batches`
  in 779035d.

`kosc_datamodule.py` and `fm_datamodule.py` have the same latent stage-gate bug but are out of scope
  for this PR — followup issue will track.

Addresses Copilot review comments r3112823274 and r3112823310.

Refs #635.

### Monitoring

- **loggers**: Make W&B opt-in, default to CSV + TensorBoard
  ([#612](https://github.com/tinaudio/synth-setter/pull/612),
  [`f2f508a`](https://github.com/tinaudio/synth-setter/commit/f2f508a9051b7726a96d68b06dd7338f01efd347))

* feat(monitoring): make W&B optional, default to TensorBoard

- Drive logger selection from env vars: if WANDB_API_KEY is unset, use TensorBoard (the new
  default). - Remove the hardcoded tinaudio fallback from configs/logger/wandb.yaml. - Add the set
  -a && source .env; set +a snippet to .env.example.

External users can now run the project end-to-end without a W&B account. Existing WANDB_API_KEY=...
  workflows continue to work unchanged.

Closes #598

* refactor(monitoring): drop W&B from default many_loggers compose

Simpler than the runtime WANDB_API_KEY gate: make the default logger composition CSV + TensorBoard
  and leave W&B as an explicit opt-in (logger=wandb). No env-var side channel in Python; what you
  configure is what you get.

- configs/logger/many_loggers.yaml: compose csv + tensorboard (drop wandb) - configs/train.yaml:
  default stays logger=many_loggers - src/utils/instantiators.py: remove the WANDB_API_KEY gate -
  tests/test_instantiators.py: deleted (no gate to test) - docs/reference/wandb-integration.md:
  update the init table

* docs: align .env.example and wandb-integration.md with null entity default

Three follow-on doc items from the same change:

- .env.example: comment out WANDB_ENTITY/WANDB_PROJECT and add guidance that WANDB_ENTITY is
  optional — leaving it unset defers to the user's W&B default entity, matching the new
  `${oc.env:WANDB_ENTITY,null}` resolver in configs/logger/wandb.yaml. The prior live value
  (WANDB_ENTITY=tinaudio) routed fresh users' runs to the upstream org. -
  docs/reference/wandb-integration.md §5 gap #1: update the resolved gap note — entity now defaults
  to `null`, not `tinaudio`. - docs/reference/wandb-integration.md Code version marker: bump from
  3e60c47/main to 0b55a9e/feat/wandb-optional-by-default so readers know which snapshot the tables
  describe.


## v0.2.1 (2026-04-20)

### Bug Fixes

- **training**: Pass weights_only=False to Trainer ckpt_path loads (PyTorch 2.6+)
  ([#634](https://github.com/tinaudio/synth-setter/pull/634),
  [`461774e`](https://github.com/tinaudio/synth-setter/commit/461774e88fb423d7d69a6460483cffbc4c4d1164))

* fix(training): pass weights_only=False to Trainer ckpt_path loads

PyTorch 2.6 flipped torch.load's default to weights_only=True, which rejects our user-defined
  checkpoint classes and breaks every Lightning trainer.fit/test/validate/predict(ckpt_path=...) in
  src/train.py and src/eval.py. Lightning 2.6.1 exposes weights_only as a public kwarg on all four
  Trainer entry points, so the minimal clean fix is to opt out at the five call sites that load our
  own (trusted) checkpoints.

A follow-up refactor tracked in #633 will centralize the policy in a TrustedLocalCheckpointIO plugin
  so future call sites inherit it.

Fixes #627

* build(deps): bump lightning minimum to 2.6.0 for weights_only kwarg

The Trainer.fit/test/validate/predict `weights_only` kwarg was introduced in Lightning 2.6.0.
  Previous pin `>=2.0.0` would hard-fail with TypeError on any install that resolved to 2.5.x or
  earlier.

Refs #627

### Build System

- Migrate skills from submodule to plugin marketplace
  ([#546](https://github.com/tinaudio/synth-setter/pull/546),
  [`eb7b36a`](https://github.com/tinaudio/synth-setter/commit/eb7b36a15aacdffcd33991d74bcea7b30393f9a0))

* build: migrate skills from submodule to plugin marketplace

* build: correct plugin name to tinaudio-synth-setter-skills

Upstream tinaudio/skills main marketplace.json renamed the plugin from synth-setter-skills to
  tinaudio-synth-setter-skills. The submodule pin we were removing was on an older branch that still
  used the old name, so the previous commit's settings.json would have silently failed to resolve
  the plugin against upstream main.

* build: update project-standards skill reference for plugin rename

Upstream tinaudio/skills main renamed the project-standards skill to synth-setter-project-standards
  (both directory path and the skill's own name frontmatter). The upstream review skill's
  orchestration list already uses the new name; update CLAUDE.md's code-review section to match so
  /review resolves correctly after the plugin migration.

* build: address Copilot review feedback on PR #546

- .devcontainer/post-create.sh: drop stale 'submodule update' from the comment describing why we
  mark the repo safe.directory (comment #3074807644) - .github/workflows/auto-approve.yml: remove
  'Claude Code Review' from workflow_run.workflows trigger list; the consumer workflow was deleted
  earlier in this PR but the trigger list still referenced it, which contradicted the updated
  docs/reference/github-actions.md dependency map (comments #3074807707 and #3074859911) -
  docs/operations/credential-rotation-guide.md: update What: to past tense and Verification: to note
  the smoke test is stale, so the section no longer contradicts the TODO added above it (comment
  #3074807731). Full rotation-procedure rewrite is still deferred to a follow-up.

- Synth-setter macOS VM ([#590](https://github.com/tinaudio/synth-setter/pull/590),
  [`5377b28`](https://github.com/tinaudio/synth-setter/commit/5377b285ab5013e8f555a7634554f8f1bf104d53))

* internal-feat(pipeline): add Tart macOS VM provisioner

Adds tart/macos.pkr.hcl — a Packer template that builds a macOS Tart VM mirroring the
  docker/ubuntu22_04 dev-base runtime: Surge XT via Homebrew cask, Python 3.10 via uv, all
  requirements.txt deps installed into a venv auto-activated on shell login, and the same smoke-test
  gates used in Docker (VST3 load check + pytest -k "not slow").

The template produces a VM that can be published manually to docker.io/tinaudio/synth-setter-macos
  for downstream dev consumption. Quick-start consumer commands live at the top of the file; full
  build and publish commands live at the bottom.

Refs #380

* internal-fix(tart): address review feedback on macos.pkr.hcl

- Drop `brew upgrade` from Homebrew bootstrap. Running it during image build makes the resulting VM
  non-reproducible: two builds on different days can diverge based on upstream formula updates. The
  subsequent `brew install` already pulls current versions for everything we need. (comment
  #3105522698) - Replace `source ~/.zprofile` with POSIX `. ~/.zprofile` in all four provisioners.
  Packer's shell provisioner default shebang is `/bin/sh`, where `source` is a bash/zsh-ism; `.`
  works in every POSIX shell. (comments #3105522713, #3105522716, #3105522720, #3105522726) -
  Capitalize and code-format the `tart --help` line in the quick-start header. (comment #3105522708)

* docs(getting-started): document Tart macOS VM path

Adds a "2h. Alternative: macOS VM (Tart)" section to the getting-started guide covering the prebuilt
  `docker.io/tinaudio/synth-setter-macos` image, prerequisites, and the advanced Packer build flow
  with overridable vars. Registers `tart/**` as a source for getting-started.md in the doc-map so
  doc-drift will flag future changes.

* internal-fix(tart): address round-2 review feedback on macos.pkr.hcl

- Pin uv to 0.11.2 via post-install assertion for parity with the Docker dev-base image. - Make the
  ~/.zshrc venv-activation append idempotent (touch + grep guard). - Drop Delete scope from the
  Docker Hub PAT recommendation (least-priv). - Soften "reproducible" wording in getting-started.md;
  spell out that brew formulas are not version-pinned. - Add security notes (template + docs)
  flagging that the VM inherits the cirruslabs base image's well-known admin/admin credentials and
  should be treated as local-only.

* internal-fix(tart): use registry-1.docker.io for Docker Hub references

Tart 2.32.1 takes the registry hostname literally and does not alias the canonical `docker.io` short
  name to `registry-1.docker.io` the way the Docker CLI does. `https://docker.io/v2/...` 302s to
  `www.docker.com/...` which returns HTML, breaking `tart clone`/`tart login`/`tart push` with:

Error: DecodingError.dataCorrupted ... Unexpected character '<' ...

Replaces every Docker Hub reference (quick-start clone, login, both push commands, docs
  getting-started.md pull command + image slug, doc-map sources entry) with
  `registry-1.docker.io/...`, which matches what actually works with Tart and what earlier drafts of
  the template used.

Tart 2.32.1 takes the registry hostname literally and does not alias the canonical 'docker.io' short
  name to 'registry-1.docker.io' the way the Docker CLI does. https://docker.io/v2/... 302s to
  www.docker.com/... which returns HTML, breaking tart clone/login/push with:

Replaces every Docker Hub reference (quick-start clone, login, both push commands, docs
  getting-started.md pull command + image slug, doc-map sources entry) with
  registry-1.docker.io/..., which matches what actually works with Tart and what earlier drafts of
  the template used.

* internal-fix(tart): pin uv via Astral installer instead of Homebrew

Homebrew's uv formula is rolling and can't reliably serve a specific historical version, so `brew
  install uv` + an exact-version assertion would start failing as soon as Homebrew bumped past
  0.11.2. Replace it with Astral's versioned installer URL (`https://astral.sh/uv/<v>/install.sh`),
  which embeds the version and is reproducible. Expose the pin as a new `uv_version` variable so the
  update path alongside docker/ubuntu22_04/Dockerfile is explicit.

Addresses review comment on PR #590.

* internal-feat(tart): symlink Surge XT into repo plugins/ dir

Matches the Docker dev-base convention (docker/ubuntu22_04/Dockerfile): after cloning the repo,
  symlink the cask-installed VST3 bundle to the repo-relative 'plugins/Surge XT.vst3' path. This is
  what the pipeline configs (configs/dataset/*.yaml), CLI --plugin_path defaults
  (src/data/vst/generate_vst_dataset.py, scripts/predict_vst_audio.py,
  scripts/surge_xt_interactive.py) and tests all assume, so users can run commands from
  ~/synth-setter without passing an absolute path.

Smoke test updated to exercise the same relative path users will hit.

* internal-fix(tart): touch ~/.zprofile before sourcing in all provisioners

Each shell provisioner sources `~/.zprofile` first, but the cirruslabs base image does not guarantee
  the file exists. Under `set -e` (Packer's default), `. ~/.zprofile` would abort provisioning
  before the build can seed the file. Prepend `touch ~/.zprofile` so the source is always safe and
  the four provisioners stay uniform.

* internal-fix(tart): hard-fail Tart build if Surge XT is not 1.3.4

Homebrew casks are rolling, so `brew install --cask surge-xt` silently upgrades to whatever the cask
  definition resolves at build time. A new Surge XT release could change parameter layout, preset
  format, or default values and silently diverge from the parameter specs in
  src/data/vst/surge_xt.py / configs.

Adds a `surge_xt_version` packer variable (default `1.3.4`) and a post-install assertion using the
  same `brew list --cask --versions` pattern the uv pin uses, so the build fails loudly when the
  cask rolls past the qualified version. Bump only after validating the new release against the
  pipeline.

* docs(tart): push both tags in a single tart push invocation

tart push accepts multiple remote refs positionally; combining :${DATE_TAG} and :latest into one
  call uploads the VM once instead of twice. The second push previously re-uploaded ~29 GB just to
  move the :latest pointer.

* internal-fix(tart): address round-3 review feedback on macos.pkr.hcl

- Drop unused `codex` Homebrew formula; it has no callers in the repo and is not part of the Docker
  dev-base parity baseline (Copilot, comment #3107388564). - Expand the `getting-started.md` vars
  list to enumerate all seven user-overridable packer variables and point readers at the template's
  `variable` blocks as the authoritative source (Copilot, comment #3107388557).

Left as-is: the uv Astral installer `curl | sh` invocation (comment #3107388568). The URL is HTTPS,
  embeds the pinned version, and the post-install `uv --version` assertion catches tampering that
  alters the resolved version; SHA256-pinning the bootstrap script would add ongoing maintenance
  cost disproportionate to a local-only dev VM's risk.

- **devcontainer**: Consolidate Dockerfile into main image stages
  ([#574](https://github.com/tinaudio/synth-setter/pull/574),
  [`6c2d22a`](https://github.com/tinaudio/synth-setter/commit/6c2d22afa307f56084619aae12b73ce020403b26))

* build(devcontainer): consolidate Dockerfile into main image stages

Move all devcontainer tooling (curl, jq, gh, nvm, node, claude-code, non-root dev user) into a new
  devcontainer-tools stage in the main Dockerfile. The .devcontainer/Dockerfile becomes a thin FROM
  extension point, eliminating three redundant apt-get update calls and the github-cli devcontainer
  feature.

Stage graph: dev-base → devcontainer-tools (new) + freeze-deps + dev-snapshot (no-op alias).

* Add target 'dev-snapshot' to Docker build args

- **devcontainer**: Enable Claude Code agent workflows in CPU+GPU containers
  ([#572](https://github.com/tinaudio/synth-setter/pull/572),
  [`75c7a7b`](https://github.com/tinaudio/synth-setter/commit/75c7a7b063b13d63fbb40ced0cf20f97f721c4fa))

* build(devcontainer): enable Claude Code agent workflows in CPU+GPU containers

Make the CPU and GPU devcontainers usable for Claude-Code-driven agent work.

- .devcontainer/Dockerfile: install curl, jq, and gh (from the official apt repo) at image build
  time, and chown -R dev:dev .git so the non-root dev user can run git against the baked repo tree.
  - .devcontainer/cpu/devcontainer.json: add runArgs --env-file .env and a read-only bind-mount of
  ~/.claude/.credentials.json into the container (copy-pasting the Claude auth challenge through
  remote -> host -> container was unreliable). - .devcontainer/gpu/devcontainer.json: same as CPU,
  plus --gpus all and --shm-size=16g. - .devcontainer/post-create.sh: when RESTRICTED_AGENT_GIT_PAT
  is set, pipe it through `gh auth login --with-token && gh auth setup-git`. Use
  ${RESTRICTED_AGENT_GIT_PAT:-} for set -u safety so the else branch is reachable when the var is
  unset and pre-commit install still runs. - .env.example: replace the now-dead GIT_PAT stub
  (removed from all consumers in #567) with a RESTRICTED_AGENT_GIT_PAT block documenting it as the
  scoped in-container agent token.

BREAKING: opening the CPU or GPU devcontainer now requires a .env file in the workspace root.
  --env-file .env makes Docker refuse to start the container otherwise. Copy .env.example to .env
  before first open.

Closes #571

* build(devcontainer): auto-create empty .env via initializeCommand

Remove the hard .env requirement introduced alongside --env-file .env. The initializeCommand runs on
  the host before the container is created, so `[ -f .env ] || touch .env` ensures .env always
  exists. An empty --env-file is a no-op for Docker, so users without a populated .env get a working
  container and users with one get their vars injected as before.

POSIX-compatible shell; the README already declares Windows unsupported.

Refs #571

* fix(devcontainer): non-fatal gh auth and ensure credentials file exists

- Wrap gh auth login in an inner conditional so a bad or placeholder token warns instead of aborting
  the entire post-create setup under set -e. This ensures pre-commit install always runs. - Extend
  initializeCommand to create ~/.claude/.credentials.json on the host if absent, preventing Docker
  from bind-mounting a directory when the file doesn't exist.

* fix(devcontainer): restrict credentials file permissions to 0600

Create ~/.claude/.credentials.json inside a subshell with umask 077 so the file gets 0600
  permissions instead of the default 0644.

- **devcontainer**: Fix git and claude auth issues
  ([#575](https://github.com/tinaudio/synth-setter/pull/575),
  [`3fd9a85`](https://github.com/tinaudio/synth-setter/commit/3fd9a858ce0361fca849e1ac5f5279f187e6d0fe))

* build(devcontainer): fix git and claude auth issues

* build(devcontainer): address copilot review on #575

- Remove dead ~/.claude/.credentials.json creation from initializeCommand in both cpu and gpu
  devcontainer configs (no longer mounted after this PR, so the mkdir+touch of the credentials stub
  was dead work). - Use printf '%s' instead of echo when piping RESTRICTED_AGENT_GIT_PAT into gh
  auth login --with-token, so a token starting with '-' can't be interpreted as an echo option.

Refs #575

- **devcontainer**: Install Claude Code CLI via userspace nvm
  ([#550](https://github.com/tinaudio/synth-setter/pull/550),
  [`f513102`](https://github.com/tinaudio/synth-setter/commit/f513102c51e96c70fcea11b880410906c1ae11fa))

* build(devcontainer): install Claude Code CLI via userspace nvm

Install @anthropic-ai/claude-code 2.1.105 inside .devcontainer/Dockerfile via a userspace nvm
  install, as the existing non-root dev user. Zero root, zero sudo, zero base-image changes —
  respects the existing "no sudo" and "system packages go in the base image" policies already
  documented in this Dockerfile.

Pins: NVM_VERSION=0.40.1, NODE_VERSION=20.18.0 (matches Anthropic's reference devcontainer),
  CLAUDE_CODE_VERSION=2.1.105.

An explicit ENV PATH entry makes `claude` resolve in any shell mode (interactive, non-interactive,
  docker exec, VSCode tasks), not just bashrc-sourcing interactive shells.

Firewall + scoped sudo (Anthropic's init-firewall.sh pattern) are deferred to #549.

Closes #548

* build(devcontainer): clone nvm at pinned tag; explicit bash SHELL

Address review feedback on PR #550:

- Clone nvm at a pinned tag instead of piping install.sh through bash. Resolves the supply-chain
  concern in pull/550#discussion_r3076693266: the cloned tree is auditable on disk, the tag is
  immutable upstream, and no unverified network content is executed during the image build.

- Set SHELL to bash -o pipefail for the nvm install layer. Resolves the implicit bash dependency in
  pull/550#discussion_r3076693291. nvm.sh currently sources under Ubuntu's /bin/sh (dash) because of
  upstream POSIX compatibility, but the dependency on bash is load-bearing and should be explicit;
  pipefail also catches any pipeline-stage failure in the clone + source + install chain.

Side effect: ~/.bashrc no longer contains nvm sourcing lines (those were written by install.sh,
  which is now skipped). This makes the verification strictly stronger — 'claude' still resolves in
  every shell mode, proving ENV PATH is independently sufficient. Users who want the 'nvm' shell
  function for ad-hoc use can source it manually with 'source $NVM_DIR/nvm.sh'.

Refs #548

- **devcontainer**: Overlay plugins/ with anonymous volume
  ([#592](https://github.com/tinaudio/synth-setter/pull/592),
  [`854bee1`](https://github.com/tinaudio/synth-setter/commit/854bee114dfa2d28b3073efb352b01f40fc114e4))

The base image bakes plugins/Surge XT.vst3 -> /usr/lib/vst3/Surge XT.vst3, but the devcontainer's
  workspaceMount shadows it with the host repo, and plugins/ is gitignored. Add an anonymous-volume
  mount at /home/build/synth-setter/plugins so Docker auto-seeds the directory from the image
  contents on container creation, without writing to the host filesystem.

Fixes #591

- **devcontainer**: Refuse to open from a git worktree
  ([#594](https://github.com/tinaudio/synth-setter/pull/594),
  [`4a99c59`](https://github.com/tinaudio/synth-setter/commit/4a99c59a2a8e51c3100fcb75e927a784566630f9))

* fix(devcontainer): refuse to open from a git worktree

A linked worktree's .git is a pointer file referencing an absolute host path into the parent repo's
  admin directory. That path is not mounted into the container, so post-create.sh fails
  mid-provision with a cryptic "not a git repository" error and leaves the container half-configured
  (no pre-commit install, no safe.directory config).

Move the .env touch into a new .devcontainer/initialize.sh that also detects the worktree pointer
  and exits with a clear error naming the two supported branch-isolation workflows: host-only
  worktree or devcontainer-from-root with `git checkout -B`. The parallel-devcontainer edge case
  (rare, not targeted by CLAUDE.md) is documented inline as a local-only patch.

Fail fast in initializeCommand, before any image build.

Fixes #593

* docs(getting-started): note the worktree hardstop in §2g

Existing doc prescient-ly warned that mounting a worktree directly does not work. Append a sentence
  noting that as of the hardstop change, the failure now surfaces as an explicit error at
  `initializeCommand` time rather than partway through `post-create`.

Refs #593

* fix(devcontainer): point initialize.sh message at docs, reference #593

The previous error suggested `git checkout -B <branch>` as workflow 2, which was technically correct
  but too narrow — docs/getting-started.md §2g already documents the recommended pattern (create a
  worktree INSIDE the container), and that works because the in-container .git is a real directory
  resolving normally.

Rephrase both the error and the adjacent comment to point at the docs rather than prescribe one
  command, and add an explicit reference to #593 for the tradeoff analysis and escape hatch.

* fix(devcontainer): reorder initialize.sh, broaden pointer-file wording

Address Copilot review feedback on PR #594:

1. Reorder: move `.env` touch below the pointer-file guard. An aborted run
  (worktree/submodule/--separate-git-dir case) no longer leaves a stray empty .env in the workspace.

2. Generalize wording: the `gitdir:*` guard actually matches any pointer-file .git (worktrees are
  just the common case; submodules and `git clone --separate-git-dir` repos have the same failure
  mode). Update the header comment and error heredoc to say "pointer file" and name worktree as the
  common case, so the error reads correctly in the rarer cases too.

Guard logic unchanged — still exits 1 for any `gitdir:` prefix, preserving the existing Level 1
  behavior verified in the initial PR comment.

* Clean up comments in initialize.sh

Removed detailed comments regarding the pointer-file .git and its implications for devcontainer
  usage.

- **devcontainer**: Split into cpu and gpu flavors
  ([#553](https://github.com/tinaudio/synth-setter/pull/553),
  [`a631c78`](https://github.com/tinaudio/synth-setter/commit/a631c788f518c785947ec8f3e5b050a28777787b))

Replace the single .devcontainer/devcontainer.json with two flavors under .devcontainer/cpu/ and
  .devcontainer/gpu/ sharing the existing Dockerfile and post-create.sh. The only difference is
  runArgs: the gpu flavor passes --gpus all so NVIDIA Container Toolkit hosts get device access.
  Both keep MODE=idle so the entrypoint runs `sleep infinity` and VS Code can attach shells.

Refs #538

- **devcontainer**: System-wide Claude Code, parameterize remoteUser, persist bash history
  ([#581](https://github.com/tinaudio/synth-setter/pull/581),
  [`0d49325`](https://github.com/tinaudio/synth-setter/commit/0d493251a398a4f7f021536c57d248e082976d52))

* build(devcontainer): move Claude Code to system-wide install and parameterize remoteUser

- docker/ubuntu22_04/Dockerfile: install Node.js (apt via NodeSource) and @anthropic-ai/claude-code
  system-wide as root before the USER dev switch. Binary lives at /usr/local/bin/claude; both root
  and dev run the same binary, each with its own $HOME/.claude state. Drops the per-user nvm install
  so root can also `npm install -g` to update. - .devcontainer/{cpu,gpu}/devcontainer.json:
  parameterize remoteUser via ${localEnv:DEVCONTAINER_USER:dev}. Default behavior unchanged (dev).
  Set DEVCONTAINER_USER=root on the host before reopening the folder in container to run sessions as
  root. dev remains unprivileged (no sudo).

* refactor Node.js Claude Code installation and persistent bash history

Updated Dockerfile to install Node.js and Claude Code CLI from NodeSource and added bash history
  persistence.

* Refactor Node.js and npm installation commands

Updated Node.js and npm installation in Dockerfile.

* Add Claude Code CLI version argument to Dockerfile

* Add volume mount for command history

* Create non-root user and set up command history

Add non-root user for development and persist bash history.

* build(devcontainer): drop post-create.sh privileges to dev when invoked as root

Prevents root-owned .git/hooks/* and .git/config landing in the bind-mounted workspace when
  DEVCONTAINER_USER=root or under Codespaces (both run postCreateCommand as root). Guard re-execs
  via runuser -u dev.

Refs #580

- **docker**: Publish public tinaudio/synth-setter image
  ([#567](https://github.com/tinaudio/synth-setter/pull/567),
  [`dbc8f85`](https://github.com/tinaudio/synth-setter/commit/dbc8f85ca977a5ee9d6b7f384dd7cc8a9b105c71))

* build(docker): rename to tinaudio/synth-setter and strip baked secrets

Delete the r2-config-base Dockerfile stage and W&B netrc bake block so the image contains no
  embedded credentials. R2_BUCKET remains as a non-sensitive build arg. Callers now provide R2
  credentials and WANDB_API_KEY at runtime via env vars.

Refs #564

* ci: switch workflows to public synth-setter image with runtime secrets

Drop BuildKit R2/W&B secret mounts from docker-build-validation, remove the private-registry
  visibility gate, add the 'latest' tag on main, and switch cache refs to
  tinaudio/synth-setter:buildcache.

Dataset and spec workflows now pipe R2 credentials into docker run via RCLONE_CONFIG_R2_* env vars.
  Docker Hub login is dropped from pull-only steps since the image is public.

* build(devcontainer): use public synth-setter image, drop baked-cred copy

The base image no longer ships credentials, so the /root -> /home/dev copy block is dead code.
  Document that R2/W&B creds must come from runtime env vars (Codespaces secrets or mounted .env).

* docs(reference): document runtime secret piping for public image

Update docker.md, docker-spec.md, and github-actions.md to reflect that R2 and W&B credentials are
  no longer baked into the image; they flow in at runtime via env vars. Remove the 'must remain
  private' invariant.

* docs(reference): add missing R2_ENDPOINT row to github-actions secrets table

All three pipeline workflows (dataset-generation, spec-materialization, test-dataset-generation)
  consume secrets.R2_ENDPOINT, but the table previously omitted the row. Follow-up to 9e3c2e6a.

* docs: document public image and runtime credential flow

Update getting-started and credential-rotation to reflect that the public image does not bake R2/W&B
  credentials. Rotation no longer requires rebuilding images.

* docs(design): rename tinaudio/perm refs and add migration plan

Update design docs to use tinaudio/synth-setter. Commit the implementation plan that drove this
  migration.

* fix(docker): stop persisting GIT_PAT in .git/config inside the image

The dev-snapshot stage previously fetched source via `git remote add origin
  "https://${token}@github.com/...`, which wrote the token into .git/config. Anyone pulling the
  public image could extract a live GitHub PAT from /home/build/synth-setter/.git/config.

Switch to `git -c http.<url>.extraheader=Authorization: Bearer ...` which passes the token
  ephemerally (only that invocation) and never persists it to config. The origin URL now stores no
  credentials.

Required for #564 — without this fix the image cannot be safely published publicly.

* fix(ci): tighten latest tag gate for workflow_dispatch

{{is_default_branch}} only checks github.ref (the branch the workflow dispatched from), not the
  git_ref input that determines which commit actually gets baked. Dispatching from main with
  git_ref=feature/foo would previously have tagged that feature branch's build as latest.

Restrict latest to: - schedule events (always on main) - workflow_dispatch where git_ref=main

* fix(docker): use x-access-token for git fetch, scrub config post-fetch

bb37dd4 attempted to avoid persisting GIT_PAT via `git -c http.<url>.extraheader=Authorization:
  Bearer ${token}`, but GitHub's git HTTPS endpoint rejects Bearer tokens (they're accepted for API
  calls but not for git operations, which require HTTP Basic). The buildx run failed with exit code
  128 on the fetch step.

Switch to the standard GitHub pattern: embed `x-access-token:${token}` in the remote URL for the
  fetch, then overwrite the URL via `git remote set-url` to scrub credentials from .git/config.
  Verified locally that (a) git's FETCH_HEAD auto-strips credentials from fetched URLs, and (b)
  after set-url, no trace of the token or x-access-token username remains in .git/.

* Delete docs/superpowers/plans/2026-04-15-public-docker-image.md

* fix(ci): gate Docker Hub login to non-PR events

On PR runs the workflow only build-validates (push: false), so the Docker Hub login is unused.
  Gating it on non-PR events means fork PRs (which never receive DOCKERHUB_USERNAME/TOKEN) can still
  exercise the Dockerfile without hitting a failure before the build starts.

Matches the existing gate on the adjacent Verify Docker Hub push access step.

* build(docker): drop GIT_PAT requirement (public repo)

The repository is being made public, so fetching the source tarball and cloning the repo inside the
  dev-snapshot stage can both happen anonymously. Remove:

- The two BuildKit secret mounts that consumed git_pat (sanity check in builder-base and the
  dev-snapshot git fetch block) - The Authorization: Bearer header on the synth-setter-src tarball
  download; curl now hits the public tarball endpoint directly - The --secret id=git_pat line from
  the Makefile recipe and the GIT_PAT variable from its usage docs - The git_pat entry from the
  docker-build-validation BuildKit secrets block (now empty — no build-time secrets at all) - The
  GIT_PAT env + make arg from the flush-investigation workflow - The GIT_PAT row from
  credential-rotation-guide, docker.md, and github-actions.md; the rotation runbook no longer
  mentions it

One less credential to rotate, one less secret to leak, and zero plumbing overhead for external
  contributors who want to build the image themselves.

* build(docker): move R2_BUCKET from build-arg to runtime env var

R2_BUCKET is non-sensitive but baking it into the image at build time freezes each published image
  to one specific bucket. Moving it to a runtime env var lets the same public image point at any
  bucket — useful for external users pointing at their own R2 account and for avoiding a rebuild
  every time the CI bucket changes.

Changes:

- Dockerfile: drop the top-level ARG R2_BUCKET and the runtime-base stage that baked it.
  dev-snapshot now inherits directly from builder-install-synth-setter-deps. - Makefile: already had
  no R2_BUCKET plumbing (cleaned up as part of the GIT_PAT removal commit). -
  docker-build-validation.yml: drop R2_BUCKET from build-args. The image_config.r2_bucket GH output
  is still generated by load_image_config but no longer consumed by the build. -
  dataset-generation.yml: read r2_bucket from configs/image/dev-snapshot.yaml at workflow time and
  pass it to docker run via -e R2_BUCKET=... This mirrors the pattern already used by
  test-dataset-generation.yml for its validate-shard job. - docs: update docker-spec.md §2/§3
  (runtime-base → direct inheritance, drop R2_BUCKET row from build-args and baked-env tables, add
  R2_BUCKET as a required env var for MODE=generate_dataset).

The pydantic ImageConfig schema still validates r2_bucket because the YAML is still the single
  source of truth for the CI bucket name — just not via a build-arg anymore.

* docs: document latest tag, correct devcontainer .env story

Three Copilot review findings on #567:

- docs/reference/docker.md Tags table: add tinaudio/synth-setter:latest row and note that
  latest/dev-snapshot are only published on dispatch/schedule (not PR builds), and latest is
  additionally gated to main-branch builds (matches the gate in 703b618). - docs/getting-started.md
  §2g: the dev container configs do NOT forward .env automatically. Users must source it manually or
  set the vars via Codespaces / Dev Container env settings. - .devcontainer/post-create.sh header:
  match the same reality — .env is not auto-loaded.

* docs: unify runtime env var enumeration across docker/spec/getting-started

The three docs had three incomplete views of the same information:

- docker.md § Runtime secrets listed 7 vars (complete) but §3.3 MODE=generate_dataset listed only
  DATASET_CONFIG + RUN_METADATA_DIR, silently omitting R2_BUCKET / RCLONE_CONFIG_R2_* /
  WANDB_API_KEY. - docker-spec.md §3.3 had R2_BUCKET but not rclone/wandb vars. - getting-started.md
  §4b told users to put RCLONE_CONFIG_R2_TYPE and RCLONE_CONFIG_R2_PROVIDER in .env, while docker.md
  said they were "fixed — set in run". Both work but the source-of-truth was mixed.

Consolidate to a single 10-row table as the canonical enumeration:

- MODE, DATASET_CONFIG, RUN_METADATA_DIR (mode dispatch + args) - R2_BUCKET (runtime bucket config,
  non-secret) - RCLONE_CONFIG_R2_TYPE, _PROVIDER (rclone constants) -
  RCLONE_CONFIG_R2_ACCESS_KEY_ID, _SECRET_ACCESS_KEY, _ENDPOINT (secrets) - WANDB_API_KEY (secret)

docker.md § Runtime environment variables is the canonical table. docker-spec.md mirrors it (with a
  note that it's kept in sync). docker.md §3.3 MODE=generate_dataset now points to the canonical
  table rather than having its own partial version. getting-started.md §4b has a complete .env
  template covering all user-provided vars in one place.

Also clarifies in all three docs that R2_BUCKET is not part of the rclone remote config — it's a
  separate bucket-name argument that generate_dataset.py interpolates into upload paths.

* refactor(docker): use git clone instead of init+remote+fetch

Functionally equivalent to the previous init+remote+fetch dance, but:

- 2 commands instead of 4 (more idiomatic, less for a reader to parse) - git clone creates a local
  `main` branch with upstream tracking wired up automatically, so shelling into the image for
  interactive development — git switch main, git pull, git rebase origin/main — just works instead
  of requiring `git checkout -b main origin/main` then `git branch --set-upstream-to=origin/main`.

The image's checkout is still pinned to SYNTH_PERMUTATIONS_GIT_REF via the detached-HEAD checkout,
  so reproducibility of the baked source is unchanged. The only runtime cost is one redundant
  working-tree write (git clone checks out main before the detach overwrites it), which is
  sub-second on synth-setter's tree size.

* revert: use git clone instead of init+remote+fetch

Reverts the git-clone refactor from 8143856. The refactor broke the build because an upstream stage
  (builder-install-synth-setter-deps, inherited by the dev-snapshot target) creates
  /home/build/synth-setter/ plugins/Surge XT.vst3 before the git step runs. `git clone URL .` aborts
  with "destination path '.' already exists and is not an empty directory", whereas `git init`
  happily creates .git/ alongside the existing plugins symlink.

docker-build-validation run 24451167961 confirmed the failure with buildx exit code 128 on the git
  clone step.

Left an inline comment in the Dockerfile explaining why we can't use `git clone .` here, so nobody
  re-attempts this refactor without first relocating the plugins symlink to the dev-snapshot stage.

* docs(ci): tighten the login-gate comment about fork PRs

The original comment claimed gating the Docker Hub login step would let fork PRs "exercise the
  Dockerfile". That's imprecise — the gate only prevents a spurious credential failure at the login
  step. Fork PRs still fail later because the in-image git fetch resolves SYNTH_PERMUTATIONS_GIT_REF
  against upstream tinaudio/synth-setter, not the fork's origin. A real fork-PR build would need to
  fetch from github.event.pull_request.head.repo.html_url; not implemented here.

The gate is still valuable: it matches the existing gate on the Verify Docker Hub push access step,
  and it prevents same-repo PRs from hitting a spurious login failure if DOCKERHUB_* secrets are
  temporarily unavailable.

Surfaced by Copilot review.

- **evaluation**: Add pandas to requirements-app.txt
  ([#579](https://github.com/tinaudio/synth-setter/pull/579),
  [`1bf8d3b`](https://github.com/tinaudio/synth-setter/commit/1bf8d3b2b77b553f68bd984c61a5895c2d67aacd))

scripts/predict_vst_audio.py imports pandas but it was missing from the app requirements, causing
  ModuleNotFoundError when running the synth matching CLI in a fresh app env.

Fixes #578

- **evaluation**: Pin eval metric dependencies
  ([#611](https://github.com/tinaudio/synth-setter/pull/611),
  [`a53c9c9`](https://github.com/tinaudio/synth-setter/commit/a53c9c91e7a9ea98a653b5dd8745b8c291a40a43))

* build(evaluation): pin eval metric dependencies

Add pesto-pitch, dtw-python, kymatio, and loguru to requirements-app.txt so
  scripts/compute_audio_metrics.py can run on a fresh checkout.

Closes #605

* build(evaluation): move loguru pin out of eval-metrics group

loguru is used broadly across the repo (src/data/vst/, several scripts), not just eval metrics. Move
  the pin into the general requirements block alphabetically and keep the eval-metrics group for
  eval-specific deps.

Addresses Copilot review feedback on #611.

### Chores

- Correct 'isse' typo in initialize.sh error message
  ([#596](https://github.com/tinaudio/synth-setter/pull/596),
  [`b5690c4`](https://github.com/tinaudio/synth-setter/commit/b5690c4cd8e8f22c89959479f85f0503f600b628))

- Gitignore .worktrees/ and /worktrees/ directories
  ([#616](https://github.com/tinaudio/synth-setter/pull/616),
  [`159ae1e`](https://github.com/tinaudio/synth-setter/commit/159ae1e36df2bcb7fcb380071f1d1d4315c35eea))

Users regularly run `git worktree add` into `.worktrees/` and `/worktrees/` subdirectories of the
  repo (per CLAUDE.md's isolated-worktree workflow). Git currently reports these as untracked, which
  creates noise on every `git status` and risks accidental staging.

Ignore both directories under a new 'Git worktrees' section.

Closes #615

- **ci**: Add Claude Code hooks for doc-drift and pr-review-resolver
  ([#587](https://github.com/tinaudio/synth-setter/pull/587),
  [`d737bca`](https://github.com/tinaudio/synth-setter/commit/d737bcae92e3ba04e90dff8c24dfe48250d5a1fb))

* chore(ci): add Claude Code hooks for doc-drift and pr-review-resolver

Two PostToolUse hooks that fire after Claude runs gh pr create or git push. Both are advisory
  (asyncRewake, exit 2 with a pointer), run a headless claude -p session invoking the matching skill
  (or an inline fallback if the skill is missing), and write the report under
  .agent-reviews/<uuid>.md.

- doc-drift on gh pr create (timeout 900s). Fallback references docs/doc-map.yaml. -
  pr-review-resolver on git push (timeout 1200s). Skips main/master; waits RESOLVER_SLEEP_SECS
  (default 360s) for CI and reviewers to settle; per-branch lockfile dedupes stacked pushes (last
  wins).

.claude/hooks/test.sh is a 10-assertion unit harness (canned stdin, PATH-stubbed claude/gh)
  covering: match/no-match, skill-missing fallback text, main-push early-exit, no-PR silent skip,
  lockfile dedupe.

.agent-reviews/ added to .gitignore.

Closes #586

* chore(ci): make hooks worktree-aware and robust to malformed stdin

Two findings from live verification on PR 587:

1. has_skill() only looked at .claude/skills/ relative to CWD, but .claude/ is gitignored and the
  plugin installs skills to the main repo checkout. From a worktree, detection always missed. Added
  lookup via git --git-common-dir so worktrees find the skills installed in the parent repo.

2. Malformed tool-input JSON (e.g. empty or truncated) caused jq to fail, and set -e + pipefail
  propagated the exit. The hook runner will never send bad JSON in practice, but fail-closed on
  malformed input is the right posture: silence, not noise.

* chore(ci): tighten hook matchers to shell-word boundaries

Live Level-1 verification (fresh claude -p session firing the real hook runner) surfaced that
  substring matching on 'gh pr create' and 'git push' also fires when those phrases appear inside an
  echo's quoted argument or a git commit message. The headless Claude itself noticed the misfire and
  declined to act on the advisory report — which is the correct defensive behavior, but the hook
  should not be triggering in the first place.

Replace the case-*substring* match with a POSIX regex that requires the phrase to sit at
  start-of-line or right after a shell operator (;, |, &, backtick, open-paren). Plain-whitespace
  prefix no longer counts, so 'echo "testing gh pr create"' no longer triggers.

Two new unit assertions cover the word-boundary behaviour (12/12). The pre-existing pr-checkbox and
  taxonomy trigger hooks in settings.json still use loose substring matching — intentionally not
  touched in this PR (out of scope).

* chore(ci): address Copilot review on PR hooks

Eight review comments from Copilot, all actionable:

- Add `if` guards at the settings.json level (comments #1, #2) so the hook commands no-op for
  unrelated Bash calls. Reuses the tight shell-word-boundary regex already in the scripts, not
  Copilot's suggested whitespace-only form — Level-1 testing proved the looser version misfires on
  quoted text inside echo args and commit messages. - Harden has_skill against unset/empty HOME and
  spaces in paths (comment #3). Skip user-global checks when HOME is empty. - Resolve the default
  branch via origin/HEAD instead of hardcoding `main` in the doc-drift diff command (comment #4).
  Exposes a new default_branch helper in _lib.sh. The Level-1 no-dry-run run surfaced the same bug:
  `git diff main...HEAD` returned files from already-merged PRs because local main was stale vs
  origin/main. - Verbose failure report on nested `claude -p` errors (comments #5, #7). Capture
  stderr, write a FAILED report with exit code, prompt, and stderr tail, and still exit 2 so the
  session is woken. New unit assertion covers the failure path. - Switch the resolver lock TOKEN
  from `$$-$(date +%s%N)` to gen_id for portability (comment #6). - Fix test.sh header comment — git
  is real, only claude/gh are stubbed (comment #8). - Add gen_id to the _lib.sh helper list in the
  header comment — a drift finding surfaced by the Level-1 doc-drift report itself.

- **ci-automation**: One-shot conda env and conda test workflow
  ([#558](https://github.com/tinaudio/synth-setter/pull/558),
  [`0b2354c`](https://github.com/tinaudio/synth-setter/commit/0b2354cb48cc309d2c04397cff00b1eef22166d5))

* chore(ci-automation): include requirements.txt from environment.yaml and add conda test workflow

Consolidates the conda dev flow so a single `conda env create -f environment.yaml` installs both the
  conda deps and the pip deps from requirements.txt — no more manual two-step setup. Adds a
  test-conda.yml workflow that exercises this path in CI so the conda env stays functional alongside
  the existing uv-based tests.

Refs #557

* chore(ci-automation): pip-install requirements-app.txt directly, not requirements.txt

conda already provides torch/torchvision/lightning/torchmetrics via precompiled binaries (the reason
  to use the conda path at all). Pulling in requirements.txt also pulls in requirements-torch.txt,
  which would redundantly re-install the torch stack through pip and risks pip/conda conflicts on
  the same packages. Swap to -r requirements-app.txt so conda owns the torch stack and pip owns
  everything else.

* chore(ci-automation): align conda torch stack specs with pip and drop overlap

Two fixes for the conda env layout:

1. Align torch-stack version specs with requirements-torch.txt. Previous conda specs
  (torchvision=0.*, torchmetrics=0.*, pytorch=2.*) allowed the resolver to pick versions older than
  the pip specs require (e.g. torchvision 0.14.x when pip wants >=0.15.0). Tighten conda to >=2.0.0
  / >=0.15.0 / >=0.11.4 / >=2.0.0 so the two installers agree on a common lower bound.

2. Drop hydra-core, rich, pre-commit, and pytest from the conda dependency list. They're already
  pulled in by `-r requirements-app.txt`, so the conda block was just duplicating them. After this
  change the conda and pip sets are non-overlapping by construction — conda owns the torch stack,
  pip owns everything else.

Also add setuptools to the conda deps so torchmetrics' import of `pkg_resources` works on a minimal
  conda env.

* chore(ci-automation): prioritize conda-forge channel to fix libtiff.so.5 import error

With `pytorch` listed first, conda was pulling pillow (a torchvision dep) from the pytorch channel,
  which ships a build that dynamically links libtiff.so.5. That lib isn't present in the minimal
  conda env, so `import PIL` blew up during torchmetrics import in the test-conda workflow:

ImportError: libtiff.so.5: cannot open shared object file: No such file or directory

Reordering the channels so `conda-forge` wins for shared deps fixes this — conda-forge's pillow
  bundles its own libtiff. The pytorch channel stays as a fallback for torch-stack builds that only
  live there.

- **code-health**: Disable plumb hooks and workflow docs
  ([#554](https://github.com/tinaudio/synth-setter/pull/554),
  [`da34fb1`](https://github.com/tinaudio/synth-setter/commit/da34fb1a58a57da6b898d1d327b7ca0707428fd6))

Light disable: remove the devcontainer plumb init, the CLAUDE.md plumb workflow block, and the
  CONTRIBUTING.md plumb section so the tool stops intercepting commits and directing contributors to
  a disabled flow.

Intentionally preserved for a follow-up decision (see #552): - plumb-dev pin in requirements-app.txt
  - docs/plumb_spec.md - .plumbignore and .plumb/ gitignore block - # plumb:req-* tags across tests/

Closes #551. Refs #552. Part of #466.

- **code-health**: Fully remove plumb tooling and artifacts
  ([#566](https://github.com/tinaudio/synth-setter/pull/566),
  [`77e2a72`](https://github.com/tinaudio/synth-setter/commit/77e2a7268b09309fae91c36b5d9119cf2f3e9c53))

Removes the plumb spec/test/code sync tool and all its artifacts from the repo. Drops the .plumb/
  metadata dir, .plumbignore, docs/plumb_spec.md, the plumb-dev dependency, and the orphan hatchling
  build-backend dep that only existed to build plumb-dev. Strips the 122 plumb:req-<hash>
  requirement annotations across 15 test files.

Closes #552

- **vst**: Isolate VST headless runtime files in mktemp dir
  ([#582](https://github.com/tinaudio/synth-setter/pull/582),
  [`7e9a252`](https://github.com/tinaudio/synth-setter/commit/7e9a2523cb45ddd5dcd1f6b28f7f906c4828eb48))

* fix(scripts): isolate VST headless runtime files in mktemp dir

Move XAUTHORITY and xvfb/xsettingsd/openbox log paths from shared /tmp/*.log (hardcoded) into the
  script's own mktemp TMP_DIR so concurrent invocations don't race on the same files. The existing
  EXIT trap already cleans up TMP_DIR, so logs and the Xauthority file are removed automatically on
  shutdown.

Refs #528

* fix(scripts): drop redundant EXIT trap and stale comment in VST headless

The initial `trap 'rm -rf "$TMP_DIR"' EXIT` is overwritten by the later `trap cleanup EXIT`, so the
  first handler never runs. `cleanup()` already calls `rm -rf "$TMP_DIR"` on its own, so removing
  the redundant trap is a pure simplification. Also drop the now-misleading "# Create temp dir for
  display number coordination" comment since TMP_DIR holds more than just the display number file.

Review feedback from copilot-pull-request-reviewer on #582.

### Continuous Integration

- Hash requirements-app.txt in test-conda cache key
  ([#570](https://github.com/tinaudio/synth-setter/pull/570),
  [`c21fde8`](https://github.com/tinaudio/synth-setter/commit/c21fde8b7bd4a6c147dcbd579d1a25171e5d0e0f))

setup-micromamba's default cache-environment key only hashes environment.yaml and does not follow
  the pip `-r requirements-app.txt` reference inside it. So PRs that change only
  requirements-app.txt leave environment.yaml byte-identical, the cached env is reused, and the
  pip-dep change is never actually installed or tested.

Set cache-environment-key explicitly to hash both environment.yaml and requirements-app.txt so the
  cache invalidates whenever either file changes. requirements-torch.txt is intentionally excluded:
  the torch specs are inlined directly in environment.yaml's conda section, so any torch version
  change already touches environment.yaml and invalidates the default key.

Closes #569

- Run devcontainer as non-root dev user ([#540](https://github.com/tinaudio/synth-setter/pull/540),
  [`c8bb6f6`](https://github.com/tinaudio/synth-setter/commit/c8bb6f62a721154e875d9a7c5ba3b35ec6eabeaa))

* ci: run devcontainer as non-root dev user with passwordless sudo

Claude Code refuses to run with --dangerously-skip-permissions as root, which blocks using it inside
  the devcontainer. Switch to the official VS Code non-root user pattern.

Add .devcontainer/Dockerfile extending tinaudio/perm:dev-snapshot with: - dev user (UID 1000) with
  passwordless sudo - chown /venv/main so `uv pip install -e .` works without sudo - copy baked R2
  (rclone.conf) and W&B (.netrc) credentials from /root into /home/dev so runtime tooling keeps
  working

Update .devcontainer/devcontainer.json to build from the Dockerfile, set remoteUser=dev, and enable
  updateRemoteUserUID so the container UID is remapped to the host user on Linux hosts.

post-create.sh needs no changes — the /venv/main chown is sufficient for `uv pip install --no-deps
  -e .` to succeed as the dev user.

Refs #539

* fix(devcontainer): scope chown, guard credential copy, --no-install-recommends

Addresses Copilot PR review feedback on #540:

- scope /venv/main chown to bin/ and site-packages/ instead of recursive (comment 3070307915) -
  avoids copy-up of the entire prebuilt ~2.5GB venv layer into a new image layer - guard
  /root/.netrc and /root/.config/rclone copies with existence checks (comment 3070307921) -
  base-image builds without the wandb_api_key BuildKit secret previously broke the devcontainer
  build - apt-get install sudo with --no-install-recommends to keep the layer small (comment
  3070307924)

* fix(devcontainer): pin base image to linux/amd64 for arm64 builders

tinaudio/perm:dev-snapshot is published only as linux/amd64. With the PR's switch from 'image:' to
  'build:', BuildKit resolves the FROM step against the builder's target platform and fails on Apple
  Silicon with "no match for platform in manifest". The old 'image:' path silently pulled amd64 and
  ran it under Rosetta; pinning --platform=linux/amd64 preserves that behavior explicitly.

Multi-arch publishing of the base image is the long-term fix and will be tracked separately; this
  unblocks arm64 devs on PR #540 today.

* fix(devcontainer): pin build target platform via devcontainer.json build.options

The FROM --platform=linux/amd64 pin added in ab252af only controlled which base-image manifest
  variant gets pulled — it does not change the build target platform. On
  cloud-tinaudio-tinaudio-builder (a multi-node Build Cloud builder with separate linux-amd64 and
  linux-arm64 workers), buildx still scheduled RUN steps on the arm64 node per the host's default
  target, and the RUN failed with "exec /bin/bash: exec format error" because the pulled amd64
  binaries cannot execute on an arm64 worker with no emulation.

Pass --platform=linux/amd64 to docker buildx build itself via devcontainer.json's build.options
  field. That sets the build target platform, so BuildKit routes the whole build (FROM pull + RUN
  steps) to an amd64-capable worker: the native linux-amd64 node on Build Cloud (no emulation, full
  speed) or Docker Desktop's local builder (amd64 RUN steps via Rosetta-for-Linux at the daemon
  level).

The FROM --platform pin is now redundant and triggered the FromPlatformFlagConstDisallowed BuildKit
  lint warning, so drop it. devcontainer.json's build.options is the single source of truth for
  platform routing, with a short comment in the Dockerfile pointing there.

* fix(devcontainer): set dev user login shell to /bin/bash

useradd defaults to /bin/sh (dash on Ubuntu) when no --shell is specified, which would give VS
  Code's integrated terminal a dash shell instead of bash — a hidden dev-UX regression vs. the
  pre-PR root user (which has /bin/bash on the base image). Automation pipelines are unaffected
  because post-create.sh, pre-commit hooks, and Makefile recipes all specify their own interpreter;
  this is strictly about the interactive terminal experience for devs using the rebuilt container.

Addresses Copilot review comment 3075047386.

* fix(devcontainer): mount workspace at baked-install path, drop chown and sudo

The base image already does 'uv pip install --no-deps -e .' from /home/build/synth-setter at bake
  time (docker/ubuntu22_04/Dockerfile:422), writing a .pth file that points at that path. The
  previous approach (cd951fc, 053b43c) mounted the workspace at /workspaces/{basename} and then
  rewrote the .pth file via a second editable install in post-create.sh — which required chowning
  /venv/main/bin and site-packages to dev so the rewrite could succeed.

Mount the host workspace at /home/build/synth-setter directly instead. The existing baked .pth is
  already correct — zero rewrites, no chown, no PYTHONPATH hack, no re-install. CI and production
  (which run the image without a mount) keep working against the baked clone; the devcontainer
  shadows the baked clone with the live workspace at the same path so Python imports resolve to the
  workspace without any consumer-side machinery.

Also drop sudo entirely. Passwordless sudo is security theatre (dev becomes root trivially), and the
  ergonomic escape hatch for missing system packages is adding them to the base image and
  rebuilding, not granting unrestricted root to the dev user.

Net Dockerfile change: 40 lines to 26 lines. Removes the scoped chown (053b43c fix #1), removes sudo
  install (053b43c fix #3), removes the FROM --platform pin comment block (c69f30c was already moved
  to devcontainer.json build.options). Keeps the credential copy with a TODO for eventual removal
  when the base image bakes credentials into /home/dev directly.

### Documentation

- Readme + getting-started install overhaul, add make link-plugins
  ([#613](https://github.com/tinaudio/synth-setter/pull/613),
  [`0fd4b03`](https://github.com/tinaudio/synth-setter/commit/0fd4b03325e6d280f8730f7974a6fb7927089cc1))

* docs(documentation): README + getting-started install overhaul, add make link-plugins

Rewrite README install section to promote `make install` as the canonical path, frame uv/pip/conda
  as interchangeable alternatives, and deduplicate against docs/getting-started.md §2.

Add `make link-plugins` target that detects Linux/macOS and symlinks the installed Surge XT VST3
  into plugins/.

Move Codespaces and Docker content to the bottom of the README. Add env-var export section and
  devcontainer-as-root note.

Closes #601 Closes #487

* docs: address review feedback on install overhaul

- README.md pip/conda note: keep each inline code span on a single line - README.md
  devcontainer-as-root note: point to the real config paths
  (.devcontainer/{cpu,gpu}/devcontainer.json); there is no .devcontainer/devcontainer.json -
  docs/getting-started.md plain-pip alternative: also run `pip install -e .` so the project itself
  is installed, matching `make install` - Makefile link-plugins: explicit destination handling —
  replace symlinks/files, error out if destination is a real directory (previously `ln -sfn` could
  create the symlink inside an existing real directory)

* feat: end-to-end make install (uv + Python 3.10 venv + deps + pre-commit)

- make install now installs uv (if missing), creates .venv/ with uv-managed Python 3.10 and --prompt
  synth-setter, installs requirements plus the project in editable mode, and registers pre-commit
  hooks (skipped when core.hooksPath is set, e.g. in the dev container). Errors out if .venv/ exists
  with a different Python version so the contract stays predictable. - Drop uv.lock: we are
  committing to `uv pip` rather than `uv sync`. uv's pyproject-driven resolution has known edge
  cases around torch indexes, transitive-dep resolution, and CPU/CUDA backend wheels that make sync
  unsuitable for this project today. The lock file would only drift. - README install flow shrinks
  from 6 steps to 5: users no longer need to install uv or create the venv manually. make install
  handles both, including fetching a Python 3.10 interpreter via uv if the user does not have one. -
  Prerequisites no longer list Python as a hard requirement — only git, curl, make, and the platform
  deps. - getting-started §2 rewritten to the uv-first canonical flow; the pip/conda/plain-venv
  walkthrough moves to a new Appendix A.

* feat: add make install-surge-xt, restructure getting-started appendices

- New target: make install-surge-xt downloads the pinned Surge XT 1.3.4 "pluginsonly" archive from
  GitHub releases, verifies md5 against the upstream checksum, caches at
  ~/.cache/synth-setter/surge-xt-1.3.4/, and extracts Surge XT.vst3 into plugins/. Skip-if-exists
  for idempotency. Linux x86_64 + macOS universal; arm64 Linux errors with a pointer to system
  install + link-plugins. - link-plugins becomes the fallback for users with a system-wide Surge XT,
  not the primary path. README + getting-started §2d updated to match. - getting-started §2d
  rewritten around install-surge-xt. §4a shrinks to a pointer at §2d (no more duplicate install
  walkthrough). - §2g (Codespaces) and §2h (Dev Container) move out of §2 into a new Appendix B:
  Container-based setup at the end of the doc. §2 now has a short pointer to Appendix B. Keeps §2
  the canonical local-setup flow and puts specialized container paths in one place alongside
  Appendix A (Manual environment setup). - README: drop Surge XT from prerequisites
  (install-surge-xt handles it); swap link-plugins for install-surge-xt in the 5-step flow; add a
  "already have Surge XT installed system-wide?" note pointing at link-plugins; update the
  Codespaces & Docker section link target to Appendix B.

* refactor: drop make link-plugins, harden install recipes, doc fixes

Addresses PR #613 Copilot review round 2.

Makefile: - Remove make link-plugins target entirely. `make install-surge-xt` covers the download
  path; users who already have a system-wide Surge XT install can run `ln -s "/path/to/Surge
  XT.vst3" "plugins/"` as a one-liner. Keeping two make paths for the same `plugins/` population was
  marginal value and accumulated review nits (missing search locations, shell-safety holes around
  `ln; echo`). - Prepend `set -e` to the install and install-surge-xt recipes so a failure in uv,
  pip, tar, or unzip does not get masked by a trailing `echo` (comments #3, #10). - Add an explicit
  elif/else for md5 detection: if neither `md5sum` nor `md5` is available, fail with a clear error
  instead of silent empty comparison (comment #7).

Docs: - §1 prerequisites: soften "all ship with macOS/Linux" — make/curl/git are standard on
  developer machines but missing on minimal/server images (comment #5). - §2b: note that pre-commit
  install is skipped when `core.hooksPath` is set (dev container case), with instructions for manual
  override (comments #1, #2). - §2d: drop the make link-plugins mention; replace with a one-line
  manual `ln -s` example for users with a system install. Add a "heads-up" call-out that
  `tests/data/vst/test_preset_params.py` and `tests/docker/test_smoke.py` still hardcode
  `/usr/lib/vst3/Surge XT.vst3`, so `pytest -m requires_vst` skips on macOS even with plugins/
  populated. Tracked in #631 (comment #6 + follow-up issue). - §4a: same link-plugins wording
  cleanup. - README: inline pre-commit skip caveat in step 4 comment; rewrite the "already have
  Surge XT system-wide" blockquote as a manual symlink one-liner. doc-map.yaml: drop link-plugins
  from the Makefile covers string.

---------

Co-authored-by: Managed via Tart <admin@Manageds-Virtual-Machine.local>

- Reflect devcontainer-tools stage, bash-history volume, post-create privilege drop
  ([#585](https://github.com/tinaudio/synth-setter/pull/585),
  [`a8cc3ed`](https://github.com/tinaudio/synth-setter/commit/a8cc3edcc1fce30bda560e22f6d1d9b7965eba4f))

* docs: reflect devcontainer-tools stage, bash-history volume, post-create privilege drop

PR #581 added a second consumable Dockerfile target (devcontainer-tools), new Makefile/CI build
  paths, bash-history persistence under /commandhistory, and a root-to-dev privilege drop in
  .devcontainer/post-create.sh. Bring the reference docs in line.

- docs/reference/docker.md: new target + tag rows, build-target example -
  docs/reference/docker-spec.md: two-target table with devcontainer-tools - docs/getting-started.md:
  accurate post-create.sh description (submodule init and workspace-editable install claims were
  already stale pre-#581) - docs/doc-map.yaml: cover .devcontainer/** under getting-started.md and
  .devcontainer/Dockerfile under docker.md

Refs #580

* fix(docs): remove accidental markdown list in docker-spec.md paragraph

mdformat parsed the line-starting '+' as a list bullet and rewrote the paragraph into a detached
  list item. Rewrite as a comma-separated inline list to restore paragraph flow.

Refs #584

* fix(docs): normalize docker-spec.md target table column widths for mdformat

mdformat trims trailing whitespace in table cells to the minimum width needed by the longest cell.
  Match that convention.

* docs: cover post-merge devcontainer drift (initialize.sh, plugins overlay, .env, submodule
  cleanup)

Four post-#581 PRs landed while #585 was open and introduced new doc drift in the same files this PR
  already touches. Fold those fixes in rather than open a follow-up:

- doc-map.yaml: add `.devcontainer/initialize.sh` (added in #594) so the worktree-hardstop file is
  auto-tracked for future drift. Update both devcontainer.json `covers:` strings to mention the new
  initializeCommand, --env-file .env, and the plugins/ anonymous overlay (#592). -
  getting-started.md §2g Prerequisites: rewrite the .env paragraph. The prior text claimed configs
  do not auto-load .env, but runArgs: ["--env-file", ".env"] has been in the configs since well
  before #581. Distinguish local (auto via --env-file) from Codespaces (forward via secrets)
  explicitly. - getting-started.md §2g Caveats: add a caveat for the plugins/ anonymous overlay
  added in #592 — without this note, a user whose host plugins/ is gitignored is surprised that
  Surge XT.vst3 still appears inside the container, and may try to drop their own plugin in plugins/
  on the host. - getting-started.md §2g: drop the stale `git submodule update` / `tinaudio/skills`
  caveat — the skills submodule was migrated to a plugin marketplace in #546, so the caveat no
  longer applies. Adjacent paragraph reworded to drop the "submodule" half of "submodule/hook
  operations". - docker.md §2 devcontainer-tools prose: name the plugins/ anonymous overlay
  alongside the /commandhistory bash-history volume, since the prose already crosses the
  docker/devcontainer boundary by enumerating one of the two mounts.

* fix(docs): correct devcontainer-tools stage parent (dev-base, not dev-snapshot)

Copilot review on PR #585 caught that two passages described `devcontainer-tools` as extending
  `dev-snapshot`, but the Dockerfile defines them as siblings — both `FROM dev-base AS <stage>`
  (docker/ubuntu22_04/Dockerfile:374 and :424). dev-base is the shared parent that holds Surge XT,
  the venv, and the synth-setter source.

Fix: - Inline comment in the make example: `dev-base + ...` instead of `dev-snapshot + ...`. -
  Surrounding prose: spell out the sibling relationship explicitly so a reader of just docker.md
  isn't surprised by the Dockerfile graph.

docs/reference/docker-spec.md already had the correct wording ("extends `dev-base`" at §2 line 70),
  so no change there.

- Update readme ([#562](https://github.com/tinaudio/synth-setter/pull/562),
  [`d8b5eff`](https://github.com/tinaudio/synth-setter/commit/d8b5eff9021bdf8a0d5d35cfe06941722f664047))

* docs: update readme

Updated acknowledgments and clarify project status.

* Update README.md

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

* Apply suggestions from code review

---------

- **documentation**: Update README for public
  ([#556](https://github.com/tinaudio/synth-setter/pull/556),
  [`8c81560`](https://github.com/tinaudio/synth-setter/commit/8c8156083a2b16c03b8ee1a716ac8ad40b905f04))

* docs(documentation): prepare README for public

Add a Status callout flagging the project as early-stage WIP, move the Ben Hayes acknowledgment up
  near the overview and link the companion ben-hayes/synth-permutations repo, add a Project Tracking
  section with real links to the project board, MVP epic, active epics, and key milestones, and
  replace the placeholder License section with a proper GPL-3.0 statement and badge.

Refs #555

* docs(documentation): curate README Documentation section with skim list

Replace the stale "coming soon" placeholders with a curated four-item skim list: getting-started,
  architecture, glossary, and the data pipeline design doc. Add pointers to docs/design/ and
  docs/reference/ for further reading.

* Revise README for acknowledgments and features

Updated acknowledgments and features sections for clarity and accuracy.

* docs(documentation): tighten README grammar, soften overclaims, add Surge XT requirement

Grammar fixes: add missing period in "et al.", replace ASCII double hyphens with em-dashes for
  consistency, drop redundant "no Windows" clause in Prerequisites.

Soften overclaims: "SOTA prior work" → "recent prior work"; "multi cloud via skypilot" → "with cloud
  support" (keeps the README from going stale mid-week while compute backend work lands); "Flow
  matching models" → "Flow matching and baseline models" to reflect VAE+RealNVP, DiT/AST, residual
  MLP, and CNN variants that also ship.

Add Surge XT 1.3.4 to Prerequisites with links to the upstream repo and the official downloads page.

- **pipeline**: Skypilot compute integration design doc
  ([#537](https://github.com/tinaudio/synth-setter/pull/537),
  [`8aeec04`](https://github.com/tinaudio/synth-setter/commit/8aeec042048b9a92a175d0e642433c3883bc63db))

* docs(pipeline): add SkyPilot compute integration design doc

Proposes replacing the planned ComputeBackend protocol and RunPodBackend with SkyPilot managed jobs
  for multi-provider GPU provisioning. Covers schema changes (compute_config field across
  DatasetConfig, train, eval), worker identity via UUID, and SkyPilot task YAML configs.

Refs #534

* docs(pipeline): address review feedback on SkyPilot design doc

- normalize header metadata to match data-pipeline.md (Status/Author/Last Updated/Tracking) - fix
  image_config.py path reference (pipeline/schemas/, not CI/) - use real CI image name tinaudio/perm
  with git-sha placeholder instead of synth-setter:latest - fix loop variable name mismatch
  (shard_batch) and clarify dict-vs-path construction

* Update SkyPilot compute integration design document

Removed mention of Lambda as an alternative provider and adjusted the multi-provider flexibility
  point.

- **readme**: Declare Windows unsupported
  ([#547](https://github.com/tinaudio/synth-setter/pull/547),
  [`84b3f74`](https://github.com/tinaudio/synth-setter/commit/84b3f74f6cb74cc9340d25c45c62d404ffe9f308))

* docs(readme): declare Windows unsupported

The sh test dependency and VST rendering tooling are POSIX-only, and CI only covers ubuntu-latest
  and macos-latest. Document this explicitly under Prerequisites so contributors don't try to set up
  the project on Windows expecting it to work.

Closes #32

* docs(getting-started): remove Windows install instructions

PR follow-up to the README Supported Platforms statement: getting-started.md still told Windows
  users to use WSL/.venv\Scripts\activate, contradicting the README. Replace the WSL note with an
  explicit "Linux or macOS only" prerequisite that points back to the README.

* Update README.md

Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>

---------

### Testing

- **datamodules**: Skip flaky test_mnist_datamodule
  ([#568](https://github.com/tinaudio/synth-setter/pull/568),
  [`fff19d6`](https://github.com/tinaudio/synth-setter/commit/fff19d62b62221d4228a6d6266d8c31a238ef4e1))

Public MNIST mirror download is unreliable; `make test` fails locally with `RuntimeError: Error
  downloading train-images-idx3-ubyte.gz`. Skip the test at the function level (covers both
  parametrizations) until the un-skip tracked by #243 lands.

Refs #243

- **testing**: Wire sweep tests to GPU runner; skip mnist-dependent tests
  ([#513](https://github.com/tinaudio/synth-setter/pull/513),
  [`b45973f`](https://github.com/tinaudio/synth-setter/commit/b45973f5787fc7b3c46cd34c0105ae7cdf979709))

* test(testing): wire sweep tests to GPU runner; skip mnist-dependent tests

test_sweeps.py had 5 tests that have never executed in any CI workflow: they were gated by
  @RunIf(sh=True) while `sh` was not in requirements, and while `test-expensive.yml` does install
  `sh` ad-hoc, it selects tests via `-m gpu` — and sweep tests had no @pytest.mark.gpu.

Furthermore, the tests shell out to `src/train.py` which inherits `trainer: gpu` via the default
  Hydra stack, so they require a GPU.

Changes: - Add `sh` to requirements-app.txt (no longer an ad-hoc install). - Replace @RunIf(sh=True)
  with @pytest.mark.gpu + @RunIf(min_gpus=1) on all 5 sweep tests — so they run on the twice-weekly
  GPU runner and skip on the nightly CPU runner. - Skip 3 of the 5 tests with a TODO(#514) pointing
  to the follow-up issue that tracks migrating them from Lightning-Hydra-Template mnist configs to
  ksin: * test_experiments — globs configs/experiment/*.yaml, pulling in example.yaml which
  overrides model=mnist. configs/model/mnist.yaml does not exist. * test_optuna_sweep +
  test_optuna_sweep_ddp_sim_wandb — use hparams_search=mnist_optuna, which sweeps
  model.net.lin{1,2,3}_size — fields only defined on SimpleDenseNet, which is not referenced by any
  active model config.

The tests themselves are legitimate coverage (experiment-compose smoke test, Optuna sweeper
  integration, Optuna+ddp_sim+wandb integration) — they just need their configs migrated.

Kept running: test_hydra_sweep and test_hydra_sweep_ddp_sim — these use the default train.yaml stack
  (data=ksin, model=ffn) and are valid.

Refs #510

* test(testing): workaround #517 — delete callbacks.lr_monitor in sweep tests

The sweep tests shell out to src/train.py with logger=[]. The default callbacks stack includes
  LearningRateMonitor, which raises MisconfigurationException at on_train_start when the trainer has
  no logger. This surfaced on the GPU runner in the test-expensive.yml run triggered against PR
  #513.

Add ~callbacks.lr_monitor to the subprocess overrides as a workaround. The fixture-based in-process
  tests avoid this via `del callbacks.lr_monitor` in conftest.py; the subprocess tests don't use the
  fixture.

Refs #517 — root-cause fix (make LearningRateMonitor a no-op without a logger) is tracked
  separately.

* no-op comment change to trigger ci

Corrected the comment about the missing mnist.yaml file.

---------

Co-authored-by: a <a@as-mac-mini.taile31224.ts.net>


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
