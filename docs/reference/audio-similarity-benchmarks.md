# Audio similarity benchmarks

Reference for the per-run audio-similarity metrics published by
[`.github/workflows/test-vst-slow.yml`](../../.github/workflows/test-vst-slow.yml)
to the benchmark chart at
**<https://tinaudio.github.io/synth-setter/dev/bench/>**.

`gh-pages` is now a **data store**, not the served branch: the
`benchmark-action` keeps appending entries to `gh-pages/dev/bench/data.js`
as before, but `gh-pages` itself is no longer rendered by Pages. The
[`docs`](../../.github/workflows/docs.yml) workflow (Phase 2 of the docs
pipeline) reads `gh-pages/dev/bench/`, merges it into the deployed
mkdocs site under `/dev/bench/`, and publishes via
`actions/deploy-pages@v4`. The chart URL is unchanged.

## Purpose

The slow VST tests in
[`tests/data/vst/test_generate_vst_dataset.py`](../../tests/data/vst/test_generate_vst_dataset.py)
exercise `make_dataset` round-trips and assert that two independent renders
of the same parameters land within phase-robust tolerances. Per-run
metric values typically come in well below the assertion thresholds, so
the assertions only catch _gross_ regressions (silence, wrong patch).

The benchmark dashboards plot the per-run values as a time series so
subtler drift becomes visible long before any threshold trips. Examples
of the kind of regressions the chart is designed to surface:

- **Surge XT version bump** in the `dev-snapshot` Docker image (the `.deb`
  install in `docker/ubuntu22_04/Dockerfile`) that quietly shifts the
  metric distribution.
- **`librosa` / `pedalboard` upgrade** changing mel-spectrogram
  computation or VST host behavior, even with identical params.
- **Regression in the renderer's determinism** in
  `src/data/vst/core.py` § `render_params()` — bug
  [#489](https://github.com/tinaudio/synth-setter/issues/489) was the
  every-other-render variance, closed by
  [#713](https://github.com/tinaudio/synth-setter/pull/713) via per-render
  plugin reload; if that fix regresses, the variance spikes here first.
- **Renderer perf regressions** — dB metrics may stay flat while
  `wall-clock-seconds-per-render` doubles.

The hard caps in `tests/data/vst/test_generate_vst_dataset.py` are the
safety net; the chart is the early warning.

## Where to find the dashboards

| Dashboard                  | URL                                                                            |
| -------------------------- | ------------------------------------------------------------------------------ |
| Live charts (both buckets) | <https://tinaudio.github.io/synth-setter/dev/bench/>                           |
| Raw data file              | <https://github.com/tinaudio/synth-setter/blob/gh-pages/dev/bench/data.js>     |
| `gh-pages` branch tree     | <https://github.com/tinaudio/synth-setter/tree/gh-pages>                       |
| Workflow runs that publish | <https://github.com/tinaudio/synth-setter/actions/workflows/test-vst-slow.yml> |
| Tracking issue             | [#703](https://github.com/tinaudio/synth-setter/issues/703)                    |
| Original bug               | [#489](https://github.com/tinaudio/synth-setter/issues/489) (closed)           |
| Fix PR                     | [#713](https://github.com/tinaudio/synth-setter/pull/713)                      |

The chart's left-hand legend lets you toggle individual metric series on
and off; the dropdown at the top selects the dashboard ("bucket").

## Two dashboards

The workflow publishes two independent dashboards, each representing a
different question.

### `VST noise floor (1 preset N renders)`

Sourced from `test_datasets_from_hardcoded_params_are_identical`. Both
stages of `make_dataset` run with the same hardcoded
`_HARDCODED_*_PARAMS` patch via `_patched_sample`, so all
`2 × num_samples` renders use _identical_ inputs. The per-pair metrics
plus an all-pairs cross-comparison expose every-other-render variance —
this was the reproducer for [#489](https://github.com/tinaudio/synth-setter/issues/489)
and is now the regression guard against its fix in
[#713](https://github.com/tinaudio/synth-setter/pull/713).

Bucket name: `VST noise floor (1 preset N renders)`<br>
JSON file (in workflow run): `vst-noise-floor-1-preset-n-renders.json`<br>
Metric name prefix: `vst-noise-floor-1-preset-n-renders/`

This is the most _sensitive_ noise-floor view — every render should
match every other render perfectly, so any drift shows up quickly.

### `VST noise floor (random preset replay)`

Sourced from `test_datasets_from_sampled_params_are_identical`. Stage 1
samples `num_samples` random patches; Stage 2 replays them via
`_patched_sample`. Each row uses _different_ params, so the metrics
compare matched pairs (`expected[i]` vs `actual[i]`) — there's no
all-pairs cross-comparison here because cross-row pairs naturally
differ.

Bucket name: `VST noise floor (random preset replay)`<br>
JSON file (in workflow run): `vst-noise-floor-random-preset-replay.json`<br>
Metric name prefix: `vst-noise-floor-random-preset-replay/`

This view is _broader_ in patch coverage but noisier between runs (each
run picks a different random sample), so trend signal is lower per
point but covers more of the patch space than the hardcoded fixture.

## Metric series

Both buckets emit the per-row "round-trip" series (five distance metrics
plus the two non-distance sentinels `num-samples` and
`wall-clock-seconds-per-render`):

| Metric                                | Computed by                                                                 | Unit        | Smaller-is-better? |
| ------------------------------------- | --------------------------------------------------------------------------- | ----------- | ------------------ |
| `multi-scale-spectral-loss-max`       | `compute_mss` (`scripts/compute_audio_metrics.py`) — multi-scale log-mel L1 | dB          | yes                |
| `dtw-aligned-mfcc-distance-max`       | `compute_wmfcc` — DTW-aligned MFCC L1 distance                              | L1          | yes                |
| `spectral-optimal-transport-max`      | `compute_sot` — Wasserstein on STFT magnitudes                              | Wasserstein | yes                |
| `rms-envelope-cosine-distance-max`    | `1 - compute_rms` — RMS envelope cosine distance                            | 1-cos       | yes                |
| `mel-spectrogram-mean-absolute-error` | mean abs diff on stored mel arrays                                          | dB          | yes                |
| `num-samples`                         | static fixture size (input parameter)                                       | count       | n/a (sentinel)     |
| `wall-clock-seconds-per-render`       | `(stage1_t + stage2_t) / (2 × num_samples)`                                 | seconds     | yes                |

The **`1 preset N renders`** bucket additionally emits five `all-pairs-*`
series — these are the **fix-regression signal for the #489
every-other-render bug**, since the per-row metrics can stay flat
while the all-pairs worst-case spikes (the bug manifested as junk on
every-other render, not on every render):

| Metric                                       | Computed by                                   | Unit        |
| -------------------------------------------- | --------------------------------------------- | ----------- |
| `all-pairs-multi-scale-spectral-loss-max`    | worst-case `compute_mss` across all pairs     | dB          |
| `all-pairs-dtw-aligned-mfcc-distance-max`    | worst-case `compute_wmfcc` across all pairs   | L1          |
| `all-pairs-spectral-optimal-transport-max`   | worst-case `compute_sot` across all pairs     | Wasserstein |
| `all-pairs-rms-envelope-cosine-distance-max` | worst-case `1 - compute_rms` across all pairs | 1-cos       |
| `all-pairs-pair-count`                       | `n × (n − 1) / 2` for `n = 2 × num_samples`   | count       |

Distance metrics are emitted as **max-over-samples** (worst-case
per-pair) for the round-trip series and **max-over-pairs** for the
all-pairs series, with `1 - min(rms_cos)` so all entries read
smaller-is-better (required by the `customSmallerIsBetter` schema).
`num-samples` and `all-pairs-pair-count` are static given the test's
fixture size; emitting them as series makes accidental fixture changes
visible alongside the metric drift they would silently cause.
`wall-clock-seconds-per-render` includes the loudness-loop retries on
Stage 1 of the random-replay test, so it's a real-throughput number
rather than a render-only one.

## Thresholds + alerting

The publish step has `alert-threshold: "150%"` and `fail-on-alert: false` — so an alert posts a comment on the offending commit if a
metric exceeds 1.5× the rolling baseline, but doesn't block CI. Once
the noise floor is well-characterized, flip `fail-on-alert: true` to
gate merges.

## Workflow wiring

```
tests/data/vst/test_generate_vst_dataset.py
   |
   |  _emit_benchmark_metrics(..., bench_filename="<prefix>.json")
   |  writes to $BENCHMARK_OUTPUT_DIR/<prefix>.json (per-test)
   v
docker run -v /tmp/bench:/bench -e BENCHMARK_OUTPUT_DIR=/bench ...
   |
   |  Surface step: cp /tmp/bench/<prefix>.json -> ${{ workspace }}/<prefix>.json
   v
benchmark-action/github-action-benchmark@v1   (one publish step per bucket)
   |
   |  fetch + commit + push gh-pages  (data store; not served directly)
   v
gh-pages branch  →  workflow_run trigger fires the `docs` workflow
                    (.github/workflows/docs.yml)
                    |
                    |  actions/checkout@v6 (ref: gh-pages, path: gh-pages-data)
                    |  cp -R gh-pages-data/dev/bench site/dev/bench
                    |  actions/configure-pages + upload-pages-artifact + deploy-pages
                    v
                    GitHub Pages  →  https://tinaudio.github.io/synth-setter/dev/bench/
```

## Operations

### Bootstrapping the chart on a new repo

`benchmark-action/github-action-benchmark@v1` can't create the
`gh-pages` branch on its own. Push an empty orphan branch first:

```bash
cd /tmp
git clone --depth 1 https://github.com/tinaudio/synth-setter.git ghpages-bootstrap
cd ghpages-bootstrap
git checkout --orphan gh-pages
git rm -rf .
echo "# Benchmark history" > README.md
git add README.md
git -c user.email="<your-noreply>@users.noreply.github.com" commit -m "Initial gh-pages bootstrap"
git push origin gh-pages
```

Then enable Pages: **Settings → Pages → Source = "GitHub Actions"**.
This is the single Pages source for the whole repo — the docs site
(mkdocs) and the benchmark chart are both served through
`actions/deploy-pages@v4` from the [`docs`](../../.github/workflows/docs.yml)
workflow. Do **not** set Source to "Deploy from a branch" pointing at
`gh-pages`: that mode is incompatible with `actions/deploy-pages@v4`
and would unpublish the docs site.

The chart will populate on the next `docs` workflow run after a
benchmark publish — either the `workflow_run` trigger fires
automatically when `test-vst-slow` completes on main, or a maintainer
can `gh workflow run Docs --ref main` to redeploy on demand.

### Publishing from a feature branch (pre-merge)

The workflow's `workflow_dispatch` accepts a `publish_metrics` boolean.
However, `gh workflow run --ref <feature-branch>` returns 404 because
the gh CLI looks up the workflow file on the default branch first, and
the standard PAT doesn't have permission for the REST `dispatches`
endpoint. So pre-merge bootstrapping uses a temporary `push:` trigger
on the feature branch + a relaxed publish-step `if:` condition; revert
both once the chart exists.

### Adding a new benchmark dashboard

To add a third bucket (e.g. `VST noise floor (sustained note)`):

1. Add a test that calls `_assert_round_trip_matches(..., benchmark_name_prefix="vst-noise-floor-sustained-note")`.
2. Add a Surface step entry that copies
   `vst-noise-floor-sustained-note.json` out of the volume.
3. Add a third publish step that reads that file with a matching
   `name:`.

The two existing dashboards share `benchmark-data-dir-path: dev/bench`
so they all write to the same `data.js`; the chart UI just adds a new
selectable bucket.

### Pruning old data

`window.BENCHMARK_DATA.entries` in
[`gh-pages/dev/bench/data.js`](https://github.com/tinaudio/synth-setter/blob/gh-pages/dev/bench/data.js)
is plain JS — surgical edits work. Three levels of cleanup:

- **Per-entry**: edit `data.js` to splice out specific runs (anomalous
  bootstrap data, runs from before a threshold change), commit,
  force-push to `gh-pages`.
- **Per-bucket reset**: replace `data.js` with a stub that has an empty
  `entries: {}` object, force-push.
- **Nuke**: delete the `gh-pages` branch and re-bootstrap.

The action also accepts `max-items-in-chart: <N>` for automatic pruning
once the chart exceeds N points — useful long-term.
