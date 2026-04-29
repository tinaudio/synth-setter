# Benchmark history

This branch is the publishing target for `benchmark-action/github-action-benchmark@v1`,
driven by `.github/workflows/test-vst-slow.yml`. It hosts the audio-similarity metric
trend chart at `https://tinaudio.github.io/synth-setter/dev/bench/`.

Don't edit by hand — the workflow appends entries to `dev/bench/data.js` on each
qualifying run.

Created during bootstrap to satisfy the action's `git switch gh-pages` precondition,
which has no fallback for missing branches.
