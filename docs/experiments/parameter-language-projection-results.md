# Parameter-language projection: local smoke study

## Verdict

All **20 runs completed** their requested optimizer budget, checkpoint reload,
and held-out evaluation. Training losses decreased in every run. The mechanical
contracts work; **a reliable semantic quality advantage is not established**.
Language variants reduced held-out losses in these two seeds, but SLAP retrieval
was unstable and flow audio metrics were mixed. Keep the feature opt-in.

Tracking: #3369. [Runner and commands](parameter-language-projection.md#research-ablations).
[Raw observations and artifact hashes](results/parameter-language-projection-smoke.json).

## Protocol

- **Immutable execution commit:** `8d425d896c0527761e5f3fe78ad9c3c26f1c239c`.
  Subsequent cache-hardening changes did not modify the running experiment worktree.
- **Dataset fingerprint:** `f2cf1cb81e8551a2acd89d594f4c727ad819baa5b1a514cd15d2c7a407c178ac`.
- **Real data:** Surge XT `surge_simple`, 32 training / 8 validation / 8 test
  examples, independently rendered with seeds 4100/4200/4300. Exact duplicate
  parameter rows across splits are rejected. Mel statistics use training only.
  The layout has 91 logical fields and 92 encoded coordinates, with no one-hot
  fields; the report does not invent a categorical accuracy for this layout.
- **Metadata:** real frozen EmbeddingGemma, revision
  `57c266a740f537b4dc058e1b0cda161fd15afa75`; native 768 and normalized 128-prefix
  vectors. The local dataset uses production Lance/statistics/artifact formats;
  it does not fabricate a pipeline completion marker. Separate real-CLI/R2 tests
  cover full finalization and publication.
- **Paired training:** seeds 101/102; batch 8; Adam at `3e-4`; width 32,
  one transformer layer, two heads; fp32 on an RTX 5060 Ti. No compilation,
  scheduler, optimal transport, or data-loader workers. All variants share
  initial numeric-head/backbone weights. Residual identity initialization and
  independently seeded controls avoid consuming the backbone RNG stream.
- **Budgets:** flow 200 optimizer steps; SLAP 1,000. Every run checks
  `trainer.global_step`; CSV histories independently contain exactly 200/1,000
  optimizer-step loss records. The last checkpoint is evaluated, not a
  validation-selected winner. No seed or treatment is discarded.
- **Held-out evaluation:** all eight test examples; flow additionally predicts
  and re-renders the same two held-out examples through the real VST. Test noise
  uses the paired seed plus 100,000. Restored language models are instantiated
  with a nonexistent metadata path before checkpoint testing.
- **Timing:** the matrix ran serially from 00:45:17 to 01:01:02 UTC on
  2026-09-10. Training-call wall time summed to 619.53 seconds. Individual timings
  include construction and final validation but exclude subsequent test/audio
  evaluation and offline text encoding. Concurrent CPU-side verification and
  warm-up effects make these observations unsuitable for speedup claims.

Pilots and the initial reporter failure are archived separately, outside the
final matrix. The reporter initially tried to average an empty one-hot field
list; a regression test now prevents undefined categorical summaries.

## Flow: 200 steps per run

Loss is the mean of each run's first/last ten logged training steps, then averaged
across seeds. Paired entries are seed 101 / seed 102. Other metrics are seed means;
lower is better. Audio means cover only two distinct held-out examples.

| Variant     | Train loss, first → last | Test parameter MSE, paired | Best-swap MSE | Audio MSS | Audio wMFCC | Audio SOT |
| ----------- | ------------------------ | -------------------------- | ------------- | --------- | ----------- | --------- |
| grouped     | 1.5360 → 1.1188          | 0.6546 / 0.6784            | 0.04581       | 19.871    | 15.866      | 0.4342    |
| learned     | 1.5348 → 1.1150          | 0.6504 / 0.6679            | 0.04403       | 19.939    | 15.840      | 0.4259    |
| random      | 1.5348 → 1.1152          | 0.6506 / 0.6684            | 0.04414       | 20.161    | 16.056      | 0.4306    |
| language128 | 1.5348 → 1.1148          | 0.6502 / 0.6656            | 0.04372       | 20.065    | 15.296      | 0.4410    |
| language768 | 1.5346 → 1.1145          | 0.6480 / 0.6666            | 0.04361       | 19.477    | 14.943      | 0.4363    |

Language-128 slightly improved parameter MSE relative to matched-capacity random
vectors in both seeds. Its mean SOT was worse, and its MSS changed in opposite
directions between seeds. Language-768 adds adapter parameters, so its comparison
with language-128 does not isolate semantic information from capacity. All flow
outputs remained unconstrained velocities; per-field prediction MSE is retained
in the raw observations.

## SLAP: 1,000 steps per run

Retrieval uses an eight-item gallery: chance R@1 is 0.125. Paired entries are seed
101 / seed 102; audio-to-parameter MRR is the seed mean. Loss aggregation matches
the flow table. Lower loss and higher retrieval scores are preferable.

| Variant     | Train loss, first → last | Test total loss, paired | Audio→parameter R@1, paired | Parameter→audio R@1, paired | Audio→parameter MRR |
| ----------- | ------------------------ | ----------------------- | --------------------------- | --------------------------- | ------------------- |
| grouped     | 1.7286 → 0.6233          | 0.9712 / 0.9630         | 0.1250 / 0.0000             | 0.1250 / 0.1250             | 0.3412              |
| learned     | 1.7290 → 0.6515          | 1.1113 / 1.0161         | 0.1250 / 0.0000             | 0.0000 / 0.2500             | 0.3121              |
| random      | 1.7290 → 0.6329          | 1.1290 / 1.0292         | 0.1250 / 0.0000             | 0.2500 / 0.1250             | 0.2955              |
| language128 | 1.7290 → 0.6170          | 0.8591 / 0.9323         | 0.2500 / 0.0000             | 0.1250 / 0.0000             | 0.3751              |
| language768 | 1.7287 → 0.6338          | 0.8390 / 0.8912         | 0.1250 / 0.0000             | 0.1250 / 0.0000             | 0.3193              |

Language variants had lower held-out total losses than the grouped baseline in
both seeds, but no consistent top-1 retrieval improvement. All treatments had
zero audio-to-parameter R@1 for seed 102. A two-seed, eight-query gallery cannot
support a claim of improved sound matching or a dimension recommendation based
on quality.

## Observed cost

Times and steps/s are means of two runs. Memory is the maximum observed CUDA
**allocated** training memory, not reserved memory or total device use. Parameter
counts exclude frozen parameters and metadata buffers; learned identity vectors
are included. These are different objectives/budgets, not a flow-versus-SLAP
throughput comparison.

| Consumer | Variant     | Trainable parameters | Train-call seconds | Steps/s | Peak allocated MiB |
| -------- | ----------- | -------------------- | ------------------ | ------- | ------------------ |
| flow     | grouped     | 73468                | 31.32              | 6.39    | 83.20              |
| flow     | learned     | 92380                | 29.46              | 6.79    | 83.51              |
| flow     | random      | 80732                | 28.38              | 7.05    | 83.38              |
| flow     | language128 | 80732                | 28.09              | 7.12    | 83.38              |
| flow     | language768 | 101212               | 28.42              | 7.05    | 83.91              |
| slap     | grouped     | 79936                | 33.90              | 29.76   | 83.17              |
| slap     | learned     | 98848                | 33.07              | 30.24   | 83.89              |
| slap     | random      | 87200                | 31.53              | 31.75   | 83.76              |
| slap     | language128 | 87200                | 33.75              | 29.63   | 83.76              |
| slap     | language768 | 107680               | 31.86              | 31.46   | 84.59              |

The opted-in width remains 128 for its smaller table/adapter, not because this
study proves it is the best width. Existing dataset and projection defaults are
unchanged. No paid compute was launched.

## What is and is not verified

Additional 300-step fixed-batch diagnostics (`15ae095cc5`) require a greater-than-90% reduction
in first-ten versus last-ten mean loss. Both pass: flow falls from 1.1272 to
0.0063; SLAP's selected cross-modal loss falls from 0.2932 to approximately zero.
Time/noise/dropout draws are fixed. For SLAP, EMA targets are frozen and the
supported cross-modal-only objective is selected, making stationary zero-loss
memorization feasible. This diagnostic does not replace the default mixed-BYOL
results above or establish retrieval quality.

The committed tests cover zero-residual equality with grouped projection,
field-local numeric gradients, semantic-gradient activation after optimization,
learned-vector updates, unrestricted flow outputs, decoder freezing, real CUDA
execution, and artifact-free checkpoint restoration. The matrix verifies finite
production training/evaluation and real rendered outcomes beyond a one-step smoke.

Final focused non-slow tests: 39 passed, 3 deselected. The curated fast suite:
1,450 passed, 6 skipped. A focused mutation run was attempted after fixing the
runner's sandbox import dependency, but stopped during baseline collection
(794 passed, one unrelated missing-preset failure tracked by #2567). No mutation
score is claimed.

This study does not establish convergence, broad synth generalization, perceptual
quality, confidence intervals, or reliable retrieval. It lacks a matched-capacity
random-768 control and shuffled-language treatment. Larger datasets, more seeds,
and stronger retrieval galleries would be needed before claiming a semantic
advantage; those experiments are not presented as completed work.

## Peer-reviewed component precedents

- He et al., [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), CVPR 2016: residual learning.
- Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html), NeurIPS 2021: feature tokenization and tabular transformers.
- Kusupati et al., [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html), NeurIPS 2022: nested embedding dimensions.

These motivate components; none establishes an improvement from this exact
language-conditioned residual projection. EmbeddingGemma's supported dimensions
and extraction contract come from its model card, not from those architecture
precedents.
