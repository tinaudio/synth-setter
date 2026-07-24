---
name: e2e-test
description: >-
  Design or review a full, real, production-path end-to-end test without test
  doubles. Use aggressively when users ask for a real E2E test, production-path
  test, no mocks/no fakes, artifact-to-consumer round trip, or integration
  coverage using real services, models, checkpoints, binaries, formats, or
  external components. Also use for requests to prove a CLI, pipeline, model,
  artifact, or service works through its actual downstream consumer, even when
  the user only says “integration test.”
---

# e2e-test — Production-Path Testing

Use this skill when the claim is that the shipped system works as deployed, not
that one component can be invoked. Exercise the public production entrypoint
and keep every load-bearing boundary real. A test that completes after a
subprocess, model load, artifact write, or upload has not established the
production contract until its real downstream consumer has used the result.

## Non-negotiable boundary

Do not use test doubles anywhere in the requested production path. This
includes mocks, `monkeypatch`, fakes, stubs, spies, dummy models, synthetic
checkpoints, intercepted subprocesses, and test-only reimplementations. Do not
patch a real dependency into behaving as if it had run.

Small deterministic input fixtures are allowed only when they traverse the
real producer and consumer. They may reduce cost; they must not replace the
model, checkpoint, binary, service, storage backend, file format, or a
production stage. Use a real, loadable checkpoint produced by the real path,
not a hand-written checkpoint-shaped file.

This requirement is stricter than ordinary unit and integration testing.
Existing tests that use `FakeVST3Plugin`, a patched subprocess, or a local
surrogate remote are useful for their own contracts but are not templates for
this skill.

## Start with repository infrastructure

1. Search `tests/`, `tests/conftest.py`, `tests/integration/`, and the target
   entrypoint's tests before creating fixtures or helpers.
2. Reuse the narrowest existing **real** harness and fixtures. In particular,
   `tests/test_train.py::test_train_eval_surge_xt` demonstrates a real VST,
   real train-produced checkpoint, and real evaluation consumer; and
   `tests/integration/test_generate_dataset_from_spec_uri_r2.py` demonstrates
   the real spec-URI CLI, Surge VST, R2, and `rclone` round trip.
3. Invoke the public CLI, public Python entrypoint, or deployed command—not a
   private helper or a copied implementation. Preserve all real formats and
   external boundaries in that path.
4. If the user requests fixed test settings, write those exact, hard-coded
   settings in the test. Do not invent dynamic pytest configuration,
   parametrization, environment-driven alternatives, or a broader matrix.

Do not create a second harness when an existing one can be extended without
weakening the real path. If no existing harness supports the required real
component, add the smallest real fixture/configuration necessary and explain
its prerequisites.

## Resource handling and test tiers

- Use the repository's registered markers rather than creating a new one:
  `slow` for expensive on-demand runs, `requires_vst` for a real VST binary,
  and `integration_r2` plus `r2` for real R2/rclone access. Combine existing
  markers when the path requires both resources.
- Let absent local checkpoints, binaries, credentials, or dependencies skip
  clearly with the missing requirement and the command/configuration needed to
  supply it. Never silently fall back to a fake, mock, random-weight model, or
  synthetic artifact.
- Before claiming VST or R2 is unavailable, follow `AGENTS.md`: probe the VST
  path and run `rclone lsd r2:`. The devcontainer is expected to provide both.
- Keep fixtures small and deterministic, cleanup bounded external state, and
  mark expensive production runs so default fast lanes remain fast.

## Assertions that prove the contract

Assert more than successful return status or path existence. Validate every
applicable boundary using the real readers/consumers:

- required artifacts and schemas/columns/keys;
- expected shapes, dtypes, and row/sample preservation;
- finite numeric values and non-degenerate outputs (for example, non-silent
  audio, non-empty predictions, or non-zero work where the contract requires
  it);
- loadability by the real downstream consumer; and
- observable downstream output after that consumer runs.

For an artifact round trip, the minimum proof is:

```
real producer → persisted production artifact/format → real consumer → downstream assertion
```

For an ML path, create or obtain the real model/checkpoint through the intended
production mechanism, load it through the public inference/evaluation path, and
validate the produced artifacts. Do not stop after training, serialization, or
checkpoint download.

## Integration verification is not ML quality evaluation

A production-path E2E test verifies wiring and safety properties: a real model
loads, a pipeline preserves rows and schemas, values are finite and
non-degenerate, and consumers can use the artifacts. It should use deterministic
small inputs and stable structural/property assertions.

Statistical ML quality evaluation is separate work: evaluate a fixed released
model on a representative held-out dataset with defined metrics, uncertainty,
and acceptance thresholds. Do not turn an E2E regression test into a noisy
quality benchmark, and do not claim quality merely because the production path
completed.

## Workflow

1. State the promised end-to-end contract and identify its public entrypoint,
   real dependencies, artifact boundary, and real downstream consumer.
2. Locate and reuse a compatible real repository harness; reject candidates
   containing a prohibited test double on the requested path.
3. Pin the smallest deterministic real inputs and any user-requested hard-coded
   settings.
4. Add the appropriate existing resource/expense markers and a clear skip for
   genuinely absent prerequisites.
5. Drive the entire real path and assert the artifact plus downstream-consumer
   properties listed above.
6. Run the focused test through its normal marker-aware command. Report the
   exact real prerequisites and any clear skip; never substitute lower-fidelity
   coverage for an unavailable production path.

## Reusable request template

> Add a production-path E2E test for `<public entrypoint>`. Reuse the existing
> real `<fixture/harness>` and run `<real dependency/model/checkpoint/binary/service>`
> with these fixed settings: `<literal settings>`. Use no mocks, monkeypatches,
> fakes, stubs, spies, dummy models, synthetic checkpoints, intercepted
> subprocesses, or test-only reimplementations. Drive
> `<real producer>` → `<production artifact/format>` → `<real downstream consumer>`
> and assert `<schema/shape/dtype/row preservation/finite/non-degenerate/downstream result>`.
> Mark it `<existing markers>` and skip clearly only when `<real prerequisite>`
> is absent. This proves integration, not statistical model quality.
