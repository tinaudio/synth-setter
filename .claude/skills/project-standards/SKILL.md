---
name: project-standards
description: Project-specific coding standards checklist for synth-permutations. Covers type safety, error handling, pipeline invariants, security, HDF5/numpy, and logging. Used by the /review skill.
---

# Project-Specific Standards Checklist

Use this checklist when reviewing code in synth-permutations. For each item, determine if it
passes or fails. Flag failures as BLOCK (must fix) or WARN (advisory).

## Type Safety

| #   | Check                                                                                                                                  | Severity |
| --- | -------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| P1  | Type annotations present on ALL function signatures                                                                                    | BLOCK    |
| P2  | No use of `Any` — use `Union`, `Optional`, or specific types                                                                           | BLOCK    |
| P3  | Pydantic `BaseModel` with `strict=True` at trust boundaries (config parsing, JSON from R2, worker reports crossing process boundaries) | BLOCK    |
| P4  | `dataclass` (not Pydantic) for internal typed containers where serialization validation is unnecessary                                 | WARN     |
| P5  | Frozen dataclasses (`frozen=True`) for immutable value objects                                                                         | WARN     |

## Error Handling

| #   | Check                                                                    | Severity |
| --- | ------------------------------------------------------------------------ | -------- |
| P6  | No bare `except:` — always catch specific exceptions                     | BLOCK    |
| P7  | No swallowed exceptions — at minimum log them                            | BLOCK    |
| P8  | R2/RunPod/rclone operations have explicit error handling                 | BLOCK    |
| P9  | External I/O operations use the centralized tenacity retry policy        | WARN     |
| P10 | Permanent failures (auth, wrong bucket) reraise immediately, not retried | WARN     |

## Pipeline Invariants

| #   | Check                                                                             | Severity |
| --- | --------------------------------------------------------------------------------- | -------- |
| P11 | Workers only write under `metadata/workers/`. Finalize only writes to `data/`     | BLOCK    |
| P12 | No writes to `data/shards/` outside finalize                                      | BLOCK    |
| P13 | All `rclone` operations use `--checksum`                                          | BLOCK    |
| P14 | Shard IDs are logical (`shard-000042`), deterministic, infrastructure-independent | BLOCK    |
| P15 | Specs are immutable after creation — no code modifies a frozen spec               | BLOCK    |
| P16 | `.valid` marker is written as the last step of shard lifecycle (commit point)     | BLOCK    |
| P17 | Worker reports and debug logs use unique `{worker_id}-{attempt_uuid}` filenames   | WARN     |
| P18 | Lifecycle markers are empty files — presence is the state, no content to parse    | WARN     |

## Security

| #   | Check                                                                                    | Severity |
| --- | ---------------------------------------------------------------------------------------- | -------- |
| P19 | No credential leaks — API keys, tokens not in code, logs, or error messages              | BLOCK    |
| P20 | No command injection via subprocess (user input not interpolated into shell commands)    | BLOCK    |
| P21 | No unsafe deserialization (pickle from untrusted sources)                                | BLOCK    |
| P22 | Secrets go through Docker BuildKit `--secret`, never baked into env vars or image layers | WARN     |

## HDF5 / Numpy

| #   | Check                                                                               | Severity |
| --- | ----------------------------------------------------------------------------------- | -------- |
| P23 | Array shapes match spec (sample rate, spectrogram bins, parameter count)            | BLOCK    |
| P24 | dtypes are correct (float32 where expected, not float64)                            | BLOCK    |
| P25 | Expected dataset keys present in HDF5 files (`audio`, `mel_spec`, `param_array`)    | BLOCK    |
| P26 | Value bounds checked (audio in [-1, 1], no NaN/Inf)                                 | WARN     |
| P27 | Array shape contracts documented in docstrings for functions accepting numpy arrays | WARN     |

## Logging

| #   | Check                                                         | Severity |
| --- | ------------------------------------------------------------- | -------- |
| P28 | `structlog` used for pipeline code (not `logging` or `print`) | WARN     |
| P29 | No `print()` statements in production code                    | BLOCK    |
| P30 | Debug logs are structured JSON (JSONL format for worker logs) | WARN     |
