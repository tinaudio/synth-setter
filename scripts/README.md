# Scripts

This directory holds **shell / Python tooling that ships outside the `synth_setter` wheel** — utilities the test suite and CI workflows shell out to, plus operator-side commands. After the [#784](https://github.com/tinaudio/synth-setter/issues/784) layout migration, every resident here lives under a categorized subdirectory; the bare `scripts/` root contains no `.sh` or `.py` files of its own.

## Layout

| Subdir              | Purpose                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `scripts/skypilot/` | SkyPilot bootstrap / diagnostics (cred writers, worker checkout, cluster-state capture)                   |
| `scripts/ci/`       | Local CI tooling (triage agent launcher, pueue job queue CLI used by `.github/workflows/job-queue*.yaml`) |

## Where Python tools moved

The Python utilities previously rooted in `scripts/` now live inside the `synth_setter` package and are invoked as `python -m synth_setter.<subpkg>.<module>`:

| Subpackage                   | Modules                                                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `synth_setter.evaluation`    | `predict_vst_audio`, `compute_audio_metrics`                                                                        |
| `synth_setter.tools`         | `surge_xt_interactive`, `docker_entrypoint`, `model_from_wandb`, `plot_param2tok`, `paramspec_to_table`, `sig_perf` |
| `synth_setter.pipeline.data` | `reshard`, `rewrite_to_latest`, `stats`, `r2_report`, `add_music2latent`                                            |

The `synth-setter-train`, `synth-setter-eval`, and `synth-setter-generate-dataset` console scripts (declared in `pyproject.toml`'s `[project.scripts]`) remain the canonical entrypoints for the train / eval / dataset-generation workflows.

## Where shell helpers moved

Container-runtime shell helpers (X11 / VST3 bootstrap) moved next to the `Dockerfile` that `COPY`s them:

| Helper                      | New home                                       |
| --------------------------- | ---------------------------------------------- |
| `run-linux-vst-headless.sh` | `docker/ubuntu22_04/run-linux-vst-headless.sh` |
| `ensure_plugin_symlinks.sh` | `docker/ubuntu22_04/ensure_plugin_symlinks.sh` |

## See also

- [`CLAUDE.md`](../CLAUDE.md) — repo layout + commit conventions.
- [`docs/architecture.md`](../docs/architecture.md) — package layout overview.
- [#784](https://github.com/tinaudio/synth-setter/issues/784) — the layout-migration epic that put these files in their current homes.
