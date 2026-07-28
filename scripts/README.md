# Scripts

This directory holds **shell / Python tooling that ships outside the `synth_setter` wheel** — utilities the test suite and CI workflows shell out to, plus operator-side commands. After the [#784](https://github.com/tinaudio/synth-setter/issues/784) layout migration, every resident lives under a categorized subdirectory **except `sync_worker_checkout.sh`**, which intentionally stays at `scripts/sync_worker_checkout.sh` — see the "Bake-lag exception" section below.

## Layout

| Subdir / file                     | Purpose                                                                                                                                                                                                                                                                           |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/skypilot/`               | SkyPilot bootstrap and diagnostics                                                                                                                                                                                                                                                |
| `scripts/ci/`                     | Local CI tooling                                                                                                                                                                                                                                                                  |
| `scripts/studiorack/`             | Post-install compatibility patches for Linux VST3 bundles and unprivileged archive installs; remove after upstream [#82](https://github.com/open-audio-stack/open-audio-stack-core/issues/82) and [#83](https://github.com/open-audio-stack/open-audio-stack-core/issues/83) ship |
| `scripts/sync_worker_checkout.sh` | Bake-lag bootstrap invoked inside worker containers before source sync                                                                                                                                                                                                            |

## Bake-lag exception: `scripts/sync_worker_checkout.sh`

`sync_worker_checkout.sh` is the bootstrap that updates the worker container's baked checkout to the PR head, so SkyPilot workers pick up entrypoint changes from a PR before `main`'s next image rebuild. Because the worker's `cd /home/build/synth-setter && bash scripts/sync_worker_checkout.sh` runs **against the previously baked image's filesystem** (i.e. main as of the last image build), the script must live at a path that the baked image already knows. Moving it to `scripts/skypilot/sync_worker_checkout.sh` in this PR would mean the next baked-image-and-after-it-is-the-PR run can't find it, defeating the bake-lag bypass. So it stays at the repo root level of `scripts/`. Once it has lived at the new path for at least one image rebuild cycle, a follow-up PR can relocate it under `scripts/skypilot/`.

## Python tools

The Python utilities live inside the `synth_setter` package and are invoked as `python -m synth_setter.<subpkg>.<module>`:

| Subpackage                   | Modules                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `synth_setter.evaluation`    | `predict_vst_audio`, `compute_audio_metrics`                                  |
| `synth_setter.tools`         | `vst_interactive`, `model_from_wandb`, `plot_param2tok`, `paramspec_to_table` |
| `synth_setter.pipeline.data` | `stats`, `add_music2latent`, `add_embeddings`                                 |
| `synth_setter.scripts`       | `load_vst3_check`                                                             |

Console scripts declared in `pyproject.toml` are the canonical entrypoints.
`synth-setter-plugins` installs, resolves, and links packages pinned in
`studiorack.json`; train, eval, and generation use their existing
`synth-setter-*` commands.

## Shell helpers

Container-runtime shell helpers (X11 / VST3 bootstrap):

| Helper                      | Location                                             |
| --------------------------- | ---------------------------------------------------- |
| `run-linux-vst-headless.sh` | `src/synth_setter/scripts/run-linux-vst-headless.sh` |
| `ensure_plugin_symlinks.sh` | `docker/ubuntu22_04/ensure_plugin_symlinks.sh`       |

`run-linux-vst-headless.sh` ships inside the `synth_setter` package and is
discovered via `synth_setter.resources.vst_headless_wrapper()`. The
`ensure_plugin_symlinks.sh` helper restores the manifest-pinned Surge alias
after a container workspace bind mount shadows `plugins/`.

## See also

- [`CLAUDE.md`](../CLAUDE.md) — repo layout + commit conventions.
- [`docs/architecture.md`](../docs/architecture.md) — package layout overview.
- [#784](https://github.com/tinaudio/synth-setter/issues/784) — the layout-migration epic that put these files in their current homes.
