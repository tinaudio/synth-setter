# GEMINI.md

synth-setter: synth inversion, sound matching, and preset-exploration tools — Python 3.10+, PyTorch Lightning, Hydra, with a distributed data pipeline on SkyPilot-managed compute (RunPod + OCI) stored in Cloudflare R2.

Shared agent instructions for Claude, Codex, and Gemini; AGENTS.md is the canonical source. Architecture: [docs/architecture.md](docs/architecture.md).

Read and follow [AGENTS.md](AGENTS.md). That file is the canonical project
instruction source for Claude, Codex, and Gemini. Keep Gemini-specific compatibility
notes in `.gemini/`; keep shared hooks and review skills under `agent/`.
