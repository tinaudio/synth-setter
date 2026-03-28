#!/usr/bin/env bash
# Docker entrypoint — dispatches on the MODE environment variable.
#
# MODE is required. The container exits with an error if MODE is unset,
# empty, or unrecognized. This prevents silent no-ops where a container
# starts, does nothing, and exits 0 without anyone noticing.
#
# Modes:
#
#   MODE=idle
#     Keeps the container alive indefinitely (exec sleep infinity).
#     Use this to attach a shell and poke around:
#       docker run -d -e MODE=idle <image>
#       docker exec -it <container> bash
#
#   MODE=passthrough
#     Runs the given command, or exits 0 if no command is provided.
#       docker run -e MODE=passthrough <image> python train.py --lr 0.01
#       docker run -e MODE=passthrough <image>   # no-op, exits 0
#
#   MODE=generate_dataset
#     Generates a VST dataset shard via generate_vst_dataset.py under headless X11.
#     Reads config from DATASET_CONFIG env var (required, path to YAML).
#     The container materializes a DataPipelineSpec, uploads spec and shard to R2.
#     spec.json is written to RUN_METADATA_DIR.
#     Optional: RUN_METADATA_DIR (default: /run-metadata).
#       docker run -e MODE=generate_dataset \
#         -e DATASET_CONFIG=configs/dataset/surge-simple-480k-10k.yaml \
#         -e RUN_METADATA_DIR=/run-metadata \
#         -v /tmp/run-metadata:/run-metadata <image>
#
# Examples:
#
#   # Debug a container interactively
#   docker run -d --name debug -e MODE=idle myimage:latest
#   docker exec -it debug bash
#   docker stop debug
#
#   # Run a one-off command through the entrypoint
#   docker run --rm -e MODE=passthrough myimage:latest python -c "import torch; print(torch.cuda.is_available())"
#
#   # CI smoke test — just check the container starts
#   docker run --rm -e MODE=passthrough myimage:latest
#
#   # Forgot to set MODE — fails fast with a helpful error
#   docker run --rm myimage:latest
#   # => Error: MODE is required. Set MODE=idle, MODE=passthrough, or MODE=generate_dataset.
#
# See also:
#   docs/reference/docker-spec.md — full spec for modes, image targets, env vars
#   Dockerfile: docker/ubuntu22_04/Dockerfile — ENTRYPOINT wiring

set -euo pipefail

mode="${MODE:-}"

case "${mode}" in
  idle)
    echo "Idle mode — sleeping indefinitely. Attach with: docker exec -it <container> bash"
    exec sleep infinity
    ;;
  passthrough)
    if [ "$#" -gt 0 ]; then
      exec "$@"
    fi
    echo "Passthrough mode — no command provided, exiting cleanly."
    exit 0
    ;;
  generate_dataset)
    : "${DATASET_CONFIG:?DATASET_CONFIG is required (path to dataset config YAML)}"
    exec scripts/run-linux-vst-headless.sh \
        python scripts/entrypoint_generate_dataset.py
    ;;
  "")
    echo "Error: MODE is required. Set MODE=idle, MODE=passthrough, or MODE=generate_dataset." >&2
    echo "Available modes: idle, passthrough, generate_dataset" >&2
    exit 1
    ;;
  *)
    echo "Error: unknown MODE '${mode}'." >&2
    echo "Available modes: idle, passthrough, generate_dataset" >&2
    exit 1
    ;;
esac
