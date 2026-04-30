#!/bin/bash
# Image ENTRYPOINT: dispatch the first arg to either the click CLI or exec it
# directly. This lets the same image serve two callers cleanly:
#
#   docker run img generate_dataset --spec ...   → routed through the click CLI
#   docker run img passthrough <cmd>             → click passthrough subcommand
#   docker run img bash -c '...'                 → exec'd directly (used by
#                                                   SkyPilot's RunPod backend,
#                                                   which sets docker_args to
#                                                   `bash -c '<base64-setup>'`)
#
# Before this wrapper, ENTRYPOINT was ["python", "/usr/local/bin/entrypoint.py"]
# and `bash` (not a click subcommand) crashed the container with
#   Error: No such command 'bash'.
# before SkyPilot could SSH in to run setup commands.
set -euo pipefail

# Known click subcommands defined in scripts/docker_entrypoint.py.
case "${1:-}" in
  idle|passthrough|generate_dataset|render_eval|train|--help|-h|"")
    exec python /usr/local/bin/entrypoint.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
