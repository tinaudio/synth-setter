#!/usr/bin/env bash
set -euo pipefail

mode="${MODE:-}"

case "${mode}" in
  idle)
    echo "Idle mode — sleeping indefinitely. Attach with: docker exec -it <container> bash"
    trap 'exit 0' TERM INT
    while true; do sleep 86400 & wait $!; done
    ;;
  passthrough)
    if [ "$#" -gt 0 ]; then
      exec "$@"
    fi
    echo "Passthrough mode — no command provided, exiting cleanly."
    exit 0
    ;;
  "")
    echo "Error: MODE is required. Set MODE=idle or MODE=passthrough." >&2
    echo "Available modes: idle, passthrough" >&2
    exit 1
    ;;
  *)
    echo "Error: unknown MODE '${mode}'." >&2
    echo "Available modes: idle, passthrough" >&2
    exit 1
    ;;
esac
