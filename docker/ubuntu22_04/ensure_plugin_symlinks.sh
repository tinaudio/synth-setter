#!/usr/bin/env bash
# Restore checkout-local aliases from exact Studiorack package installs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHONPATH="${repo_root}/src" python -m synth_setter.cli.plugins \
  --manifest "${repo_root}/studiorack.json" \
  --links-dir "${repo_root}/plugins" \
  link --plugin surge-synthesizer/surge
