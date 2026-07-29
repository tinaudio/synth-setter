#!/usr/bin/env bash
# Restore checkout-local aliases from exact Studiorack package installs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
plugin_cli=(
  env "PYTHONPATH=${repo_root}/src" python -m synth_setter.cli.plugins
  --manifest "${repo_root}/studiorack.json"
  --links-dir "${repo_root}/plugins"
)
if [[ -d "/usr/lib/vst3/Surge XT.vst3" ]]; then
  "${plugin_cli[@]}" adopt \
    --plugin surge-synthesizer/surge \
    --bundle-path "/usr/lib/vst3/Surge XT.vst3"
fi
"${plugin_cli[@]}" link --plugin surge-synthesizer/surge
