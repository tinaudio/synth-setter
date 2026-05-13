#!/usr/bin/env bash
# Recreate the plugins/<plugin>.vst3 symlink that `docker run -v $(pwd):...`
# overlays. Override target via SYNTH_SETTER_PLUGIN_PATH.
set -euo pipefail

target="${SYNTH_SETTER_PLUGIN_PATH:-/usr/lib/vst3/Surge XT.vst3}"
if [[ ! -e "$target" ]]; then
  echo "ensure_plugin_symlinks.sh: target does not exist: $target" >&2
  exit 1
fi
mkdir -p plugins
ln -sf "$target" "plugins/$(basename "$target")"
