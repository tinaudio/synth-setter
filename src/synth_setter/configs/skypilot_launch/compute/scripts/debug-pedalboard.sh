#!/usr/bin/env bash
# SSH cwd is ~/sky_workdir (synced from the dispatching checkout via
# task.workdir); ensure_plugin_symlinks.sh restores the Surge XT symlink the
# synced workdir hides.
set -euo pipefail
bash docker/ubuntu22_04/ensure_plugin_symlinks.sh
bash src/synth_setter/scripts/run-linux-vst-headless.sh python -c '
from pedalboard import VST3Plugin
plugin = VST3Plugin("plugins/Surge XT.vst3")
print(f"plugin loaded: name={plugin.name!r} version={plugin.version!r}")
print("skypilot-debug variant=pedalboard-load done")
'
