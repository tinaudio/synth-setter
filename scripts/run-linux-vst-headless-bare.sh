#!/usr/bin/env bash
#
# Diagnostic variant 5 of run-linux-vst-headless.sh: bare wrapper.
# Strips out everything (no Xvfb, no xsettingsd, no openbox, no dbus,
# no traps, no temp dir). Just `exec "$@"`. Should behave identically
# to running the command directly (the noop pure-sky variant).
#
# Hypothesis: if even this trivial wrapper hangs, the wrapper file
# itself has some side effect we haven't isolated (env propagation,
# file mode quirk, shebang interpreter difference) — the X stack /
# dbus chain is a red herring. Expected outcome: PASS, matching the
# noop pure-sky variant.

set -euo pipefail

exec "$@"
