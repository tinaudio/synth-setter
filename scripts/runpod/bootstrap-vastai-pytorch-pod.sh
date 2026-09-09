#!/usr/bin/env bash
# Bring a RunPod pod running the stock vastai/pytorch image to parity with
# tinaudio/synth-setter:devcontainer-tools (docker/ubuntu22_04/Dockerfile,
# devcontainer-tools stage + .devcontainer/post-create.sh) without using the
# repo's Docker/devcontainer tooling.
#
# The script is self-fetching: it clones tinaudio/synth-setter itself, so the
# only thing the pod needs up front is this one file. The repo is public, so
# no token is required to fetch either.
#
#   SS_BRANCH=internal-feat/runpod-vastai-bootstrap   # branch that carries this script
#   SS_URL="https://raw.githubusercontent.com/tinaudio/synth-setter/${SS_BRANCH}/scripts/runpod/bootstrap-vastai-pytorch-pod.sh"
#
# Option A — run automatically at pod startup (works on RunPod, not only
# Vast). PROVISIONING_SCRIPT is read by the image itself, not by the Vast host:
# the entrypoint's boot step /etc/vast_boot.d/75-provisioning-manifest.sh hands
# it to the bundled `provisioner`, which downloads and runs it as root once
# (guarded by /.provisioning_complete). In the RunPod template add the env vars
#   PROVISIONING_SCRIPT=${SS_URL}
#   SS_GIT_REF=main            # optional: branch to check out
# Output goes to the provisioner's log under /var/log/portal (no
# /workspace/bootstrap.log in this mode). Delete /.provisioning_complete to
# force a re-run on the next boot.
#
# Option B — run once over SSH after the pod is up:
#   curl -fsSL "${SS_URL}" -o /tmp/bootstrap.sh
#   bash /tmp/bootstrap.sh 2>&1 | tee /workspace/bootstrap.log
#
# Every step is idempotent, so re-running after a failure resumes cheaply.
#
# Tunables (env vars):
#   SS_REPO_DIR        checkout path            (default /workspace/synth-setter,
#                      the persistent volume; /home/build/synth-setter is
#                      symlinked to it and SSH logins land there)
#   SS_GIT_REF         ref to check out         (default main)
#   SS_VENV            project venv             (default $SS_REPO_DIR/.venv; the
#                      devcontainer bakes /venv/main, but vastai's /venv/main
#                      hosts the image's own services, so we leave it alone)
#   SS_TORCH_BACKEND   uv extra                 (default cu128)
#   SS_SKIP_AGENT_TOOLS=1   skip codex/claude/pi/hermes/agy
#   SS_SKIP_DOOM=1          skip Doom Emacs (slowest optional piece)
#   SS_RUN_TESTS=1          run `pytest -k "not slow"` at the end (image does)
#   RESTRICTED_AGENT_GIT_PAT  if set, `gh auth login` + `gh auth setup-git`
set -euo pipefail

# ---------------------------------------------------------------- pins ----
# Mirrors docker/ubuntu22_04/Dockerfile ARGs; bump both places together.
readonly UV_VERSION=0.11.28
readonly PYTHON_VERSION=3.12.13
readonly NODE_VERSION=24.15.0
readonly PUEUE_VERSION=v4.0.4
readonly PUEUE_SHA256=c1b10d7e4e62211075ddd0e1dc3e8cbfc5a43d662cb3be7402a28504e23fcb51
readonly PUEUED_SHA256=5afeff6adbafb909e8d54e2caff158e6966c2adffa2c09e60fd631cc51b60390
readonly ZELLIJ_VERSION=v0.44.3
readonly ZELLIJ_SHA256=0f7c346788627f506c0a28296517768633cff24fc822a739f8264b640ecad751
readonly INFISICAL_VERSION=0.38.0
readonly INFISICAL_SHA256=b77813070e5b59ecdebd399f2d7efbb0158aabbf5d6fba679a1f32e6f3e9d03f
readonly KR106_VERSION=v2.5.13
readonly KR106_GIT_REF=bc15caee5843ab238a25d0969e68d57db2b1615f
readonly DOOM_EMACS_GIT_SHA=04b2956bafd883e5da7c530f8e0fbd22b1fe7af8
readonly HERMES_GIT_REF=v2026.7.7.2
readonly HERMES_GIT_SHA=9de9c25f620ff7f1ce0fd5457d596052d5159596
readonly HERMES_INSTALLER_SHA256=a93c65b01ea392e179cf872e182bd01a2b65c0c15f17833e9f9569033ef10e07

SS_REPO_DIR="${SS_REPO_DIR:-/workspace/synth-setter}"
SS_GIT_REF="${SS_GIT_REF:-main}"
SS_VENV="${SS_VENV:-$SS_REPO_DIR/.venv}"
SS_TORCH_BACKEND="${SS_TORCH_BACKEND:-cu128}"
readonly IMAGE_WORKDIR=/home/build/synth-setter
readonly VST3_DIR=/usr/lib/vst3
readonly STUDIORACK_PLUGINS_DIR=/opt/studiorack
readonly PY_INSTALL_DIR=/opt/uv/python
readonly PROFILE=/etc/profile.d/synth-setter.sh

export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC LANG=C.UTF-8
export UV_PYTHON_INSTALL_DIR="$PY_INSTALL_DIR"
export STUDIORACK_PLUGINS_DIR
# The vastai image pre-activates /venv/main; drop it so uv/npm target ours.
unset VIRTUAL_ENV
export PATH="/usr/local/bin:$PATH"

WARNINGS=()
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*" >&2; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; WARNINGS+=("$*"); }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

sha_check() { echo "$1  $2" | sha256sum -c - >/dev/null || die "checksum mismatch for $2"; }

# apt package names drift between 22.04 and 24.04; install what exists and
# report the rest instead of failing the whole step on one rename. `apt-cache
# policy` is the reliable probe: pure-virtual packages (libasound2 on 24.04)
# and obsoleted ones (libgl1-mesa-glx) still print via `apt-cache show`.
apt_install() {
  local pkg present=() missing=()
  for pkg in "$@"; do
    if apt-cache policy "$pkg" 2>/dev/null | grep -q '^ *Candidate: [^(]'; then
      present+=("$pkg")
    else
      missing+=("$pkg")
    fi
  done
  if (( ${#missing[@]} )); then warn "apt packages unavailable on this Ubuntu: ${missing[*]}"; fi
  apt-get install -y --no-install-recommends "${present[@]}"
}

# ---------------------------------------------------------- preflight ----
[[ "$(id -u)" -eq 0 ]] || die "run as root (RunPod starts vastai/pytorch as root)"
[[ "$(uname -m)" == x86_64 ]] || die "only x86_64 is supported (KR-106/pueue/zellij pins)"
. /etc/os-release
log "Ubuntu ${VERSION_ID} on $(uname -m); repo=$SS_REPO_DIR ref=$SS_GIT_REF venv=$SS_VENV"

# RunPod injects PUBLIC_KEY into /root/.ssh with modes sshd refuses ("bad
# ownership or modes for file authorized_keys"). Normalise first so SSH works
# even if a later step fails.
if [[ -d /root/.ssh ]]; then
  chown -R root:root /root/.ssh
  chmod 700 /root/.ssh
  [[ -f /root/.ssh/authorized_keys ]] && chmod 600 /root/.ssh/authorized_keys
fi

# ------------------------------------------------------- 1. apt layer ----
log "1/12 apt: build toolchain, VST runtime, headless X stack, CLI tools"
apt-get update --error-on=any
apt_install ca-certificates curl wget gnupg git jq unzip xz-utils sudo \
  software-properties-common make ninja-build flex build-essential cmake pkg-config \
  python3 python3-dev python3-pip rsync openssh-client rclone
# JUCE build deps (KR-106 source build) — same list as the Dockerfile.
apt_install libx11-dev libxcb-cursor-dev libxcb-keysyms1-dev libxcb-util-dev \
  libxkbcommon-dev libxkbcommon-x11-dev xcb libgtk-3-dev libwebkit2gtk-4.0-dev \
  libwebkit2gtk-4.1-dev libcurl4-openssl-dev libasound2-dev libjack-dev \
  libfreetype-dev libfontconfig1-dev libxinerama-dev libxrandr-dev libgl-dev \
  libgl1-mesa-dev mesa-common-dev libxcursor-dev
# Runtime libs for headless VST3 loading + the Xvfb/xsettingsd/openbox stack.
# 24.04 renamed several to *t64; both spellings are listed and the probe above
# keeps whichever exists.
apt_install dbus-x11 libasound2 libasound2t64 libcairo2 libcurl4 libcurl4t64 \
  libfontconfig1 libfreetype6 libgl1-mesa-dri libgl1-mesa-glx libgl1 \
  libgtk-3-0 libgtk-3-0t64 libjack-jackd2-0 libjack-jackd2-0t64 libx11-6 \
  libxcb-cursor0 libxcb-keysyms1 libxcb-util1 libxcursor1 libxinerama1 \
  libxkbcommon-x11-0 libxrandr2 mesa-utils openbox x11-utils x11-xserver-utils \
  xauth xclip xdg-utils xsettingsd xvfb
# rclone 1.53 (apt) execs bare `fusermount`, which only fuse 2.x ships; fuse3
# Breaks fuse, so keep whichever the image already has.
if dpkg -s fuse3 >/dev/null 2>&1; then
  warn "fuse3 already installed; skipping fuse (2.x) — rclone mount may need a fusermount3 shim"
else
  apt_install fuse
fi
# CLI tools from the devcontainer-tools stage.
apt_install btop bubblewrap emacs-nox fd-find ripgrep tmux
[[ -e /usr/local/bin/fd ]] || ln -s /usr/bin/fdfind /usr/local/bin/fd
if ! command -v gh >/dev/null; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    -o /usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list
  apt-get update && apt-get install -y --no-install-recommends gh
fi
apt-get clean

# ------------------------------------------------------ 2. uv + python ----
log "2/12 uv ${UV_VERSION} + CPython ${PYTHON_VERSION}"
if [[ "$(uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv python install "$PYTHON_VERSION"

# --------------------------------------------------------- 3. checkout ----
log "3/12 checkout ${SS_GIT_REF} -> ${SS_REPO_DIR}"
git config --global --add safe.directory "$SS_REPO_DIR" 2>/dev/null || true
if [[ ! -d "$SS_REPO_DIR/.git" ]]; then
  mkdir -p "$SS_REPO_DIR"
  git -C "$SS_REPO_DIR" init
  git -C "$SS_REPO_DIR" remote add origin https://github.com/tinaudio/synth-setter.git
fi
git -C "$SS_REPO_DIR" fetch origin
if git -C "$SS_REPO_DIR" show-ref --verify -q "refs/heads/$SS_GIT_REF"; then
  # Rerun: fast-forward only, so local commits on the pod are never discarded.
  git -C "$SS_REPO_DIR" checkout -q "$SS_GIT_REF"
  git -C "$SS_REPO_DIR" pull -q --ff-only origin "$SS_GIT_REF" || warn "could not fast-forward $SS_GIT_REF; left as is"
else
  git -C "$SS_REPO_DIR" checkout -B "$SS_GIT_REF" "origin/$SS_GIT_REF" 2>/dev/null \
    || git -C "$SS_REPO_DIR" checkout --detach "$SS_GIT_REF"
fi
# The image's WORKDIR is /home/build/synth-setter; alias it so paths, docs,
# and muscle memory match while the clone itself lives on the /workspace volume.
mkdir -p "$(dirname "$IMAGE_WORKDIR")"
[[ "$IMAGE_WORKDIR" == "$SS_REPO_DIR" ]] || ln -sfn "$SS_REPO_DIR" "$IMAGE_WORKDIR"
cd "$SS_REPO_DIR"

# ------------------------------------------------------- 4. python env ----
log "4/12 uv sync --extra ${SS_TORCH_BACKEND} --group dev -> ${SS_VENV}"
if [[ ! -x "$SS_VENV/bin/python" ]] \
   || ! "$SS_VENV/bin/python" -c "import sys; raise SystemExit(sys.version_info[:3] != tuple(map(int, '${PYTHON_VERSION}'.split('.'))))"; then
  rm -rf "$SS_VENV"
  uv venv --python "$PYTHON_VERSION" --prompt synth-setter "$SS_VENV"
fi
UV_PROJECT_ENVIRONMENT="$SS_VENV" uv sync --frozen --extra "$SS_TORCH_BACKEND" --no-default-groups --group dev
export VIRTUAL_ENV="$SS_VENV" PATH="$SS_VENV/bin:$PATH"
python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), sys.version)"

# -------------------------------------------------------- 5. node + npm ----
log "5/12 Node ${NODE_VERSION} (studiorack + agent CLIs)"
if [[ "$(node --version 2>/dev/null)" != "v${NODE_VERSION}" ]]; then
  tarball="node-v${NODE_VERSION}-linux-x64.tar.xz"
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/${tarball}" -o "/tmp/${tarball}"
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt" \
    | grep " ${tarball}\$" | (cd /tmp && sha256sum -c -) || die "node tarball checksum mismatch"
  tar -xJf "/tmp/${tarball}" -C /usr/local --strip-components=1 \
    --exclude='CHANGELOG.md' --exclude='LICENSE' --exclude='README.md'
  rm "/tmp/${tarball}"
fi
node --version && npm --version

# ------------------------------------------------------- 6. VST3 plugins ----
log "6/12 Studiorack plugins -> ${STUDIORACK_PLUGINS_DIR}, links in ${VST3_DIR}"
npm ci
mkdir -p "$VST3_DIR" "$STUDIORACK_PLUGINS_DIR"
readonly HEADLESS=src/synth_setter/scripts/run-linux-vst-headless.sh
chmod +x "$HEADLESS"
studiorack() { synth-setter-plugins --manifest studiorack.json --links-dir "$VST3_DIR" "$@"; }
# Studiorack's validator instantiates each plugin after download. JUCE synths
# (Six Sines, OB-Xf) touch X11 on init and hang against a forwarded or absent
# DISPLAY, so every install runs inside the Xvfb/xsettingsd/dbus wrapper with
# the login environment's DISPLAY dropped.
for spec in "surge-synthesizer/surge:Surge XT" "asb2m10/dexed:Dexed" \
            "baconpaul/six-sines:Six Sines" "surge-synthesizer/ob-xf:OB-Xf"; do
  plugin="${spec%%:*}"; bundle="${spec#*:}"
  [[ -d "$VST3_DIR/${bundle}.vst3" ]] \
    || env -u DISPLAY "$HEADLESS" synth-setter-plugins --manifest studiorack.json \
         --links-dir "$VST3_DIR" install --plugin "$plugin"
done
if [[ ! -d "$VST3_DIR/CardinalSynth.vst3" ]]; then
  env -u DISPLAY "$HEADLESS" synth-setter-plugins --manifest studiorack-cardinal.json \
    --links-dir "$VST3_DIR" install --plugin distrho/cardinal
fi
# KR-106: upstream release zips need glibc 2.38, so build from source like the image.
if [[ ! -d "$VST3_DIR/Ultramaster KR-106.vst3" ]]; then
  cache="$HOME/.cache/synth-setter/ultramaster-kr106-${KR106_VERSION}"
  src="$cache/src"; build="$cache/build"
  mkdir -p "$src"
  if ! git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$src" init
    git -C "$src" remote add origin https://github.com/kayrockscreenprinting/ultramaster_kr106.git
  fi
  git -C "$src" fetch --depth 1 origin "$KR106_GIT_REF"
  git -C "$src" checkout --detach FETCH_HEAD
  git -C "$src" submodule update --init --recursive --depth 1
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DKR106_COPY_AFTER_BUILD=OFF
  cmake --build "$build" --config Release --target KR106_VST3 --parallel "$(nproc)"
  bundle="$build/KR106_artefacts/Release/VST3/Ultramaster KR-106.vst3"
  [[ -d "$bundle" ]] || die "KR-106 build produced no bundle at $bundle"
  studiorack adopt --plugin kayrockscreenprinting/ultramaster-kr106 --bundle-path "$bundle"
  studiorack link --plugin kayrockscreenprinting/ultramaster-kr106
fi
# Load-check every bundle and project it into plugins/ exactly like the image.
mkdir -p plugins
for entry in "Surge XT|" "CardinalSynth|" "Dexed|" "OB-Xf|" "Six Sines|Six Sines" "Ultramaster KR-106|Ultramaster KR-106"; do
  name="${entry%%|*}"; pname="${entry#*|}"
  [[ -d "$VST3_DIR/${name}.vst3" ]] || die "${name}.vst3 missing from ${VST3_DIR}"
  if [[ -n "$pname" ]]; then
    env -u DISPLAY "$HEADLESS" python -X faulthandler src/synth_setter/scripts/load_vst3_check.py "$VST3_DIR/${name}.vst3" "$pname"
  else
    env -u DISPLAY "$HEADLESS" python -X faulthandler src/synth_setter/scripts/load_vst3_check.py "$VST3_DIR/${name}.vst3"
  fi
  ln -sfn "$VST3_DIR/${name}.vst3" "plugins/${name}.vst3"
done

# ------------------------------------------------------------- 7. pueue ----
log "7/12 pueue ${PUEUE_VERSION}"
if ! pueue --version 2>/dev/null | grep -q "${PUEUE_VERSION#v}"; then
  base="https://github.com/Nukesor/pueue/releases/download/${PUEUE_VERSION}"
  wget -q -O /usr/local/bin/pueue  "${base}/pueue-x86_64-unknown-linux-musl"
  wget -q -O /usr/local/bin/pueued "${base}/pueued-x86_64-unknown-linux-musl"
  sha_check "$PUEUE_SHA256" /usr/local/bin/pueue
  sha_check "$PUEUED_SHA256" /usr/local/bin/pueued
  chmod +x /usr/local/bin/pueue /usr/local/bin/pueued
fi

# ------------------------------------------------- 8. zellij + infisical ----
log "8/12 zellij ${ZELLIJ_VERSION} + infisical ${INFISICAL_VERSION}"
if ! zellij --version 2>/dev/null | grep -q "${ZELLIJ_VERSION#v}"; then
  wget -q -O /tmp/zellij.tar.gz \
    "https://github.com/zellij-org/zellij/releases/download/${ZELLIJ_VERSION}/zellij-x86_64-unknown-linux-musl.tar.gz"
  sha_check "$ZELLIJ_SHA256" /tmp/zellij.tar.gz
  tar -xzf /tmp/zellij.tar.gz -C /usr/local/bin zellij && chmod +x /usr/local/bin/zellij
  rm /tmp/zellij.tar.gz
fi
if ! infisical --version 2>/dev/null | grep -q "$INFISICAL_VERSION"; then
  package="infisical_${INFISICAL_VERSION}_linux_amd64.deb"
  curl -fsSL "https://dl.cloudsmith.io/public/infisical/infisical-cli/deb/debian/pool/any-version/main/i/in/infisical_${INFISICAL_VERSION}/${package}" -o "/tmp/${package}"
  sha_check "$INFISICAL_SHA256" "/tmp/${package}"
  dpkg -i "/tmp/${package}" && rm "/tmp/${package}"
fi

# ------------------------------------------------------ 9. agent CLIs ----
if [[ "${SS_SKIP_AGENT_TOOLS:-0}" != 1 ]]; then
  log "9/12 codex, claude-code, pi, hermes, agy"
  npm install -g @openai/codex@latest @anthropic-ai/claude-code@latest
  npm install -g --ignore-scripts @earendil-works/pi-coding-agent@latest
  npm cache clean --force
  codex --version && claude --version && pi --version
  if ! command -v hermes >/dev/null; then
    curl -fsSL "https://raw.githubusercontent.com/NousResearch/hermes-agent/${HERMES_GIT_REF}/scripts/install.sh" -o /tmp/hermes-install.sh
    sha_check "$HERMES_INSTALLER_SHA256" /tmp/hermes-install.sh
    env -u VIRTUAL_ENV -u UV_PYTHON_INSTALL_DIR bash /tmp/hermes-install.sh \
      --branch "$HERMES_GIT_REF" --commit "$HERMES_GIT_SHA" --skip-browser \
      || warn "hermes install failed (non-fatal)"
    rm -f /tmp/hermes-install.sh
  fi
  if ! command -v agy >/dev/null && [[ ! -x "$HOME/.local/bin/agy" ]]; then
    { curl -fsSL https://antigravity.google/cli/install.sh -o /tmp/agy-install.sh \
      && bash /tmp/agy-install.sh; } || warn "agy install failed (non-fatal)"
    rm -f /tmp/agy-install.sh
  fi
else
  log "9/12 agent CLIs skipped (SS_SKIP_AGENT_TOOLS=1)"
fi

# ------------------------------------------------------- 10. doom emacs ----
if [[ "${SS_SKIP_DOOM:-0}" != 1 ]]; then
  log "10/12 Doom Emacs @ ${DOOM_EMACS_GIT_SHA:0:7}"
  emacs_dir="$HOME/.config/emacs"
  if [[ ! -x "$emacs_dir/bin/doom" ]]; then
    mkdir -p "$emacs_dir" "$HOME/.local/bin"
    git -C "$emacs_dir" init
    git -C "$emacs_dir" remote add origin https://github.com/doomemacs/doomemacs.git
    git -C "$emacs_dir" fetch --depth 1 origin "$DOOM_EMACS_GIT_SHA"
    git -C "$emacs_dir" checkout --detach "$DOOM_EMACS_GIT_SHA"
    git -C "$emacs_dir" submodule update --init --recursive
    "$emacs_dir/bin/doom" install --force --no-env --no-hooks || warn "doom install failed (non-fatal)"
  fi
else
  log "10/12 Doom Emacs skipped (SS_SKIP_DOOM=1)"
fi

# ------------------------------------------- 11. shell env + dotfiles ----
log "11/12 profile, bashrc, tmux/zellij configs, codex defaults"
cat > "$PROFILE" <<EOF
# synth-setter pod environment — mirrors the devcontainer image ENV block.
export SYNTH_SETTER_PLUGIN_PATH="${VST3_DIR}/Surge XT.vst3"
export STUDIORACK_PLUGINS_DIR="${STUDIORACK_PLUGINS_DIR}"
export UV_PYTHON_INSTALL_DIR="${PY_INSTALL_DIR}"
export HYDRA_FULL_ERROR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DATA_DIR="\$HOME/.cache/wandb"
export PYTEST_XDIST_AUTO_NUM_WORKERS=4
export PATH="\$PATH:\$HOME/.local/bin:\$HOME/.config/emacs/bin"
# Devcontainers get .env via --env-file; here we source it when present.
if [[ -f "${SS_REPO_DIR}/.env" ]]; then set -a; . "${SS_REPO_DIR}/.env"; set +a; fi
EOF
grep -qs "$PROFILE" "$HOME/.bashrc" || printf '\n. %s\n' "$PROFILE" >> "$HOME/.bashrc"

# Per-worktree venv isolation, verbatim from .devcontainer/post-create.sh (#1339).
if ! grep -qs 'Per-worktree venv isolation' "$HOME/.bashrc"; then
  cat >>"$HOME/.bashrc" <<'EOF'

# Per-worktree venv isolation — see .devcontainer/post-create.sh.
__ss_root="$PWD"
while [[ "$__ss_root" != "/" && ! -f "$__ss_root/.project-root" ]]; do
  __ss_root="$(dirname "$__ss_root")"
done
if [[ -f "$__ss_root/.venv/bin/activate" && "${VIRTUAL_ENV:-}" != "$__ss_root/.venv" ]]; then
  unset VIRTUAL_ENV
  source "$__ss_root/.venv/bin/activate"
fi
unset __ss_root
EOF
fi
# Land interactive logins in the checkout with its venv active, like the
# image's WORKDIR. Appended after the image's own `cd ${WORKSPACE}` line so it
# wins; interactive-only so scp/rsync/non-login commands keep their cwd.
grep -qs "ss-login-landing" "$HOME/.bashrc" || cat >>"$HOME/.bashrc" <<EOF
# ss-login-landing
if [[ \$- == *i* && -d "${IMAGE_WORKDIR}" ]]; then cd "${IMAGE_WORKDIR}"; fi
[[ -z "\${VIRTUAL_ENV:-}" && -f "${SS_VENV}/bin/activate" ]] && source "${SS_VENV}/bin/activate"
EOF
# Persisted history + agent autonomy defaults, as post-create.sh seeds them.
if ! grep -qs 'HISTFILE=/commandhistory' "$HOME/.bashrc"; then
  mkdir -p /commandhistory && touch /commandhistory/.bash_history
  printf '%s\n' "export PROMPT_COMMAND='history -a'" "export HISTFILE=/commandhistory/.bash_history" >> "$HOME/.bashrc"
fi
install -m 0644 .devcontainer/tmux.conf "$HOME/.tmux.conf"
install -D -m 0644 .devcontainer/zellij.kdl "$HOME/.config/zellij/config.kdl"
if [[ ! -f "$HOME/.codex/config.toml" ]]; then
  mkdir -p "$HOME/.codex"
  printf 'approval_policy = "never"\nsandbox_mode = "danger-full-access"\n' > "$HOME/.codex/config.toml"
fi
grep -qsF 'agy()' "$HOME/.bashrc" || printf '\nagy() { command agy --dangerously-skip-permissions "$@"; }\n' >> "$HOME/.bashrc"

# --------------------------------------------- 12. git hooks + gh auth ----
log "12/12 pre-commit hooks, gh auth, verification"
for scope in --system --global --local; do git config "$scope" --unset-all core.hooksPath 2>/dev/null || true; done
pre-commit install --hook-type pre-commit --hook-type pre-push
if [[ -n "${RESTRICTED_AGENT_GIT_PAT:-}" ]]; then
  pat="${RESTRICTED_AGENT_GIT_PAT//[\"\']/}"
  if printf '%s' "$pat" | gh auth login --with-token; then gh auth setup-git; else warn "gh auth login failed"; fi
else
  warn "RESTRICTED_AGENT_GIT_PAT unset; gh/git push are unauthenticated"
fi
scripts/dev/link-skills.sh || true

python -c "import synth_setter, torch; print('synth_setter from', synth_setter.__file__); print('cuda devices', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || warn "nvidia-smi unavailable"
[[ -f .env ]] || warn "no ${SS_REPO_DIR}/.env — copy .env.example and fill R2/W&B creds (sourced by ${PROFILE})"
if [[ "${SS_RUN_TESTS:-0}" == 1 ]]; then
  "$HEADLESS" pytest -k "not slow" -q
fi

log "DONE. Open a fresh shell (or: . ~/.bashrc) and run 'make test-fast' from ${SS_REPO_DIR}."
if (( ${#WARNINGS[@]} )); then
  printf 'Warnings:\n' >&2; printf '  - %s\n' "${WARNINGS[@]}" >&2
fi
