# ==============================================================================
# Quick start — pull and run the prebuilt VM:
#
#   brew install cirruslabs/cli/tart
#   tart clone registry-1.docker.io/tinaudio/synth-setter-macos:latest synth-setter-macos
#   tart run synth-setter-macos                          # GUI window opens
#
#   # In another terminal:
#   ssh admin@$(tart ip synth-setter-macos)              # password: admin
#
# Security note: the VM inherits the cirruslabs base image's well-known
# admin/admin credentials. Treat this as a local-only dev VM. If the host is
# on a shared or untrusted network, change the password in the GUI on first
# boot, or add an SSH key to ~admin/.ssh/authorized_keys and disable
# PasswordAuthentication in /etc/ssh/sshd_config before exposing port 22.
#
# See https://tart.run/faq/ for ssh troubleshooting information.
# Run `tart --help` for additional commands.
# To build and publish a new image yourself, see the instructions at the
# bottom of this file.
# ==============================================================================


packer {
  required_plugins {
    tart = {
      version = ">= 1.12.0"
      source  = "github.com/cirruslabs/tart"
    }
  }
}

variable "base_image_digest" {
  type    = string
  default = "sha256:6abd551a46da4e595b6a9f678535a8f1bbd61bdc275a363265cd39281d3abdef"
}

variable "synth_setter_git_ref" {
  type        = string
  default     = "main"
  description = "Git ref (branch, tag, or SHA) to check out for synth-setter. Prefer a SHA for reproducible builds."
}

variable "python_version" {
  type        = string
  default     = "3.11"
  description = "Python interpreter version installed and pinned via uv."
}

variable "uv_version" {
  type        = string
  default     = "0.11.2"
  description = "uv version installed via Astral's versioned installer (https://astral.sh/uv/<version>/install.sh). Keep in sync with docker/ubuntu22_04/Dockerfile (`ghcr.io/astral-sh/uv:<version>`) so Tart and Docker dev-base resolve identical wheels via `uv sync --frozen`."
}

variable "vm_name" {
  type        = string
  default     = "synth-setter-macos"
  description = "Name of the built Tart VM. Used with `tart run <name>` and `tart ip <name>`."
}

variable "surge_xt_version" {
  type        = string
  default     = "1.3.4"
  description = "Required Surge XT version. Asserted after `brew install --cask surge-xt` so the build fails loudly if Homebrew's cask has rolled past this — bump only after validating the new release against the pipeline."
}

variable "codex_version" {
  type        = string
  default     = "latest"
  description = "OpenAI Codex CLI version installed via `npm install -g @openai/codex@<version>`. Matches the CODEX_VERSION arg in docker/ubuntu22_04/Dockerfile's devcontainer-tools stage."
}

# ---------------------------------------------------------------------------
# Third-party VST3 synths baked into the image — parity with the
# `vst3-synths-fetch` stage in docker/ubuntu22_04/Dockerfile. Each asset is
# pinned by version and SHA256 (refresh both atomically by running
# `shasum -a 256` against the upstream macOS asset). Licensing review applies
# to bumps for the same reason it does in Docker — these third-party binaries
# ship in the published Tart image: Dexed (GPL-3.0), OB-Xf (GPL-3.0), Six Sines
# (MIT).
# ---------------------------------------------------------------------------

variable "dexed_version" {
  type        = string
  default     = "0.9.8"
  description = "Dexed (asb2m10/dexed, GPL-3.0) release version. Templated into both the asset URL and the .pkg filename inside the macOS DMG. Matches DEXED_VERSION in docker/ubuntu22_04/Dockerfile."
}

variable "dexed_macos_sha256" {
  type        = string
  default     = "3be32f98e56b40d9555a4069368c7307b8bae3368459bb087cf6195ae7538704"
  description = "SHA256 of the asb2m10/dexed `dexed-<dexed_version>-macos.zip` asset. Refresh atomically with `dexed_version`."
}

variable "obxf_version" {
  type        = string
  default     = "v1.0.3"
  description = "OB-Xf (surge-synthesizer/OB-Xf, GPL-3.0) release tag. Matches OBXF_VERSION in docker/ubuntu22_04/Dockerfile."
}

variable "obxf_macos_sha256" {
  type        = string
  default     = "e6f52f453c476dfafbd3225b80c5c3ec341ec20be5748c6f70c01cef14e7e7a8"
  description = "SHA256 of the surge-synthesizer/OB-Xf `ob-xf-macOS-<obxf_version>.dmg` asset. Refresh atomically with `obxf_version`."
}

variable "six_sines_version" {
  type        = string
  default     = "v1.1.0"
  description = "Six Sines (baconpaul/six-sines, MIT) release tag. Matches SIX_SINES_VERSION in docker/ubuntu22_04/Dockerfile."
}

variable "six_sines_macos_asset" {
  type        = string
  default     = "six-sines-macOS-2025-03-18-43d10b2.dmg"
  description = "Six Sines macOS DMG asset filename — upstream embeds a date+SHA, not the release tag, in the filename. Matches the per-platform SIX_SINES_ASSET in docker/ubuntu22_04/Dockerfile."
}

variable "six_sines_macos_sha256" {
  type        = string
  default     = "d6fdf29179cd5b35aa024f25912a50bd14df59fa73cea9ce3da6f3fbf1abe55a"
  description = "SHA256 of the Six Sines macOS DMG. Refresh atomically with `six_sines_macos_asset`."
}

source "tart-cli" "tart" {
  vm_base_name = "ghcr.io/cirruslabs/macos-tahoe-base@${var.base_image_digest}"
  vm_name      = var.vm_name
  cpu_count    = 4
  memory_gb    = 8
  # Credentials Packer uses to SSH into the VM to run provisioners.
  # These are the defaults baked into cirruslabs base images. If you
  # change the password inside a provisioner, subsequent provisioners
  # in the same build will fail to reconnect.
  ssh_password = "admin"
  ssh_username = "admin"
  ssh_timeout  = "120s"
}

build {
  sources = ["source.tart-cli.tart"]

  # CLI + GUI tools. Parity with the CLI stack in the Docker
  # devcontainer-tools stage, adapted to macOS: surge-xt ships as a cask
  # (.vst3 installed to /Library/Audio/Plug-Ins/VST3/Surge XT.vst3); tmux
  # + zellij mirror the multiplexers baked into devcontainer-tools; node
  # provides the `npm` runtime used to install the Codex CLI below.
  provisioner "shell" {
    inline = [
      "touch ~/.zprofile && . ~/.zprofile",
      "brew --version",
      "brew update",
      "brew install git gh jq rclone bats-core tmux zellij node",
      # Install uv from Astral's versioned installer rather than `brew install uv`.
      # Homebrew's uv formula is rolling, so it cannot reliably hold a specific
      # version; the Astral installer URL embeds the version and is reproducible.
      # Keep ${var.uv_version} in sync with docker/ubuntu22_04/Dockerfile so
      # `uv sync --frozen` resolves identical wheels in Docker and Tart.
      "curl -LsSf https://astral.sh/uv/${var.uv_version}/install.sh | sh",
      "grep -qxF 'export PATH=\"$HOME/.local/bin:$PATH\"' ~/.zprofile || printf '\\nexport PATH=\"$HOME/.local/bin:$PATH\"\\n' >> ~/.zprofile",
      ". ~/.zprofile",
      "test \"$(uv --version | awk '{print $2}')\" = \"${var.uv_version}\"",
      "brew install --cask claude-code",
      # OpenAI Codex CLI (parity with devcontainer-tools' per-user npm install).
      # macOS users own /opt/homebrew (Apple Silicon) so a system-wide
      # `npm install -g` works without the per-user prefix the Docker image needs.
      "npm install -g \"@openai/codex@${var.codex_version}\"",
      "codex --version",
      # Antigravity (Google) ships the standalone `agy` binary, not on npm; its
      # installer fetches a SHA512-verified release into ~/.local/bin (upstream
      # latest — no version flag). Fetched to a file, not piped to bash, so a curl
      # failure is distinguishable from a script failure. Mirrors devcontainer-tools.
      "curl -fsSL https://antigravity.google/cli/install.sh -o /tmp/agy-install.sh",
      "bash /tmp/agy-install.sh",
      "rm /tmp/agy-install.sh",
      "agy --version",
      "brew install --cask surge-xt",
      # Hard-fail if Homebrew's cask resolves to a Surge XT version we haven't
      # qualified against the pipeline. `brew list --cask --versions` prints
      # `surge-xt <version>`; extract the second field and assert equality.
      "test \"$(brew list --cask --versions surge-xt | awk '{print $2}')\" = \"${var.surge_xt_version}\"",
    ]
  }

  # Third-party VST3 synths — parity with the `vst3-synths-fetch` stage in
  # docker/ubuntu22_04/Dockerfile. Each .pkg lands at
  # /Library/Audio/Plug-Ins/VST3/<Bundle>.vst3 (sibling to the cask-installed
  # Surge XT). Passwordless sudo for the `admin` user is part of the
  # cirruslabs/macos-tahoe-base contract — same root the Surge XT cask uses.
  provisioner "shell" {
    inline = [
      "touch ~/.zprofile && . ~/.zprofile",
      # Dexed (asb2m10/dexed, GPL-3.0) — the macOS release ships as a zip
      # wrapping a DMG; mount the inner DMG and run the embedded multi-pkg.
      "curl -fsSL -o /tmp/dexed.zip 'https://github.com/asb2m10/dexed/releases/download/v${var.dexed_version}/dexed-${var.dexed_version}-macos.zip'",
      "echo '${var.dexed_macos_sha256}  /tmp/dexed.zip' | shasum -a 256 -c -",
      "rm -rf /tmp/dexed && unzip -q /tmp/dexed.zip -d /tmp/dexed",
      "rm -rf /tmp/dexed-mnt && hdiutil attach -nobrowse -quiet -mountpoint /tmp/dexed-mnt '/tmp/dexed/dexed-${var.dexed_version}-macos.dmg'",
      "sudo installer -pkg '/tmp/dexed-mnt/dexed-macOS-${var.dexed_version}.pkg' -target /",
      "hdiutil detach -quiet /tmp/dexed-mnt && rm -rf /tmp/dexed /tmp/dexed.zip",
      "test -d '/Library/Audio/Plug-Ins/VST3/Dexed.vst3'",
      # OB-Xf (surge-synthesizer/OB-Xf, GPL-3.0) — DMG ships the multi-pkg
      # directly.
      "curl -fsSL -o /tmp/obxf.dmg 'https://github.com/surge-synthesizer/OB-Xf/releases/download/${var.obxf_version}/ob-xf-macOS-${var.obxf_version}.dmg'",
      "echo '${var.obxf_macos_sha256}  /tmp/obxf.dmg' | shasum -a 256 -c -",
      "rm -rf /tmp/obxf-mnt && hdiutil attach -nobrowse -quiet -mountpoint /tmp/obxf-mnt /tmp/obxf.dmg",
      "sudo installer -pkg '/tmp/obxf-mnt/ob-xf-macOS-${var.obxf_version}.pkg' -target /",
      "hdiutil detach -quiet /tmp/obxf-mnt && rm /tmp/obxf.dmg",
      "test -d '/Library/Audio/Plug-Ins/VST3/OB-Xf.vst3'",
      # Six Sines (baconpaul/six-sines, MIT) — pkg name shares the asset's
      # date+SHA basename, so derive the .pkg filename from
      # ${var.six_sines_macos_asset} by swapping the trailing `.dmg`.
      "curl -fsSL -o /tmp/six-sines.dmg 'https://github.com/baconpaul/six-sines/releases/download/${var.six_sines_version}/${var.six_sines_macos_asset}'",
      "echo '${var.six_sines_macos_sha256}  /tmp/six-sines.dmg' | shasum -a 256 -c -",
      "rm -rf /tmp/six-sines-mnt && hdiutil attach -nobrowse -quiet -mountpoint /tmp/six-sines-mnt /tmp/six-sines.dmg",
      "sudo installer -pkg '/tmp/six-sines-mnt/${regex_replace(var.six_sines_macos_asset, "\\.dmg$", ".pkg")}' -target /",
      "hdiutil detach -quiet /tmp/six-sines-mnt && rm /tmp/six-sines.dmg",
      "test -d '/Library/Audio/Plug-Ins/VST3/Six Sines.vst3'",
    ]
  }

  # Install and pin Python via uv.
  provisioner "shell" {
    inline = [
      "touch ~/.zprofile && . ~/.zprofile",
      "uv python install ${var.python_version}",
      "uv python pin ${var.python_version}",
    ]
  }

  # Clone the repo, use venv with all runtime deps (parity with Docker dev-base
  # stage). On macOS we resolve torch from PyPI's MPS-capable wheel — no backend
  # extra on Apple Silicon (the sys_platform marker in [tool.uv.sources] would
  # not match anyway). Mirrors test-mps.yml's `uv sync --frozen` invocation.
  provisioner "shell" {
    inline = [
      "touch ~/.zprofile && . ~/.zprofile",
      "git clone https://github.com/tinaudio/synth-setter.git ~/synth-setter",
      "cd ~/synth-setter && git checkout ${var.synth_setter_git_ref}",
      "cd ~/synth-setter && uv venv --python ${var.python_version}",
      "cd ~/synth-setter && uv sync --frozen",
      # Mirror the Docker dev-base convention: symlink each system-wide VST3
      # bundle to the repo-relative `plugins/<Name>.vst3` path that configs,
      # CLI `--plugin_path` defaults, and tests all assume. See
      # docker/ubuntu22_04/Dockerfile's `ln -s` loop in
      # builder-install-synth-setter-deps.
      "mkdir -p ~/synth-setter/plugins",
      "ln -sfn '/Library/Audio/Plug-Ins/VST3/Surge XT.vst3' ~/synth-setter/'plugins/Surge XT.vst3'",
      "ln -sfn '/Library/Audio/Plug-Ins/VST3/Dexed.vst3' ~/synth-setter/'plugins/Dexed.vst3'",
      "ln -sfn '/Library/Audio/Plug-Ins/VST3/OB-Xf.vst3' ~/synth-setter/'plugins/OB-Xf.vst3'",
      "ln -sfn '/Library/Audio/Plug-Ins/VST3/Six Sines.vst3' ~/synth-setter/'plugins/Six Sines.vst3'",
      # Auto-activate the venv for every interactive shell so tools installed
      # into .venv/bin (pre-commit, pyright, pytest, ruff, etc.) are on PATH
      # from login without a manual `source .venv/bin/activate`.
      "touch ~/.zshrc && (grep -qxF 'source ~/synth-setter/.venv/bin/activate' ~/.zshrc || printf '\\nsource ~/synth-setter/.venv/bin/activate\\n' >> ~/.zshrc)",
      # Diagnostic + wandb env defaults — parity with the ENV block in
      # docker/ubuntu22_04/Dockerfile's python-base / devcontainer-tools.
      # PYTORCH_CUDA_ALLOC_CONF is omitted: macOS uses MPS, not CUDA.
      "grep -qxF 'export HYDRA_FULL_ERROR=1' ~/.zprofile || printf '\\nexport HYDRA_FULL_ERROR=1\\n' >> ~/.zprofile",
      "grep -qxF 'export PYTHONDONTWRITEBYTECODE=1' ~/.zprofile || printf '\\nexport PYTHONDONTWRITEBYTECODE=1\\n' >> ~/.zprofile",
      "grep -qxF 'export PYTHONFAULTHANDLER=1' ~/.zprofile || printf '\\nexport PYTHONFAULTHANDLER=1\\n' >> ~/.zprofile",
      "grep -qxF 'export PYTHONUNBUFFERED=1' ~/.zprofile || printf '\\nexport PYTHONUNBUFFERED=1\\n' >> ~/.zprofile",
      "grep -qxF 'export WANDB_DATA_DIR=\"$HOME/.cache/wandb\"' ~/.zprofile || printf '\\nexport WANDB_DATA_DIR=\"$HOME/.cache/wandb\"\\n' >> ~/.zprofile",
    ]
  }

  # Smoke tests — mirror the two gates from Docker dev-base. No xvfb wrapper
  # is needed; the macOS VM has a native window server. One subprocess per
  # plugin because sequential in-process VST3 loads crash order-dependently
  # (#1649) — same isolation reason Docker uses the load_vst3_check.py loop.
  # Six Sines needs an explicit `plugin_name` because the bundle exposes more
  # than one factory class; Surge XT, Dexed, and OB-Xf each ship a sole
  # plugin so `plugin_name=None` is the right default.
  provisioner "shell" {
    inline = [
      "touch ~/.zprofile && . ~/.zprofile",
      "cd ~/synth-setter && .venv/bin/python -X faulthandler -c \"from synth_setter.data.vst.core import load_plugin; load_plugin('plugins/Surge XT.vst3')\"",
      "cd ~/synth-setter && .venv/bin/python -X faulthandler -c \"from synth_setter.data.vst.core import load_plugin; load_plugin('plugins/Dexed.vst3')\"",
      "cd ~/synth-setter && .venv/bin/python -X faulthandler -c \"from synth_setter.data.vst.core import load_plugin; load_plugin('plugins/OB-Xf.vst3')\"",
      "cd ~/synth-setter && .venv/bin/python -X faulthandler -c \"from synth_setter.data.vst.core import load_plugin; load_plugin('plugins/Six Sines.vst3', plugin_name='Six Sines')\"",
      "cd ~/synth-setter && .venv/bin/pytest -k 'not slow' -v",
    ]
  }
}

# ==============================================================================
# Publishing a new image to Docker Hub (tinaudio/synth-setter-macos):
#   https://hub.docker.com/repository/docker/tinaudio/synth-setter-macos/general
#
#   # 1. One-time: create a Docker Hub personal access token with
#   #    Read, Write scopes at https://hub.docker.com/settings/security
#   #    and use it as your password in step 2. Delete is not required
#   #    for the push flow documented here.
#
#   # 2. Log in (credentials are stored by tart for subsequent pushes).
#   tart login registry-1.docker.io
#   # Username: <your-dockerhub-username>
#   # Password: <access-token-from-step-1>
#
#   # 3. Build the VM (requires Apple Silicon Mac).
#
#   #   3a. Install Homebrew if you don't have it.
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
#   #   3b. Install Tart and Packer.
#   brew install cirruslabs/cli/tart
#   brew install packer
#
#   #   3c. Fetch the Tart plugin declared below.
#   packer init tart/macos.pkr.hcl
#
#   #   3d. (Optional) Sanity-check the template before building.
#   packer validate tart/macos.pkr.hcl
#
#   #   3e. Build the VM (pulls the base image on first run, then provisions).
#   #       For a reproducible image, pin the repo to a specific commit:
#   #         packer build -var "synth_setter_git_ref=<40-char-sha>" tart/macos.pkr.hcl
#   packer build tart/macos.pkr.hcl
#
#   #   3f. (Optional) Smoke-test the built VM locally before publishing.
#   tart run synth-setter-macos                          # GUI window opens
#   # In another terminal:
#   ssh admin@$(tart ip synth-setter-macos)              # password: admin
#
#   #   One-time host setup if you're on a headless Mac or want to skip the
#   #   Sequoia/Tahoe "Local Network" permission prompt during `tart run`:
#   sudo defaults write com.apple.network.local-network AllowedEthernetLocalNetworkAddresses -array "10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16"
#   sudo defaults write com.apple.network.local-network AllowedWiFiLocalNetworkAddresses -array "10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16"
#   sudo reboot
#
#   # 4. Push two tags — :latest as the moving pointer, and a dated tag as
#   #    an immutable rollback point. The local VM name comes from var.vm_name.
#   DATE_TAG="$(date -u +%Y-%m-%d)"
#   tart push synth-setter-macos \
#   registry-1.docker.io/tinaudio/synth-setter-macos:${DATE_TAG} \
#   registry-1.docker.io/tinaudio/synth-setter-macos:latest
#
#   # 5. (Optional) Free local disk once the push succeeds.
#   tart delete synth-setter-macos
#
# ==============================================================================
