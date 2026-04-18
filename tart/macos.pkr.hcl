# ==============================================================================
# Quick start — pull and run the prebuilt VM:
#
#   brew install cirruslabs/cli/tart
#   tart clone docker.io/tinaudio/synth-setter-macos:latest synth-setter-macos
#   tart run synth-setter-macos                          # GUI window opens
#
#   # In another terminal:
#   ssh admin@$(tart ip synth-setter-macos)              # password: admin
#
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

variable "torch_backend" {
  type        = string
  default     = "cpu"
  description = "Value passed to `uv pip install --torch-backend`. Tart VMs have no GPU, so keep `cpu`."
}

variable "python_version" {
  type        = string
  default     = "3.10"
  description = "Python interpreter version installed and pinned via uv."
}

variable "vm_name" {
  type        = string
  default     = "synth-setter-macos"
  description = "Name of the built Tart VM. Used with `tart run <name>` and `tart ip <name>`."
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
  # (.vst3 installed to /Library/Audio/Plug-Ins/VST3/Surge XT.vst3).
  provisioner "shell" {
    inline = [
      ". ~/.zprofile",
      "brew --version",
      "brew update",
      "brew install git gh jq rclone uv codex bats-core",
      "brew install --cask claude-code",
      "brew install --cask surge-xt",
    ]
  }

  # Install and pin Python via uv.
  provisioner "shell" {
    inline = [
      ". ~/.zprofile",
      "uv python install ${var.python_version}",
      "uv python pin ${var.python_version}",
    ]
  }

  # Clone the repo, use venv with all runtime deps (parity with Docker dev-base
  # stage).
  provisioner "shell" {
    inline = [
      ". ~/.zprofile",
      "git clone https://github.com/tinaudio/synth-setter.git ~/synth-setter",
      "cd ~/synth-setter && git checkout ${var.synth_setter_git_ref}",
      "cd ~/synth-setter && uv venv --python ${var.python_version}",
      "cd ~/synth-setter && uv pip install --torch-backend ${var.torch_backend} -r requirements.txt",
      "cd ~/synth-setter && uv pip install --no-deps -e .",
      # Auto-activate the venv for every interactive shell so tools installed
      # into .venv/bin (pre-commit, pyright, pytest, ruff, etc.) are on PATH
      # from login without a manual `source .venv/bin/activate`.
      "printf '\\nsource ~/synth-setter/.venv/bin/activate\\n' >> ~/.zshrc",
    ]
  }

  # Smoke tests — mirror the two gates from Docker dev-base. No xvfb wrapper
  # is needed; the macOS VM has a native window server.
  provisioner "shell" {
    inline = [
      ". ~/.zprofile",
      "cd ~/synth-setter && .venv/bin/python -X faulthandler -c \"from src.data.vst.core import load_plugin; load_plugin('/Library/Audio/Plug-Ins/VST3/Surge XT.vst3')\"",
      "cd ~/synth-setter && .venv/bin/pytest -k 'not slow' -v",
    ]
  }
}

# ==============================================================================
# Publishing a new image to Docker Hub (tinaudio/synth-setter-macos):
#   https://hub.docker.com/repository/docker/tinaudio/synth-setter-macos/general
#
#   # 1. One-time: create a Docker Hub personal access token with
#   #    Read, Write, Delete scopes at https://hub.docker.com/settings/security
#   #    and use it as your password in step 2.
#
#   # 2. Log in (credentials are stored by tart for subsequent pushes).
#   tart login docker.io
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
#   tart push synth-setter-macos docker.io/tinaudio/synth-setter-macos:${DATE_TAG}
#   tart push synth-setter-macos docker.io/tinaudio/synth-setter-macos:latest
#
#   # 5. (Optional) Free local disk once the push succeeds.
#   tart delete synth-setter-macos
#
# ==============================================================================
