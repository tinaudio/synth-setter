# syntax=docker/dockerfile:1.6
ARG BUILD_MODE=source

# release-xt/1.3.4 SHA
ARG SURGE_GIT_REF=f7b97c682ade0b87da85ca5968b63d5c7c98e68d
# Default to main branch
ARG SYNTH_PERMUTATIONS_GIT_REF=main
# ubuntu:22.04 digest retrieved via:
#      docker pull ubuntu:22.04
#      docker inspect ubuntu:22.04 --format='{{index .RepoDigests 0}}'
ARG BASE_IMAGE=ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751
# TODO(ktinubu): Rename to something more useful and distrinct from BASE_IMAGE
# TODO(ktinubu): Check for unset TARGETARCH or TARGETPLATFORM. Fallback to amd, linux/amd but add tag to container
# TODO(ktinubu): Add appropriate image labels.
FROM ${BASE_IMAGE} AS amd64-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /tmp-artifacts
RUN echo "x86_64" > /tmp-artifacts/cmake-arch-name.txt
RUN echo "1.3.4/surge-xt-linux-x64-1.3.4.deb" > /tmp-artifacts/surge-package-filepath.txt

FROM ${BASE_IMAGE} AS arm64-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_MODE
RUN echo "TARGETARCH=${TARGETARCH:-<unset>}" >&2
RUN echo "TARGETPLATFORM=${TARGETPLATFORM:-<unset>}" >&2
WORKDIR /tmp-artifacts
RUN echo "aarch64" > /tmp-artifacts/cmake-arch-name.txt
RUN if [ "${BUILD_MODE:-}" = "prebuilt" ] && [ "${TARGETARCH:-}" = "arm64" ]; then \
      echo "ERROR: arm64 prebuilt package is not currently available. Set BUILD_MODE=source." >&2; \
      exit 1; \
    fi
FROM ${BASE_IMAGE} AS arch-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /artifacts
ARG TARGETARCH
WORKDIR /artifacts

COPY --from=arm64-vars /tmp-artifacts/ /tmp-artifacts/arm64/
COPY --from=amd64-vars /tmp-artifacts/ /tmp-artifacts/amd64/

# `cp` with Trailing dot copies everything inside the source to dest.
RUN case "${TARGETARCH:-}" in \
      arm64) cp -av  /tmp-artifacts/arm64/. /artifacts ;; \
      amd64) cp -av /tmp-artifacts/amd64/. /artifacts ;; \
      *) echo "ERROR: TARGETARCH must be amd64 or arm64 (got '${TARGETARCH:-<unset>}')" >&2; exit 1 ;; \
    esac

# TODO: add appropriate build labels
# TODO: Consider taking python runtime from `python:3.10-*` stage.
# This would involving copying`/usr/local`.
# TODO: Add diagram of stages and dependencies.
# TODO(deduplicate package lists)
# ==========================================
FROM ${BASE_IMAGE} AS builder-base
WORKDIR /home/build/
ENV LANG=C.UTF-8
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Non-interactive configuration of tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN { echo 'tzdata tzdata/Areas select Etc'; echo 'tzdata tzdata/Zones/Etc select UTC'; } | debconf-set-selections

ARG TARGETARCH
ARG BUILD_MODE
# Validate build args
RUN case "${TARGETARCH:-}" in \
      amd64|arm64) ;; \
      *) echo "ERROR: TARGETARCH must be amd64 or arm64 (got '${TARGETARCH:-<unset>}')" >&2; exit 1 ;; \
    esac
RUN case "${BUILD_MODE}" in \
      source|prebuilt) ;; \
      *) echo "ERROR: BUILD_MODE must be source or prebuilt (got '${BUILD_MODE}')" >&2; exit 1 ;; \
    esac
# Install base build tools (gcc-12, g++-12, ninja, flex) shared by all builder stages.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  gcc-12 \
  g++-12 \
  git \
  ninja-build \
  flex


# ==========================================
# Source build: installs CMake 3.28 and JUCE prerequisites, clones
# surge-synthesizer/surge at SURGE_GIT_REF, builds Surge XT from source,
# installs it system-wide, and symlinks the VST3 into /usr/lib/vst3/.
# ==========================================
FROM builder-base AS builder-install-surge-from-source
COPY --from=arch-vars /artifacts/cmake-arch-name.txt /artifacts/cmake-arch-name.txt
RUN mkdir /home/build/cmake-install
WORKDIR /home/build/cmake-install
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget


RUN cmake_arch_name="$(cat /artifacts/cmake-arch-name.txt )"; \
    echo "cmake-arch-name: ${cmake_arch_name}"; \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-${cmake_arch_name}.sh && \
    chmod 755 ./cmake-3.28.1-linux-${cmake_arch_name}.sh && \
    ./cmake-3.28.1-linux-${cmake_arch_name}.sh --skip-license --prefix=/usr
# Then install the prerequistes for a juce build
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        vim \
        cmake \
        ninja-build \
              devscripts \
              libx11-dev \
              libxcb-cursor-dev \
              libxcb-keysyms1-dev \
              libxcb-util-dev \
              libxkbcommon-dev \
              libxkbcommon-x11-dev \
              xcb \
              libgtk-3-dev \
              libwebkit2gtk-4.0 \
              libwebkit2gtk-4.0-dev \
              libcurl4-openssl-dev \
              alsa \
              alsa-tools \
              libasound2-dev \
              libjack-dev \
              libfreetype-dev \
              libfontconfig1-dev \
              libfontconfig1-dev \
              libxinerama-dev \
              libxcb-xinerama0 \
              libxinerama1 \
              x11proto-xinerama-dev \
              libxrandr-dev \
              libgl1-mesa-dev \
              libxcursor-dev \
              libxcursor1 \
              libxcb-cursor-dev \
              libxcb-cursor0 \
              rpm

# Finally, pick gcc12
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

RUN chmod 777 /home/build
RUN mkdir /home/build/dev
RUN mkdir /home/build/plugin
WORKDIR /home/build/plugin

WORKDIR /home/build/plugin
# Now clone and build surge from source
RUN git clone https://github.com/surge-synthesizer/surge.git
WORKDIR /home/build/plugin/surge
ARG SURGE_GIT_REF
RUN git checkout --detach ${SURGE_GIT_REF}
RUN git submodule update --init --recursive
RUN --mount=type=cache,id=surge-cmake-build,target=/home/build/plugin/surge/ignore/sxt \
    cmake -Bignore/sxt -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
RUN --mount=type=cache,id=surge-cmake-build,target=/home/build/plugin/surge/ignore/sxt \
    cmake --build ignore/sxt --config Release --parallel $(nproc)
RUN --mount=type=cache,id=surge-cmake-build,target=/home/build/plugin/surge/ignore/sxt \
    cmake --install ignore/sxt
RUN cmake_arch_name="$(cat /artifacts/cmake-arch-name.txt )";  \
  mkdir -p /usr/lib/vst3; \
  # Check both multiarch vst3 dirs for the installed Surge XT.vst3 and link it to /usr/lib/vst3/Surge XT.vst3
  if [ -e "/usr/lib/${cmake_arch_name}-linux-gnu/vst3/Surge XT.vst3" ]; then \
    ln -sfn "/usr/lib/${cmake_arch_name}-linux-gnu/vst3/Surge XT.vst3" "/usr/lib/vst3/Surge XT.vst3"; \
  else \
    echo "ERROR: Surge XT.vst3 not found in /usr/lib/${cmake_arch_name}-linux-gnu/ vst3 dirs" >&2; \
    exit 1; \
  fi
RUN rm -rf /home/build/plugin/surge /home/build/cmake-install

# ==========================================
# Prebuilt install: downloads the Surge XT .deb for the target arch and
# installs it together with its runtime dependencies.
# ==========================================
FROM builder-base AS builder-install-surge-from-prebuilt
WORKDIR /home/build
COPY --from=arch-vars /artifacts/surge-package-filepath.txt /surge-package-filepath.txt
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget
RUN pkg="$(cat /surge-package-filepath.txt)"; \
    echo $pkg; \
    wget "https://github.com/surge-synthesizer/releases-xt/releases/download/${pkg}"

# Add any additional dependencies needed to install the prebuilt package
# TODO: check to see which of the installs below are actually needed to install the prebuilt package,  remove the ones that aren't needed.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libcairo-dev \
    libxkbcommon-x11-dev \
    libxkbcommon-dev \
    libxcb-cursor-dev \
    libxcb-keysyms1-dev \
    libxcb-util-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libasound2-dev \
    libjack-jackd2-dev
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    pkg="$(cat /surge-package-filepath.txt)" && \
    fname="${pkg##*/}" && \
    apt-get update && apt-get install -y --no-install-recommends "./${fname}" && \
    rm "./${fname}"

# ==========================================
# VST3 synth fetch: SHA256-pinned prebuilt Linux VST3 bundles, staged under
# /plugins-staging for the runtime stage to copy. Upstream ships x86_64-only
# binaries, so non-amd64 builds stage nothing and downstream steps skip.
# Version pins are glibc-bound: Ubuntu 22.04 ships glibc 2.35, so assets
# built against glibc 2.38 (Ubuntu 24.04 runners) cannot load — that rules
# out Dexed > 0.9.8 and every OB-Xd / Ultramaster KR-106 Linux release.
# ==========================================
FROM ${BASE_IMAGE} AS vst3-synths-fetch
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ARG TARGETARCH
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    unzip \
    wget
WORKDIR /plugins-staging

ARG DEXED_VERSION=0.9.8
ARG DEXED_SHA256=5d026f53504f9303ae2a4a635cf6fdfc50ab9c947cbc0a20ecb5c8f323402dab
RUN if [ "${TARGETARCH:-}" != "amd64" ]; then echo "skipping Dexed: x86_64-only asset" >&2; exit 0; fi; \
    wget -q -O /tmp/dexed.zip "https://github.com/asb2m10/dexed/releases/download/v${DEXED_VERSION}/dexed-${DEXED_VERSION}-lnx.zip" && \
    echo "${DEXED_SHA256}  /tmp/dexed.zip" | sha256sum -c - && \
    unzip -q /tmp/dexed.zip -d /tmp/dexed && \
    mv "/tmp/dexed/dexed-${DEXED_VERSION}-lnx/Dexed.vst3" . && \
    rm -rf /tmp/dexed.zip /tmp/dexed

ARG VITAL_VERSION=1_5_5
ARG VITAL_SHA256=68f3c7e845f3d7a5b44a83adeb6e34ef221503df00e7964f7d5a1f132a252d13
RUN if [ "${TARGETARCH:-}" != "amd64" ]; then echo "skipping Vital: x86_64-only asset" >&2; exit 0; fi; \
    wget -q -O /tmp/vital.zip "https://builds.vital.audio/VitalAudio/vital/${VITAL_VERSION}/VitalInstaller.zip" && \
    echo "${VITAL_SHA256}  /tmp/vital.zip" | sha256sum -c - && \
    unzip -q /tmp/vital.zip "VitalInstaller/lib/vst3/Vital.vst3/*" -d /tmp/vital && \
    mv "/tmp/vital/VitalInstaller/lib/vst3/Vital.vst3" . && \
    rm -rf /tmp/vital.zip /tmp/vital

ARG SIX_SINES_VERSION=v1.1.0
ARG SIX_SINES_ASSET=six-sines-linux-2025-03-18-43d10b2.tgz
ARG SIX_SINES_SHA256=fae7c1c325fde7ed49c978358397cb4bcf69012c4e6eefe2a5968fe6a36d0421
RUN if [ "${TARGETARCH:-}" != "amd64" ]; then echo "skipping Six Sines: x86_64-only asset" >&2; exit 0; fi; \
    wget -q -O /tmp/six-sines.tgz "https://github.com/baconpaul/six-sines/releases/download/${SIX_SINES_VERSION}/${SIX_SINES_ASSET}" && \
    echo "${SIX_SINES_SHA256}  /tmp/six-sines.tgz" | sha256sum -c - && \
    tar -xzf /tmp/six-sines.tgz -C . "./Six Sines.vst3" && \
    rm /tmp/six-sines.tgz

ARG CARDINAL_VERSION=26.02
ARG CARDINAL_SHA256=657df0beeec04184de7359cbd3e173a36eeab78077e1e26da81405632f98ec25
RUN if [ "${TARGETARCH:-}" != "amd64" ]; then echo "skipping Cardinal: x86_64-only asset" >&2; exit 0; fi; \
    wget -q -O /tmp/cardinal.tgz "https://github.com/DISTRHO/Cardinal/releases/download/${CARDINAL_VERSION}/Cardinal-linux-x86_64-${CARDINAL_VERSION}.tar.gz" && \
    echo "${CARDINAL_SHA256}  /tmp/cardinal.tgz" | sha256sum -c - && \
    tar -xzf /tmp/cardinal.tgz -C . "Cardinal.vst3" && \
    rm /tmp/cardinal.tgz

# ==========================================
# Source stage: download synth-setter at a specific git ref via tarball.
# Uses GitHub archive endpoint — no full clone, no .git metadata.
# ==========================================
FROM builder-base AS synth-setter-src

ARG SYNTH_PERMUTATIONS_GIT_REF
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl
WORKDIR /home/build/synth-setter
# Best practice: use a commit SHA for SYNTH_PERMUTATIONS_GIT_REF for deterministic builds.
# Anonymous API access — the repository is public.
RUN curl -fsSL \
      "https://api.github.com/repos/tinaudio/synth-setter/tarball/${SYNTH_PERMUTATIONS_GIT_REF}" \
      -o /tmp/src.tgz && \
    tar -xzf /tmp/src.tgz --strip-components=1 && \
    rm /tmp/src.tgz


# ==========================================
# Python base: Python + venv on top of the chosen Surge install.
# ==========================================
FROM builder-install-surge-from-${BUILD_MODE} AS python-base
# Check that vst3 was installed.
RUN ls /usr/lib/vst3/ && test -n "$(ls -A /usr/lib/vst3/Surge\ XT.vst3 2>/dev/null)" \
    || (echo "ERROR: Surge XT.vst3 not found in /usr/lib/vst3/" >&2; exit 1)

# Point tests and ensure_plugin_symlinks.sh at the system VST3 path baked above,
# so container callers don't have to pass `-e SYNTH_SETTER_PLUGIN_PATH=...`.
ENV SYNTH_SETTER_PLUGIN_PATH="/usr/lib/vst3/Surge XT.vst3"

# Install Python and create venv
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev
RUN python3 --version && \
    python3 -c "import sys; exit(0) if sys.version_info[:2] == (3,10) else exit(1)"
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/
ENV VIRTUAL_ENV=/venv/main
# HYDRA_FULL_ERROR=1 — full Python traceback instead of Hydra's trimmed output
# PYTHONDONTWRITEBYTECODE=1 — no .pyc files written in the image
# PYTHONFAULTHANDLER=1 — C/CUDA segfault traceback (pairs with HYDRA_FULL_ERROR)
# PYTHONUNBUFFERED=1 — real-time stdout/stderr under Docker/SLURM log capture
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — reduces CUDA allocator fragmentation (no-op on CPU)
ENV HYDRA_FULL_ERROR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python --version && \
    python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

# ==========================================
# Shared runtime base: Python + runtime deps on top of python-base.
# Both dev and final-stage inherit from this stage.
# ==========================================
FROM python-base AS builder-install-synth-setter-deps
# Install Python dependencies via `uv sync --frozen` against the committed
# lockfile. uv's cache (mounted below) persists torch wheels (~2.5 GB)
# across builds. Cache scope: only pyproject.toml + uv.lock + README.md
# (referenced by [project].readme) enter this layer, so unrelated source
# edits do not bust the torch-wheels cache. The .github/workflows/
# uv-lock-check.yml matrix guards the upstream invariant.
#
# TORCH_BACKEND selects the `cpu`/`cu128` extra → pytorch-* index per
# [tool.uv.sources]; build-arg from docker-build-validation.yml.
# UV_PROJECT_ENVIRONMENT points uv at /venv/main (created in python-base).
COPY --from=synth-setter-src /home/build/synth-setter/pyproject.toml /artifacts/
COPY --from=synth-setter-src /home/build/synth-setter/uv.lock /artifacts/
COPY --from=synth-setter-src /home/build/synth-setter/README.md /artifacts/
ARG TORCH_BACKEND=cu128
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /artifacts && \
    UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV \
    uv sync --frozen --extra ${TORCH_BACKEND} --no-default-groups --group dev --no-install-project && \
    rm -rf /artifacts/pyproject.toml /artifacts/uv.lock /artifacts/README.md

# Runtime apt packages: headless VST loading, rclone for R2 uploads, and the
# toolchain SkyPilot's RunPod backend needs to provision and stage file_mounts
# over SSH (rsync, openssh-client) plus a system pip3 (sky shells out to `pip3`
# directly, which the uv-managed venv at /venv/main does not expose).
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      xvfb \
      xauth \
      dbus-x11 \
      xsettingsd \
      libgl1-mesa-dri \
      libgl1-mesa-glx \
      mesa-utils \
      openbox \
      x11-utils \
      x11-xserver-utils \
      rclone \
      rsync \
      openssh-client \
      python3-pip; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Install pueue task queue manager (https://github.com/Nukesor/pueue). Static
# musl binaries from the upstream release page — no apt repo and no rust
# toolchain pulled into the image. Used by `scripts/ci/job_queue.py` as the
# job-queue backend.
#
# Integrity: each binary is verified against a pinned SHA256 (per arch) before
# chmod. Bump these alongside PUEUE_VERSION — values come from running
# `sha256sum pueue-* pueued-*` against the upstream release assets.
ARG PUEUE_VERSION=v4.0.4
ARG PUEUE_SHA256_X86_64=c1b10d7e4e62211075ddd0e1dc3e8cbfc5a43d662cb3be7402a28504e23fcb51
ARG PUEUED_SHA256_X86_64=5afeff6adbafb909e8d54e2caff158e6966c2adffa2c09e60fd631cc51b60390
ARG PUEUE_SHA256_AARCH64=759bf5100a51024997111c6913aaf3330a0cdfd893ff552dcf429ae9b5e01e09
ARG PUEUED_SHA256_AARCH64=332c5ef74270b64aeaf04894c8c04826f3422eb7d50dbd1a8e0706d74a42f653
RUN case "${TARGETARCH:-}" in \
      amd64) pueue_arch=x86_64;  pueue_sha=${PUEUE_SHA256_X86_64};  pueued_sha=${PUEUED_SHA256_X86_64}  ;; \
      arm64) pueue_arch=aarch64; pueue_sha=${PUEUE_SHA256_AARCH64}; pueued_sha=${PUEUED_SHA256_AARCH64} ;; \
      *) echo "ERROR: unsupported TARGETARCH for pueue: ${TARGETARCH:-<unset>}" >&2; exit 1 ;; \
    esac && \
    base="https://github.com/Nukesor/pueue/releases/download/${PUEUE_VERSION}" && \
    wget -q -O /usr/local/bin/pueue  "${base}/pueue-${pueue_arch}-unknown-linux-musl"  && \
    wget -q -O /usr/local/bin/pueued "${base}/pueued-${pueue_arch}-unknown-linux-musl" && \
    echo "${pueue_sha}  /usr/local/bin/pueue"   | sha256sum -c - && \
    echo "${pueued_sha}  /usr/local/bin/pueued" | sha256sum -c - && \
    chmod +x /usr/local/bin/pueue /usr/local/bin/pueued && \
    pueue --version && \
    pueued --version

WORKDIR /artifacts
COPY --from=synth-setter-src /home/build/synth-setter/src/synth_setter/scripts/run-linux-vst-headless.sh /artifacts/
COPY --from=synth-setter-src /home/build/synth-setter/src/synth_setter/data/vst/core.py /artifacts/
RUN chmod +x ./run-linux-vst-headless.sh
# Validate that Surge XT loads correctly under headless X11
RUN ./run-linux-vst-headless.sh \
    python -X faulthandler -c "from core import load_plugin; load_plugin('/usr/lib/vst3/Surge XT.vst3')"

RUN mkdir -p /home/build/synth-setter/plugins && \
    ln -s "/usr/lib/vst3/Surge XT.vst3" "/home/build/synth-setter/plugins/Surge XT.vst3"

COPY --from=vst3-synths-fetch /plugins-staging/ /usr/lib/vst3/
COPY --from=synth-setter-src /home/build/synth-setter/src/synth_setter/scripts/load_vst3_check.py /artifacts/

# Validate each fetched synth loads under headless X11, then symlink it into
# plugins/ like Surge XT above (no-op on non-amd64 — nothing staged). One
# wrapper invocation per plugin: sequential in-process loads crash
# order-dependently. "bundle|plugin_name" pins multi-plugin bundles.
RUN if [ "${TARGETARCH:-}" != "amd64" ]; then echo "skipping VST3 synth validation on ${TARGETARCH:-<unset>}" >&2; exit 0; fi; \
    for entry in "Dexed|" "Vital|" "Six Sines|Six Sines" "Cardinal|"; do \
      name="${entry%%|*}"; \
      ./run-linux-vst-headless.sh \
        python -X faulthandler load_vst3_check.py "/usr/lib/vst3/${name}.vst3" "${entry#*|}"; \
      ln -s "/usr/lib/vst3/${name}.vst3" "/home/build/synth-setter/plugins/${name}.vst3"; \
    done

WORKDIR /home/build/synth-setter

# ==========================================
# dev-base: deps + live git clone at SYNTH_PERMUTATIONS_GIT_REF.
#
# This image contains NO baked credentials and no R2 bucket binding —
# it is safe to publish on public registries. Callers provide R2 +
# W&B credentials and the target R2 bucket at runtime via env vars
# (see docs/reference/docker.md § Runtime secrets).
#
# Useful for CI or one-off runs without mounting local source.
# Usage: make docker-build-dev-snapshot GIT_REF=<sha>
# ==========================================
FROM builder-install-synth-setter-deps AS dev-base

ARG SYNTH_PERMUTATIONS_GIT_REF
ENV SYNTH_PERMUTATIONS_GIT_REF=${SYNTH_PERMUTATIONS_GIT_REF}

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends git ca-certificates jq

WORKDIR /home/build/synth-setter

# The repository is public, so git fetch works anonymously. No credential
# needs to enter or leave the image.
#
# We use init+remote+fetch instead of `git clone .` because an upstream
# stage (`builder-install-synth-setter-deps`, inherited by this target)
# already creates `plugins/Surge XT.vst3` under this WORKDIR. `git clone`
# refuses to clone into a non-empty directory, whereas `git init` is
# happy to create `.git/` alongside the existing plugins symlink.
# See: https://github.com/tinaudio/synth-setter/pull/567 commit 8143856
# which attempted the clone refactor and failed with curl exit 128.
RUN git init && \
    git remote add origin "https://github.com/tinaudio/synth-setter.git" && \
    git fetch origin && \
    git checkout -B main "${SYNTH_PERMUTATIONS_GIT_REF}" && \
    git branch --set-upstream-to=origin/main main

# Install the project package so `import src` works from any working directory.
# --no-deps pairs with the `--no-install-project` on the earlier `uv sync` above —
# deps are already in the venv; this step installs the package editable.
RUN uv pip install --no-deps -e .

# Run not-slow unit tests (mocked subprocess + LocalFakeUploader — no VST or R2 needed)
RUN pytest -k "not slow" -v

# No ENTRYPOINT: SkyPilot's RunPod backend launches the pod with
# `dockerArgs: "bash -c '<base64-setup>'"`, and a baked-in ENTRYPOINT would
# prepend the entrypoint program, turning that into
#   <entrypoint> bash -c '...'
# which crashes before SkyPilot can SSH in. Default CMD is /bin/bash so
# `docker run img` (no args) lands in a shell. Callers invoke specific
# console scripts (`synth-setter-generate-dataset`, etc.) explicitly.
CMD ["/bin/bash"]

# ==========================================
# devcontainer-tools: dev-base + CLI tools + non-root user.
# Published as tinaudio/synth-setter:devcontainer-tools.
# ==========================================
FROM dev-base AS devcontainer-tools

# --- CLI tools (as root) ---
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      curl \
      tmux \
  && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
  && apt-get update && apt-get install -y --no-install-recommends gh

# --- Node.js (system-wide) ---
# CLI agents (claude-code, codex) install per-user below: a root-owned /usr/local/bin
# copy would shadow dev's auto-updated ~/.npm-global copy in PATH, pinning the version.
COPY --from=node:20-bullseye-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:20-bullseye-slim /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && node --version \
 && npm --version

# Zellij terminal multiplexer (https://github.com/zellij-org/zellij) — not in
# the Ubuntu 22.04 apt repos, so install the upstream static musl binary,
# verified against a pinned per-arch SHA256 (run `sha256sum` on the release
# tarballs to refresh these when bumping ZELLIJ_VERSION).
ARG TARGETARCH
ARG ZELLIJ_VERSION=v0.44.3
ARG ZELLIJ_SHA256_X86_64=0f7c346788627f506c0a28296517768633cff24fc822a739f8264b640ecad751
ARG ZELLIJ_SHA256_AARCH64=15e6534d42644d66973d136c590c49739dcfd6a1a2a0d3d917973f16c81b45fb
RUN case "${TARGETARCH:-}" in \
      amd64) zellij_arch=x86_64;  zellij_sha=${ZELLIJ_SHA256_X86_64}  ;; \
      arm64) zellij_arch=aarch64; zellij_sha=${ZELLIJ_SHA256_AARCH64} ;; \
      *) echo "ERROR: unsupported TARGETARCH for zellij: ${TARGETARCH:-<unset>}" >&2; exit 1 ;; \
    esac && \
    url="https://github.com/zellij-org/zellij/releases/download/${ZELLIJ_VERSION}/zellij-${zellij_arch}-unknown-linux-musl.tar.gz" && \
    wget -q -O /tmp/zellij.tar.gz "$url" && \
    echo "${zellij_sha}  /tmp/zellij.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/zellij.tar.gz -C /usr/local/bin zellij && \
    rm /tmp/zellij.tar.gz && \
    chmod +x /usr/local/bin/zellij && \
    zellij --version

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash -m $USERNAME
RUN chown -R $USER_UID:$USER_GID .git
# Directories only: pip writes are governed by the parent dir's permission bit,
# and skipping files avoids overlayfs copy-up of ~2.5 GB of prebuilt deps.
RUN find /venv/main -type d -exec chown $USER_UID:$USER_GID {} +

# Persist bash history for non-root dev: user
RUN mkdir -p /commandhistory \
 && touch /commandhistory/.bash_history \
 && chown -R "$USERNAME:$USERNAME" /commandhistory \
 && printf '%s\n' \
      "export PROMPT_COMMAND='history -a'" \
      "export HISTFILE=/commandhistory/.bash_history" \
      >> "/home/$USERNAME/.bashrc"

# --- Non-root dev user ---
USER $USERNAME

# Appended, not prepended: root-default devcontainer sessions must not resolve
# these dev-writable dirs ahead of system binaries.
ENV PATH="$PATH:/home/${USERNAME}/.npm-global/bin:/home/${USERNAME}/.local/bin"

# wandb's default staging dir (~/.local/share/wandb) isn't writable at runtime in
# the devcontainer (see #1423); stage under .cache, which wandb creates on demand.
ENV WANDB_DATA_DIR="/home/${USERNAME}/.cache/wandb"

# dev's `npm install -g` would EACCES on the root-owned global tree (/usr/local);
# a per-user prefix sends it to a writable one, so the in-app updater can replace
# claude-code in place. $HOME is unset after USER, so export it per-RUN (not ENV —
# that leaks to root sessions).
ARG CLAUDE_CODE_VERSION=latest
ARG CODEX_VERSION=latest
RUN export HOME="/home/${USERNAME}" \
 && npm config set prefix "/home/${USERNAME}/.npm-global" \
 && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
 && npm install -g "@openai/codex@${CODEX_VERSION}" \
 && npm cache clean --force \
 && claude --version \
 && codex --version

# Antigravity (Google) ships the standalone `agy` binary, not on npm; its
# installer fetches a SHA512-verified release into ~/.local/bin (upstream
# latest — no version flag). Fetched to a file, not piped to bash, so a curl
# failure is distinguishable from a script failure.
RUN export HOME="/home/${USERNAME}" \
 && curl -fsSL https://antigravity.google/cli/install.sh -o /tmp/agy-install.sh \
 && bash /tmp/agy-install.sh \
 && rm /tmp/agy-install.sh \
 && agy --version

FROM dev-base AS freeze-deps
# Record package versions for audit / reproducibility
RUN mkdir -p /usr/local/share/audit && \
    uv pip freeze | sort > /usr/local/share/audit/pip-freeze.txt && \
    dpkg-query -W -f='${Package}=${Version}\n' | sort > /usr/local/share/audit/dpkg.txt

FROM dev-base AS dev-snapshot
