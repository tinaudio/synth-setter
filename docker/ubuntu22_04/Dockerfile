# syntax=docker/dockerfile:1.6
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_MODE=source

# release-xt/1.3.4 SHA
ARG SURGE_GIT_REF=f7b97c682ade0b87da85ca5968b63d5c7c98e68d
# Experiment branch commit SHA
ARG SYNTH_PERMUTATIONS_GIT_REF=experiment
ARG BASE_IMAGE=ubuntu:22.04
# TODO(khaledt): Rename to something more useful and distrinct from BASE_IMAGE
ARG IMAGE=dev-snapshot

FROM ${BASE_IMAGE} AS amd64-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /tmp-artifacts
RUN echo "x86_64" > /tmp-artifacts/cmake-arch-name.txt
RUN echo "1.3.4/surge-xt-linux-x64-1.3.4.deb" > /tmp-artifacts/surge-package-filepath.txt

FROM ${BASE_IMAGE} AS arm64-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ARG TARGETARCH
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
# TODO: Consider taking python runtime from `python:3.10-*` stage.
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
ARG IMAGE
# Validate build args
RUN case "${TARGETARCH:-}" in \
      amd64|arm64) ;; \
      *) echo "ERROR: TARGETARCH must be amd64 or arm64 (got '${TARGETARCH:-<unset>}')" >&2; exit 1 ;; \
    esac
RUN case "${BUILD_MODE}" in \
      source|prebuilt) ;; \
      *) echo "ERROR: BUILD_MODE must be source or prebuilt (got '${BUILD_MODE}')" >&2; exit 1 ;; \
    esac
RUN case "${IMAGE}" in \
      "dev-snapshot" | "dev-live" | "prod") ;; \
      *) echo "ERROR: IMAGE must be dev-snapshot, dev-live, or prod (got '${IMAGE}')" >&2; exit 1 ;; \
    esac
RUN --mount=type=secret,id=git_pat \
    set -eu; \
    if [ ! -s /run/secrets/git_pat ]; then \
        echo "ERROR: git_pat secret is missing or empty" >&2; \
        exit 1; \
    fi;

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
    apt-get update && apt-get install -y --no-install-recommends ./surge-xt-linux-x64-1.3.4.deb \
    && rm surge-xt-linux-x64-1.3.4.deb
RUN ls

# ==========================================
# Source stage: download synth-permutations at a specific git ref via tarball.
# Uses GitHub archive endpoint — no full clone, no .git metadata.
# ==========================================
FROM builder-base AS synth-permutations-src

ARG SYNTH_PERMUTATIONS_GIT_REF
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl
WORKDIR /home/build/synth-permutations
# Best practice: use a commit SHA for SYNTH_PERMUTATIONS_GIT_REF for deterministic builds.
RUN --mount=type=secret,id=git_pat \
    set +x; \
    token="$(cat /run/secrets/git_pat)"; \
    curl -fsSL \
      -H "Authorization: Bearer ${token}" \
      "https://api.github.com/repos/ktinubu/synth-permutations/tarball/${SYNTH_PERMUTATIONS_GIT_REF}" \
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

# Install Python and create venv
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip
RUN python3 --version && \
    python3 -c "import sys; exit(0) if sys.version_info[:2] == (3,10) else exit(1)"
ENV VIRTUAL_ENV=/venv/main
# PIP_DISABLE_PIP_VERSION_CHECK=1 — suppresses version check noise in all layers
# PYTHONDONTWRITEBYTECODE=1 — no .pyc files written in the image
# PYTHONUNBUFFERED=1 — stdout/stderr stream immediately (important for logs)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip virtualenv \
    && python3 -m virtualenv $VIRTUAL_ENV \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python --version && \
    python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

# ==========================================
# Wheels stage: build wheels for all Python deps (online, with index access).
# TODO(khaled): Consider  separate RUN layers so the torch download cache is
# independent of app (changing requirements-app.txt won't re-download torch wheels.
# ==========================================
FROM python-base AS wheels
COPY --from=synth-permutations-src /home/build/synth-permutations/requirements*.txt /artifacts/
ARG TORCH_INDEX_URL
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip wheel \
        --wheel-dir /wheels \
        --extra-index-url ${TORCH_INDEX_URL} \
        -r /artifacts/requirements.txt

# ==========================================
# Shared runtime base: Python + runtime deps on top of python-base.
# Both dev and final-stage inherit from this stage.
# ==========================================
FROM python-base AS builder-install-synth-permutations-deps
# Install Python dependencies from pre-built wheels (offline, deterministic).
# Use --mount=from=wheels to bind the wheels stage's /wheels dir during the RUN
# without copying it into a layer (avoids the COPY+rm layer bloat anti-pattern).
COPY --from=synth-permutations-src /home/build/synth-permutations/requirements*.txt /artifacts/
RUN --mount=from=wheels,source=/wheels,target=/wheels \
    pip install \
        --no-index \
        --find-links=/wheels \
        -r /artifacts/requirements.txt && \
    rm -rf /artifacts/requirements*.txt

# Runtime apt packages for headless VST loading
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    xauth \
    dbus-x11 \
    xsettingsd \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    mesa-utils \
    openbox \
    x11-utils \
    x11-xserver-utils

WORKDIR /home/build/synth-permutations

# ==========================================
# Dev target: Surge + Python deps pre-installed, no baked-in source code.
# Usage: make docker-build-dev-live GIT_PAT=<token> && make docker-run-dev
# TODO: consider a lighter dev base without Surge for faster builds when VST is not needed.
# ==========================================
FROM builder-install-synth-permutations-deps AS dev-live
CMD ["bash"]

# ==========================================
# prod: full self-contained image — deps + baked-in source at SYNTH_PERMUTATIONS_GIT_REF.
# ==========================================
# TODO: Add linting.
FROM builder-install-synth-permutations-deps AS prod

# Copy full source (after deps for caching)
COPY --from=synth-permutations-src /home/build/synth-permutations /home/build/synth-permutations

# ==========================================
# dev-snapshot: deps + live git clone at SYNTH_PERMUTATIONS_GIT_REF.
# Useful for CI or one-off runs without mounting local source.
# Usage: make docker-build-dev-snapshot GIT_REF=<sha> GIT_PAT=<token>
# ==========================================
FROM builder-install-synth-permutations-deps AS dev-snapshot

ARG SYNTH_PERMUTATIONS_GIT_REF

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends git ca-certificates

WORKDIR /home/build/synth-permutations

RUN --mount=type=secret,id=git_pat \
    set +x; \
    token="$(cat /run/secrets/git_pat)"; \
    git clone https://${token}@github.com/ktinubu/synth-permutations.git . && \
    git checkout --detach "${SYNTH_PERMUTATIONS_GIT_REF}"

FROM ${IMAGE} AS test
RUN chmod +x scripts/run-linux-vst-headless.sh
RUN scripts/run-linux-vst-headless.sh \
    python -X faulthandler -c "from src.data.vst import load_plugin; load_plugin('/usr/lib/vst3/Surge XT.vst3')"
# Record package versions
RUN mkdir -p /usr/local/share/audit && \
    python -m pip freeze --all | sort > /usr/local/share/audit/pip-freeze.txt && \
    dpkg-query -W -f='${Package}=${Version}\n' | sort > /usr/local/share/audit/dpkg.txt
