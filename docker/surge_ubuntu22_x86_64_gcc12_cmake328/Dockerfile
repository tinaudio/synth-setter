# syntax=docker/dockerfile:1
ARG TARGETARCH
ARG BUILD_MODE
ARG GIT_PAT
#TODO: figure out best way to restrict TARGETARCH and BUILD_MODE to valid values, 
# and error if invalid values are used. 
# ==========================================
FROM ubuntu:22.04 AS base
ENV LANG C.UTF-8
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Non-interactive configuration of tzdata
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
RUN { echo 'tzdata tzdata/Areas select Etc'; echo 'tzdata tzdata/Zones/Etc select UTC'; } | debconf-set-selections

# Create unprivileged user to build toolchains and plugins
RUN groupadd -g 1000 build
RUN useradd --create-home --uid 1000 --gid 1000 --shell /bin/bash build

# and install it
RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN apt-get update 
RUN apt-get install -y gcc-12 g++-12 git ninja-build flex

FROM base AS builder-amd64-source-cmake
RUN mkdir /home/build/cmake-install
WORKDIR /home/build/cmake-install
RUN apt-get install -y wget
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh
RUN chmod 755 ./cmake-3.28.1-linux-x86_64.sh
RUN ./cmake-3.28.1-linux-x86_64.sh --skip-license --prefix=/usr

FROM base AS builder-arm64-source-cmake
RUN mkdir /home/build/cmake-install
WORKDIR /home/build/cmake-install
RUN apt-get install -y wget
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-aarch64.sh
RUN chmod 755 ./cmake-3.28.1-linux-aarch64.sh
RUN ./cmake-3.28.1-linux-aarch64.sh --skip-license --prefix=/usr

FROM builder-${TARGETARCH}-source-cmake AS builder-common-source
# Then install the prerequistes for a juce build
RUN apt-get install -y git vim 
RUN apt-get install -y cmake ninja-build 
RUN apt-get install -y \
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
              libxcb-cursor0

RUN apt-get install -y rpm

# Finally, pick gcc12
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

RUN chmod 777 /home/build
RUN mkdir /home/build/dev
RUN mkdir /home/build/plugin
WORKDIR /home/build/plugin

FROM builder-common-source AS builder-source-install
WORKDIR /home/build/plugin
# Now clone and build surge from source
RUN git clone https://github.com/surge-synthesizer/surge.git
WORKDIR /home/build/plugin/surge
RUN git checkout release-xt/1.3.4
RUN git submodule update --init --recursive
RUN cmake -Bignore/sxt -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
RUN cmake --build ignore/sxt --config Release --parallel 8
RUN cmake --install ignore/sxt

FROM base AS builder-amd64-prebuilt
RUN apt-get update && apt-get install -y wget
RUN wget https://github.com/surge-synthesizer/releases-xt/releases/download/1.3.4/surge-xt-linux-x64-1.3.4.deb


FROM base AS builder-arm64-prebuilt
RUN echo "ERROR: arm64 prebuilt package is not currently available. Set BUILD_MODE=source." >&2 && exit 1

FROM builder-${TARGETARCH}-prebuilt AS builder-prebuilt-install
# Add any additional dependencies needed to install the prebuilt package
# TODO: check to see which of the installs below are actually needed to install the prebuilt package,  remove the ones that aren't needed.
RUN apt-get update && apt-get install -y \
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
RUN apt-get install -y ./surge-xt-linux-x64-1.3.4.deb && rm surge-xt-linux-x64-1.3.4.deb

# TODO: Add test stage with python that uses pedalboard to load to plugin.

FROM builder-${BUILD_MODE}-install AS final-stage
# Check that vst3 was installed.
RUN ls /usr/lib/vst3/ && test -n "$(ls -A /usr/lib/vst3/Surge\ XT.vst3 2>/dev/null)"  || (echo "ERROR: Surge\ XT.vst3 not found in /usr/lib/vst3/" >&2; exit 1)
# Clone the repository using the provided GIT_PAT
# TODO: use buildx secrets to pass GIT_PAT.
ARG GIT_PAT
RUN if [ -z "$GIT_PAT" ]; then echo "ERROR: GIT_PAT build-arg is required and cannot be empty." >&2; exit 1; fi
RUN git clone https://${GIT_PAT}@github.com/ktinubu/synth-permutations.git /home/build/synth-permutations
# Set the working directory to the root of the cloned repository
WORKDIR /home/build/synth-permutations
RUN git checkout experiment
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
# Install Python dependencies from the cloned repo's requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# TODO: consider using user:build instead of root
# Use root for now since some scripts require permissions to create/write to files 
# USER build