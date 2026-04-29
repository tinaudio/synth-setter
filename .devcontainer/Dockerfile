FROM tinaudio/synth-setter:devcontainer-tools

# Extension point for per-developer customizations.
# All tools are baked into tinaudio/synth-setter:devcontainer-tools
# (built from docker/ubuntu22_04/Dockerfile, devcontainer-tools stage).
#
# To add a package without modifying the base image:
#
#   USER root
#   RUN apt-get update && apt-get install -y <package> \
#     && rm -rf /var/lib/apt/lists/*
#   USER dev
