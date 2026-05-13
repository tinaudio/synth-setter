#!/usr/bin/env python

from setuptools import find_packages, setup

# Legacy `console_scripts` (`train_command`, `eval_command`) were removed in
# the Phase 2 src-layout migration (#989). Console scripts are now declared in
# `pyproject.toml` under `[project.scripts]`. Full `setup.py` deletion happens
# in Phase 5 of #784.
setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
)
