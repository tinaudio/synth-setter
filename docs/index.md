# synth-permutations config reference

This site documents the project's Hydra config schemas — the Pydantic models
that define how datasets, training runs, and pipeline stages are configured.
For now it covers a single model, the **dataset spec**, which describes a full
dataset materialization (layout, render config, and runtime provenance). More
models will be added here as the documentation effort expands; each entry in
the **Config Reference** tab is auto-generated from the model's source via
[mkdocstrings](https://mkdocstrings.github.io/), so field types and docstrings
stay in sync with the code.
