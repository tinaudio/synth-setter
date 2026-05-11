# Dataset Spec

`DatasetSpec` is the unified Pydantic model that describes a dataset
materialization end-to-end: layout, render configuration, and the runtime
provenance fields (git SHA, run ID, R2 prefix, creation timestamp) that get
filled in when a spec is first constructed and then preserved on round-trips
through R2. The same model serves as both the YAML-shaped config that Hydra
composes and the materialized artifact that workers re-validate from JSON, so
it is the single trust boundary for dataset configuration.

::: pipeline.schemas.spec.DatasetSpec
