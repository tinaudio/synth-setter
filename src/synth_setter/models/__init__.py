"""Flow matching and baseline models for synth parameter estimation."""

from synth_setter.models.dynamo_typecheck import install_dynamo_typecheck_bypass

# Every compiled module traces through jaxtyped code, so install before any model is built.
install_dynamo_typecheck_bypass()
