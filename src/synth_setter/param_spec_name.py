"""Domain identifier for dynamically registered parameter specifications.

Use ``ParamSpecName`` for runtime registry keys and
``ValidatedParamSpecName`` at Pydantic boundaries::

    def resolve(name: ParamSpecName) -> object: ...
"""

from collections.abc import Mapping
from typing import Annotated, NewType

from pydantic import AfterValidator

ParamSpecName = NewType("ParamSpecName", str)

_ONSET_DURATION_SUFFIX = "_onset_duration"


def onset_duration_param_spec_name(name: str) -> ParamSpecName:
    """Name the onset-duration twin of an endpoint-timed spec.

    :param name: Registry key of a spec whose note window is two independent endpoints.
    :returns: Registry key of the same spec with onset-duration note timing.
    """
    return ParamSpecName(f"{name}{_ONSET_DURATION_SUFFIX}")


def base_param_spec_name(name: str) -> ParamSpecName:
    """Return the name keying a spec's timing-independent artifacts.

    Parameter maps, presets, and source digests describe the synth parameters, which an
    onset-duration twin shares with the endpoint-timed spec it derives from.

    :param name: Any registry key.
    :returns: ``name`` itself, or the endpoint-timed spec an onset-duration twin derives from.
    """
    return ParamSpecName(name.removesuffix(_ONSET_DURATION_SUFFIX))


def is_onset_duration_param_spec_name(name: str) -> bool:
    """Return whether a registry key names an onset-duration twin.

    :param name: Any registry key.
    :returns: ``True`` for ``<base>_onset_duration`` names.
    """
    return name.endswith(_ONSET_DURATION_SUFFIX)


def require_trained_param_spec(
    hyperparameters: Mapping[str, object], decoding_spec_name: str
) -> None:
    """Reject decoding a checkpoint's parameter rows under a spec it was not trained on.

    A checkpoint that predates spec recording counts as the endpoint-timed spec, since
    onset-duration specs are always recorded.

    :param hyperparameters: Checkpoint hyperparameters; ``param_spec`` records the spec.
    :param decoding_spec_name: Spec about to decode the checkpoint's parameter rows.
    :raises ValueError: The checkpoint was trained on another spec.
    """
    trained = hyperparameters.get("param_spec")
    if trained is None:
        trained = base_param_spec_name(decoding_spec_name)
    if trained != decoding_spec_name:
        raise ValueError(
            f"checkpoint parameter spec {trained!r} does not match {decoding_spec_name!r}"
        )


def _reject_blank(value: ParamSpecName) -> ParamSpecName:
    """Reject blank names without changing registry identity.

    :param value: Candidate registry key from a Pydantic trust boundary.
    :returns: The original nonblank key, including any surrounding whitespace.
    :raises ValueError: If the key contains only whitespace.
    """
    if not value.strip():
        raise ValueError("param spec name must not be blank")
    return value


ValidatedParamSpecName = Annotated[ParamSpecName, AfterValidator(_reject_blank)]
