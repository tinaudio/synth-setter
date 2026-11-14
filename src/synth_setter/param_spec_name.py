"""Domain identifier for dynamically registered parameter specifications.

Use ``ParamSpecName`` for runtime registry keys and
``ValidatedParamSpecName`` at Pydantic boundaries::

    def resolve(name: ParamSpecName) -> object: ...
"""

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
