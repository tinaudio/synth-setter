"""Domain identifier for dynamically registered parameter specifications.

Use ``ParamSpecName`` for runtime registry keys and
``ValidatedParamSpecName`` at Pydantic boundaries::

    def resolve(name: ParamSpecName) -> object: ...
"""

from typing import Annotated, NewType

from pydantic import AfterValidator

ParamSpecName = NewType("ParamSpecName", str)

ONSET_DURATION_PARAM_SPEC_SUFFIX = "_onset_duration"
LEGACY_ENDPOINT_PARAM_SPEC_NAMES = frozenset(
    ParamSpecName(name)
    for name in (
        "cardinal",
        "faust_bright_organ",
        "faust_bubble",
        "faust_church_organ",
        "faust_filter_osc",
        "obxf",
        "surge_4",
        "surge_simple",
        "surge_xt",
        "torchsynth_adsr",
        "torchsynth_full",
        "torchsynth_simple",
        "ultramaster_kr106",
        "ultramaster_kr106_onehot",
        "ultramaster_kr106_single_note",
    )
)


def onset_duration_param_spec_name(name: ParamSpecName) -> ParamSpecName:
    """Return the versioned identity for onset-duration timing semantics.

    :param name: Legacy endpoint parameter-spec identity.
    :returns: Corresponding onset-duration identity.
    """
    return ParamSpecName(f"{name}{ONSET_DURATION_PARAM_SPEC_SUFFIX}")


def legacy_endpoint_param_spec_name(name: ParamSpecName) -> ParamSpecName:
    """Return the parameter-map and synth-source identity shared by a timing variant.

    :param name: Legacy or onset-duration parameter-spec identity.
    :returns: Legacy identity used by renderer resources.
    """
    if name.endswith(ONSET_DURATION_PARAM_SPEC_SUFFIX):
        return ParamSpecName(name.removesuffix(ONSET_DURATION_PARAM_SPEC_SUFFIX))
    return name


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
