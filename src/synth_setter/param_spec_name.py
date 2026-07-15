"""Domain identifier for dynamically registered parameter specifications.

Use ``ParamSpecName`` for runtime registry keys and
``ValidatedParamSpecName`` at Pydantic boundaries::

    def resolve(name: ParamSpecName) -> object: ...
"""

from typing import Annotated, NewType, TypeAlias

from pydantic import AfterValidator

ParamSpecName = NewType("ParamSpecName", str)


def _reject_blank(value: ParamSpecName) -> ParamSpecName:
    """Reject blank names without changing registry identity.

    :param value: Candidate registry key from a Pydantic trust boundary.
    :returns: The original nonblank key, including any surrounding whitespace.
    :raises ValueError: If the key contains only whitespace.
    """
    if not value.strip():
        raise ValueError("param spec name must not be blank")
    return value


ValidatedParamSpecName: TypeAlias = Annotated[ParamSpecName, AfterValidator(_reject_blank)]
