"""Property-based tests using Hypothesis."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException


@pytest.mark.hypothesis
@pytest.mark.slow
@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(st.integers(), st.text(), st.none(), st.floats(allow_nan=False)),
        min_size=0,
        max_size=20,
    )
)
@settings(max_examples=50)
def test_arbitrary_dict_does_not_crash_omegaconf(
    random_dict: dict[str, int | str | None | float],
) -> None:
    """OmegaConf.create should handle or reject arbitrary dicts without crashing."""
    try:
        cfg = OmegaConf.create(random_dict)
    except OmegaConfBaseException:
        return

    # If it accepts the dict, it should be convertible back without errors
    OmegaConf.to_container(cfg)
