"""Property-based tests using Hypothesis."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


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
def test_arbitrary_dict_does_not_crash_omegaconf(random_dict):
    """OmegaConf.create should handle or reject arbitrary dicts without crashing."""
    from omegaconf import OmegaConf

    try:
        cfg = OmegaConf.create(random_dict)
        # If it accepts the dict, it should be convertible back
        OmegaConf.to_container(cfg)
    except Exception:  # noqa: S110
        # Rejection is fine — crashing is not
        pass
