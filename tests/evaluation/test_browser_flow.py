"""Validate browser prediction transport before native rendering."""

import pytest
from pydantic import ValidationError

from synth_setter.evaluation.browser_flow import BrowserPrediction


@pytest.mark.parametrize("params", [[float("nan")], [float("inf")], ["0.5"]])
def test_browser_prediction_invalid_numbers_rejected(params: list[object]) -> None:
    """Nonfinite and coerced values cannot reach the renderer.

    :param params: Untrusted prediction values.
    """
    with pytest.raises(ValidationError):
        BrowserPrediction.model_validate(
            {"token": "session", "params": params, "sketch_groups": []}
        )


@pytest.mark.parametrize("token", ["tökén", "", "bad token"])
def test_browser_prediction_malformed_session_token_rejected(token: str) -> None:
    """Reject capabilities that cannot be emitted by the URL-safe session generator.

    :param token: Malformed capability from an untrusted browser request.
    """
    with pytest.raises(ValidationError, match="token"):
        BrowserPrediction.model_validate({"token": token, "params": [0.1], "sketch_groups": []})


@pytest.mark.parametrize("groups", ["pitch", [1], [["pitch"]], None])
def test_browser_prediction_malformed_sketch_groups_rejected(groups: object) -> None:
    """The page must report the applied groups as a list of names, never coerced.

    :param groups: Untrusted group selection from a browser request.
    """
    with pytest.raises(ValidationError, match="sketch_groups"):
        BrowserPrediction.model_validate(
            {"token": "session", "params": [0.1], "sketch_groups": groups}
        )


def test_browser_prediction_missing_sketch_groups_rejected() -> None:
    """A prediction that omits the applied groups cannot be attributed in provenance."""
    with pytest.raises(ValidationError, match="sketch_groups"):
        BrowserPrediction.model_validate({"token": "session", "params": [0.1]})
