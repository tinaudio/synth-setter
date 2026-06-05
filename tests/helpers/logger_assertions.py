"""Crash-gate assertion over a loguru ``logger`` patched onto a module.

loguru output does not propagate to pytest's ``caplog``, so a test that wants
to prove a render worker never logged a crash has to stub the module's
``logger`` with a recording mock and inspect its calls. Two VST end-to-end
tests (``tests/data/vst/test_fake_plugin_e2e.py`` and
``tests/data/vst/test_always_on_integration.py``) share this idiom; this
module is its single owner.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@contextmanager
def assert_no_logger_exceptions(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> Iterator[MagicMock]:
    """Patch ``module.logger`` with a recording mock and assert no ``.exception`` fired.

    The mock wraps the real logger, so log lines still emit while their calls
    are recorded. On clean exit the body must not have triggered
    ``logger.exception`` — the structural crash gate that ``caplog`` cannot
    observe for loguru. Only ``.exception`` is gated, matching what the call
    sites assert; the yielded mock exposes the rest for finer checks.

    :param monkeypatch: Active pytest monkeypatch fixture; the patch is reverted
        when it tears down.
    :param module: Module whose ``logger`` attribute is stubbed for the body.
    :yields: The recording mock, so callers may assert on other logger calls.
    :ytype: MagicMock
    """
    fake_logger = MagicMock(wraps=module.logger)
    monkeypatch.setattr(module, "logger", fake_logger)
    yield fake_logger
    assert fake_logger.exception.call_count == 0, (
        f"unexpected logger.exception calls: {fake_logger.exception.call_args_list}"
    )
