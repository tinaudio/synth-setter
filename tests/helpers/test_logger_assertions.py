"""Unit tests for the ``assert_no_logger_exceptions`` crash-gate helper."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from tests.helpers.logger_assertions import assert_no_logger_exceptions


def _module_with_logger() -> ModuleType:
    """Build a stand-in module exposing a no-op ``logger`` with loguru-like methods.

    :returns: An object usable where ``assert_no_logger_exceptions`` expects a
        module with a ``logger`` attribute.
    """
    logger = SimpleNamespace(exception=lambda *a, **k: None, error=lambda *a, **k: None)
    return cast(ModuleType, SimpleNamespace(logger=logger))


def _raise_boom() -> None:
    """Raise ``RuntimeError`` to drive the helper's body-error restore path.

    :raises RuntimeError: Always, with message ``boom``.
    """
    raise RuntimeError("boom")


def test_assert_no_logger_exceptions_no_calls_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body that never calls ``logger.exception`` exits cleanly.

    :param monkeypatch: Pytest monkeypatch fixture passed to the helper.
    """
    module = _module_with_logger()
    with assert_no_logger_exceptions(monkeypatch, module) as fake_logger:
        module.logger.error("non-fatal")
    assert fake_logger.error.call_count == 1


def test_assert_no_logger_exceptions_on_exception_call_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body that calls ``logger.exception`` trips the crash gate on exit.

    :param monkeypatch: Pytest monkeypatch fixture passed to the helper.
    """
    module = _module_with_logger()
    with pytest.raises(AssertionError, match="logger.exception"):
        with assert_no_logger_exceptions(monkeypatch, module):
            module.logger.exception("boom")


def test_assert_no_logger_exceptions_restores_logger_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The patched ``logger`` is restored when the context exits, not at teardown.

    :param monkeypatch: Pytest monkeypatch fixture passed to the helper.
    """
    module = _module_with_logger()
    original = module.logger
    with assert_no_logger_exceptions(monkeypatch, module):
        assert module.logger is not original
    assert module.logger is original


def test_assert_no_logger_exceptions_restores_logger_on_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body that raises still restores the original ``logger`` on exit.

    :param monkeypatch: Pytest monkeypatch fixture passed to the helper.
    """
    module = _module_with_logger()
    original = module.logger
    with pytest.raises(RuntimeError, match="boom"):
        with assert_no_logger_exceptions(monkeypatch, module):
            _raise_boom()
    assert module.logger is original
