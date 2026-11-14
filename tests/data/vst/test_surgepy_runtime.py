"""SurgePy runtime identity contracts that need no native extension."""

from __future__ import annotations

import pytest

from synth_setter.data.vst.surgepy_runtime import (
    ensure_surgepy_runtime,
    surgepy_version_matches,
)


def test_surgepy_version_matches_identical_stamps_accepts() -> None:
    """An unabbreviated match is still a match."""
    assert surgepy_version_matches("1.3.master.f7b97c68", "1.3.master.f7b97c68")


@pytest.mark.parametrize(
    ("actual", "configured"),
    [
        ("1.3.master.f7b97c682", "1.3.master.f7b97c68"),
        ("1.3.master.f7b97c68", "1.3.master.f7b97c682"),
    ],
)
def test_surgepy_version_matches_differing_abbreviation_length_accepts(
    actual: str,
    configured: str,
) -> None:
    """One commit abbreviated to different lengths is one identity.

    :param actual: Stamp reported by the installed extension.
    :param configured: Stamp pinned by the synth identity group.
    """
    assert surgepy_version_matches(actual, configured)


def test_surgepy_version_matches_differing_release_prefix_rejects() -> None:
    """A different Surge release is a different renderer."""
    assert not surgepy_version_matches("1.4.master.f7b97c68", "1.3.master.f7b97c68")


def test_surgepy_version_matches_differing_commit_rejects() -> None:
    """A different commit at equal abbreviation length is a different renderer."""
    assert not surgepy_version_matches("1.3.master.a1b2c3d4", "1.3.master.f7b97c68")


def test_surgepy_version_matches_commit_diverging_past_shared_prefix_rejects() -> None:
    """Sharing a leading run of hex digits is not sharing a commit."""
    assert not surgepy_version_matches("1.3.master.f7b97c69", "1.3.master.f7b97c68")


def test_surgepy_version_matches_abbreviation_at_git_floor_accepts() -> None:
    """Seven digits is the shortest abbreviation git emits, so it still identifies."""
    assert surgepy_version_matches("1.3.master.f7b97c6", "1.3.master.f7b97c68")


def test_surgepy_version_matches_abbreviation_below_git_floor_rejects() -> None:
    """A stamp too short to identify a commit never widens the match."""
    assert not surgepy_version_matches("1.3.master.f7b97c", "1.3.master.f7b97c68")


def test_surgepy_version_matches_non_commit_release_stamps_compare_exactly() -> None:
    """A release version without a commit component keeps exact comparison."""
    assert not surgepy_version_matches("1.3", "1.3.4")


def test_surgepy_version_matches_non_hex_final_component_rejects() -> None:
    """A trailing word is a release channel, not an abbreviated commit."""
    assert not surgepy_version_matches("1.3.master.nightly", "1.3.master.nightlybuild")


@pytest.mark.requires_surgepy
def test_ensure_surgepy_runtime_accepts_pin_abbreviated_differently_than_the_build() -> None:
    """The installed extension satisfies a pin abbreviated at another length.

    Reproduces the CI condition in #3391: the same pinned commit reaches the Ubuntu
    lane as ``f7b97c682`` and this host as ``f7b97c68``.
    """
    import surgepy

    installed = surgepy.getVersion()
    release, _, commit = installed.rpartition(".")
    widened = f"{release}.{commit}0" if len(commit) == 8 else f"{release}.{commit[:8]}"

    ensure_surgepy_runtime("surgepy", widened)


@pytest.mark.requires_surgepy
def test_ensure_surgepy_runtime_rejects_a_different_pinned_commit() -> None:
    """A pin naming another commit still fails closed."""
    with pytest.raises(RuntimeError, match="does not match configured"):
        ensure_surgepy_runtime("surgepy", "1.3.master.a1b2c3d4")
