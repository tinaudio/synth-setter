"""Behavior tests for the ``--verify`` post-draft battery (issue #1596).

Covers the pure checks (verdict precedence, spec-text analysis, classifier
audit, markdown rendering); the subprocess-backed checks are exercised
end-to-end in ``test_introspect_cli_register.py``.
"""

from __future__ import annotations

from pathlib import Path

from synth_setter.data.vst.verification import (
    VerificationReport,
    check_classifier_against_plugin,
    check_spec_text,
)

from tests.data.vst._introspect_fakes import IntrospectFakeParameter, IntrospectFakePlugin


def test_report_with_block_findings_verdict_is_blocked() -> None:
    """Any BLOCK finding makes the whole report verdict BLOCKED."""
    report = VerificationReport("fake_synth")
    report.block("spec module is oversized")
    report.warn("plugin_path is absolute")

    assert report.verdict() == "BLOCKED — not committable as-is"


def test_report_with_only_warn_findings_verdict_is_committable() -> None:
    """WARN-only findings keep the output committable but flag hand-tuning."""
    report = VerificationReport("fake_synth")
    report.warn("renderer_version is 'unknown'")

    assert report.verdict() == "COMMITTABLE with WARN findings (hand-tuning needed)"


def test_report_with_only_passes_verdict_is_clean() -> None:
    """A report with passes and no findings is CLEAN."""
    report = VerificationReport("fake_synth")
    report.ok("registry import + sample() OK")

    assert report.verdict() == "CLEAN"


def test_spec_text_check_flags_oversized_module_as_block(tmp_path: Path) -> None:
    """A spec module above the check-added-large-files limit records a BLOCK.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = tmp_path / "big_param_spec.py"
    spec.write_text("# filler\n" * 70_000)  # ~630 KB
    report = VerificationReport("big")

    check_spec_text(spec, report)

    assert any("500 KB" in b for b in report.blocks)


def test_spec_text_check_flags_oversized_onehot_categorical_as_warn(tmp_path: Path) -> None:
    """A categorical wider than the str draft cap is flagged as a likely sweep.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    values = "\n".join(f'                "{v / 10:.1f} cents",' for v in range(-250, 251))
    spec = tmp_path / "sweep_param_spec.py"
    spec.write_text(
        f'CategoricalParameter(\n    name="tune_cents",\n    values=[\n{values}\n    ],\n)\n'
    )
    report = VerificationReport("sweep")

    check_spec_text(spec, report)

    assert any("tune_cents" in w and "onehot" in w for w in report.warns)


def test_spec_text_check_flags_duplicate_parameter_names_as_warn(tmp_path: Path) -> None:
    """Two emitted parameters with the same name flag a sanitization collision.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = tmp_path / "dup_param_spec.py"
    spec.write_text(
        'ContinuousParameter(name="osc_1", min=0.0, max=1.0),\n'
        'ContinuousParameter(name="osc_1", min=0.0, max=1.0),\n'
    )
    report = VerificationReport("dup")

    check_spec_text(spec, report)

    assert any("duplicate" in w and "osc_1" in w for w in report.warns)


def test_spec_text_check_missing_module_is_block(tmp_path: Path) -> None:
    """A missing spec module is a BLOCK, not a crash.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    report = VerificationReport("ghost")

    check_spec_text(tmp_path / "ghost_param_spec.py", report)

    assert any("missing" in b for b in report.blocks)


def test_spec_text_check_clean_module_records_no_findings(tmp_path: Path) -> None:
    """A small draft with unique names and modest categoricals passes silently.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    spec = tmp_path / "ok_param_spec.py"
    spec.write_text('ContinuousParameter(name="cutoff", min=0.0, max=1.0),\n')
    report = VerificationReport("ok")

    check_spec_text(spec, report)

    assert report.blocks == []
    assert report.warns == []


def test_classifier_audit_flags_binary_float_drafted_continuous(tmp_path: Path) -> None:
    """A two-value float drafted continuous is flagged as a likely on/off switch.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    plugin = IntrospectFakePlugin({"osc1_reset": IntrospectFakeParameter(float, [0.0, 1.0])})
    spec = tmp_path / "odin_param_spec.py"
    spec.write_text('ContinuousParameter(name="osc1_reset", min=0.0, max=1.0),\n')
    report = VerificationReport("odin")

    check_classifier_against_plugin(plugin, spec, report)

    assert any("osc1_reset" in w and "2 valid_values" in w for w in report.warns)


def test_classifier_audit_flags_small_discrete_float_drafted_continuous(tmp_path: Path) -> None:
    """A stepped float selector drafted continuous is flagged with its cardinality.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    plugin = IntrospectFakePlugin(
        {"osc1_octave": IntrospectFakeParameter(float, [float(v) for v in range(-4, 5)])}
    )
    spec = tmp_path / "odin_param_spec.py"
    spec.write_text('ContinuousParameter(name="osc1_octave", min=0.0, max=1.0),\n')
    report = VerificationReport("odin")

    check_classifier_against_plugin(plugin, spec, report)

    assert any("osc1_octave(9)" in w for w in report.warns)


def test_classifier_audit_accepts_dense_float_drafted_continuous(tmp_path: Path) -> None:
    """A dense float sweep drafted continuous is the correct draft — no findings.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    plugin = IntrospectFakePlugin(
        {"cutoff": IntrospectFakeParameter(float, [i / 100 for i in range(101)])}
    )
    spec = tmp_path / "ok_param_spec.py"
    spec.write_text('ContinuousParameter(name="cutoff", min=0.0, max=1.0),\n')
    report = VerificationReport("ok")

    check_classifier_against_plugin(plugin, spec, report)

    assert report.warns == []


def test_classifier_audit_ignores_str_params_drafted_continuous(tmp_path: Path) -> None:
    """A str numeric sweep drafted continuous is the cap working as intended.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    plugin = IntrospectFakePlugin(
        {"tune_cents": IntrospectFakeParameter(str, [f"{v}.0 cents" for v in range(-100, 101)])}
    )
    spec = tmp_path / "obxf_param_spec.py"
    spec.write_text('ContinuousParameter(name="tune_cents", min=0.0, max=1.0),\n')
    report = VerificationReport("obxf")

    check_classifier_against_plugin(plugin, spec, report)

    assert report.warns == []


def test_markdown_report_carries_verdict_findings_and_artifacts(tmp_path: Path) -> None:
    """The rendered report names the synth, verdict, artifact paths, and findings.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    report = VerificationReport("fake_synth")
    report.block("spec module is oversized")
    report.warn("renderer_version is 'unknown'")
    report.ok("Hydra render=fake_synth composes into a valid RenderConfig")

    text = report.to_markdown([tmp_path / "fake_synth_param_spec.py"])

    assert "# Introspection verification — `fake_synth`" in text
    assert "**Verdict: BLOCKED — not committable as-is**" in text
    assert "fake_synth_param_spec.py" in text
    assert "## BLOCK (1)" in text
    assert "## WARN (1)" in text
    assert "## PASS (1)" in text
