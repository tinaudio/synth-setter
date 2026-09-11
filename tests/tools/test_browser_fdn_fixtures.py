"""Tests for the browser FDN golden-fixture generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfilt

from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.evaluation.compute_audio_metrics import (
    compute_acoustic_parameter_metrics_mono_only,
    compute_mss,
    compute_octave_edc_rmse_db_mono_only,
    compute_octave_rt60_log_rmse_mono_only,
)
from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.tools.browser_fdn_fixtures import (
    FIXTURE_DIR,
    SAMPLE_RATE,
    build_fixtures,
    main,
    synthetic_impulse_response,
)


def test_synthetic_impulse_response_when_seeded_is_deterministic_unit_peak() -> None:
    """Same recipe, same samples, unit peak."""
    first = synthetic_impulse_response(seed=7, samples=2048, decay_seconds=0.5)
    second = synthetic_impulse_response(seed=7, samples=2048, decay_seconds=0.5)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2048,)
    assert np.max(np.abs(first)) == pytest.approx(1.0)


def test_synthetic_impulse_response_when_seed_differs_changes_samples() -> None:
    """Different seeds give different noise."""
    first = synthetic_impulse_response(seed=7, samples=2048, decay_seconds=0.5)
    second = synthetic_impulse_response(seed=8, samples=2048, decay_seconds=0.5)

    assert not np.array_equal(first, second)


def test_build_fixtures_sketch_matches_public_extractor() -> None:
    """Golden sketch equals the public extractor output."""
    fixtures = build_fixtures()

    expected = extract_reverb_sketch(
        synthetic_impulse_response(**fixtures["golden"]["target"]["recipe"]), SAMPLE_RATE
    )

    np.testing.assert_array_equal(
        np.asarray(fixtures["golden"]["target"]["sketch"], dtype=np.float32), expected
    )


def test_build_fixtures_metrics_match_public_metric_functions() -> None:
    """Golden metrics equal the public metric functions."""
    fixtures = build_fixtures()
    golden = fixtures["golden"]
    target = synthetic_impulse_response(**golden["target"]["recipe"])[None, :]
    pred = synthetic_impulse_response(**golden["pred"]["recipe"])[None, :]

    metrics = golden["metrics"]
    assert metrics["mss"] == compute_mss(target, pred, SAMPLE_RATE)
    assert metrics["octave_edc_rmse_db"] == compute_octave_edc_rmse_db_mono_only(
        target, pred, SAMPLE_RATE
    )
    assert metrics["octave_rt60_log_rmse"] == compute_octave_rt60_log_rmse_mono_only(
        target, pred, SAMPLE_RATE
    )
    acoustic = compute_acoustic_parameter_metrics_mono_only(target, pred, SAMPLE_RATE)
    assert metrics["t30_mape"] == acoustic["t30_mape"]
    assert metrics["c50_mae_db"] == acoustic["c50_mae_db"]


def test_build_fixtures_decode_matches_public_decoder() -> None:
    """Golden decode equals the public decoder output."""
    fixtures = build_fixtures()
    decode = fixtures["golden"]["decode"]
    spec = resolve_param_spec(ParamSpecName("pyfdn_n8_mono_householder"))

    synth_params, _note = decode_model_output(
        np.asarray(decode["model_row"], dtype=np.float32), spec
    )

    assert decode["delays"] == np.asarray(synth_params["delays"]).tolist()
    assert decode["input_matrix"] == np.asarray(synth_params["input_matrix"]).ravel().tolist()
    assert decode["output_matrix"] == np.asarray(synth_params["output_matrix"]).ravel().tolist()
    assert decode["direct_matrix"] == float(np.asarray(synth_params["direct_matrix"]).item())
    assert decode["rt_dc_seconds"] == synth_params["post_delay.rt_dc_seconds"]
    assert decode["rt_nyquist_seconds"] == synth_params["post_delay.rt_nyquist_seconds"]


def test_build_fixtures_octave_sos_filters_a_unit_impulse_like_scipy() -> None:
    """Stored SOS rows are usable scipy sections."""
    fixtures = build_fixtures()
    impulse = np.zeros(4096)
    impulse[0] = 1.0

    sos = np.asarray(fixtures["octave_bands"]["sketch"][3])
    filtered = sosfilt(sos, impulse)

    assert sos.shape == (4, 6)
    assert fixtures["octave_bands"]["sketch_centres_hz"][3] == 500.0
    assert np.isfinite(filtered).all()
    assert np.abs(filtered).max() > 0.0


def test_committed_fixtures_match_regenerated_fixtures() -> None:
    """The checked-in JSON must equal what the current Python references produce."""
    fixtures = build_fixtures()

    for name, payload in fixtures.items():
        committed = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert committed == payload, f"{name}.json drifted; rerun the generator"


def test_main_when_output_dir_given_writes_both_fixture_files(tmp_path: Path) -> None:
    """The CLI writes both fixture files.

    :param tmp_path: Pytest temporary directory.
    """
    main(["--output", str(tmp_path)])

    assert (tmp_path / "golden.json").is_file()
    assert (tmp_path / "octave_bands.json").is_file()
    written = json.loads((tmp_path / "golden.json").read_text(encoding="utf-8"))
    assert written["sample_rate"] == SAMPLE_RATE
