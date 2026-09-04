"""Behavioral tests for the lance-free pyFDN sketch pool-worker entry."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np

from synth_setter.features.pyfdn_controls import extract_reverb_sketch
from synth_setter.pipeline.data.pyfdn_sketch_worker import extract_reverb_sketch_row


def test_extract_reverb_sketch_row_matches_public_extractor_bit_exact() -> None:
    """The worker entry adds no numeric policy on top of the public transform."""
    sample_rate = 44_100
    rng = np.random.default_rng(7)
    time = np.arange(4 * sample_rate, dtype=np.float64) / sample_rate
    ir = rng.standard_normal(time.size) * np.exp(-6.0 * time)

    actual = extract_reverb_sketch_row(ir, float(sample_rate))

    np.testing.assert_array_equal(actual, extract_reverb_sketch(ir, float(sample_rate)))


def test_extract_reverb_sketch_row_suppresses_mixing_time_warning() -> None:
    """An IR without a detectable mixing time extracts without warning noise."""
    sample_rate = 44_100
    impulse = np.zeros(4 * sample_rate, dtype=np.float64)
    impulse[0] = 1.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_reverb_sketch_row(impulse, float(sample_rate))

    assert not [w for w in caught if "Mixing time" in str(w.message)]


def test_worker_module_import_never_pulls_in_lance() -> None:
    """Spawn workers unpickle this module; its import chain must exclude lance."""
    probe = (
        "import sys\n"
        "import synth_setter.pipeline.data.pyfdn_sketch_worker\n"
        "assert 'lance' not in sys.modules, 'worker import chain pulled in lance'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
