"""Comparison against a reference NeuroML simulator.

Most of these tests exercise the comparison machinery on synthetic traces, so
they run anywhere. The handful that actually invoke ``jnml`` are skipped when
the reference simulator is not installed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tarjuman.validate import (
    DEFAULT_TOLERANCES,
    MIN_SPIKING_RANGE_MV,
    TraceComparison,
    ValidationResult,
    auto_threshold,
    compare_traces,
    reference_available,
    spike_times,
    validate_lems,
)

DATA = Path(__file__).parent / "data"
NEEDS_REFERENCE = pytest.mark.skipif(
    not reference_available(), reason="jnml is not installed"
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------- #
# Synthetic traces
# --------------------------------------------------------------------------- #
def _spiking_trace(
    duration: float = 200.0,
    dt: float = 0.01,
    period: float = 20.0,
    phase: float = 0.0,
    rest: float = -65.0,
    peak: float = 40.0,
    width: float = 1.0,
):
    """A train of Gaussian 'spikes' on a flat baseline."""
    time = np.arange(0.0, duration, dt)
    voltage = np.full_like(time, rest)
    for spike in np.arange(period, duration - period, period):
        voltage += (peak - rest) * np.exp(
            -0.5 * ((time - spike - phase) / width) ** 2
        )
    return time, voltage


def test_spike_times_are_interpolated_between_samples():
    """Sub-sample interpolation matters: the step is the size of the effect."""
    time = np.array([0.0, 1.0, 2.0])
    voltage = np.array([-10.0, -10.0, 10.0])
    # Crossing zero exactly halfway between t=1 and t=2.
    assert spike_times(time, voltage, 0.0)[0] == pytest.approx(1.5)


def test_spike_times_finds_every_crossing_once():
    time, voltage = _spiking_trace(duration=200.0, period=20.0)
    expected = len(np.arange(20.0, 180.0, 20.0))
    assert spike_times(time, voltage, 0.0).size == expected


def test_auto_threshold_sits_between_rest_and_peak():
    _, voltage = _spiking_trace(rest=-65.0, peak=40.0)
    threshold = auto_threshold(voltage)
    assert -65.0 < threshold < 40.0


def test_auto_threshold_handles_an_integrate_and_fire_range():
    """An IaF cell never reaches 0 mV; a fixed threshold would call it silent."""
    time = np.arange(0.0, 300.0, 0.01)
    # Sawtooth between -70 and -50.
    voltage = -70.0 + 20.0 * ((time % 30.0) / 30.0)
    threshold = auto_threshold(voltage)
    assert threshold is not None
    assert -70.0 < threshold < -50.0
    assert spike_times(time, voltage, threshold).size == 10


def test_auto_threshold_ignores_the_initial_transient():
    """A model may start at a voltage it never returns to.

    NeuroML's ``iafTauCell`` starts at its own threshold; taking the plain
    maximum would put the detection level above the entire rest of the trace.
    """
    time = np.arange(0.0, 300.0, 0.01)
    voltage = -70.0 + 15.0 * ((time % 30.0) / 30.0)
    voltage[0] = -50.0  # the initial condition, never reached again
    threshold = auto_threshold(voltage)
    assert threshold < -55.0
    assert spike_times(time, voltage, threshold).size > 5


def test_auto_threshold_rejects_a_flat_trace():
    voltage = -65.0 + np.random.default_rng(0).normal(0, 0.1, 1000)
    assert auto_threshold(voltage) is None


def test_flat_traces_are_compared_point_wise():
    time = np.arange(0.0, 100.0, 0.1)
    reference = -65.0 + 0.0 * time
    candidate = reference + 0.25
    comparison = compare_traces((time, reference), (time, candidate), name="v")
    assert comparison.spike_threshold is None
    assert comparison.subthreshold_rms == pytest.approx(0.25, rel=1e-6)
    assert "non-spiking" in comparison.summary()


def test_identical_traces_compare_as_identical():
    time, voltage = _spiking_trace()
    comparison = compare_traces((time, voltage), (time, voltage), name="v")
    assert comparison.max_abs_diff == 0.0
    assert comparison.spikes_match
    assert comparison.spike_time_shift == 0.0
    assert comparison.passed()


def test_a_timing_shift_is_not_counted_as_an_amplitude_error():
    """The point of the spike-aware comparison.

    Two traces identical but for a 0.2 ms shift differ by tens of mV at the
    spike edges; a point-wise RMS would call that a gross failure, when the
    models agree and only the discretisation differs.
    """
    time, reference = _spiking_trace(phase=0.0, width=0.4)
    _, candidate = _spiking_trace(phase=0.3, width=0.4)

    comparison = compare_traces((time, reference), (time, candidate), name="v")
    assert comparison.max_abs_diff > 20.0  # point-wise: looks catastrophic
    assert comparison.subthreshold_rms < 0.5  # away from spikes: agrees
    assert comparison.spikes_match
    assert comparison.spike_time_shift == pytest.approx(0.3, abs=0.02)
    assert comparison.passed()


def test_a_missing_spike_fails_however_small_the_voltage_error():
    time, reference = _spiking_trace(period=20.0)
    _, candidate = _spiking_trace(period=25.0)  # fires less often
    comparison = compare_traces((time, reference), (time, candidate), name="v")
    assert not comparison.spikes_match
    assert not comparison.passed()


def test_drift_separates_a_constant_offset_from_a_rate_difference():
    time, reference = _spiking_trace(period=20.0)
    # Same count, every spike shifted by the same amount: offset, no drift.
    _, offset = _spiking_trace(period=20.0, phase=0.3)
    constant = compare_traces((time, reference), (time, offset), name="v")
    assert abs(constant.spike_time_drift) < 0.01

    # A different period: the shift grows with every interval.
    _, faster = _spiking_trace(period=20.1)
    growing = compare_traces((time, reference), (time, faster), name="v")
    assert abs(growing.spike_time_drift) > 0.05


def test_traces_on_different_time_bases_are_resampled():
    time_a, voltage_a = _spiking_trace(dt=0.01)
    time_b, voltage_b = _spiking_trace(dt=0.025)
    comparison = compare_traces((time_a, voltage_a), (time_b, voltage_b), name="v")
    assert comparison.spikes_match
    assert comparison.subthreshold_rms < 0.5


def test_non_voltage_traces_skip_spike_logic():
    time = np.arange(0.0, 100.0, 0.1)
    gate = 0.5 + 0.4 * np.sin(time / 5.0)
    comparison = compare_traces(
        (time, gate), (time, gate * 1.01), name="m", is_voltage=False
    )
    assert comparison.n_spikes_reference == 0
    assert comparison.correlation == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Tolerances and reporting
# --------------------------------------------------------------------------- #
def test_spike_count_tolerance_is_exact_by_default():
    assert DEFAULT_TOLERANCES["spike_count"] == 0
    comparison = TraceComparison(
        name="v", n_spikes_reference=5, n_spikes_tarjuman=6, spike_threshold=0.0
    )
    assert not comparison.passed()


def test_tolerances_can_be_relaxed():
    comparison = TraceComparison(
        name="v",
        n_spikes_reference=5,
        n_spikes_tarjuman=5,
        subthreshold_rms=2.0,
        spike_threshold=0.0,
    )
    assert not comparison.passed()
    assert comparison.passed({"subthreshold_rms_mv": 3.0})


def test_result_fails_when_any_trace_fails():
    good = TraceComparison(name="a", spike_threshold=0.0)
    bad = TraceComparison(
        name="b", n_spikes_reference=1, n_spikes_tarjuman=0, spike_threshold=0.0
    )
    assert ValidationResult(model="m", comparisons=[good]).passed
    assert not ValidationResult(model="m", comparisons=[good, bad]).passed


def test_result_with_no_comparisons_is_not_a_pass():
    """An empty comparison set means nothing was checked, not that all is well."""
    assert not ValidationResult(model="m").passed


def test_summary_mentions_the_engine():
    result = ValidationResult(
        model="m",
        comparisons=[TraceComparison(name="v", spike_threshold=0.0)],
        engine="neuron",
        reference_seconds=2.0,
        tarjuman_seconds=1.0,
    )
    assert "neuron" in result.summary()
    assert result.speedup == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Against the real reference simulator
# --------------------------------------------------------------------------- #
@NEEDS_REFERENCE
def test_hh_cell_matches_the_reference_simulator():
    """The headline check: a converted NeuroML model against jnml itself."""
    result = validate_lems(DATA / "LEMS_hh_single_comp.xml", timeout=600)
    assert result.error is None, result.error
    voltage = next(c for c in result.comparisons if c.name == "v")
    assert voltage.spikes_match
    assert voltage.subthreshold_rms < 1.0
    assert result.passed, result.summary()


@NEEDS_REFERENCE
def test_gate_variables_match_the_reference_simulator():
    result = validate_lems(DATA / "LEMS_hh_single_comp.xml", timeout=600)
    gates = [c for c in result.comparisons if c.name in ("m", "n")]
    assert len(gates) == 2
    for gate in gates:
        assert gate.correlation > 0.85
        assert gate.max_abs_diff < 1.0  # gates live in [0, 1]


# --------------------------------------------------------------------------- #
# Thresholds taken from the model rather than guessed
# --------------------------------------------------------------------------- #
def test_spike_thresholds_come_from_the_model():
    """A NeuroML cell declares its own spikeThresh; prefer it to a guess."""
    import tarjuman
    from tarjuman.validate import model_spike_thresholds

    model = tarjuman.from_neuroml(DATA / "hh_single_comp.nml", network_id="net1")
    thresholds = model_spike_thresholds(model)
    assert thresholds == {"hhpop": pytest.approx(-20.0)}  # <spikeThresh value="-20mV"/>


def test_declared_threshold_beats_a_trace_guess_on_subthreshold_traces():
    """The case that made a passing model look broken.

    A compartment receiving only sub-threshold EPSPs has no spikes. A threshold
    inferred from such a trace lands among the EPSP peaks and counts them,
    inventing a spike-count mismatch; the model's own threshold does not.
    """
    time = np.arange(0.0, 300.0, 0.01)
    # EPSPs rising to -50 mV on a -65 mV baseline: nothing reaches -20 mV.
    voltage = -65.0 + 15.0 * np.exp(-0.5 * ((time % 50.0 - 10.0) / 3.0) ** 2)

    guessed = auto_threshold(voltage)
    assert guessed is not None and spike_times(time, voltage, guessed).size > 0

    declared = -20.0  # what the cell's spikeThresh says
    assert spike_times(time, voltage, declared).size == 0

    comparison = compare_traces(
        (time, voltage), (time, voltage * 1.0), name="v", threshold=declared
    )
    assert comparison.n_spikes_reference == 0
    assert comparison.passed()


def test_detection_mode_is_chosen_once_from_the_reference():
    """Both traces must be measured the same way or the counts are meaningless.

    An integrate-and-fire cell is clamped at its threshold, so one simulator
    may register crossings while another registers none and falls back to
    resets. Letting each trace pick for itself produced a spurious off-by-one.
    """
    from tarjuman.validate import detection_mode

    time = np.arange(0.0, 300.0, 0.01)
    sawtooth = -70.0 + 15.0 * ((time % 30.0) / 30.0)

    # Clamped exactly at threshold: no crossing, so resets are the marker.
    assert detection_mode(time, sawtooth, -55.0) == "reset"
    # Overshooting the threshold: crossings are available.
    assert detection_mode(time, sawtooth, -60.0) == "crossing"

    comparison = compare_traces(
        (time, sawtooth), (time, sawtooth + 1e-9), name="v", threshold=-55.0
    )
    assert comparison.detection == "reset"
    assert comparison.spikes_match


def test_reset_times_ignores_a_slow_repolarisation():
    """An action potential must not be mistaken for an integrate-and-fire reset."""
    from tarjuman.validate import reset_times

    time, action_potentials = _spiking_trace(width=1.0)
    assert reset_times(time, action_potentials).size == 0

    # A one-sample drop of most of the range is a reset.
    time2 = np.arange(0.0, 100.0, 0.01)
    sawtooth = -70.0 + 20.0 * ((time2 % 25.0) / 25.0)
    assert reset_times(time2, sawtooth).size == 3


def test_iaf_tau_cell_reports_its_missing_current_term():
    """NeuroML's iafTauCell ignores injected current; a Jaxley compartment cannot."""
    from tarjuman import ir
    from tarjuman.cells import build_point_cell
    from tarjuman.report import ConversionReport

    report = ConversionReport()
    build_point_cell(
        ir.PointCell(
            id="t",
            kind="iafTauCell",
            params=dict(tau=10.0, leakReversal=-50.0, thresh=-55.0, reset=-70.0),
        ),
        report,
    )
    assert any("no current term" in note.message for note in report.approximations)


def test_availability_requires_a_runnable_reference(monkeypatch):
    """The script existing is not the same as the simulator working.

    `pip install pyneuroml` puts `jnml` on the PATH with or without a Java
    runtime, and macOS ships a `java` stub whose only job is to report that
    Java is missing. Treating the script's presence as readiness turned a
    missing JDK into confusing test failures instead of skips.
    """
    import subprocess

    from tarjuman import validate as module

    module.reference_available.cache_clear()
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/local/bin/jnml")

    def failing_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args, returncode=1, stdout="", stderr="Unable to locate a Java Runtime."
        )

    monkeypatch.setattr(module.subprocess, "run", failing_run)
    assert module.reference_available() is False

    problem = module.reference_problem()
    assert problem is not None and "java" in problem.lower()
    module.reference_available.cache_clear()


def test_no_reference_problem_is_reported_when_it_works(monkeypatch):
    import subprocess

    from tarjuman import validate as module

    module.reference_available.cache_clear()
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/local/bin/jnml")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "jNeuroML v0.12.0", ""),
    )
    assert module.reference_available() is True
    assert module.reference_problem() is None
    module.reference_available.cache_clear()


def test_neuron_is_found_where_pip_puts_it(monkeypatch, tmp_path):
    """jnml reads NEURON_HOME and a list of 1990s install prefixes, not the PATH.

    A pip-installed NEURON matches none of those prefixes, so `pip install
    neuron` used to end in the bare Java error "Could not find NEURON home
    directory!" while `nrniv` sat on the PATH all along.
    """
    import sys
    import types

    from tarjuman import validate as module

    package = tmp_path / "site-packages" / "neuron"
    home = package / ".data"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "nrniv").write_text("#!/bin/sh\n")

    monkeypatch.delenv("NEURON_HOME", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    stub = types.ModuleType("neuron")
    stub.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "neuron", stub)

    assert module.neuron_home() == home
    assert module.neuron_problem() is None

    # And from the PATH, for a NEURON that was not installed by pip.
    monkeypatch.setitem(sys.modules, "neuron", None)
    monkeypatch.setattr(
        module.shutil, "which", lambda name: str(home / "bin" / "nrniv")
    )
    assert module.neuron_home() == home

    environment = module._reference_environment("neuron")
    assert environment["NEURON_HOME"] == str(home)
    # The wheel's own nrngui and nrnivmodl scripts fall back to the temporary
    # prefix they were built under unless NRNHOME is set.
    assert environment["NRNHOME"] == str(home)


def test_a_missing_neuron_is_explained_rather_than_left_to_java(monkeypatch):
    from tarjuman import validate as module

    monkeypatch.delenv("NEURON_HOME", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "neuron", None)

    assert module.neuron_home() is None
    problem = module.neuron_problem()
    assert problem is not None and "pip install neuron" in problem

    with pytest.raises(module.ReferenceUnavailable):
        module._reference_environment("neuron")
    # jLEMS needs none of this, so it is left alone.
    assert "NEURON_HOME" not in module._reference_environment("jlems")


def test_a_missing_file_is_reported_once(tmp_path):
    """Not once per reference engine, and not as a stack trace."""
    result = validate_lems(tmp_path / "LEMS_NotHere.xml")
    assert not result.passed
    assert result.error.startswith("no such file")
    assert "jlems" not in result.error and "neuron" not in result.error
