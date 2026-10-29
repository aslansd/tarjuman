"""Abstract cells driven by their own LEMS dynamics.

``izhikevich2007Cell``, ``adExIaFCell`` and the integrate-and-fire cells models
define for themselves all read their synaptic input with ``<DerivedVariable
select="synapses[*]/i" reduce="add"/>``, which is the construct that used to
keep every one of them out.  These tests cover the mechanism that resolves it,
the regime flattening ``adExIaFCell`` needs, and the two numerical traps that
a spiking membrane equation sets for an implicit solver.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import tarjuman
from tarjuman import ir
from tarjuman.cells import build_point_cell
from tarjuman.cells_lems import (
    LemsPointCell,
    flatten_spiking_regimes,
    make_lems_point_cell,
)
from tarjuman.errors import ParseError, UnsupportedComponentError
from tarjuman.lems.builtins import builtin_registry, cell_component_types
from tarjuman.lems.component_types import read_component_type, resolve_inheritance
from tarjuman.report import ConversionReport

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _resolved(name: str):
    registry = builtin_registry()
    return resolve_inheritance(registry[name], registry)


IZHIKEVICH = {
    "C": "100pF",
    "v0": "-60mV",
    "k": "0.7nS_per_mV",
    "vr": "-60mV",
    "vt": "-40mV",
    "vpeak": "35mV",
    "a": "0.03per_ms",
    "b": "-2nS",
    "c": "-50mV",
    "d": "100pA",
}
ADEX = {
    "C": "281pF",
    "gL": "30nS",
    "EL": "-70.6mV",
    "reset": "-48.5mV",
    "VT": "-50.4mV",
    "thresh": "-40.4mV",
    "delT": "2mV",
    "tauw": "40ms",
    "a": "4nS",
    "b": "0.08nA",
    "refract": "0ms",
}


#: NeuroML's adExRebound. Its detection threshold sits 24 mV above ``VT``,
#: which leaves a band of voltages where the membrane conductance is already
#: regenerative but no reset has fired — the band an implicit solver cannot
#: cross. adExBurst2 above has no such band: it resets at -40.4 mV, before the
#: feedback ever becomes dangerous.
ADEX_REBOUND = {
    **ADEX,
    "EL": "-60mV",
    "reset": "-51mV",
    "VT": "-54mV",
    "thresh": "-30mV",
    "tauw": "150ms",
    "a": "200nS",
    "b": "0.1nA",
}


def _cell(kind: str, attributes: dict, component_types=None, area=100.0):
    point_cell = ir.PointCell(
        id=f"test_{kind}",
        kind=kind,
        custom=ir.CustomComponent(
            id=f"test_{kind}", component_type=kind, attributes=dict(attributes)
        ),
    )
    return make_lems_point_cell(point_cell, component_types or {}, area)


# --------------------------------------------------------------------------- #
# The canonical definitions
# --------------------------------------------------------------------------- #
def test_the_abstract_cell_types_come_from_neuroml_not_a_transcription():
    """``_cell_types.py`` is generated, so the dynamics are NeuroML's own.

    Reimplementing ``izhikevich2007Cell`` in Python would put a transcription
    between the model and the simulation; the generated module keeps the
    ComponentType verbatim and feeds it to the interpreter instead.
    """
    types = cell_component_types()
    assert {"iafCell", "izhikevich2007Cell", "adExIaFCell"} <= set(types)

    izhikevich = _resolved("izhikevich2007Cell")
    # The 2007 form: C dv/dt = k (v - vr)(v - vt) - u + i, du/dt = a(b(v-vr) - u).
    assert izhikevich.time_derivatives["v"] == "iMemb / C"
    assert "k * (v-vr) * (v-vt)" in next(
        d.value for d in izhikevich.derived_variables if d.name == "iMemb"
    )
    assert set(izhikevich.state_variables) == {"v", "u"}
    assert izhikevich.parameters["k"] == "conductance_per_voltage"


def test_a_cell_that_gathers_its_synaptic_input_is_interpretable():
    """The construct that blocked these cell types is resolved, not refused."""
    for name in ("iafCell", "izhikevich2007Cell", "adExIaFCell"):
        assert _resolved(name).gathered_variables == ["iSyn"]

    channel = _cell("izhikevich2007Cell", IZHIKEVICH)
    assert channel._gathered == {"iSyn": "synapses[*]/i"}


def test_gathering_anything_other_than_synaptic_current_is_still_refused():
    """Jaxley's solver assembles the synaptic sum; it assembles nothing else."""
    component = read_component_type(
        __import__("xml.etree.ElementTree", fromlist=["X"]).fromstring(
            """
            <ComponentType name="odd" extends="baseCellMembPotCap">
              <Dynamics>
                <StateVariable name="v" dimension="voltage" exposure="v"/>
                <DerivedVariable name="peers" dimension="voltage"
                                 select="children[*]/v" reduce="add"/>
                <TimeDerivative variable="v" value="peers / C"/>
              </Dynamics>
            </ComponentType>
            """
        )
    )
    with pytest.raises(UnsupportedComponentError, match="children"):
        _cell("odd", {"C": "100pF"}, {"odd": component})


# --------------------------------------------------------------------------- #
# Units and initial conditions
# --------------------------------------------------------------------------- #
def test_parameters_are_read_with_the_dimensions_the_type_declares():
    """Not inferred from the written unit, which is ambiguous for bare numbers."""
    channel = _cell("adExIaFCell", ADEX)
    params = channel.channel_params
    prefix = channel.name
    assert params[f"{prefix}_C"] == pytest.approx(281.0)  # pF
    assert params[f"{prefix}_gL"] == pytest.approx(0.03)  # 30 nS as uS
    assert params[f"{prefix}_a"] == pytest.approx(0.004)  # 4 nS as uS
    assert params[f"{prefix}_b"] == pytest.approx(0.08)  # 0.08 nA
    assert params[f"{prefix}_thresh"] == pytest.approx(-40.4)  # mV


def test_the_resting_potential_comes_from_onstart():
    """``OnStart`` is where NeuroML puts a cell's initial voltage."""
    assert _cell("adExIaFCell", ADEX).resting_potential == pytest.approx(-70.6)
    assert _cell("izhikevich2007Cell", IZHIKEVICH).resting_potential == pytest.approx(
        -60.0
    )


def test_a_cell_without_a_capacitance_is_refused_with_a_reason():
    """Dimensionless cells have no C, and Jaxley's membrane equation needs one."""
    with pytest.raises(UnsupportedComponentError, match="dimensionless"):
        _cell("fitzHughNagumoCell", {"I": "0.1"})


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def test_the_refractory_regime_is_flattened_into_plain_dynamics():
    component, regimes = flatten_spiking_regimes(_resolved("adExIaFCell"))
    assert regimes is not None
    assert regimes.spike_test == "v .gt. thresh"
    assert regimes.entry == [("v", "reset"), ("w", "w + b")]
    # The exit is timed against the simulation clock, which a Jaxley channel
    # cannot read; the duration is counted down from this parameter instead.
    assert regimes.refract_parameter == "refract"
    # `lastSpikeTime` only timestamped the spike for that timed exit.
    assert "lastSpikeTime" not in component.state_variables
    # The integrating regime's equations are now the component's own.
    assert component.time_derivatives["v"] == "iMemb / C"
    assert not component.unsupported


def test_an_unrecognised_regime_arrangement_is_refused_not_guessed():
    """Running the wrong dynamics silently would be worse than not running."""
    component = _resolved("adExIaFCell")
    component.regimes = component.regimes[:1]  # one regime, no initial
    with pytest.raises(UnsupportedComponentError, match="regime"):
        flatten_spiking_regimes(component)


def test_a_component_without_regimes_passes_through_unchanged():
    component = _resolved("izhikevich2007Cell")
    flattened, regimes = flatten_spiking_regimes(component)
    assert regimes is None and flattened is component


# --------------------------------------------------------------------------- #
# The two numerical traps
# --------------------------------------------------------------------------- #
def _params(channel, capacitance_uf_per_cm2=2.81):
    params = {k: jnp.array(v) for k, v in channel.channel_params.items()}
    params["capacitance"] = jnp.array(capacitance_uf_per_cm2)
    return params


def test_the_gathered_synaptic_term_is_zero_so_the_solver_can_supply_it():
    """Evaluating it twice would count every synapse twice.

    At rest an AdEx cell's own current is nil, so whatever ``compute_current``
    returns there is the leak alone — not a synaptic contribution.
    """
    channel = _cell("adExIaFCell", ADEX)
    states = {k: jnp.array(v) for k, v in channel.channel_states.items()}
    at_rest = float(
        channel.compute_current(states, jnp.array(-70.6), _params(channel))
    )
    assert abs(at_rest) < 1e-3  # mA/cm2


def test_the_voltage_is_left_to_jaxley_until_a_reset_fires():
    """Returning a voltage every step would override the solver's own.

    Jaxley takes whatever ``update_states`` returns as ``"v"``, so a cell that
    writes one unconditionally never feels its stimulus or its synapses.
    """
    channel = _cell("izhikevich2007Cell", IZHIKEVICH)
    states = {k: jnp.array(v) for k, v in channel.channel_states.items()}
    params = _params(channel)

    quiet = channel.update_states(states, 0.01, jnp.array(-60.0), params)
    assert float(quiet["v"]) == pytest.approx(-60.0)
    assert float(quiet[f"{channel.name}_spikes"]) == 0.0

    # Above vpeak the OnCondition assigns v = c and u = u + d.
    spiking = channel.update_states(states, 0.01, jnp.array(40.0), params)
    assert float(spiking["v"]) == pytest.approx(-50.0)  # c
    assert float(spiking[f"{channel.name}_spikes"]) == 1.0
    assert float(spiking[f"{channel.name}_u"]) == pytest.approx(0.1, abs=1e-6)  # d


def test_the_regenerative_part_of_a_spike_is_integrated_explicitly():
    """An implicit step inverts the conductance, and AdEx makes it negative.

    ``exp((v - VT) / delT)`` turns the membrane conductance regenerative about
    12 mV above ``VT``, while ``thresh`` sits another 12 mV above that — so the
    voltage would spend several implicit steps in a region where ``C/dt + dI/dv``
    is negative and the step runs backwards, reaching hundreds of millivolts the
    wrong way. The cell detects that region and integrates it itself.
    """
    channel = _cell("adExIaFCell", ADEX_REBOUND)
    states = {k: jnp.array(v) for k, v in channel.channel_states.items()}
    params = _params(channel)
    runaway = f"{channel.name}_runaway"

    # Well below threshold the equation is tame and Jaxley does the solving.
    calm = channel.update_states(states, 0.025, jnp.array(-50.0), params)
    assert float(calm[runaway]) == 0.0
    assert float(calm["v"]) == pytest.approx(-50.0)  # left to the solver

    # Inside the regenerative band it does not, and it carries the voltage
    # across itself without yet reaching the reset.
    steep = channel.update_states(states, 0.025, jnp.array(-42.0), params)
    assert float(steep[runaway]) == 1.0
    assert float(steep["v"]) > -42.0
    assert float(steep[f"{channel.name}_spikes"]) == 0.0

    # Having taken the step over, it hands the solver no current, so the
    # voltage it wrote is not advanced twice.
    taken = dict(states)
    taken[runaway] = jnp.array(1.0)
    assert float(channel.compute_current(taken, jnp.array(-42.0), params)) == 0.0

    # A cell whose threshold sits below the band never needs any of this.
    tame = _cell("adExIaFCell", ADEX)
    tame_states = {k: jnp.array(v) for k, v in tame.channel_states.items()}
    stepped = tame.update_states(
        tame_states, 0.025, jnp.array(-41.0), _params(tame)
    )
    assert float(stepped[f"{tame.name}_runaway"]) == 0.0


def test_explicit_substepping_stops_at_threshold_rather_than_overflowing():
    """The equation diverges to infinity in finite time; NeuroML stops there too."""
    channel = _cell("adExIaFCell", ADEX)
    states = {k: jnp.array(v) for k, v in channel.channel_states.items()}
    params = _params(channel)
    # A voltage deep in the divergence: the exponential here is astronomical.
    result = channel.update_states(states, 0.025, jnp.array(-35.0), params)
    assert np.isfinite(float(result["v"]))
    assert float(result["v"]) == pytest.approx(-48.5)  # reset
    assert float(result[f"{channel.name}_spikes"]) == 1.0


def test_a_refractory_period_holds_the_voltage_and_stops_the_current():
    """The refractory regime declares no TimeDerivative for v: no equation, no current."""
    channel = _cell("adExIaFCell", {**ADEX, "refract": "5ms"})
    states = {k: jnp.array(v) for k, v in channel.channel_states.items()}
    params = _params(channel)
    refractory = f"{channel.name}_refractory"

    fired = channel.update_states(states, 0.025, jnp.array(-35.0), params)
    assert float(fired[refractory]) == pytest.approx(5.0)
    assert float(fired["v"]) == pytest.approx(-48.5)

    held = dict(states)
    held[refractory] = jnp.array(3.0)
    during = channel.update_states(held, 0.025, jnp.array(-60.0), params)
    assert float(during[refractory]) == pytest.approx(2.975)
    assert float(during["v"]) == pytest.approx(-48.5)  # held at reset
    assert float(channel.compute_current(held, jnp.array(-60.0), params)) == 0.0


# --------------------------------------------------------------------------- #
# Through the builder
# --------------------------------------------------------------------------- #
def test_a_point_cell_becomes_a_one_compartment_jaxley_cell():
    report = ConversionReport()
    point_cell = ir.PointCell(
        id="izh",
        kind="izhikevich2007Cell",
        custom=ir.CustomComponent(
            id="izh", component_type="izhikevich2007Cell", attributes=IZHIKEVICH
        ),
    )
    cell, morphology = build_point_cell(point_cell, report)
    assert len(cell.nodes) == 1
    assert float(cell.nodes.iloc[0]["v"]) == pytest.approx(-60.0)
    assert morphology.segment_to_branch == {0: 0}
    # The report says which cell was interpreted and what that costs.
    notes = " ".join(note.message for note in report.notes)
    assert "LEMS dynamics" in notes and "state variables" in notes
    assert any("izhikevich2007Cell" in note.component for note in report.notes)


def test_the_iaf_family_still_uses_its_own_validated_channel():
    """The generic path is for everything else; iafCell keeps its channel."""
    from tarjuman.cells import IntegrateAndFire

    report = ConversionReport()
    cell, _ = build_point_cell(
        ir.PointCell(
            id="iaf",
            kind="iafCell",
            params={
                "C": 10.0,
                "leakConductance": 0.01,
                "leakReversal": -70.0,
                "thresh": -50.0,
                "reset": -70.0,
            },
        ),
        report,
    )
    assert any(isinstance(c, IntegrateAndFire) for c in cell.channels)


def test_a_subtype_named_in_the_type_attribute_is_honoured():
    """``<iafCell type="iafActivityCell">`` is c302 parameter set B's cell.

    Reading the tag and ignoring ``type`` builds a plain integrate-and-fire
    cell and drops the state variable the subtype adds — the same numbers, a
    different model, and no warning.
    """
    from tarjuman.reader import read_neuroml_string

    source = """
        <neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="d">
          <ComponentType name="iafActivityCell" extends="iafCell">
            <Exposure name="activity" dimension="none"/>
            <Parameter name="tau1" dimension="time"/>
            <Dynamics>
              <StateVariable name="v" exposure="v" dimension="voltage"/>
              <StateVariable name="activity" exposure="activity" dimension="none"/>
              <DerivedVariable name="iSyn" dimension="current" exposure="iSyn"
                               select="synapses[*]/i" reduce="add"/>
              <DerivedVariable name="iMemb" dimension="current" exposure="iMemb"
                               value="leakConductance * (leakReversal - v) + iSyn"/>
              <DerivedVariable name="target" dimension="none"
                               value="(v - reset) / (thresh - reset)"/>
              <TimeDerivative variable="v" value="iMemb / C"/>
              <TimeDerivative variable="activity" value="(target-activity)/tau1"/>
              <OnStart><StateAssignment variable="v" value="reset"/></OnStart>
              <OnCondition test="v .gt. thresh">
                <StateAssignment variable="v" value="reset"/>
              </OnCondition>
            </Dynamics>
          </ComponentType>
          <iafCell type="iafActivityCell" id="generic" C="3pF" thresh="-30mV"
                   reset="-50mV" leakConductance="0.1nS" leakReversal="-50mV"
                   tau1="50ms"/>
        </neuroml>
        """
    report = ConversionReport()
    read = read_neuroml_string(source, report=report)
    assert read.point_cells["generic"].kind == "iafActivityCell"

    cell, _ = build_point_cell(
        read.point_cells["generic"], report, component_types=read.component_types
    )
    channel = next(c for c in cell.channels if isinstance(c, LemsPointCell))
    assert "activity" in channel._state_dimensions


# --------------------------------------------------------------------------- #
# Against the reference simulator
# --------------------------------------------------------------------------- #
from tarjuman.validate import reference_available, validate_lems  # noqa: E402

NEEDS_REFERENCE = pytest.mark.skipif(
    not reference_available(), reason="jnml is not installed"
)


@NEEDS_REFERENCE
def test_izhikevich2007_matches_the_reference_simulator():
    """Tonic and bursting, voltage and recovery variable, against jnml itself."""
    result = validate_lems(DATA / "LEMS_izhikevich2007.xml", timeout=900)
    assert result.error is None, result.error
    for name in ("vTonic", "vBurst"):
        voltage = next(c for c in result.comparisons if c.name == name)
        assert voltage.spikes_match, voltage
        assert voltage.subthreshold_rms < 1.0, voltage
    for name in ("uTonic", "uBurst"):
        recovery = next(c for c in result.comparisons if c.name == name)
        assert recovery.correlation > 0.95, recovery
    assert result.passed, result.summary()


@NEEDS_REFERENCE
def test_the_adex_rebound_burst_matches_the_reference_simulator():
    """The case the regenerative band decides.

    jnml puts three spikes in this rebound burst, and stays at three for every
    time step from 0.025 ms down. An implicit solve that integrated the
    regenerative region would not reach three at any step — it reaches a
    voltage of several hundred millivolts the wrong way instead.
    """
    result = validate_lems(DATA / "LEMS_adex_rebound.xml", timeout=900)
    assert result.error is None, result.error
    voltage = next(c for c in result.comparisons if c.name == "v")
    assert voltage.n_spikes_reference == 3, voltage
    assert voltage.spikes_match, voltage
    assert voltage.subthreshold_rms < 1.0, voltage
    adaptation = next(c for c in result.comparisons if c.name == "w")
    assert adaptation.correlation > 0.95, adaptation
    assert result.passed, result.summary()
