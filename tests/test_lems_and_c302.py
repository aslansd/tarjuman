"""The LEMS interpreter: custom ComponentTypes, ion pools, and point cells.

These are the features that make real models work — the NeuroML core examples
mostly use the standard component types, but the C. elegans connectome models
(c302) define their own gates, synapses and calcium pools, and half of their
parameter sets are built from integrate-and-fire cells.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import tarjuman
from tarjuman import ir
from tarjuman.cells import IntegrateAndFire, build_point_cell
from tarjuman.errors import ParseError, UnsupportedComponentError
from tarjuman.lems import (
    CompiledDynamics,
    base_kind_of,
    compile_expression,
    expression_symbols,
    read_component_type,
    resolve_inheritance,
)
from tarjuman.lems.builtins import builtin_registry
from tarjuman.report import ConversionReport
from tarjuman.synapses.lems_synapse import LemsSynapse

DATA = Path(__file__).parent / "data"
CALCIUM_MODEL = DATA / "calcium_custom_gate.nml"


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expression, context, expected",
    [
        ("2^3^2", {}, 512.0),  # right associative
        ("-3 + 4 * 2 / (1 - 5)^2", {}, -2.5),
        ("1 +((q-1) * alpha)", {"q": 0.5, "alpha": 0.282473}, 0.8587635),
        ("exp(x) * ln(y)", {"x": 0.0, "y": jnp.e}, 1.0),
        ("v .gt. thresh", {"v": -20.0, "thresh": -30.0}, 1.0),
        ("v .lt. thresh", {"v": -20.0, "thresh": -30.0}, 0.0),
        ("(a .gt. 0) .and. (b .gt. 0)", {"a": 1.0, "b": -1.0}, 0.0),
        ("(a .gt. 0) .or. (b .gt. 0)", {"a": 1.0, "b": -1.0}, 1.0),
        ("1/(1 + exp(beta * (vth - vpeer)))", {"beta": 0.25, "vth": -20.0, "vpeer": -20.0}, 0.5),
    ],
)
def test_lems_expressions_evaluate(expression, context, expected):
    function = compile_expression(expression)
    value = function({key: jnp.array(value) for key, value in context.items()})
    assert float(value) == pytest.approx(expected, rel=1e-5)


def test_expression_symbols_are_reported():
    assert expression_symbols("(iCa/surfaceArea) * rho - concentration/tau") == {
        "iCa",
        "surfaceArea",
        "rho",
        "concentration",
        "tau",
    }


def test_unknown_function_is_rejected():
    with pytest.raises(ParseError, match="Unknown function"):
        compile_expression("wiggle(v)")


def test_undefined_symbol_is_reported_clearly():
    function = compile_expression("a + b")
    with pytest.raises(ParseError, match="not defined here"):
        function({"a": jnp.array(1.0)})


# --------------------------------------------------------------------------- #
# ComponentTypes
# --------------------------------------------------------------------------- #
CUSTOM_GATE_XML = """
<ComponentType name="customHGate" extends="gateHHtauInf">
    <Parameter name="alpha" dimension="none"/>
    <Parameter name="k" dimension="concentration"/>
    <Parameter name="ca_half" dimension="concentration"/>
    <Constant name="SEC" dimension="time" value="1s"/>
    <Requirement name="caConc" dimension="concentration"/>
    <Dynamics>
        <DerivedVariable name="inf" dimension="none" exposure="inf"
                         value="1 / (1 + (exp( (ca_half - caConc) / k)))"/>
        <DerivedVariable name="tau" dimension="time" exposure="tau" value="0 * SEC"/>
        <DerivedVariable name="q" exposure="q" dimension="none" value="inf"/>
        <DerivedVariable name="fcond" exposure="fcond" dimension="none"
                         value="1 +((q-1) * alpha)"/>
    </Dynamics>
</ComponentType>
"""


def _component(xml: str):
    import xml.etree.ElementTree as ET

    return read_component_type(ET.fromstring(xml))


def test_component_type_is_read_with_its_dynamics():
    component = _component(CUSTOM_GATE_XML)
    assert component.extends == "gateHHtauInf"
    assert set(component.parameters) == {"alpha", "k", "ca_half"}
    assert component.constants["SEC"] == (1.0, "time")  # SI: one second
    assert [derived.name for derived in component.derived_variables] == [
        "inf",
        "tau",
        "q",
        "fcond",
    ]


def test_custom_gate_resolves_to_the_gate_family():
    registry = builtin_registry()
    component = _component(CUSTOM_GATE_XML)
    registry[component.name] = component
    assert base_kind_of(component, registry) == "gate"


def test_compiled_dynamics_matches_the_closed_form():
    """c302's calcium-dependent gate, against the expression written by hand."""
    registry = builtin_registry()
    component = _component(CUSTOM_GATE_XML)
    registry[component.name] = component
    resolved = resolve_inheritance(component, registry)

    ca_half, k, alpha = 6.41889e-8, -1.00056e-8, 0.282473
    dynamics = CompiledDynamics(
        resolved, {"alpha": alpha, "k": k, "ca_half": ca_half}
    )
    for calcium in (0.0, 5e-8, 6.41889e-8, 1e-7):
        values = dynamics.evaluate({}, {"caConc": calcium})
        expected_inf = 1.0 / (1.0 + np.exp((ca_half - calcium) / k))
        assert float(values["inf"]) == pytest.approx(expected_inf, rel=1e-5)
        assert float(values["fcond"]) == pytest.approx(
            1.0 + (expected_inf - 1.0) * alpha, rel=1e-5
        )


def test_gathered_variables_only_fail_when_instantiated():
    """A ComponentType using <DerivedVariable select=...> must not break parsing."""
    component = _component(
        """
        <ComponentType name="iafActivityCell" extends="baseCell">
            <Parameter name="C" dimension="capacitance"/>
            <Dynamics>
                <StateVariable name="v" dimension="voltage"/>
                <DerivedVariable name="iSyn" select="synapses[*]/i" reduce="add"/>
                <TimeDerivative variable="v" value="iSyn / C"/>
            </Dynamics>
        </ComponentType>
        """
    )
    assert component.gathered_variables == ["iSyn"]
    assert not component.is_interpretable
    with pytest.raises(ParseError, match="select"):
        CompiledDynamics(component, {"C": 1e-12})


def test_exponential_euler_step_is_exact_for_a_linear_decay():
    component = _component(
        """
        <ComponentType name="decay" extends="concentrationModel">
            <Parameter name="tau" dimension="time"/>
            <Dynamics>
                <StateVariable name="x" dimension="none"/>
                <TimeDerivative variable="x" value="-x / tau"/>
            </Dynamics>
        </ComponentType>
        """
    )
    dynamics = CompiledDynamics(component, {"tau": 0.01})  # 10 ms, in seconds
    state = {"x": jnp.array(1.0)}
    for _ in range(10):  # ten steps of 1 ms
        state = dynamics.step(state, {}, 1e-3)
    assert float(state["x"]) == pytest.approx(np.exp(-1.0), rel=1e-3)


# --------------------------------------------------------------------------- #
# Calcium pools and calcium-dependent gates
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def calcium_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tarjuman.from_neuroml(CALCIUM_MODEL)


def test_species_becomes_a_pump_with_its_own_state(calcium_model):
    module = calcium_model.module
    assert "CaPool" in [pump.name for pump in module.base.pumps]
    assert "CaCon_i" in module.base.pumped_ions
    assert "CaCon_i" in module.nodes.columns


def test_calcium_channels_share_one_current(calcium_model):
    """A pool reads `iCa`, the total calcium current, as NeuroML specifies."""
    channels = {channel.name: channel for channel in calcium_model.module.base.channels}
    assert channels["ca_boyle_all"].current_name == "i_Ca"
    assert channels["k_slow_all"].current_name != "i_Ca"


def test_reversal_potential_does_not_collide_with_a_gate_named_e(calcium_model):
    """Boyle's calcium channel has a gate called 'e'; so did our erev parameter."""
    channels = {channel.name: channel for channel in calcium_model.module.base.channels}
    calcium = channels["ca_boyle_all"]
    assert calcium.channel_params["ca_boyle_all_erev"] == pytest.approx(40.0)
    assert "ca_boyle_all_e" in calcium.channel_states
    assert not set(calcium.channel_params) & set(calcium.channel_states)


def test_calcium_rises_with_current_and_decays_with_its_time_constant(calcium_model):
    result = tarjuman.simulate(
        calcium_model,
        t_max=400.0,
        delta_t=0.05,
        records=["pop[0]/v", "pop[0]/caConc"],
    )
    calcium = np.asarray(result["pop[0]/caConc"])
    voltage = np.asarray(result["pop[0]/v"])
    step = 0.05

    assert calcium.min() >= 0.0  # a pool cannot go negative
    during = calcium[int(200 / step)]
    assert during > 0.0
    assert voltage[int(200 / step)] > voltage[int(40 / step)]  # the pulse depolarises

    # Well after the pulse the calcium current is negligible and the pool is a
    # pure exponential with the model's decayConstant.
    decay_constant = 11.5943
    start = calcium[int(300 / step)]
    after = calcium[int((300 + decay_constant) / step)]
    assert after / start == pytest.approx(np.exp(-1.0), rel=0.1)


def test_custom_gate_conductance_follows_fcond(calcium_model):
    """`fcond = 1 + (q-1)*alpha` is a partial block, not a gate in series."""
    channels = {channel.name: channel for channel in calcium_model.module.base.channels}
    calcium = channels["ca_boyle_all"]
    params = dict(calcium.channel_params)
    states = {name: jnp.array(0.0) for name in calcium.channel_states}
    states["ca_boyle_all_e"] = jnp.array(1.0)
    states["ca_boyle_all_f"] = jnp.array(1.0)
    states["ca_boyle_all_h"] = jnp.array(1.0)

    # With calcium at zero the gate is open (inf -> 1) and fcond -> 1.
    states["CaCon_i"] = jnp.array(0.0)
    open_current = float(calcium.compute_current(states, jnp.array(0.0), params))
    # With calcium high the gate closes, but fcond floors at 1 - alpha = 0.717.
    states["CaCon_i"] = jnp.array(1e-6)
    blocked_current = float(calcium.compute_current(states, jnp.array(0.0), params))
    assert blocked_current / open_current == pytest.approx(1.0 - 0.282473, rel=1e-2)


# --------------------------------------------------------------------------- #
# Custom synapses
# --------------------------------------------------------------------------- #
GRADED_SYNAPSE_XML = """
<ComponentType name="gradedSynapse2" extends="baseGradedSynapse">
    <Parameter name="conductance" dimension="conductance"/>
    <Parameter name="ar" dimension="per_time"/>
    <Parameter name="ad" dimension="per_time"/>
    <Parameter name="beta" dimension="per_voltage"/>
    <Parameter name="vth" dimension="voltage"/>
    <Parameter name="erev" dimension="voltage"/>
    <Exposure name="i" dimension="current"/>
    <Requirement name="v" dimension="voltage"/>
    <Dynamics>
        <StateVariable name="s" dimension="none"/>
        <DerivedVariable name="vpeer" dimension="voltage" select="peer/v"/>
        <DerivedVariable name="phi" dimension="none" value="1/(1 + exp(beta * (vth - vpeer)))"/>
        <DerivedVariable name="i" exposure="i" value="weight * conductance * s * (erev-v)"/>
        <TimeDerivative variable="s" value="ar*phi*(1-s) - ad*s"/>
    </Dynamics>
</ComponentType>
"""


def _graded_synapse():
    registry = builtin_registry()
    component = _component(GRADED_SYNAPSE_XML)
    registry[component.name] = component
    resolved = resolve_inheritance(component, registry)
    return LemsSynapse(
        resolved,
        {  # SI
            "conductance": 5e-9,
            "ar": 0.5,
            "ad": 20.0,
            "beta": 0.25e3,
            "vth": -20e-3,
            "erev": 0.0,
        },
        name="gs",
    )


def test_peer_selector_supplies_the_presynaptic_voltage():
    synapse = _graded_synapse()
    assert synapse.synapse_states == {"gs_s": 0.0}
    assert synapse.synapse_params["gs_conductance"] == pytest.approx(5e-3)  # uS


def test_graded_synapse_reaches_its_analytic_steady_state():
    """s -> ar*phi / (ar*phi + ad) for a held presynaptic voltage."""
    synapse = _graded_synapse()
    params = dict(synapse.synapse_params)
    states = dict(synapse.synapse_states)
    pre, post, dt = 0.0, -45.0, 0.05

    for _ in range(40000):  # 2 s, far longer than 1/ad
        states.update(synapse.update_states(states, dt, pre, post, params))

    ar, ad, beta, vth = 0.5e-3, 20e-3, 0.25, -20.0  # per ms, per mV
    phi = 1.0 / (1.0 + np.exp(beta * (vth - pre)))
    expected = ar * phi / (ar * phi + ad)
    assert float(states["gs_s"]) == pytest.approx(expected, rel=1e-3)


def test_graded_synapse_current_has_jaxleys_sign():
    synapse = _graded_synapse()
    params = dict(synapse.synapse_params)
    states = {"gs_s": jnp.array(1.0)}
    # erev = 0 mV, post = -45 mV: NeuroML's i = g*(erev - v) is inward
    # (positive); Jaxley's convention is outward-positive, so this is negative.
    current = float(synapse.compute_current(states, 0.0, -45.0, params))
    assert current < 0.0
    assert current == pytest.approx(5e-3 * (-45.0 - 0.0), rel=1e-3)


def test_unresolvable_selector_is_rejected_with_an_explanation():
    registry = builtin_registry()
    component = _component(
        GRADED_SYNAPSE_XML.replace('select="peer/v"', 'select="synapses[*]/i"')
    )
    registry[component.name] = component
    with pytest.raises(UnsupportedComponentError, match="peer/v"):
        LemsSynapse(
            resolve_inheritance(component, registry),
            {
                "conductance": 5e-9,
                "ar": 0.5,
                "ad": 20.0,
                "beta": 0.25e3,
                "vth": -20e-3,
                "erev": 0.0,
            },
            name="gs",
        )


# --------------------------------------------------------------------------- #
# Point cells
# --------------------------------------------------------------------------- #
def _iaf_cell(**overrides):
    params = {
        "C": 3.0,  # pF
        "leakConductance": 1e-4,  # uS == 0.1 nS
        "leakReversal": -50.0,
        "thresh": -30.0,
        "reset": -50.0,
    }
    params.update(overrides)
    return ir.PointCell(id="iaf", kind="iafCell", params=params)


def test_iaf_cell_has_the_right_membrane_time_constant():
    """tau = C/g must survive the translation into densities."""
    cell, _ = build_point_cell(_iaf_cell(), ConversionReport())
    node = cell.nodes.iloc[0]
    area = 2 * np.pi * node["radius"] * node["length"]  # um2
    # capacitance uF/cm2, gLeak S/cm2 -> tau in ms
    channel = cell.base.channels[0]
    tau = node["capacitance"] / channel.channel_params["iaf_gLeak"] * 1e-3
    assert area == pytest.approx(100.0, rel=1e-6)
    assert tau == pytest.approx(3.0 / 0.1, rel=1e-6)  # C/g = 3pF / 0.1nS = 30 ms


def test_iaf_cell_fires_at_the_analytic_rate():
    """ISI = tau * ln((v_inf - reset) / (v_inf - thresh))."""
    import jaxley as jx

    report = ConversionReport()
    cell, _ = build_point_cell(_iaf_cell(), report)
    dt, t_max, current = 0.01, 500.0, 0.004  # nA == 4 pA
    cell.branch(0).comp(0).record("v", verbose=False)
    cell.branch(0).comp(0).stimulate(
        jx.step_current(0.0, t_max, current, dt, t_max), verbose=False
    )
    cell.init_states(delta_t=dt)
    voltage = np.asarray(jx.integrate(cell, delta_t=dt, t_max=t_max))[0]
    time = np.arange(voltage.shape[0]) * dt

    crossings = np.where((voltage[:-1] < -31.0) & (voltage[1:] >= -31.0))[0]
    assert len(crossings) > 5
    measured = np.diff(time[crossings]).mean()

    tau = 30.0  # ms
    # e_leak + I/g: nA / uS is mV.
    v_infinity = -50.0 + current / 1e-4
    expected = tau * np.log((v_infinity + 50.0) / (v_infinity + 30.0))
    assert measured == pytest.approx(expected, rel=0.02)
    assert report.approximations  # the reset is an approximation, and says so


def test_iaf_reset_lands_on_the_reset_potential():
    import jaxley as jx

    cell, _ = build_point_cell(_iaf_cell(), ConversionReport())
    dt = 0.01
    cell.branch(0).comp(0).record("v", verbose=False)
    cell.branch(0).comp(0).stimulate(
        jx.step_current(0.0, 200.0, 0.004, dt, 200.0), verbose=False
    )
    cell.init_states(delta_t=dt)
    voltage = np.asarray(jx.integrate(cell, delta_t=dt, t_max=200.0))[0]
    assert voltage.max() < -29.0  # never overshoots threshold
    assert voltage.min() == pytest.approx(-50.0, abs=0.2)


def test_refractory_period_lengthens_the_interval():
    import jaxley as jx

    intervals = {}
    for refract in (0.0, 10.0):
        point_cell = ir.PointCell(
            id="iaf",
            kind="iafRefCell",
            params={
                "C": 3.0,
                "leakConductance": 1e-4,
                "leakReversal": -50.0,
                "thresh": -30.0,
                "reset": -50.0,
                "refract": refract,
            },
        )
        cell, _ = build_point_cell(point_cell, ConversionReport())
        dt = 0.025
        cell.branch(0).comp(0).record("v", verbose=False)
        cell.branch(0).comp(0).stimulate(
            jx.step_current(0.0, 400.0, 0.004, dt, 400.0), verbose=False
        )
        cell.init_states(delta_t=dt)
        voltage = np.asarray(jx.integrate(cell, delta_t=dt, t_max=400.0))[0]
        crossings = np.where((voltage[:-1] < -31.0) & (voltage[1:] >= -31.0))[0]
        intervals[refract] = np.diff(crossings).mean() * dt

    assert intervals[10.0] == pytest.approx(intervals[0.0] + 10.0, rel=0.1)


def test_unsupported_point_cell_says_why():
    point_cell = ir.PointCell(id="izh", kind="izhikevich2007Cell", params={})
    with pytest.raises(UnsupportedComponentError, match="izhikevich2007Cell"):
        build_point_cell(point_cell, ConversionReport())
