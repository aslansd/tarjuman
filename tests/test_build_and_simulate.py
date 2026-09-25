"""Building and running converted models, and checking they are *correct*.

The important tests here are the ones that pin the converted model to
something independent: Jaxley's own Hodgkin-Huxley channel, and the analytic
membrane time constant of a passive compartment.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pytest
from jaxley.channels import HH

import tarjuman
from tarjuman import ir
from tarjuman.channels import make_channel
from tarjuman.channels.expressions import exp_linear_rate, exp_rate, sigmoid_rate
from tarjuman.synapses import ExpTwoSynapse, GapJunction, make_synapse

DATA = Path(__file__).parent / "data"
HH_MODEL = DATA / "hh_single_comp.nml"
ANALOG_MODEL = DATA / "two_cell_analog.nml"
HH_LEMS = DATA / "LEMS_hh_single_comp.xml"


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------- #
# Rate expressions
# --------------------------------------------------------------------------- #
def test_rate_forms_match_their_lems_definitions():
    v = jnp.array(-42.0)
    assert float(exp_rate(v, 4.0, -65.0, -18.0)) == pytest.approx(
        4.0 * np.exp((-42.0 + 65.0) / -18.0), rel=1e-6
    )
    assert float(sigmoid_rate(v, 1.0, -35.0, 10.0)) == pytest.approx(
        1.0 / (1.0 + np.exp(-(-42.0 + 35.0) / 10.0)), rel=1e-6
    )
    x = (-42.0 + 40.0) / 10.0
    assert float(exp_linear_rate(v, 1.0, -40.0, 10.0)) == pytest.approx(
        x / (1.0 - np.exp(-x)), rel=1e-6
    )


def test_exp_linear_rate_is_finite_at_its_singularity():
    """``x / (1 - exp(-x))`` is removable at x = 0 and must not produce NaN."""
    value = exp_linear_rate(jnp.array(-40.0), 1.0, -40.0, 10.0)
    assert float(value) == pytest.approx(1.0, abs=1e-5)
    gradient = jax.grad(lambda v: exp_linear_rate(v, 1.0, -40.0, 10.0))(
        jnp.array(-40.0)
    )
    assert np.isfinite(float(gradient))


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
def test_channel_exposes_every_rate_parameter():
    document = tarjuman.read_neuroml(HH_MODEL)
    density = ir.ChannelDensity(
        id="naChans", ion_channel="naChan", cond_density=0.12, erev=50.0
    )
    channel = make_channel(document.ion_channels["naChan"], density)
    assert channel.name == "naChans"
    assert channel.channel_params["naChans_gbar"] == pytest.approx(0.12)
    # The reversal potential is "_erev", not "_e": a gate may be called "e".
    assert channel.channel_params["naChans_erev"] == pytest.approx(50.0)
    assert channel.channel_params["naChans_m_alpha_midpoint"] == pytest.approx(-40.0)
    assert set(channel.channel_states) == {"naChans_m", "naChans_h"}


def test_q10_scales_the_time_constant():
    gate = ir.Gate(
        id="n",
        instances=1,
        kind="gateHHrates",
        forward_rate=ir.Rate("HHExpRate", 1.0, -50.0, 10.0),
        reverse_rate=ir.Rate("HHExpRate", 1.0, -50.0, -10.0),
        q10=ir.Q10(kind="q10ExpTemp", q10_factor=3.0, experimental_temp=279.45),
    )
    channel_ir = ir.IonChannel(id="k", kind="ionChannelHH", gates=[gate])
    cold = make_channel(channel_ir, temperature=279.45)
    warm = make_channel(channel_ir, temperature=289.45)  # +10 K -> x3
    assert cold.channel_params["k_n_rateScale"] == pytest.approx(1.0)
    assert warm.channel_params["k_n_rateScale"] == pytest.approx(3.0)


def test_converted_hh_matches_jaxleys_own_hh_channel():
    """The strongest check available: same model, two independent code paths.

    NML2_SingleCompHHCell uses the textbook Hodgkin-Huxley rate equations, which
    are also what ``jaxley.channels.HH`` implements.  Converted and built-in
    must therefore produce the same voltage trace.
    """
    dt, t_max = 0.01, 200.0
    stimulus = jx.step_current(50.0, 100.0, 0.08, dt, t_max)

    model = tarjuman.from_neuroml(HH_MODEL, network_id="net1")
    model.module.delete_stimuli()
    model.view("hhpop", 0).stimulate(stimulus, verbose=False)
    result = tarjuman.simulate(
        model,
        t_max=t_max,
        delta_t=dt,
        records=["hhpop[0]/v"],
        apply_stimuli=False,
    )
    converted = np.asarray(result["hhpop[0]/v"])

    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    cell.set("length", 17.841242)
    cell.set("radius", 17.841242 / 2)
    cell.set("capacitance", 1.0)
    cell.set("axial_resistivity", 30.0)
    cell.set("v", -65.0)
    for key, value in {
        "HH_gNa": 0.12,
        "HH_gK": 0.036,
        "HH_gLeak": 3e-4,
        "HH_eNa": 50.0,
        "HH_eK": -77.0,
        "HH_eLeak": -54.3,
    }.items():
        cell.set(key, value)
    cell.branch(0).comp(0).record("v", verbose=False)
    cell.branch(0).comp(0).stimulate(stimulus, verbose=False)
    cell.init_states(delta_t=dt)
    reference = np.asarray(jx.integrate(cell, delta_t=dt, t_max=t_max))[0]

    n = min(len(converted), len(reference))
    assert np.abs(converted[:n] - reference[:n]).max() < 0.5  # mV
    spikes = lambda v: int(np.sum((v[:-1] < 0) & (v[1:] >= 0)))
    assert spikes(converted) == spikes(reference) > 0


def test_passive_compartment_has_the_analytic_time_constant():
    """tau = Rm * Cm = cm / g_leak, independent of geometry."""
    dt, t_max = 0.01, 40.0
    model = tarjuman.from_neuroml(HH_MODEL, network_id="net1")
    module = model.module
    module.set("naChans_gbar", 0.0)
    module.set("kChans_gbar", 0.0)
    module.set("v", -65.0)
    result = tarjuman.simulate(
        model, t_max=t_max, delta_t=dt, records=["hhpop[0]/v"], apply_stimuli=False
    )
    v = np.asarray(result["hhpop[0]/v"])

    e_leak, v0 = -54.3, -65.0
    tau_expected = 1.0 / 3e-4 * 1e-3  # (1/g) * cm, in ms
    decay = (v - e_leak) / (v0 - e_leak)
    tau_fit = -result.time[1:] / np.log(np.clip(decay[1:], 1e-9, None))
    assert np.median(tau_fit[100:]) == pytest.approx(tau_expected, rel=0.02)


# --------------------------------------------------------------------------- #
# Synapses
# --------------------------------------------------------------------------- #
def test_exp_two_synapse_peaks_at_gbase():
    """One event must give a peak conductance of exactly gbase * weight."""
    synapse = ExpTwoSynapse(
        name="syn", erev=0.0, gbase=1e-3, tau_rise=0.5, tau_decay=5.0
    )
    params = dict(synapse.synapse_params)
    states = dict(synapse.synapse_states)
    dt = 0.001
    peak = 0.0
    for step in range(20000):
        pre_v = 10.0 if step == 10 else -70.0
        states.update(synapse.update_states(states, dt, pre_v, -70.0, params))
        conductance = params["syn_gbase"] * (states["syn_B"] - states["syn_A"])
        peak = max(peak, float(conductance))
    assert peak == pytest.approx(1e-3, rel=2e-3)


def test_synapse_fires_once_per_threshold_crossing():
    synapse = ExpTwoSynapse(name="syn", gbase=1.0, tau_rise=0.5, tau_decay=5.0)
    params = dict(synapse.synapse_params)
    states = dict(synapse.synapse_states)
    # Ten steps above threshold is one crossing, not ten.
    for _ in range(10):
        states.update(synapse.update_states(states, 0.025, 10.0, -70.0, params))
    single = float(states["syn_B"])
    states = dict(synapse.synapse_states)
    for step in range(10):
        pre_v = 10.0 if step == 0 else -70.0
        states.update(synapse.update_states(states, 0.025, pre_v, -70.0, params))
    assert single == pytest.approx(float(states["syn_B"]), rel=1e-6)


def test_gap_junction_current_is_ohmic_and_antisymmetric():
    junction = GapJunction(name="gj", conductance=0.5)
    params = dict(junction.synapse_params)
    forward = junction.compute_current({}, -60.0, -70.0, params)
    backward = junction.compute_current({}, -70.0, -60.0, params)
    assert float(forward) == pytest.approx(0.5 * (-70.0 + 60.0))
    assert float(forward) == pytest.approx(-float(backward))


def test_unsupported_synapse_type_raises():
    from tarjuman.errors import UnsupportedComponentError

    with pytest.raises(UnsupportedComponentError, match="no Jaxley equivalent"):
        make_synapse(ir.Synapse(id="s", kind="doubleSynapse"))


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def test_analog_network_builds_gap_junctions_in_both_directions():
    model = tarjuman.from_neuroml(ANALOG_MODEL)
    edges = model.module.edges
    assert sorted(edges["type"]) == ["gj1", "gj1", "gs1"]
    gap = edges[edges["type"] == "gj1"]
    pre, post = gap["pre_index"].tolist(), gap["post_index"].tolist()
    assert set(zip(pre, post)) == {(pre[0], post[0]), (post[0], pre[0])}


def test_silent_synapse_endpoint_contributes_no_edge():
    model = tarjuman.from_neuroml(ANALOG_MODEL)
    assert "silent1" not in set(model.module.edges["type"])


def test_channel_density_respects_its_segment_group():
    model = tarjuman.from_neuroml(ANALOG_MODEL)
    nodes = model.module.nodes
    soma = nodes[nodes["global_branch_index"] == 0]
    dendrite = nodes[nodes["global_branch_index"] == 1]
    assert soma["naChans"].all()
    assert not dendrite["naChans"].any()
    assert dendrite["leak"].all()  # the leak has no segmentGroup: it is everywhere


def test_conversion_report_records_geometry_and_counts():
    model = tarjuman.from_neuroml(ANALOG_MODEL)
    report = model.report
    assert (report.n_cells, report.n_branches, report.n_compartments) == (3, 6, 18)
    assert report.n_synapses == 3
    assert report.is_faithful


def test_locating_a_neuroml_position_in_jaxley():
    model = tarjuman.from_neuroml(ANALOG_MODEL)
    # Segment 1 is the 50 um dendrite, split into 5 compartments.
    assert model.locate("popA", 0, 1, 0.1) == (0, 1, 0)
    assert model.locate("popA", 0, 1, 0.9) == (0, 1, 4)
    assert model.locate("popB", 0, 0, 0.5)[0] == 1


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def test_run_lems_reproduces_the_requested_columns():
    result, model = tarjuman.run_lems(HH_LEMS)
    assert list(result.traces) == ["v", "m", "n"]
    assert result.time[-1] == pytest.approx(300.0)
    v = result["v"]
    assert v.min() < -70.0 and v.max() > 0.0  # the cell spikes
    assert 0.0 <= result["m"].min() and result["m"].max() <= 1.0


def test_run_lems_writes_si_units_like_jnml(tmp_path):
    result, _ = tarjuman.run_lems(HH_LEMS)
    path = result.save(tmp_path / "out.dat")
    data = np.loadtxt(path)
    assert data[-1, 0] == pytest.approx(0.3)  # 300 ms in seconds
    assert data[:, 1].min() > -0.1  # volts, not millivolts


def test_stimulus_timing_follows_the_pulse_generator():
    model = tarjuman.from_neuroml(HH_MODEL, network_id="net1")
    result = tarjuman.simulate(model, t_max=300.0, delta_t=0.025)
    v = np.asarray(list(result.traces.values())[0])
    before = v[: int(90 / 0.025)]
    during = v[int(120 / 0.025) : int(190 / 0.025)]
    assert before.max() < -50.0  # pulseGen1 starts at 100 ms
    assert during.max() > 0.0


def test_gates_start_at_steady_state_as_neuroml_specifies():
    model = tarjuman.from_neuroml(HH_MODEL, network_id="net1")
    model.module.init_states(delta_t=0.025)
    m = model.module.nodes["naChans_m"].iloc[0]
    # alpha_m / (alpha_m + beta_m) at -65 mV for the standard HH equations.
    assert m == pytest.approx(0.0529, abs=1e-3)


# --------------------------------------------------------------------------- #
# The reason for doing this on Jaxley at all
# --------------------------------------------------------------------------- #
def test_converted_model_is_differentiable_in_its_channel_parameters():
    model = tarjuman.from_neuroml(HH_MODEL, network_id="net1")
    module = model.module
    module.delete_recordings()
    module.delete_stimuli()
    module.init_states(delta_t=0.025)
    module.cell(0).branch(0).comp(0).record("v", verbose=False)
    module.cell(0).branch(0).comp(0).stimulate(
        jx.step_current(2.0, 10.0, 0.2, 0.025, 20.0), verbose=False
    )
    module.make_trainable("naChans_gbar")
    parameters = module.get_parameters()

    def peak_voltage(parameters):
        voltages = jx.integrate(module, params=parameters, delta_t=0.025, t_max=20.0)
        return jnp.max(voltages)

    gradient = jax.grad(peak_voltage)(parameters)
    value = float(list(gradient[0].values())[0].ravel()[0])
    assert np.isfinite(value) and value != 0.0


# --------------------------------------------------------------------------- #
# Synaptic delays and batched connection
# --------------------------------------------------------------------------- #
DELAYED_MODEL = DATA / "delayed_chemical.nml"


def test_delayed_synapse_hands_over_the_voltage_n_steps_late():
    from tarjuman.synapses import DelayedSynapse, ExpTwoSynapse

    inner = ExpTwoSynapse(name="syn", gbase=1.0, tau_rise=0.5, tau_decay=5.0)
    synapse = DelayedSynapse(inner, delay_steps=4, initial_voltage=-70.0)
    params = dict(synapse.synapse_params)
    states = dict(synapse.synapse_states)

    responses = []
    for step in range(8):
        pre_v = 10.0 if step == 0 else -70.0  # one spike, at step 0
        states.update(synapse.update_states(states, 0.025, pre_v, -70.0, params))
        responses.append(float(states["syn_B"]))

    # The wrapped synapse sees the spike only once it reaches the end of the
    # register: nothing happens for the first four steps.
    assert all(value == 0.0 for value in responses[:4])
    assert responses[4] > 0.0


def test_delay_is_built_when_the_time_step_is_known():
    model = tarjuman.from_neuroml(DELAYED_MODEL, delta_t=0.025)
    assert sorted(set(model.module.edges["type"])) == ["ampa_delay200"]  # 5 ms / 0.025
    assert any("shift registers" in note.message for note in model.report.notes)


def test_delay_is_dropped_when_the_time_step_is_unknown():
    model = tarjuman.from_neuroml(DELAYED_MODEL)
    assert sorted(set(model.module.edges["type"])) == ["ampa"]
    assert any(
        "dropped" in note.message and "delay" in note.message
        for note in model.report.approximations
    )


def test_delay_shifts_the_postsynaptic_response_by_the_right_amount():
    lags = {}
    for delta_t in (None, 0.025):
        model = tarjuman.from_neuroml(
            DELAYED_MODEL, **({"delta_t": delta_t} if delta_t else {})
        )
        result = tarjuman.simulate(
            model, t_max=120.0, delta_t=0.025, records=["pre[0]/v", "post[0]/v"]
        )
        pre = np.asarray(result["pre[0]/v"])
        post = np.asarray(result["post[0]/v"])
        spike = result.time[np.argmax(pre > 0.0)]
        response = result.time[np.argmax(post > -64.9)]
        lags[delta_t] = response - spike

    assert lags[0.025] - lags[None] == pytest.approx(5.0, abs=0.05)


def test_batched_connection_pairs_the_right_compartments():
    """One `connect` per projection must give the same edges as one per connection."""
    from tarjuman import read_neuroml_string
    from tarjuman.builder import build

    text = """
    <neuroml id="doc">
      <expOneSynapse id="syn" gbase="1nS" erev="0mV" tauDecay="2ms"/>
      <cell id="c">
        <morphology id="m">
          <segment id="0"><proximal x="0" y="0" z="0" diameter="10"/>
            <distal x="0" y="10" z="0" diameter="10"/></segment>
        </morphology>
      </cell>
      <network id="net">
        <population id="p" component="c" size="4"/>
        <projection id="proj" presynapticPopulation="p" postsynapticPopulation="p" synapse="syn">
          <connectionWD id="0" preCellId="../p[3]" postCellId="../p[0]" weight="2" delay="0ms"/>
          <connectionWD id="1" preCellId="../p[0]" postCellId="../p[2]" weight="1" delay="0ms"/>
          <connectionWD id="2" preCellId="../p[1]" postCellId="../p[3]" weight="7" delay="0ms"/>
        </projection>
      </network>
    </neuroml>
    """
    model = build(read_neuroml_string(text), network_id="net")
    edges = model.module.edges
    assert list(zip(edges["pre_index"], edges["post_index"])) == [(3, 0), (0, 2), (1, 3)]
    assert edges["syn_weight"].tolist() == [2.0, 1.0, 7.0]
