# SPDX-License-Identifier: Apache-2.0
"""Handwritten Jaxley model for Ex25_MultiComp.

Equations and model values are transcribed from the original NeuroML files.
Upstream revisions and licenses are listed in models.json.
Voltages are in mV, times in ms, and conductance densities in S/cm2.
"""

import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Channel
from jaxley.solver_gate import solve_inf_gate_exponential
from jaxley.synapses import Synapse


def exp_linear(x):
    denominator = jnp.where(x == 0, 1.0, 1.0 - jnp.exp(-x))
    return jnp.where(x == 0, 1.0, x / denominator)


def rates(alpha, beta, q10=1.0):
    return alpha / (alpha + beta), 1.0 / ((alpha + beta) * q10)


class OhmicChannel(Channel):
    gates = ()
    calcium_current = False

    def __init__(self, name, gbar, erev, q10=1.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.current_name = "i_Ca" if self.calcium_current else "i_" + name
        self.channel_params = {
            name + "_gbar": gbar,
            name + "_erev": erev,
            name + "_q10": q10,
        }
        self.channel_states = {name + "_" + gate: 0.0 for gate in self.gates}

    def kinetics(self, v, states, params):
        return {}

    def init_state(self, states, v, params, delta_t):
        return {
            self.name + "_" + gate: jnp.broadcast_to(inf, jnp.shape(v))
            for gate, (inf, tau) in self.kinetics(v, states, params).items()
        }

    def update_states(self, states, dt, v, params):
        return {
            self.name + "_" + gate: solve_inf_gate_exponential(
                states[self.name + "_" + gate], dt, inf, tau
            )
            for gate, (inf, tau) in self.kinetics(v, states, params).items()
        }

    def opening(self, states):
        return 1.0

    def compute_current(self, states, v, params):
        return (
            params[self.name + "_gbar"]
            * self.opening(states)
            * (v - params[self.name + "_erev"])
        )


class Leak(OhmicChannel):
    pass


class HHNa(OhmicChannel):
    gates = ("m", "h")

    def kinetics(self, v, states, params):
        q10 = params[self.name + "_q10"]
        return {
            "m": rates(
                exp_linear((v + 40.0) / 10.0), 4.0 * jnp.exp(-(v + 65.0) / 18.0), q10
            ),
            "h": rates(
                0.07 * jnp.exp(-(v + 65.0) / 20.0),
                1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0)),
                q10,
            ),
        }

    def opening(self, states):
        return states[self.name + "_m"] ** 3 * states[self.name + "_h"]


class HHK(OhmicChannel):
    gates = ("n",)

    def kinetics(self, v, states, params):
        return {
            "n": rates(
                0.1 * exp_linear((v + 55.0) / 10.0),
                0.125 * jnp.exp(-(v + 65.0) / 80.0),
                params[self.name + "_q10"],
            )
        }

    def opening(self, states):
        return states[self.name + "_n"] ** 4


class ExpTwoSynapse(Synapse):
    def __init__(self, name, gbase, rise, decay):
        super().__init__(name)
        self.synapse_params = {
            name + "_gbase": gbase,
            name + "_rise": rise,
            name + "_decay": decay,
            name + "_erev": 0.0,
            name + "_threshold": -20.0,
        }
        self.synapse_states = {name + "_A": 0.0, name + "_B": 0.0, name + "_above": 0.0}

    def update_states(self, states, dt, pre_voltage, post_voltage, params):
        p = self.name + "_"
        rise, decay = params[p + "rise"], params[p + "decay"]
        peak = jnp.log(decay / rise) / (1.0 / rise - 1.0 / decay)
        event = (pre_voltage > params[p + "threshold"]) & (states[p + "above"] == 0.0)
        above = jnp.where(
            pre_voltage < params[p + "threshold"],
            0.0,
            jnp.where(event, 1.0, states[p + "above"]),
        )
        jump = event / (jnp.exp(-peak / decay) - jnp.exp(-peak / rise))
        return {
            p + "A": states[p + "A"] * jnp.exp(-dt / rise) + jump,
            p + "B": states[p + "B"] * jnp.exp(-dt / decay) + jump,
            p + "above": above,
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        p = self.name + "_"
        return (
            params[p + "gbase"]
            * (states[p + "B"] - states[p + "A"])
            * (post_voltage - params[p + "erev"])
        )


class NMDASynapse(ExpTwoSynapse):
    def __init__(self):
        super().__init__("NMDA", 0.0008, 1.0, 13.3333)
        self.synapse_params.update(
            NMDA_blockConcentration=1.2,
            NMDA_scalingConc=1.9205441817997078,
            NMDA_scalingVolt=16.129032258064516,
        )

    def compute_current(self, states, pre_voltage, post_voltage, params):
        current = super().compute_current(states, pre_voltage, post_voltage, params)
        return current / (
            1.0
            + params["NMDA_blockConcentration"]
            / params["NMDA_scalingConc"]
            * jnp.exp(-post_voltage / params["NMDA_scalingVolt"])
        )


DT = 0.005
DURATION = 140.0


def build_model(dt=DT, duration=DURATION):
    cell = jx.Cell(
        [jx.Branch(jx.Compartment(), ncomp=n) for n in (1, 1, 9)],
        parents=[-1, 0, 1],
    )
    cell.branch(0).set("length", 10.0)
    cell.branch(0).set("radius", 5.0)
    cell.branch(1).set("length", 10.0)
    cell.branch(1).set("radius", 1.5)
    cell.branch(2).set("length", 30.0 / 9)
    for compartment in range(9):
        cell.branch(2).comp(compartment).set(
            "radius", 1.375 if compartment < 3 else 1.0
        )
    cell.set("capacitance", 1.0)
    cell.set("axial_resistivity", 100000.0)
    cell.set("v", -65.0)
    cell.insert(Leak("leak", 0.0003, -54.3))
    cell.insert(HHNa("naChans", 0.12, 50.0))
    cell.insert(HHK("kChans", 0.036, -77.0))
    net = jx.Network([cell, cell, cell])
    ampa = ExpTwoSynapse("AMPA", 0.0003, 0.03, 0.5)
    nmda = NMDASynapse()
    pre0 = net.cell(0).branch(0).comp(0)
    pre2 = net.cell(2).branch(0).comp(0)
    post = net.cell(1)
    # Segment 3 spans the last 20 um of branch 2. Its source fractions 0.25,
    # 0.3 and 0.5 therefore select compartments 4, 4 and 6, respectively.
    jx.connect(pre0, post.branch(0).comp(0), ampa)
    jx.connect(pre0, post.branch(2).comp(4), ampa)
    jx.connect(pre2, post.branch(0).comp(0), ampa)
    jx.connect(pre2, post.branch(1).comp(0), ampa)
    jx.connect(pre2, post.branch(2).comp(4), ampa)
    jx.connect(pre0, post.branch(0).comp(0), nmda)
    jx.connect(pre0, post.branch(2).comp(6), nmda)
    jx.connect(pre2, post.branch(0).comp(0), nmda)
    jx.connect(pre2, post.branch(1).comp(0), nmda)
    jx.connect(pre2, post.branch(2).comp(4), nmda)
    pre0.stimulate(jx.step_current(20.0, 100.0, 0.2, dt, duration), verbose=False)
    pre2.stimulate(jx.step_current(30.0, 100.0, 0.18, dt, duration), verbose=False)
    return net, {0: -20.0, 1: -20.0, 2: -20.0}
