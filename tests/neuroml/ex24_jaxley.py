# SPDX-License-Identifier: Apache-2.0
"""Handwritten Jaxley model for Ex24_FractionalConductance.

Equations and model values are transcribed from the original NeuroML files.
Upstream revisions and licenses are listed in models.json.
Voltages are in mV, times in ms, and conductance densities in S/cm2.
"""

import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Channel
from jaxley.solver_gate import solve_inf_gate_exponential


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


class SlowK(OhmicChannel):
    gates = ("a", "b_bb", "b_bb1")

    def kinetics(self, v, states, params):
        alpha = 0.0052 * 13.1 * exp_linear((v - 11.1) / 13.1)
        beta = 0.01938 * jnp.exp(-(v + 1.27) / 71.0) - 0.0053
        inf = 1.0 / (1.0 + jnp.exp((v + 58.0) / 11.0))
        return {
            "a": rates(alpha, beta, params[self.name + "_q10"]),
            "b_bb": (
                inf,
                360.0
                + (1010.0 + 23.7 * (v + 54.0)) * jnp.exp(-((v + 75.0) ** 2) / 48.0**2),
            ),
            "b_bb1": (
                inf,
                2350.0 + 1380.0 * jnp.exp(-0.01118 * v) - 210.0 * jnp.exp(-0.0306 * v),
            ),
        }

    def opening(self, states):
        return states[self.name + "_a"] ** 2 * (
            0.5 * states[self.name + "_b_bb"] + 0.5 * states[self.name + "_b_bb1"]
        )


DT = 0.01
DURATION = 200.0


def build_model(dt=DT, duration=DURATION):
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.set("length", 20.0)
    cell.set("radius", 5.0)
    cell.set("capacitance", 1.0)
    cell.set("axial_resistivity", 100.0)
    cell.set("v", -65.0)
    cell.insert(HHNa("naChans", 0.12, 50.0))
    cell.insert(HHK("kChans", 0.036, -77.0))
    cell.insert(SlowK("k_slowChans", 0.036, -77.0, 2.3 ** ((34.0 - 21.0) / 10.0)))
    cell.insert(Leak("leakChans", 0.0003, -54.3))
    net = jx.Network([cell])
    soma = net.cell(0).branch(0).comp(0)
    soma.stimulate(jx.step_current(25.0, 50.0, 0.1, dt, duration), verbose=False)
    soma.stimulate(jx.step_current(100.0, 50.0, -0.1, dt, duration), verbose=False)
    return net, {0: 0.0}
