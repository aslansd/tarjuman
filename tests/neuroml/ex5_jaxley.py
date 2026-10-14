# SPDX-License-Identifier: Apache-2.0
"""Handwritten Jaxley model for Ex5_DetCell.

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


def hh_cell(q10=1.0):
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    # NML2_SingleCompHHCell: the cylinder preserves the spherical soma's area.
    cell.set("length", 17.841242)
    cell.set("radius", 17.841242 / 2)
    cell.set("capacitance", 1.0)
    cell.set("axial_resistivity", 30.0)
    cell.set("v", -65.0)
    cell.insert(Leak("leak", 0.0003, -54.3))
    cell.insert(HHNa("naChans", 0.12, 50.0, q10))
    cell.insert(HHK("kChans", 0.036, -77.0, q10))
    return cell


DT = 0.01
DURATION = 300.0


def build_model(dt=DT, duration=DURATION):
    net = jx.Network([hh_cell()])
    current = jx.step_current(100.0, 100.0, 0.08, dt, duration)
    net.cell(0).branch(0).comp(0).stimulate(current, verbose=False)
    return net, {0: -20.0}
