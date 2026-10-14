# SPDX-License-Identifier: Apache-2.0
"""Handwritten Jaxley model for Ex15_CaDynamics.

Equations and model values are transcribed from the original NeuroML files.
Upstream revisions and licenses are listed in models.json.
Voltages are in mV, times in ms, and conductance densities in S/cm2.
"""

import math

import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Channel
from jaxley.pumps import Pump
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


class GranuleNa(OhmicChannel):
    gates = ("m", "h")

    def kinetics(self, v, states, params):
        q10 = params[self.name + "_q10"]
        am = 1.5 * jnp.exp((v + 28.999999999999996) / 12.345679)
        bm = 1.5 * jnp.exp(-(v + 28.999999999999996) / 15.1515)
        ah = 0.12 * jnp.exp(-(v + 40.0) / 11.23596)
        bh = 0.12 * jnp.exp((v + 40.0) / 11.23596)
        return {
            "m": (am / (am + bm), jnp.maximum(1.0 / (am + bm), 0.05) / q10),
            "h": (ah / (ah + bh), jnp.maximum(1.0 / (ah + bh), 0.225) / q10),
        }

    def opening(self, states):
        return states[self.name + "_m"] ** 3 * states[self.name + "_h"]


class GranuleKDr(OhmicChannel):
    gates = ("m", "h")

    def kinetics(self, v, states, params):
        shifted = v - 10.0
        q10 = params[self.name + "_q10"]
        am = 0.17 * jnp.exp(0.073 * (shifted + 38.0))
        bm = 0.17 * jnp.exp(-0.018 * (shifted + 38.0))
        ah = jnp.where(
            shifted > -46.0,
            0.00076,
            0.0007 + 0.000065 * jnp.exp(-0.08 * (shifted + 46.0)),
        )
        bh = 0.0011 / (1.0 + jnp.exp(-0.0807 * (shifted + 44.0)))
        return {"m": rates(am, bm, q10), "h": rates(ah, bh, q10)}

    def opening(self, states):
        return states[self.name + "_m"] ** 4 * states[self.name + "_h"]


class GranuleH(OhmicChannel):
    gates = ("n",)

    def kinetics(self, v, states, params):
        return {
            "n": rates(
                0.0008 * jnp.exp(-(v + 65.0) / 11.00110011),
                0.0008 * jnp.exp((v + 65.0) / 11.00110011),
                params[self.name + "_q10"],
            )
        }

    def opening(self, states):
        return states[self.name + "_n"]


class GranuleCa(OhmicChannel):
    gates = ("m", "h")
    calcium_current = True

    def kinetics(self, v, states, params):
        shifted = v - 10.0
        q10 = params[self.name + "_q10"]
        am = 1.6 / (1.0 + jnp.exp(-(v - 15.0) / 13.88888889))
        bm = 0.1 * exp_linear((v - 1.1000000000000003) / -5.0)
        decay = 0.005 * jnp.exp(-0.05 * (shifted + 60.0))
        ah = jnp.where(shifted < -60.0, 0.005, decay)
        bh = jnp.where(shifted < -60.0, 0.0, 0.005 - decay)
        return {"m": rates(am, bm, q10), "h": rates(ah, bh, q10)}

    def opening(self, states):
        return states[self.name + "_m"] ** 2 * states[self.name + "_h"]


class GranuleKA(OhmicChannel):
    gates = ("m", "h")

    def kinetics(self, v, states, params):
        shifted = v - 10.0
        return {
            "m": (
                1.0 / (1.0 + jnp.exp(-(v + 36.699999999999996) / 19.8)),
                0.410 * jnp.exp((shifted + 43.5) / -42.8) + 0.167,
            ),
            "h": (
                1.0 / (1.0 + jnp.exp((v + 68.8) / 8.4)),
                10.8
                + 0.03 * shifted
                + 1.0
                / (
                    57.9 * jnp.exp(0.127 * shifted) + 134e-6 * jnp.exp(-0.059 * shifted)
                ),
            ),
        }

    def opening(self, states):
        return states[self.name + "_m"] ** 3 * states[self.name + "_h"]


class GranuleKCa(OhmicChannel):
    gates = ("m",)

    def __init__(self, name, gbar, erev, q10=1.0):
        super().__init__(name, gbar, erev, q10)
        self.channel_states["CaCon_i"] = 7.55e-5

    def kinetics(self, v, states, params):
        shifted = v - 10.0
        calcium = states["CaCon_i"]
        alpha = 2.5 * calcium / (calcium + 1.5e-3 * jnp.exp(-0.085 * shifted))
        beta = 1.5 / (1.0 + calcium / (1.5e-4 * jnp.exp(-0.077 * shifted)))
        return {"m": rates(alpha, beta, params[self.name + "_q10"])}

    def opening(self, states):
        return states[self.name + "_m"]


class GranuleCaPool(Pump):
    def __init__(self):
        super().__init__("Gran_CaPool")
        self.ion_name, self.current_name = "CaCon_i", "i_Gran_CaPool"
        self.channel_params = {
            "Gran_CaPool_rest": 7.55e-5,
            "Gran_CaPool_tau": 10.0,
            "Gran_CaPool_shell": 0.084,
        }
        self.channel_states = {"CaCon_i": 7.55e-5, "i_Ca": 0.0}

    def init_state(self, states, v, params, delta_t):
        return {"CaCon_i": 7.55e-5}

    def update_states(self, states, dt, v, params):
        return {"CaCon_i": jnp.maximum(states["CaCon_i"], 0.0)}

    def compute_current(self, states, concentration, params):
        area = 2.0 * math.pi * params["radius"] * params["length"] * 1e-12
        radius = jnp.sqrt(area / (4.0 * 3.14159))
        inner = radius - params["Gran_CaPool_shell"] * 1e-6
        volume = 4.0 * radius**3 * 3.14159 / 3.0 - 4.0 * inner**3 * 3.14159 / 3.0
        inward_current = -10.0 * states["i_Ca"] * area
        return (concentration - params["Gran_CaPool_rest"]) / params[
            "Gran_CaPool_tau"
        ] - inward_current / (2.0 * 96485.3 * volume) / 1000.0


DT = 0.001
DURATION = 700.0


def build_model(dt=DT, duration=DURATION):
    q10 = 3.0 ** ((32.0 - 17.350264793) / 10.0)
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.set("length", 10.0)
    cell.set("radius", 5.0)
    cell.set("capacitance", 1.0)
    cell.set("axial_resistivity", 100.0)
    cell.set("v", -65.0)
    cell.insert(Leak("GranPassiveCond_all", 0.0330033e-3, -65.0))
    cell.insert(GranuleKDr("Gran_KDr_98_all", 8.89691e-3, -90.0, q10))
    cell.insert(GranuleNa("Gran_NaF_98_all", 55.7227e-3, 55.0, q10))
    cell.insert(GranuleH("Gran_H_98_all", 0.03090506e-3, -42.0, q10))
    cell.insert(GranuleCa("Gran_CaHVA_98_all", 0.9084216e-3, 80.0, q10))
    cell.insert(GranuleKA("Gran_KA_98_all", 1.14567e-3, -90.0))
    cell.insert(GranuleKCa("Gran_KCa_98_all", 17.9811e-3, -90.0, q10))
    net = jx.Network([cell])
    net.cell(0).insert(GranuleCaPool())
    current = jx.step_current(100.0, 500.0, 0.01, dt, duration)
    net.cell(0).branch(0).comp(0).stimulate(current, verbose=False)
    return net, {0: 0.0}
