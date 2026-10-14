# SPDX-License-Identifier: Apache-2.0
"""Handwritten Jaxley c302 C Oscillator model.

Equations and parameters: cell_C.xml, Channels.xml, Cells.xml and Synapses.xml.
Connections: c302_C_Oscillator.net.nml; inputs: LEMS_c302_C_Oscillator.xml.
c302 sources are MIT; NeuroML core definitions are LGPL-3.0.
Jaxley's public Channel/Pump/Synapse APIs use Apache-2.0.
Source revisions and licenses are listed in models.json.
"""

import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import Channel
from jaxley.pumps import Pump
from jaxley.solver_gate import solve_inf_gate_exponential
from jaxley.synapses import Synapse

DT = 0.05
DURATION = 1000.0


class OhmicChannel(Channel):
    gates = ()

    def __init__(self, name, gbar, erev):
        self.current_is_in_mA_per_cm2 = True
        super().__init__("ref_" + name)
        self.current_name = self.name + "_current"
        self.channel_params = {
            self.name + "_gbar": gbar,
            self.name + "_erev": erev,
        }
        self.channel_states = {self.name + "_" + gate: 0.0 for gate in self.gates}

    def kinetics(self, v):
        return {}

    def init_state(self, states, v, params, delta_t):
        return {
            self.name + "_" + gate: inf for gate, (inf, tau) in self.kinetics(v).items()
        }

    def update_states(self, states, dt, v, params):
        return {
            self.name + "_" + gate: solve_inf_gate_exponential(
                states[self.name + "_" + gate], dt, inf, tau
            )
            for gate, (inf, tau) in self.kinetics(v).items()
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
    def __init__(self, gbar, erev):
        super().__init__("Leak", gbar, erev)


class KSlow(OhmicChannel):
    gates = ("n",)

    def __init__(self, gbar, erev):
        super().__init__("k_slow", gbar, erev)

    def kinetics(self, v):
        return {"n": (1.0 / (1.0 + jnp.exp(-(v - 19.8741) / 15.8512)), 25.0007)}

    def opening(self, states):
        return states[self.name + "_n"]


class KFast(OhmicChannel):
    gates = ("p", "q")

    def __init__(self, gbar, erev):
        super().__init__("k_fast", gbar, erev)

    def kinetics(self, v):
        return {
            "p": (1.0 / (1.0 + jnp.exp(-(v + 8.05232) / 7.42636)), 2.25518),
            "q": (1.0 / (1.0 + jnp.exp(-(v + 15.6456) / -9.97468)), 149.963),
        }

    def opening(self, states):
        return states[self.name + "_p"] ** 4 * states[self.name + "_q"]


class CaBoyle(OhmicChannel):
    gates = ("e", "f")

    def __init__(self, gbar, erev):
        super().__init__("ca_boyle", gbar, erev)
        self.current_name = "i_Ca"
        self.channel_states["CaCon_i"] = 0.0

    def kinetics(self, v):
        return {
            "e": (1.0 / (1.0 + jnp.exp(-(v + 3.3568) / 6.74821)), 0.100027),
            "f": (1.0 / (1.0 + jnp.exp(-(v - 25.1815) / -5.03176)), 150.88),
        }

    def opening(self, states):
        # customHGate is an algebraic function of the current calcium level.
        h = 1.0 / (1.0 + jnp.exp((6.41889e-8 - states["CaCon_i"]) / -1.00056e-8))
        return (
            states[self.name + "_e"] ** 2
            * states[self.name + "_f"]
            * (1.0 + (h - 1.0) * 0.282473)
        )


class CaPool(Pump):
    def __init__(self, initial, rest, tau, rho):
        super().__init__("ref_pool")
        self.initial = initial
        self.ion_name = "CaCon_i"
        self.current_name = "ref_pool_flux"
        self.channel_params = {
            "ref_pool_rest": rest,
            "ref_pool_tau": tau,
            "ref_pool_rho": rho,
        }
        self.channel_states = {"CaCon_i": initial, "i_Ca": 0.0}

    def init_state(self, states, v, params, delta_t):
        return {"CaCon_i": self.initial}

    def update_states(self, states, dt, v, params):
        return {"CaCon_i": jnp.maximum(states["CaCon_i"], 0.0), "i_Ca": states["i_Ca"]}

    def compute_current(self, states, concentration, params):
        # Inward A/m2 = -10 * outward mA/cm2. Divide the SI time derivative
        # by 1000 for milliseconds. Pump returns the negative derivative.
        return (concentration - params["ref_pool_rest"]) / params[
            "ref_pool_tau"
        ] + 0.01 * params["ref_pool_rho"] * states["i_Ca"]


class ExpTwoSynapse(Synapse):
    def __init__(self, name, gbase, erev, rise, decay):
        super().__init__("ref_" + name)
        self.synapse_params = {
            self.name + "_gbase": gbase,
            self.name + "_erev": erev,
            self.name + "_rise": rise,
            self.name + "_decay": decay,
            self.name + "_weight": 1.0,
            self.name + "_threshold": -20.0,
        }
        self.synapse_states = {
            self.name + "_A": 0.0,
            self.name + "_B": 0.0,
            self.name + "_above": 0.0,
        }

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        p = self.name + "_"
        rise, decay = params[p + "rise"], params[p + "decay"]
        peak = jnp.log(decay / rise) / (1 / rise - 1 / decay)
        factor = 1 / (jnp.exp(-peak / decay) - jnp.exp(-peak / rise))
        higher = pre_voltage > params[p + "threshold"]
        event = higher & (states[p + "above"] < 0.5)
        above = jnp.where(
            higher,
            1.0,
            jnp.where(pre_voltage < params[p + "threshold"], 0.0, states[p + "above"]),
        )
        jump = event * params[p + "weight"] * factor
        # Exact between-event decay, then the NeuroML OnEvent jump on the
        # Jaxley time grid. No empirical delay or trace alignment is used.
        return {
            p + "A": states[p + "A"] * jnp.exp(-delta_t / rise) + jump,
            p + "B": states[p + "B"] * jnp.exp(-delta_t / decay) + jump,
            p + "above": above.astype(float),
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        p = self.name + "_"
        g = params[p + "gbase"] * (states[p + "B"] - states[p + "A"])
        return g * (post_voltage - params[p + "erev"])


class GapJunction(Synapse):
    def __init__(self, name, gbase):
        super().__init__("ref_" + name)
        self.synapse_params = {
            self.name + "_gbase": gbase,
            self.name + "_weight": 1.0,
        }
        self.synapse_states = {}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        return {}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        return (
            params[self.name + "_weight"]
            * params[self.name + "_gbase"]
            * (post_voltage - pre_voltage)
        )


CELLS = (
    "AVBL",
    "AVBR",
    "DA2",
    "DA3",
    "DB2",
    "DB3",
    "DD2",
    "DD3",
    "VA2",
    "VA3",
    "VB2",
    "VB3",
    "VD2",
    "VD3",
)

# Presynaptic cell, postsynaptic cell, connection weight. Gap junctions
# contain both directions because Jaxley synapse connections are directed.
CONNECTIONS = {
    "neuron_to_neuron_exc_syn": [
        ("AVBL", "AVBR", 1.0),
        ("AVBL", "VA2", 1.0),
        ("AVBL", "VB2", 1.0),
        ("AVBR", "AVBL", 1.0),
        ("AVBR", "VD3", 1.0),
        ("DA2", "VD2", 11.0),
        ("DA2", "VD3", 5.0),
        ("DA3", "DB3", 1.0),
        ("DA3", "DD2", 1.0),
        ("DA3", "VD3", 25.0),
        ("DB2", "DA3", 5.0),
        ("DB2", "DD2", 2.0),
        ("DB2", "VD3", 23.0),
        ("DB3", "DD2", 4.0),
        ("DB3", "DD3", 10.0),
        ("VA2", "VA3", 2.0),
        ("VA2", "VD2", 8.0),
        ("VA3", "DD2", 11.0),
        ("VA3", "VD2", 3.0),
        ("VA3", "VD3", 3.0),
        ("VB2", "DD2", 1.0),
        ("VB2", "VA2", 1.0),
        ("VB2", "VD2", 9.0),
        ("VB2", "VD3", 3.0),
        ("VB3", "DD2", 37.0),
    ],
    "neuron_to_neuron_inh_syn": [
        ("DD2", "DA3", 1.0),
        ("DD2", "VD3", 1.0),
        ("VD2", "VA2", 6.0),
        ("VD2", "VB2", 3.0),
        ("VD3", "VA3", 2.0),
        ("VD3", "VB2", 2.0),
    ],
    "neuron_to_neuron_elec_syn": [
        ("AVBL", "AVBR", 3.0),
        ("AVBR", "AVBL", 3.0),
        ("AVBL", "DB3", 1.0),
        ("DB3", "AVBL", 1.0),
        ("AVBL", "VB2", 3.0),
        ("VB2", "AVBL", 3.0),
        ("AVBR", "AVBL", 3.0),
        ("AVBL", "AVBR", 3.0),
        ("AVBR", "DB2", 1.0),
        ("DB2", "AVBR", 1.0),
        ("AVBR", "DB3", 1.0),
        ("DB3", "AVBR", 1.0),
        ("AVBR", "VB2", 1.0),
        ("VB2", "AVBR", 1.0),
        ("AVBR", "VB3", 1.0),
        ("VB3", "AVBR", 1.0),
        ("DB2", "AVBR", 1.0),
        ("AVBR", "DB2", 1.0),
        ("DB2", "DB3", 6.0),
        ("DB3", "DB2", 6.0),
        ("DB3", "AVBL", 1.0),
        ("AVBL", "DB3", 1.0),
        ("DB3", "AVBR", 1.0),
        ("AVBR", "DB3", 1.0),
        ("DB3", "DB2", 6.0),
        ("DB2", "DB3", 6.0),
        ("DD2", "DD3", 2.0),
        ("DD3", "DD2", 2.0),
        ("DD3", "DD2", 2.0),
        ("DD2", "DD3", 2.0),
        ("VD2", "VA2", 3.0),
        ("VA2", "VD2", 3.0),
        ("VD2", "VD3", 2.0),
        ("VD3", "VD2", 2.0),
        ("VD3", "VD2", 2.0),
        ("VD2", "VD3", 2.0),
        ("VA2", "VD2", 3.0),
        ("VD2", "VA2", 3.0),
        ("VB2", "AVBL", 3.0),
        ("AVBL", "VB2", 3.0),
        ("VB2", "AVBR", 1.0),
        ("AVBR", "VB2", 1.0),
        ("VB2", "VB3", 1.0),
        ("VB3", "VB2", 1.0),
        ("VB3", "AVBR", 1.0),
        ("AVBR", "VB3", 1.0),
        ("VB3", "VB2", 1.0),
        ("VB2", "VB3", 1.0),
    ],
}


def build_model(dt=DT, duration=DURATION):
    cells = []
    for _ in CELLS:
        cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
        cell.set("length", 5.0)
        cell.set("radius", 2.5)
        cell.set("capacitance", 1.0)
        cell.set("axial_resistivity", 100.0)
        cell.set("v", -45.0)
        cell.insert(Leak(gbar=0.005e-3, erev=-50.0))
        cell.insert(KSlow(gbar=3e-3, erev=-60.0))
        cell.insert(KFast(gbar=0.0711643917483308e-3, erev=-60.0))
        cell.insert(CaBoyle(gbar=3e-3, erev=40.0))
        cells.append(cell)
    net = jx.Network(cells)
    for index in range(len(CELLS)):
        net.cell(index).insert(
            CaPool(initial=0.0, rest=0.0, tau=11.5943, rho=0.000238919)
        )
    synapses = {
        "neuron_to_neuron_exc_syn": ExpTwoSynapse(
            "neuron_to_neuron_exc_syn", gbase=0.0001, erev=0.0, rise=1.0, decay=5.0
        ),
        "neuron_to_neuron_inh_syn": ExpTwoSynapse(
            "neuron_to_neuron_inh_syn", gbase=0.0001, erev=-60.0, rise=2.0, decay=30.0
        ),
        "neuron_to_neuron_elec_syn": GapJunction(
            "neuron_to_neuron_elec_syn", gbase=5e-7
        ),
    }
    for name, edges in CONNECTIONS.items():
        start = len(net.edges)
        synapse = synapses[name]
        jx.connect(
            net.select(nodes=[CELLS.index(pre) for pre, post, weight in edges]),
            net.select(nodes=[CELLS.index(post) for pre, post, weight in edges]),
            synapse,
        )
        for offset, (pre, post, weight) in enumerate(edges):
            net.select(edges=[start + offset]).set(synapse.name + "_weight", weight)
    current = jx.step_current(100.0, 800.0, 0.004, dt, duration)
    net.cell(CELLS.index("AVBL")).branch(0).comp(0).stimulate(current, verbose=False)
    net.cell(CELLS.index("AVBR")).branch(0).comp(0).stimulate(current, verbose=False)
    return net, {index: -20.0 for index in range(len(CELLS))}
