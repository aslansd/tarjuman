"""Jaxley synapses that reproduce the NeuroML 2 synapse component types.

NeuroML synapses are *event driven*: a presynaptic spike delivers an event and
the conductance waveform is advanced by the ``OnEvent`` block.  Jaxley has no
event system — a synapse sees the presynaptic voltage at every step and nothing
else.  tarjuman bridges the two by detecting threshold crossings inside the
synapse itself: each synapse keeps the previous presynaptic voltage as a state,
and an upward crossing of ``spikeThresh`` counts as one event.  The resulting
conductance waveform is then identical to the NeuroML one, up to the time
discretisation of the crossing.

Two consequences are worth stating plainly, and tarjuman records both in the
conversation report:

* ``delay`` on a ``connection`` is not represented (Jaxley has no delay line);
* a presynaptic cell that sits above threshold for several steps still produces
  exactly one event, but a spike narrower than ``dt`` can be missed.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array
from jaxley.solver_gate import save_exp
from jaxley.synapses import Synapse

from .. import ir
from ..errors import UnsupportedComponentError
from .lems_synapse import LemsSynapse, make_lems_synapse

__all__ = [
    "LemsSynapse",
    "ExpTwoSynapse",
    "BlockingPlasticSynapse",
    "GapJunction",
    "GradedSynapse",
    "SilentSynapse",
    "ExpOneSynapse",
    "AlphaSynapse",
    "make_synapse",
    "SUPPORTED_SYNAPSE_TYPES",
    "DEFAULT_SPIKE_THRESHOLD",
]

DEFAULT_SPIKE_THRESHOLD = 0.0  # mV
_VERY_NEGATIVE = -1e3  # mV, initial "previous voltage" so step 0 never fires


class _EventSynapse(Synapse):
    """Base class handling threshold detection and the shared parameters."""

    def __init__(
        self,
        name: Optional[str] = None,
        erev: float = 0.0,
        gbase: float = 1e-4,
        threshold: float = DEFAULT_SPIKE_THRESHOLD,
        weight: float = 1.0,
    ):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gbase": float(gbase),  # uS
            f"{prefix}_erev": float(erev),  # mV
            f"{prefix}_threshold": float(threshold),  # mV
            f"{prefix}_weight": float(weight),
        }
        self.synapse_states = {f"{prefix}_pre_v_prev": _VERY_NEGATIVE}

    def _detect_spike(self, states, pre_voltage, params) -> Array:
        """1.0 on an upward threshold crossing since the previous step."""
        prefix = self._name
        threshold = params[f"{prefix}_threshold"]
        previous = states[f"{prefix}_pre_v_prev"]
        crossed = jnp.logical_and(previous < threshold, pre_voltage >= threshold)
        return crossed.astype(jnp.result_type(float))


class ExpTwoSynapse(_EventSynapse):
    """NeuroML ``expTwoSynapse``: a bi-exponential conductance waveform.

    ``g = gbase * (B - A)`` with ``A`` and ``B`` decaying with ``tauRise`` and
    ``tauDecay``; each event adds ``weight * waveformFactor`` to both, so that
    the peak conductance of a single event is exactly ``gbase * weight``.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        erev: float = 0.0,
        gbase: float = 1e-4,
        tau_rise: float = 0.1,
        tau_decay: float = 2.0,
        threshold: float = DEFAULT_SPIKE_THRESHOLD,
        weight: float = 1.0,
    ):
        super().__init__(name, erev=erev, gbase=gbase, threshold=threshold, weight=weight)
        prefix = self._name
        self.synapse_params[f"{prefix}_tauRise"] = float(tau_rise)
        self.synapse_params[f"{prefix}_tauDecay"] = float(tau_decay)
        self.synapse_states[f"{prefix}_A"] = 0.0
        self.synapse_states[f"{prefix}_B"] = 0.0

    @staticmethod
    def _waveform_factor(tau_rise, tau_decay):
        """``1 / (-exp(-tp/tauRise) + exp(-tp/tauDecay))`` as in Synapses.xml."""
        ratio = tau_decay / tau_rise
        peak_time = jnp.log(ratio) * (tau_rise * tau_decay) / (tau_decay - tau_rise)
        return 1.0 / (
            -save_exp(-peak_time / tau_rise) + save_exp(-peak_time / tau_decay)
        )

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        prefix = self._name
        tau_rise = params[f"{prefix}_tauRise"]
        tau_decay = params[f"{prefix}_tauDecay"]
        spike = self._detect_spike(states, pre_voltage, params)
        increment = spike * params[f"{prefix}_weight"] * self._waveform_factor(
            tau_rise, tau_decay
        )
        new_a = states[f"{prefix}_A"] * save_exp(-delta_t / tau_rise) + increment
        new_b = states[f"{prefix}_B"] * save_exp(-delta_t / tau_decay) + increment
        return {
            f"{prefix}_A": new_a,
            f"{prefix}_B": new_b,
            f"{prefix}_pre_v_prev": pre_voltage,
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        conductance = params[f"{prefix}_gbase"] * (
            states[f"{prefix}_B"] - states[f"{prefix}_A"]
        )
        return conductance * (post_voltage - params[f"{prefix}_erev"])


class BlockingPlasticSynapse(ExpTwoSynapse):
    """NeuroML ``blockingPlasticSynapse`` with a ``voltageConcDepBlockMechanism``.

    The bi-exponential conductance is scaled by

    ``blockFactor = 1 / (1 + (blockConcentration / scalingConc) * exp(-v / scalingVolt))``

    evaluated at the postsynaptic voltage, which is the standard Mg block of an
    NMDA receptor.  Short-term plasticity mechanisms, if the NeuroML component
    has any, are *not* reproduced and are reported as an approximation.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        erev: float = 0.0,
        gbase: float = 1e-4,
        tau_rise: float = 1.0,
        tau_decay: float = 13.3,
        block_concentration: float = 1.2,
        scaling_conc: float = 1.9205441817997078,
        scaling_volt: float = 16.129032258064516,
        threshold: float = DEFAULT_SPIKE_THRESHOLD,
        weight: float = 1.0,
    ):
        super().__init__(
            name=name,
            erev=erev,
            gbase=gbase,
            tau_rise=tau_rise,
            tau_decay=tau_decay,
            threshold=threshold,
            weight=weight,
        )
        prefix = self._name
        self.synapse_params[f"{prefix}_blockConcentration"] = float(block_concentration)
        self.synapse_params[f"{prefix}_scalingConc"] = float(scaling_conc)
        self.synapse_params[f"{prefix}_scalingVolt"] = float(scaling_volt)

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        block_factor = 1.0 / (
            1.0
            + (
                params[f"{prefix}_blockConcentration"]
                / params[f"{prefix}_scalingConc"]
            )
            * save_exp(-post_voltage / params[f"{prefix}_scalingVolt"])
        )
        conductance = (
            block_factor
            * params[f"{prefix}_gbase"]
            * (states[f"{prefix}_B"] - states[f"{prefix}_A"])
        )
        return conductance * (post_voltage - params[f"{prefix}_erev"])


class ExpOneSynapse(_EventSynapse):
    """NeuroML ``expOneSynapse``: instantaneous rise, single exponential decay."""

    def __init__(
        self,
        name: Optional[str] = None,
        erev: float = 0.0,
        gbase: float = 1e-4,
        tau_decay: float = 2.0,
        threshold: float = DEFAULT_SPIKE_THRESHOLD,
        weight: float = 1.0,
    ):
        super().__init__(name, erev=erev, gbase=gbase, threshold=threshold, weight=weight)
        prefix = self._name
        self.synapse_params[f"{prefix}_tauDecay"] = float(tau_decay)
        self.synapse_states[f"{prefix}_g"] = 0.0

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        prefix = self._name
        spike = self._detect_spike(states, pre_voltage, params)
        decayed = states[f"{prefix}_g"] * save_exp(
            -delta_t / params[f"{prefix}_tauDecay"]
        )
        new_g = decayed + spike * params[f"{prefix}_weight"] * params[f"{prefix}_gbase"]
        return {f"{prefix}_g": new_g, f"{prefix}_pre_v_prev": pre_voltage}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        return states[f"{prefix}_g"] * (post_voltage - params[f"{prefix}_erev"])


class AlphaSynapse(_EventSynapse):
    """NeuroML ``alphaSynapse``: equal rise and decay time ``tau``.

    Integrated with the exact solution of the coupled pair
    ``g' = (e*A - g)/tau``, ``A' = -A/tau``.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        erev: float = 0.0,
        gbase: float = 1e-4,
        tau: float = 2.0,
        threshold: float = DEFAULT_SPIKE_THRESHOLD,
        weight: float = 1.0,
    ):
        super().__init__(name, erev=erev, gbase=gbase, threshold=threshold, weight=weight)
        prefix = self._name
        self.synapse_params[f"{prefix}_tau"] = float(tau)
        self.synapse_states[f"{prefix}_g"] = 0.0
        self.synapse_states[f"{prefix}_A"] = 0.0

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        prefix = self._name
        tau = params[f"{prefix}_tau"]
        spike = self._detect_spike(states, pre_voltage, params)
        a = states[f"{prefix}_A"] + spike * params[f"{prefix}_weight"] * params[
            f"{prefix}_gbase"
        ]
        decay = save_exp(-delta_t / tau)
        new_g = decay * (states[f"{prefix}_g"] + jnp.e * a * delta_t / tau)
        return {
            f"{prefix}_g": new_g,
            f"{prefix}_A": a * decay,
            f"{prefix}_pre_v_prev": pre_voltage,
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        return states[f"{prefix}_g"] * (post_voltage - params[f"{prefix}_erev"])


class GapJunction(Synapse):
    """NeuroML ``gapJunction`` / ``linearGradedSynapse``: ohmic electrical coupling.

    The current into the postsynaptic compartment is
    ``weight * conductance * (post_v - pre_v)`` in Jaxley's outward-positive
    convention.  A Jaxley synapse is one-directional, so tarjuman instantiates
    two of these for every symmetric NeuroML ``electricalConnection``.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        conductance: float = 1e-4,
        weight: float = 1.0,
    ):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_conductance": float(conductance),  # uS
            f"{prefix}_weight": float(weight),
        }
        self.synapse_states = {}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        return {}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        return (
            params[f"{prefix}_weight"]
            * params[f"{prefix}_conductance"]
            * (post_voltage - pre_voltage)
        )


class GradedSynapse(Synapse):
    """NeuroML ``gradedSynapse``: graded (analog) transmission.

    ``s`` follows ``inf(v_pre) = 1 / (1 + exp((Vth - v_pre) / delta))`` with
    time constant ``(1 - inf) / k``, and the current is
    ``weight * conductance * s * (post_v - erev)``.  This is the synapse type
    the *C. elegans* connectome models in NeuroML use for their chemical
    connections, alongside gap junctions.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        conductance: float = 1e-4,
        delta: float = 5.0,
        k: float = 0.025,
        v_th: float = -35.0,
        erev: float = 0.0,
        weight: float = 1.0,
    ):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_conductance": float(conductance),  # uS
            f"{prefix}_delta": float(delta),  # mV
            f"{prefix}_k": float(k),  # 1/ms
            f"{prefix}_Vth": float(v_th),  # mV
            f"{prefix}_erev": float(erev),  # mV
            f"{prefix}_weight": float(weight),
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        prefix = self._name
        inf = 1.0 / (
            1.0
            + save_exp((params[f"{prefix}_Vth"] - pre_voltage) / params[f"{prefix}_delta"])
        )
        # NeuroML clamps s to inf when tau would become vanishingly small.
        one_minus_inf = jnp.maximum(1.0 - inf, 1e-4)
        tau = one_minus_inf / params[f"{prefix}_k"]
        decay = save_exp(-delta_t / tau)
        new_s = states[f"{prefix}_s"] * decay + inf * (1.0 - decay)
        return {f"{prefix}_s": jnp.where(1.0 - inf < 1e-4, inf, new_s)}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        prefix = self._name
        conductance = (
            params[f"{prefix}_weight"]
            * params[f"{prefix}_conductance"]
            * states[f"{prefix}_s"]
        )
        return conductance * (post_voltage - params[f"{prefix}_erev"])


class SilentSynapse(Synapse):
    """NeuroML ``silentSynapse``: the inert endpoint of an analog connection."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.synapse_params = {f"{self._name}_weight": 1.0}
        self.synapse_states = {}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        return {}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        return 0.0 * post_voltage


#: NeuroML component types with a Jaxley equivalent.
SUPPORTED_SYNAPSE_TYPES = {
    "expOneSynapse",
    "expTwoSynapse",
    "alphaSynapse",
    "blockingPlasticSynapse",
    "gapJunction",
    "linearGradedSynapse",
    "gradedSynapse",
    "silentSynapse",
}


def make_synapse(
    synapse: ir.Synapse,
    name: Optional[str] = None,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    component_types: Optional[dict] = None,
) -> Synapse:
    """Build the Jaxley synapse matching a NeuroML synapse component.

    Args:
        synapse: The IR synapse.
        name: Jaxley name for the synapse type; defaults to the NeuroML id.
        threshold: Presynaptic spike-detection threshold in mV.

    Returns:
        A :class:`jaxley.synapses.Synapse` instance.

    Raises:
        UnsupportedComponentError: If the component type has no equivalent.
    """
    name = name or synapse.id
    params = synapse.params
    kind = synapse.kind

    if synapse.custom is not None:
        # A synapse type the model defines for itself.
        from .lems_synapse import make_lems_synapse

        return make_lems_synapse(synapse, component_types or {}, name=name)

    if kind == "blockingPlasticSynapse" and "blockConcentration" in params:
        return BlockingPlasticSynapse(
            name=name,
            erev=params.get("erev", 0.0),
            gbase=params.get("gbase", 1e-4),
            tau_rise=params.get("tauRise", 1.0),
            tau_decay=params.get("tauDecay", 13.3),
            block_concentration=params["blockConcentration"],
            scaling_conc=params["scalingConc"],
            scaling_volt=params["scalingVolt"],
            threshold=threshold,
        )
    if kind in ("expTwoSynapse", "blockingPlasticSynapse"):
        return ExpTwoSynapse(
            name=name,
            erev=params.get("erev", 0.0),
            gbase=params.get("gbase", 1e-4),
            tau_rise=params.get("tauRise", 0.1),
            tau_decay=params.get("tauDecay", 2.0),
            threshold=threshold,
        )
    if kind == "expOneSynapse":
        return ExpOneSynapse(
            name=name,
            erev=params.get("erev", 0.0),
            gbase=params.get("gbase", 1e-4),
            tau_decay=params.get("tauDecay", 2.0),
            threshold=threshold,
        )
    if kind == "alphaSynapse":
        return AlphaSynapse(
            name=name,
            erev=params.get("erev", 0.0),
            gbase=params.get("gbase", 1e-4),
            tau=params.get("tau", 2.0),
            threshold=threshold,
        )
    if kind in ("gapJunction", "linearGradedSynapse"):
        return GapJunction(name=name, conductance=params.get("conductance", 1e-4))
    if kind == "gradedSynapse":
        return GradedSynapse(
            name=name,
            conductance=params.get("conductance", 1e-4),
            delta=params.get("delta", 5.0),
            k=params.get("k", 0.025),
            v_th=params.get("Vth", -35.0),
            erev=params.get("erev", 0.0),
        )
    if kind == "silentSynapse":
        return SilentSynapse(name=name)
    raise UnsupportedComponentError(
        f"Synapse type '{kind}' (id '{synapse.id}') has no Jaxley equivalent yet. "
        f"Supported types: {', '.join(sorted(SUPPORTED_SYNAPSE_TYPES))}."
    )
