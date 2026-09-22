"""Synapses whose dynamics come from a custom LEMS ComponentType.

The C. elegans connectome models are the motivating case.  c302 parameter sets
from C0 onwards do not use ``expTwoSynapse`` for their chemical connections:
they define their own graded synapse, following Kunert et al. (2017)::

    <ComponentType name="gradedSynapse2" extends="baseGradedSynapse">
      <DerivedVariable name="vpeer" select="peer/v"/>
      <DerivedVariable name="phi" value="1/(1 + exp(beta * (vth - vpeer)))"/>
      <DerivedVariable name="i" exposure="i" value="weight * conductance * s * (erev-v)"/>
      <TimeDerivative variable="s" value="ar*phi*(1-s) - ad*s"/>
    </ComponentType>

A LEMS synapse reads its peer's voltage through ``select="peer/v"``.  That is
exactly what a Jaxley synapse is handed, so the gathered variable is supplied
directly rather than treated as unsupported.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxley.synapses import Synapse

from .. import ir
from ..errors import UnsupportedComponentError
from ..lems.component_types import ComponentType
from ..lems.runtime import CompiledDynamics
from ..units import _SI_TO_JAXLEY, convert as from_si, to_si

__all__ = ["LemsSynapse", "PEER_SELECTORS"]

#: ``select`` expressions a synapse can resolve from the presynaptic side.
PEER_SELECTORS = {
    "peer/v": ("vpeer", "voltage"),
    "peer/V": ("vpeer", "voltage"),
}


class LemsSynapse(Synapse):
    """A Jaxley synapse whose behaviour is defined by a LEMS ComponentType.

    Args:
        component: The inheritance-resolved component type.
        parameters_si: Instance parameters in SI units.
        name: Jaxley name for this synapse type; defaults to the NeuroML id.
        weight: Default connection weight.
    """

    def __init__(
        self,
        component: ComponentType,
        parameters_si: dict[str, float],
        name: Optional[str] = None,
        weight: float = 1.0,
    ):
        super().__init__(name or component.name)
        prefix = self._name
        self.component = component

        gathered = {
            derived.name: derived.select
            for derived in component.derived_variables
            if derived.is_gathered
        }
        unresolved = {
            name: select
            for name, select in gathered.items()
            if select not in PEER_SELECTORS
        }
        if unresolved:
            raise UnsupportedComponentError(
                f"Synapse type '{component.name}' gathers {sorted(unresolved)} from "
                f"{sorted(unresolved.values())}, which tarjuman cannot resolve; only "
                f"{sorted(PEER_SELECTORS)} are available to a Jaxley synapse."
            )
        self._peer_variables = {
            name: PEER_SELECTORS[select] for name, select in gathered.items()
        }

        self.dynamics = CompiledDynamics(
            component, parameters_si, provided=set(gathered)
        )

        self.synapse_params = {f"{prefix}_weight": float(weight)}
        self._si_factor: dict[str, float] = {}
        for parameter, value_si in parameters_si.items():
            dimension = component.parameters.get(parameter, "none")
            factor = _SI_TO_JAXLEY.get(dimension, 1.0)
            self.synapse_params[f"{prefix}_{parameter}"] = float(value_si * factor)
            self._si_factor[parameter] = 1.0 / factor

        self.synapse_states = {
            f"{prefix}_{state}": 0.0 for state in component.state_variables
        }
        if not self.synapse_states:
            # Jaxley expects at least one state per synapse type.
            self.synapse_states = {f"{prefix}_unused": 0.0}

        self._current_dimension = component.exposures.get("i", "current")

    # -- context ---------------------------------------------------------- #
    def _context(self, pre_voltage, post_voltage, params: dict) -> dict:
        prefix = self._name
        context = {
            name: params[f"{prefix}_{name}"] * factor
            for name, factor in self._si_factor.items()
        }
        context["weight"] = params[f"{prefix}_weight"]
        context["v"] = to_si(post_voltage, "voltage")
        for name, (_, dimension) in self._peer_variables.items():
            context[name] = to_si(pre_voltage, dimension)
        context.setdefault("vpeer", to_si(pre_voltage, "voltage"))
        return context

    def _states_si(self, states: dict) -> dict:
        prefix = self._name
        return {
            name: to_si(states[f"{prefix}_{name}"], variable.dimension)
            for name, variable in self.component.state_variables.items()
        }

    # -- Synapse interface ------------------------------------------------ #
    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        if not self.component.state_variables:
            return {}
        context = self._context(pre_voltage, post_voltage, params)
        stepped = self.dynamics.step(
            self._states_si(states), context, to_si(delta_t, "time")
        )
        prefix = self._name
        return {
            f"{prefix}_{name}": from_si(
                value, self.component.state_variables[name].dimension
            )
            for name, value in stepped.items()
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        context = self._context(pre_voltage, post_voltage, params)
        values = self.dynamics.evaluate(self._states_si(states), context)
        if "i" not in values:
            raise UnsupportedComponentError(
                f"Synapse type '{self.component.name}' exposes no current 'i'."
            )
        current = from_si(values["i"], self._current_dimension)
        # NeuroML synaptic currents are inward-positive; Jaxley's are
        # outward-positive, like its channel currents.
        return -current

    def init_state(self, states, v, params, delta_t):
        return {}


def make_lems_synapse(
    synapse: ir.Synapse,
    component_types: dict,
    name: Optional[str] = None,
) -> LemsSynapse:
    """Build a :class:`LemsSynapse` from a custom synapse component instance."""
    from ..lems.builtins import builtin_registry
    from ..units import parse_quantity

    registry = builtin_registry()
    registry.update(component_types)
    component = registry.get(synapse.kind)
    if component is None:
        raise UnsupportedComponentError(
            f"Synapse '{synapse.id}' has type '{synapse.kind}', which is neither a "
            "NeuroML core type tarjuman implements nor a ComponentType defined in "
            "the document."
        )

    attributes = synapse.custom.attributes if synapse.custom else {}
    context = f"{synapse.kind} '{synapse.id}'"
    parameters_si: dict[str, float] = {}
    for parameter, dimension in component.parameters.items():
        text = attributes.get(parameter)
        if text is None:
            raise UnsupportedComponentError(
                f"{context} is missing the parameter '{parameter}'."
            )
        parameters_si[parameter] = parse_quantity(
            text, dimension if dimension != "none" else None, context
        )[0]

    return LemsSynapse(
        component,
        parameters_si,
        name=name or synapse.id,
        weight=float(attributes.get("weight", 1.0)),
    )
