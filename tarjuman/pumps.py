"""NeuroML concentration models as Jaxley pumps.

A NeuroML ``<species>`` attaches a concentration model to a cell: the calcium
that flows through calcium channels accumulates in a pool, decays back towards
a resting level, and — this is the point — feeds back into channels whose gates
depend on ``caConc``.  Every c302 parameter set from C onwards works this way,
following Boyle & Cohen (2008).

Jaxley calls anything that modifies an ion concentration a
:class:`~jaxley.pumps.Pump`, and integrates it implicitly from the linearised
concentration derivative.  This module wraps a compiled LEMS
``concentrationModel`` in that interface, so built-in models
(``fixedFactorConcentrationModel``, ``decayingPoolConcentrationModel``) and
model-specific ones (c302's ``muscleConcentrationModel``) go through the same
path.

Units: the LEMS dynamics are evaluated in SI, since that is the unit system
their expressions assume; parameters are exposed to the user in Jaxley units
and converted with fixed factors on the way in.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
from jaxley.pumps import Pump

from . import ir
from .errors import ParseError
from .lems.component_types import ComponentType
from .lems.runtime import CompiledDynamics
from .units import _SI_TO_JAXLEY, to_si

__all__ = ["NeuroMLConcentrationModel", "make_concentration_model", "ion_state_name"]

#: mA/cm2 * um2 -> A
_CURRENT_DENSITY_AREA_TO_AMPS = 1e-11
#: um2 -> m2
_UM2_TO_M2 = 1e-12


def ion_state_name(ion: str) -> str:
    """Jaxley's state name for an intracellular ion concentration."""
    return f"{ion.capitalize()}Con_i"


def ion_current_name(ion: str) -> str:
    """Jaxley's state name for the summed current of one ion species."""
    return f"i_{ion.capitalize()}"


class NeuroMLConcentrationModel(Pump):
    """A Jaxley pump whose dynamics come from a NeuroML concentration model.

    Args:
        component: The (inheritance-resolved) LEMS component type.
        parameters_si: Instance parameters in SI units.
        name: Jaxley name; defaults to the NeuroML id.
        ion: Ion whose concentration this model tracks.
        initial_concentration: Initial intracellular concentration in mM.
        initial_ext_concentration: Initial extracellular concentration in mM.
    """

    def __init__(
        self,
        component: ComponentType,
        parameters_si: dict[str, float],
        name: Optional[str] = None,
        ion: str = "ca",
        initial_concentration: float = 0.0,
        initial_ext_concentration: float = 2.0,
    ):
        super().__init__(name or component.name)
        prefix = self._name

        self.component = component
        self.ion = ion
        self.ion_name = ion_state_name(ion)
        self.current_name = f"i_{prefix}"
        self._ion_current_name = ion_current_name(ion)

        self.dynamics = CompiledDynamics(component, parameters_si)

        # Parameters are exposed in Jaxley units; `_to_si` converts them back
        # for the LEMS expressions, which assume SI.
        self.channel_params = {}
        self._si_factor: dict[str, float] = {}
        for parameter, value_si in parameters_si.items():
            dimension = component.parameters.get(parameter, "none")
            factor = _SI_TO_JAXLEY.get(dimension, 1.0)
            self.channel_params[f"{prefix}_{parameter}"] = float(value_si * factor)
            self._si_factor[parameter] = 1.0 / factor

        self.channel_states = {
            self.ion_name: float(initial_concentration),
            self._ion_current_name: 0.0,
        }
        self._initial_concentration = float(initial_concentration)
        self._initial_ext_concentration = float(initial_ext_concentration)
        self.META = {
            "mechanism": "NeuroML concentration model",
            "component_type": component.name,
            "ion": ion,
        }

    # -- helpers ---------------------------------------------------------- #
    def _si_parameters(self, params: dict) -> dict:
        prefix = self._name
        return {
            parameter: params[f"{prefix}_{parameter}"] * factor
            for parameter, factor in self._si_factor.items()
        }

    def _requirements(self, states: dict, params: dict) -> dict:
        """Build the SI context the LEMS dynamics needs."""
        area_um2 = 2.0 * math.pi * params["radius"] * params["length"]
        current_density = states.get(self._ion_current_name, 0.0)
        # NeuroML currents are inward-positive; Jaxley's are outward-positive.
        current_amps = (
            -current_density * area_um2 * _CURRENT_DENSITY_AREA_TO_AMPS
        )
        return {
            "iCa": current_amps,
            f"i{self.ion.capitalize()}": current_amps,
            "surfaceArea": area_um2 * _UM2_TO_M2,
            "initialConcentration": to_si(self._initial_concentration, "concentration"),
            "initialExtConcentration": to_si(
                self._initial_ext_concentration, "concentration"
            ),
            "extConcentration": to_si(
                self._initial_ext_concentration, "concentration"
            ),
        }

    # -- Pump interface --------------------------------------------------- #
    def update_states(self, states: dict, dt, v, params: dict) -> dict:
        """Jaxley integrates the concentration; this applies the floor at zero.

        NeuroML's concentration models carry
        ``<OnCondition test="concentration .lt. 0">`` with a state assignment
        back to zero.  A state assignment after the step is exactly what this
        is: an ion pool cannot be driven negative by an outward current.
        """
        return {
            self.ion_name: jnp.maximum(states[self.ion_name], 0.0),
            self._ion_current_name: states[self._ion_current_name],
        }

    def compute_current(self, states: dict, modified_state, params: dict):
        """Return ``-d[ion]/dt`` in mM/ms, as Jaxley's implicit step expects."""
        si_parameters = dict(self._si_parameters(params))
        requirements = self._requirements(states, params)
        context = dict(si_parameters)
        context.update(requirements)
        # `modified_state` is the concentration Jaxley is linearising around.
        context["concentration"] = to_si(modified_state, "concentration")

        for name, function in self.dynamics._derived:
            context[name] = function(context)

        derivative_si = self.dynamics._time_derivatives["concentration"](context)
        # mol/m3/s -> mM/ms
        derivative = derivative_si * 1e-3
        return -derivative

    def init_state(self, states, v, params, delta_t):
        return {self.ion_name: self._initial_concentration}


def make_concentration_model(
    model: ir.ConcentrationModel,
    component_types: dict,
    species: Optional[ir.Species] = None,
    name: Optional[str] = None,
) -> NeuroMLConcentrationModel:
    """Build the Jaxley pump for one NeuroML concentration model.

    Args:
        model: The IR concentration model (with its raw attributes).
        component_types: Known component types, including the built-in ones.
        species: The ``<species>`` that places it on a cell, for the initial
            concentrations.
        name: Jaxley name; defaults to the NeuroML id.

    Returns:
        A :class:`NeuroMLConcentrationModel` ready for ``Module.insert()``.
    """
    from .lems.builtins import builtin_registry

    registry = builtin_registry()
    registry.update({k: v for k, v in component_types.items()})
    component = registry.get(model.kind)
    if component is None:
        raise ParseError(
            f"Concentration model '{model.id}' has type '{model.kind}', which is "
            "neither a NeuroML core type tarjuman implements nor a ComponentType "
            "defined in the document."
        )

    context = f"{model.kind} '{model.id}'"
    parameters_si: dict[str, float] = {}
    for parameter, dimension in component.parameters.items():
        text = model.attributes.get(parameter)
        if text is None:
            raise ParseError(f"{context} is missing the parameter '{parameter}'.")
        from .units import parse_quantity

        parameters_si[parameter] = parse_quantity(
            text, dimension if dimension != "none" else None, context
        )[0]

    return NeuroMLConcentrationModel(
        component,
        parameters_si,
        name=name or model.id,
        ion=model.ion or (species.ion if species else "ca"),
        initial_concentration=species.initial_concentration if species else 0.0,
        initial_ext_concentration=species.initial_ext_concentration if species else 2.0,
    )
