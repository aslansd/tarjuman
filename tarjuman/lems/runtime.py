"""Compile a LEMS ``<Dynamics>`` block into JAX functions.

The compiled object evaluates derived variables in dependency order, advances
state variables, and applies ``OnCondition`` assignments — all in SI units,
which is the unit system LEMS expressions are written in.  Callers convert at
the boundary with :func:`si_context` and :func:`from_si`.

State variables are advanced by exponential Euler with local linearisation: the
time derivative is evaluated twice per step to estimate ``a = d(dx/dt)/dx``,
and the step is then exact for the linear case (every gate and every
concentration pool in practice) and stable when it is not.  Jaxley linearises
channel currents the same way.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

from ..errors import ParseError
from ..units import convert as si_to_jaxley
from ..units import to_si as jaxley_to_si
from .component_types import ComponentType
from .expressions import parse_expression

__all__ = ["CompiledDynamics", "si_context", "from_si", "to_si"]

#: Relative perturbation used to linearise a time derivative, with an absolute
#: floor.  It is relative because states range from volts to micromolar
#: concentrations, and a fixed absolute step either loses all precision to
#: float32 cancellation or steps clear out of the state's range.
_LINEARISATION_STEP = 1e-3
_LINEARISATION_FLOOR = 1e-9
_MIN_RATE = 1e-12


def to_si(value, dimension: str):
    """Convert a value in Jaxley units into SI."""
    return jaxley_to_si(value, dimension)


def from_si(value, dimension: str):
    """Convert an SI value into Jaxley units."""
    return si_to_jaxley(value, dimension)


def si_context(values: dict, dimensions: dict[str, str]) -> dict:
    """Convert a dictionary of Jaxley-unit values into SI, by dimension."""
    return {
        name: to_si(value, dimensions.get(name, "none"))
        for name, value in values.items()
    }


class CompiledDynamics:
    """The dynamics of one LEMS component instance, compiled to JAX.

    Args:
        component: The (inheritance-resolved) component type.
        parameters: Instance parameter values **in SI units**, keyed by the
            parameter names the component type declares.
    """

    def __init__(
        self,
        component: ComponentType,
        parameters: dict[str, float],
        provided: Optional[set[str]] = None,
        external_states: Optional[set[str]] = None,
    ):
        self.component = component
        self.parameters = dict(parameters)
        #: Gathered (``select``) variables the caller supplies itself; a Jaxley
        #: synapse, for instance, is handed the presynaptic voltage that a LEMS
        #: synapse reads with ``select="peer/v"``.
        self.provided = set(provided or ())
        #: State variables the caller integrates rather than this runtime. A
        #: point cell's ``v`` is the case: Jaxley's own solver advances the
        #: membrane potential, because only it knows the stimulus and synaptic
        #: currents flowing into the compartment. Such a variable is still read
        #: by every expression and may still be assigned by an ``OnCondition``
        #: — which is how a spike reset reaches Jaxley.
        self.external_states = set(external_states or ())

        unresolved_gathered = [
            name for name in component.gathered_variables if name not in self.provided
        ]
        if component.unsupported or unresolved_gathered:
            reasons = list(component.unsupported)
            if unresolved_gathered:
                reasons.append(
                    "<DerivedVariable select=...> gathering from child components "
                    f"({', '.join(unresolved_gathered)}); in Jaxley the "
                    "synaptic input a cell receives is assembled by the network, "
                    "not read by the cell"
                )
            raise ParseError(
                f"ComponentType '{component.name}' uses features tarjuman cannot "
                f"interpret: {'; '.join(reasons)}."
            )

        missing = set(component.parameters) - set(self.parameters)
        if missing:
            raise ParseError(
                f"ComponentType '{component.name}' needs parameters "
                f"{sorted(missing)}, which the component instance does not give."
            )

        self.constants = {name: value for name, (value, _) in component.constants.items()}
        #: State variables this runtime integrates, in declaration order.
        self.state_names = [
            name
            for name in component.state_variables
            if name not in self.external_states
        ]

        self._derived = self._compile_derived()
        self._time_derivatives = {
            name: parse_expression(expression)[0]
            for name, expression in component.time_derivatives.items()
        }
        self._on_start = [
            (variable, parse_expression(expression)[0])
            for action in component.events
            if action.kind == "start"
            for variable, expression in action.assignments
        ]
        self._on_condition = [
            (
                parse_expression(action.test)[0],
                [
                    (variable, parse_expression(expression)[0])
                    for variable, expression in action.assignments
                ],
            )
            for action in component.events
            if action.kind == "condition" and action.test
        ]
        self._on_event = [
            (variable, parse_expression(expression)[0])
            for action in component.events
            if action.kind == "event"
            for variable, expression in action.assignments
        ]

    # -- compilation ------------------------------------------------------ #
    def _compile_derived(self) -> list[tuple[str, Callable]]:
        """Derived variables, ordered so each is computed after its inputs."""
        entries: list[tuple[str, Callable, set[str]]] = []
        for derived in self.component.derived_variables:
            if derived.is_gathered:
                continue
            function, symbols = parse_expression(derived.value)
            entries.append((derived.name, function, symbols))
        for conditional in self.component.conditional_derived:
            function, symbols = self._compile_conditional(conditional)
            entries.append((conditional.name, function, symbols))

        known = (
            set(self.parameters)
            | set(self.constants)
            | set(self.component.requirements)
            | set(self.component.state_variables)
            | self.provided
            | {"weight"}
        )
        ordered: list[tuple[str, Callable]] = []
        remaining = list(entries)
        while remaining:
            progressed = False
            for entry in list(remaining):
                name, function, symbols = entry
                if symbols <= known | {name}:
                    ordered.append((name, function))
                    known.add(name)
                    remaining.remove(entry)
                    progressed = True
            if not progressed:
                # A genuine cycle, or a symbol supplied by a child component.
                unresolved = {
                    name: sorted(symbols - known) for name, _, symbols in remaining
                }
                raise ParseError(
                    f"ComponentType '{self.component.name}': cannot order derived "
                    f"variables; unresolved references {unresolved}."
                )
        return ordered

    def _compile_conditional(self, conditional) -> tuple[Callable, set[str]]:
        cases = []
        symbols: set[str] = set()
        for condition, value in conditional.cases:
            value_function, value_symbols = parse_expression(value)
            symbols |= value_symbols
            if condition is None:
                cases.append((None, value_function))
            else:
                test_function, test_symbols = parse_expression(condition)
                symbols |= test_symbols
                cases.append((test_function, value_function))

        def evaluate(context):
            # Last case wins as the default; earlier conditions take priority.
            result = None
            for test, value_function in reversed(cases):
                value = value_function(context)
                result = value if result is None else result
                if test is not None:
                    result = jnp.where(test(context) > 0, value, result)
            return result

        return evaluate, symbols

    # -- evaluation ------------------------------------------------------- #
    def context(self, states: dict, requirements: dict) -> dict:
        """Build the evaluation context (all in SI) for one step."""
        context = dict(self.constants)
        context.update(self.parameters)
        context.update(requirements)
        context.update(states)
        return context

    def evaluate(self, states: dict, requirements: dict) -> dict:
        """Evaluate every derived variable; returns the full context."""
        context = self.context(states, requirements)
        for name, function in self._derived:
            context[name] = function(context)
        return context

    def initial_states(self, requirements: dict) -> dict:
        """Apply ``OnStart``, giving the initial value of each state variable.

        Externally integrated variables start from ``requirements`` where the
        caller gave a value, and are returned if ``OnStart`` assigns them: a
        point cell's resting potential comes from this.
        """
        states = {name: jnp.array(0.0) for name in self.state_names}
        for name in self.external_states:
            states[name] = requirements.get(name, jnp.array(0.0))
        context = self.evaluate(states, requirements)
        for variable, function in self._on_start:
            states[variable] = function(context)
            context[variable] = states[variable]
        return states

    def step(
        self,
        states: dict,
        requirements: dict,
        dt: float,
        apply_conditions: bool = True,
    ) -> dict:
        """Advance the state variables by ``dt`` seconds (SI).

        ``states`` may include the externally integrated variables; they are
        read but not advanced, and are returned with whatever an
        ``OnCondition`` assigned to them.

        Args:
            states: State values in SI.
            requirements: Everything else the expressions read, in SI.
            dt: Time step in seconds.
            apply_conditions: Whether to apply ``OnCondition`` assignments
                here. A caller that must test them against a different
                voltage than the one it integrated on — see
                :meth:`apply_conditions` — passes ``False`` and calls that
                itself.
        """
        integrable = {
            name: derivative
            for name, derivative in self._time_derivatives.items()
            if name not in self.external_states
        }
        if not integrable:
            return (
                self._apply_conditions(dict(states), requirements)
                if apply_conditions
                else dict(states)
            )

        context = self.evaluate(states, requirements)
        updated = dict(states)
        for name, derivative in integrable.items():
            value = states[name]
            f0 = derivative(context)

            step = jnp.maximum(
                jnp.abs(value) * _LINEARISATION_STEP, _LINEARISATION_FLOOR
            )
            perturbed = dict(context)
            perturbed[name] = value + step
            for derived_name, function in self._derived:
                perturbed[derived_name] = function(perturbed)
            f1 = derivative(perturbed)

            slope = (f1 - f0) / step
            safe_slope = jnp.where(jnp.abs(slope) < _MIN_RATE, _MIN_RATE, slope)
            exponential = value + (f0 / safe_slope) * jnp.expm1(safe_slope * dt)
            explicit = value + dt * f0
            updated[name] = jnp.where(
                jnp.abs(slope) < _MIN_RATE, explicit, exponential
            )
        return (
            self._apply_conditions(updated, requirements)
            if apply_conditions
            else updated
        )

    def apply_conditions(self, states: dict, requirements: dict) -> dict:
        """Apply ``OnCondition`` assignments to ``states`` (all in SI).

        Separate from :meth:`step` because a caller may need to test the
        conditions against a state it did not integrate: a point cell tests
        its spike condition against the voltage the step is *about* to
        produce, since that is what NeuroML's own explicit integrator does.
        """
        return self._apply_conditions(dict(states), requirements)

    def derivative(self, name: str, states: dict, requirements: dict):
        """Evaluate one ``TimeDerivative`` in SI units."""
        if name not in self._time_derivatives:
            raise ParseError(
                f"ComponentType '{self.component.name}' gives no TimeDerivative "
                f"for '{name}'."
            )
        context = self.evaluate(states, requirements)
        return self._time_derivatives[name](context)

    def _apply_conditions(self, states: dict, requirements: dict) -> dict:
        if not self._on_condition:
            return states
        context = self.evaluate(states, requirements)
        for test, assignments in self._on_condition:
            fires = test(context) > 0
            for variable, function in assignments:
                if variable in states:
                    states[variable] = jnp.where(
                        fires, function(context), states[variable]
                    )
        return states

    def apply_event(self, states: dict, requirements: dict, fired) -> dict:
        """Apply ``OnEvent`` assignments where ``fired`` is truthy."""
        if not self._on_event:
            return states
        context = self.evaluate(states, requirements)
        updated = dict(states)
        for variable, function in self._on_event:
            if variable in states:
                updated[variable] = jnp.where(
                    fired > 0, function(context), states[variable]
                )
        return updated

    def exposure(self, name: str, states: dict, requirements: dict):
        """Evaluate one exposure by name, in SI units."""
        context = self.evaluate(states, requirements)
        if name not in context:
            raise ParseError(
                f"ComponentType '{self.component.name}' exposes no '{name}'."
            )
        return context[name]
