"""Abstract cells driven by their own LEMS dynamics.

``izhikevich2007Cell``, ``adExIaFCell`` and the integrate-and-fire cells a
model defines itself are all LEMS ComponentTypes carrying a membrane equation,
a few state variables and a reset rule.  tarjuman already has an interpreter
for LEMS dynamics, so rather than reimplement each cell in Python this module
hands the canonical ``<Dynamics>`` to that interpreter and splits the work with
Jaxley along the one line that matters:

* **Jaxley integrates the voltage.**  Only it knows the stimulus and synaptic
  currents arriving at the compartment, so the membrane equation stays in its
  hands.  The cell contributes the current those equations describe through
  ``compute_current``, which is the LEMS ``dv/dt`` multiplied by ``C`` — with
  the gathered synaptic term set to zero, because Jaxley is about to add the
  real one.

* **The interpreter integrates everything else** — the adaptation current, the
  recovery variable, whatever the component declares — and applies the
  ``OnCondition`` assignments, including the one that writes ``v`` on a spike.
  A non-capacitive Jaxley channel may return ``"v"`` from ``update_states``,
  so the reset is exact rather than approximated.

That split is what makes ``<DerivedVariable select="synapses[*]/i"
reduce="add"/>`` work.  A LEMS cell reads its total synaptic current with it,
and the obvious reading — that the cell needs a sum Jaxley assembles
elsewhere — is what blocked these cell types.  But the cell does not need to
*read* the sum: it needs the sum to reach its membrane equation, and Jaxley
puts it there already.  Zeroing the term and letting the solver supply it
gives the same equation.

The consequence is that any exposure derived from the gathered current — a
LEMS ``iSyn`` or ``iMemb`` exposure — cannot be recorded, since this cell
genuinely never computes it.  Recording one is refused rather than answered
with a number that leaves out the synapses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import jaxley as jx
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxley.channels import Channel

from . import ir
from .errors import UnsupportedComponentError
from .lems.component_types import ComponentType, Regime
from .lems.expressions import parse_expression
from .lems.runtime import CompiledDynamics
from .morphology import BranchSpec, CellMorphology
from .report import ConversionReport
from .units import _SI_TO_JAXLEY, convert as from_si, to_si

__all__ = [
    "LemsPointCell",
    "SpikingRegimes",
    "flatten_spiking_regimes",
    "make_lems_point_cell",
    "GATHERED_CURRENTS",
]

#: ``select`` expressions that gather the current arriving from attached
#: synapses.  Jaxley's solver assembles exactly this sum, so the LEMS term is
#: evaluated as zero and the solver's own contribution stands in for it.
GATHERED_CURRENTS = {
    "synapses[*]/i": "current",
    "synapses[*]/I": "none",
}

#: Amps -> mA/cm2 for a compartment of 1 um2.  (1e3 mA/A) / (1e-8 cm2/um2)
_AMPS_PER_UM2_TO_MA_PER_CM2 = 1e11

#: How much positive feedback in the membrane equation is too much for an
#: implicit step.  ``d(dv/dt)/dv * dt`` above this means the voltage is in the
#: regenerative part of a spike, where Jaxley's implicit solve inverts a
#: negative conductance and sends the voltage the wrong way; the cell
#: integrates its own voltage explicitly there instead.  Half a step of
#: doubling is already far more feedback than any sub-threshold dynamics has.
RUNAWAY_FEEDBACK = 0.5

#: Explicit sub-steps taken across the regenerative region in one Jaxley step.
RUNAWAY_SUBSTEPS = 16

#: Voltage nudge, in volts, for measuring the membrane equation's feedback.
_FEEDBACK_PROBE = 1e-6


@dataclass
class SpikingRegimes:
    """The one ``<Regime>`` arrangement tarjuman interprets.

    ``adExIaFCell`` expresses its refractory period as two regimes: an
    integrating one that transitions out when the voltage crosses threshold,
    and a refractory one whose ``OnEntry`` performs the reset and which
    transitions back after a fixed time.  Everything needed to run that is
    here, with the time-based exit rewritten as a countdown so the cell never
    needs to know the simulation clock.

    Attributes:
        spike_test: The integrating regime's exit condition, e.g.
            ``v .gt. thresh``.
        entry: ``(variable, expression)`` assignments applied on the spike.
        refract_parameter: Name of the parameter holding the refractory
            duration, or ``None`` where the arrangement has no timed exit.
        held: State variables frozen during the refractory period — those the
            integrating regime advances but the refractory one does not.
    """

    spike_test: str
    entry: list[tuple[str, str]]
    refract_parameter: Optional[str]
    held: set[str]


def flatten_spiking_regimes(
    component: ComponentType,
) -> tuple[ComponentType, Optional[SpikingRegimes]]:
    """Resolve a two-regime spiking cell into plain dynamics.

    Args:
        component: An inheritance-resolved component type.

    Returns:
        ``(component, regimes)``.  Without regimes the component is returned
        unchanged and ``regimes`` is ``None``.  With the recognised
        arrangement, a copy whose ``time_derivatives`` are the integrating
        regime's and whose ``unsupported`` is cleared, plus the
        :class:`SpikingRegimes` describing the spike and the refractory hold.

    Raises:
        UnsupportedComponentError: If the component uses regimes in any other
            way.  Guessing at unfamiliar regime structure would run dynamics
            that are not the model's.
    """
    if not component.regimes:
        return component, None

    context = f"ComponentType '{component.name}'"
    # `resolve_inheritance` concatenates each ancestor's regimes, so the same
    # regime can appear more than once; keep the most derived definition.
    regimes: dict[str, Regime] = {}
    for regime in component.regimes:
        regimes[regime.name] = regime
    if len(regimes) != 2:
        raise UnsupportedComponentError(
            f"{context} has {len(regimes)} LEMS regimes "
            f"({', '.join(sorted(regimes))}). tarjuman interprets only the "
            "two-regime spike-and-refractory arrangement NeuroML's adExIaFCell "
            "uses; any other regime structure would need a state machine "
            "tarjuman does not implement."
        )

    initial = [regime for regime in regimes.values() if regime.initial]
    if len(initial) != 1:
        raise UnsupportedComponentError(
            f"{context} has {len(initial)} regimes marked initial; "
            "exactly one is needed to know where the cell starts."
        )
    integrating = initial[0]
    refractory = next(r for r in regimes.values() if r is not integrating)

    exits = integrating.transitions
    if len(exits) != 1 or exits[0].transition != refractory.name:
        raise UnsupportedComponentError(
            f"{context}: regime '{integrating.name}' must have exactly one "
            f"transition, into '{refractory.name}', for tarjuman to read it as "
            "a spike."
        )
    spike_test = exits[0].test
    if not spike_test:
        raise UnsupportedComponentError(
            f"{context}: the transition out of '{integrating.name}' has no test."
        )

    refract_parameter = _refractory_parameter(component, refractory, context)
    entry = [
        (variable, expression)
        for variable, expression in refractory.on_entry
        # The spike time is bookkeeping for the timed exit, which is a
        # countdown here, so the variable has nothing left to do.
        if variable not in _spike_time_variables(refractory)
    ]

    flattened = ComponentType(
        name=component.name,
        extends=component.extends,
        description=component.description,
        parameters=dict(component.parameters),
        constants=dict(component.constants),
        derived_parameters=dict(component.derived_parameters),
        requirements=dict(component.requirements),
        exposures=dict(component.exposures),
        texts=list(component.texts),
        state_variables={
            name: variable
            for name, variable in component.state_variables.items()
            if name not in _spike_time_variables(refractory)
        },
        derived_variables=list(component.derived_variables),
        conditional_derived=list(component.conditional_derived),
        time_derivatives={
            **component.time_derivatives,
            **integrating.time_derivatives,
        },
        events=[action for action in component.events if action.kind != "entry"],
        regimes=[],
        unsupported=[
            reason
            for reason in component.unsupported
            if not reason.startswith("Regime")
        ],
    )
    held = set(integrating.time_derivatives) - set(refractory.time_derivatives)
    return flattened, SpikingRegimes(
        spike_test=spike_test,
        entry=entry,
        refract_parameter=refract_parameter,
        held=held,
    )


def _threshold_parameter(
    component: ComponentType, condition: str
) -> Optional[str]:
    """The voltage parameter a spike condition compares the voltage against.

    ``v .gt. thresh`` gives ``thresh``; ``v .gt. vpeak`` gives ``vpeak``.  It
    is the one parameter of dimension ``voltage`` the condition mentions, and
    without exactly one there is no single value to clamp the membrane current
    at, so the clamp is simply not applied.
    """
    symbols = set(
        condition.replace(".", " ").replace("(", " ").replace(")", " ").split()
    )
    candidates = sorted(
        name
        for name, dimension in component.parameters.items()
        if dimension == "voltage" and name in symbols
    )
    return candidates[0] if len(candidates) == 1 else None


def _reset_condition(component: ComponentType) -> Optional[str]:
    """The ``OnCondition`` test under which the voltage is assigned.

    ``izhikevich2007Cell`` writes ``v`` from ``OnCondition test="v .gt.
    vpeak"``.  For the duration of that condition the voltage is set by
    assignment, so there is no membrane equation to integrate — which matters
    because the equation being left behind is one that diverges.
    """
    for action in component.events:
        if action.kind != "condition" or not action.test:
            continue
        if any(variable == "v" for variable, _ in action.assignments):
            return action.test
    return None


def _spike_time_variables(refractory: Regime) -> set[str]:
    """State variables the refractory regime uses only to timestamp the spike."""
    exits = refractory.transitions
    tests = " ".join(action.test or "" for action in exits)
    return {
        variable
        for variable, _ in refractory.on_entry
        if variable in tests.replace("+", " ").replace(".", " ").split()
    }


def _refractory_parameter(
    component: ComponentType, refractory: Regime, context: str
) -> Optional[str]:
    """Which parameter holds the refractory duration.

    The exit condition is written against the simulation clock —
    ``t .gt. lastSpikeTime + refract`` — which a Jaxley channel has no access
    to, so the duration is counted down instead.  The duration is the one
    parameter of dimension ``time`` that the exit condition mentions.
    """
    exits = refractory.transitions
    if not exits:
        return None
    if len(exits) != 1:
        raise UnsupportedComponentError(
            f"{context}: regime '{refractory.name}' has {len(exits)} "
            "transitions out; tarjuman reads one timed exit."
        )
    mentioned = set(
        (exits[0].test or "").replace("+", " ").replace("-", " ").replace(".", " ").split()
    )
    candidates = sorted(
        name
        for name, dimension in component.parameters.items()
        if dimension == "time" and name in mentioned
    )
    if len(candidates) != 1:
        raise UnsupportedComponentError(
            f"{context}: cannot tell how long regime '{refractory.name}' lasts. "
            f"Its exit condition is '{exits[0].test}', which mentions "
            f"{candidates or 'no'} parameter of dimension 'time'; tarjuman needs "
            "exactly one to count the refractory period down."
        )
    return candidates[0]


class LemsPointCell(Channel):
    """A point cell whose membrane equation comes from a LEMS ComponentType.

    Args:
        component: The inheritance-resolved, regime-flattened component type.
        parameters_si: Instance parameters in SI units.
        area_um2: Surface area of the compartment the cell becomes, in um2.
        name: Jaxley name; defaults to the component type's name.
        regimes: The spike-and-refractory arrangement, where the component
            uses one.
    """

    def __init__(
        self,
        component: ComponentType,
        parameters_si: dict[str, float],
        area_um2: float,
        name: Optional[str] = None,
        regimes: Optional[SpikingRegimes] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name or component.name)
        prefix = self._name
        self.component = component
        self.regimes = regimes
        self.area_um2 = float(area_um2)

        if "v" not in component.state_variables:
            dimensionless = "V" in component.state_variables
            raise UnsupportedComponentError(
                f"Cell type '{component.name}' declares no state variable 'v', so "
                "tarjuman cannot tell which quantity is the membrane potential"
                + (
                    ". It declares 'V' instead, which marks a dimensionless cell "
                    "(one extending baseCellMembPotDL): those carry no capacitance "
                    "and are driven by dimensionless currents, which tarjuman does "
                    "not support yet."
                    if dimensionless
                    else "."
                )
            )
        if "v" not in component.time_derivatives:
            raise UnsupportedComponentError(
                f"Cell type '{component.name}' gives no TimeDerivative for 'v', "
                "so it describes no membrane equation tarjuman can hand to "
                "Jaxley's solver."
            )
        if "C" not in parameters_si:
            raise UnsupportedComponentError(
                f"Cell type '{component.name}' has no capacitance 'C'. tarjuman "
                "drives abstract cells through Jaxley's own membrane equation, "
                "which needs one; dimensionless cells (those extending "
                "baseCellMembPotDL) are not supported yet."
            )

        self._gathered = {
            derived.name: derived.select
            for derived in component.derived_variables
            if derived.is_gathered
        }
        unresolved = {
            name: select
            for name, select in self._gathered.items()
            if select not in GATHERED_CURRENTS
        }
        if unresolved:
            raise UnsupportedComponentError(
                f"Cell type '{component.name}' gathers {sorted(unresolved)} from "
                f"{sorted(unresolved.values())}. tarjuman resolves "
                f"{sorted(GATHERED_CURRENTS)}, which Jaxley's solver assembles; "
                "anything else would have to be read by the cell itself."
            )

        self.dynamics = CompiledDynamics(
            component,
            parameters_si,
            provided=set(self._gathered),
            external_states={"v"},
        )
        #: The condition under which the voltage is *assigned* rather than
        #: integrated: the regime's exit test, or the ``OnCondition`` that
        #: writes ``v``. While it holds, NeuroML's dynamics give no
        #: ``TimeDerivative`` for ``v`` at all, so there is no membrane current
        #: to hand Jaxley — see :meth:`compute_current`.
        condition = regimes.spike_test if regimes else _reset_condition(component)
        self._spike_test = parse_expression(condition)[0] if condition else None
        #: The voltage parameter that condition compares against — ``thresh``
        #: for an AdEx cell, ``vpeak`` for an Izhikevich one. The membrane
        #: current is evaluated at no more than this; see
        #: :meth:`compute_current`.
        self._threshold_parameter = (
            _threshold_parameter(component, condition) if condition else None
        )
        self._entry = (
            [
                (variable, parse_expression(expression)[0])
                for variable, expression in regimes.entry
            ]
            if regimes
            else []
        )

        self._capacitance_si = float(parameters_si["C"])
        #: Membrane capacitance in pF, for sizing the Jaxley compartment.
        self.capacitance_pf = float(from_si(self._capacitance_si, "capacitance"))
        self._si_factor: dict[str, float] = {}
        self.channel_params = {}
        for parameter, value_si in parameters_si.items():
            dimension = component.parameters.get(parameter, "none")
            factor = _SI_TO_JAXLEY.get(dimension, 1.0)
            self.channel_params[f"{prefix}_{parameter}"] = float(value_si * factor)
            self._si_factor[parameter] = 1.0 / factor

        #: The interpreter's own state variables, in Jaxley units.
        self._state_dimensions = {
            name: variable.dimension
            for name, variable in component.state_variables.items()
            if name != "v"
        }
        initial = self.initial_states_si()
        self.channel_states = {
            f"{prefix}_{name}": float(
                from_si(initial.get(name, 0.0), dimension)
            )
            for name, dimension in self._state_dimensions.items()
        }
        self.channel_states[f"{prefix}_spikes"] = 0.0
        #: Set on a step the cell integrated its own voltage through; see
        #: :meth:`_predict`.
        self.channel_states[f"{prefix}_runaway"] = 0.0
        if regimes is not None and regimes.refract_parameter is not None:
            self.channel_states[f"{prefix}_refractory"] = 0.0
        # Jaxley groups membrane currents by this name; without one they all
        # share a `None` key and the state pytree cannot even be sorted.
        self.current_name = f"i_{prefix}"
        #: Resting potential in mV, from the component's ``OnStart``.
        self.resting_potential = float(from_si(initial.get("v", 0.0), "voltage"))

        self.META = {
            "mechanism": f"NeuroML abstract cell '{component.name}'",
            "reference": component.description,
        }

    # -- helpers ---------------------------------------------------------- #
    def initial_states_si(self) -> dict:
        """``OnStart`` values, in SI, including ``v`` where it is assigned."""
        requirements = dict.fromkeys(self._gathered, 0.0)
        requirements["v"] = 0.0
        return self.dynamics.initial_states(requirements)

    def _context(self, v, params: dict) -> dict:
        """The SI evaluation context, with the gathered synaptic current zero."""
        prefix = self._name
        context = {
            name: params[f"{prefix}_{name}"] * factor
            for name, factor in self._si_factor.items()
        }
        # Jaxley's solver adds the stimulus and synaptic currents to this
        # compartment itself, so the LEMS term that would double-count them is
        # evaluated as zero.
        for name, select in self._gathered.items():
            context[name] = jnp.zeros_like(jnp.asarray(v, dtype=float))
        context["v"] = to_si(v, "voltage")
        return context

    def _states_si(self, states: dict, v) -> dict:
        prefix = self._name
        values = {
            name: to_si(states[f"{prefix}_{name}"], dimension)
            for name, dimension in self._state_dimensions.items()
        }
        values["v"] = to_si(v, "voltage")
        return values

    # -- Channel interface ------------------------------------------------ #
    def compute_current(
        self, states: dict[str, Array], v, params: dict[str, Array]
    ) -> Array:
        """The membrane current the LEMS equations describe, in mA/cm2.

        ``dv/dt`` is the equation the component gives; multiplied by ``C`` it
        is the current driving it, which is what Jaxley's solver wants.  The
        gathered synaptic term is zero here because the solver is adding the
        real one.

        During a refractory period the current is zero, which is the regime's
        own statement: ``adExIaFCell``'s refractory regime declares no
        ``TimeDerivative`` for ``v``, so there is no equation to integrate.

        Nothing here clamps the regenerative part of the equation, because by
        the time the voltage could reach it the cell has already been reset —
        see :meth:`update_states`, which runs first.
        """
        prefix = self._name
        context = self._context(v, params)
        amps = self._capacitance_si * self.dynamics.derivative(
            "v", self._states_si(states, v), context
        )
        density = amps * _AMPS_PER_UM2_TO_MA_PER_CM2 / self.area_um2
        # LEMS drives the voltage up with a positive current; Jaxley's channel
        # currents are outward-positive, like its own Leak.
        density = -density

        refractory = f"{prefix}_refractory"
        if refractory in states:
            density = jnp.where(states[refractory] > 0.0, 0.0, density)
        # Where `update_states` integrated the voltage itself, it has already
        # written the result, so the solver must not move it again.
        return jnp.where(states[f"{prefix}_runaway"] > 0.0, 0.0, density)

    def update_states(
        self, states: dict[str, Array], dt: float, v, params: dict[str, Array]
    ) -> dict[str, Array]:
        """Advance the interpreter's states and apply the reset.

        The reset is tested against the voltage the step is *about* to
        produce, not the one it starts from, because that is the test NeuroML
        itself makes.  jLEMS integrates explicitly: it forms ``v + dt dv/dt``,
        finds it above threshold, and stores the reset value — so the overshoot
        never appears in its output either.

        Doing the same here is also what keeps the solve well conditioned. An
        AdEx cell's ``exp((v - VT) / delT)`` makes the membrane conductance
        regenerative — negative, and growing without bound — a few millivolts
        above ``VT``.  Jaxley infers that conductance from a finite difference
        and inverts it, so once ``|g|`` passes ``C / dt`` the implicit step
        changes sign and the voltage flies off to hundreds of millivolts the
        wrong way.  Resetting on the step the crossing is *predicted* keeps the
        voltage out of that region entirely: Jaxley never evaluates the
        equation there, because ``update_states`` runs before the currents are
        computed and has already written the reset.

        The prediction leaves out the stimulus and synaptic currents, which
        this cell never sees.  Where it matters — within a step of threshold,
        on the steep part of the exponential — the cell's own current exceeds
        those by orders of magnitude.
        """
        prefix = self._name
        dt_si = to_si(dt, "time")
        context = self._context(v, params)
        before = self._states_si(states, v)

        # Integrate the cell's own states on the voltage it actually has.
        stepped = self.dynamics.step(before, context, dt_si, apply_conditions=False)
        updated = {
            f"{prefix}_{name}": from_si(stepped[name], dimension)
            for name, dimension in self._state_dimensions.items()
        }

        # Then test the reset against the voltage this step is heading for.
        predicted, runaway = self._predict(before, context, dt_si)
        voltage, spiked, updated = self._reset(
            states, dt, v, params, predicted, context, updated
        )
        # Where the cell integrated its own voltage but did not reach the
        # reset, that voltage still stands — and Jaxley must be told not to
        # advance it again.
        voltage = jnp.where(
            jnp.logical_and(runaway, spiked == 0),
            from_si(predicted["v"], "voltage"),
            voltage,
        )
        updated["v"] = voltage
        updated[f"{prefix}_spikes"] = spiked
        updated[f"{prefix}_runaway"] = jnp.asarray(runaway, dtype=float)
        return updated

    def _predict(self, before: dict, context: dict, dt_si: float):
        """Where this step is heading, and whether the cell is mid-spike.

        Returns ``(states with the predicted v, runaway)``.  Normally the
        prediction is one explicit step, used only to test the reset
        condition — jLEMS integrates explicitly, so testing the condition on
        ``v + dt dv/dt`` is the test NeuroML itself makes, and the overshoot
        appears in neither simulator's output.

        In the regenerative part of a spike the prediction is the answer rather
        than a test.  A spiking cell's membrane equation has positive feedback
        there by design — AdEx's ``exp((v - VT) / delT)`` doubles every
        ``delT`` — and an implicit step inverts ``C/dt + dI/dv``, so once the
        feedback passes ``C/dt`` the step changes sign and the voltage flies
        off hundreds of millivolts the wrong way.  The region is not narrow
        enough to skip over: AdEx's conductance turns regenerative around
        12 mV above ``VT`` while the detection threshold sits another 12 mV
        above that, several steps away.  So the cell integrates those steps
        itself, explicitly and finely, which is what NeuroML's own integrator
        does; :meth:`compute_current` then yields the step to it.
        """
        derivative = self.dynamics.derivative("v", before, context)
        step = dict(before)
        step["v"] = before["v"] + dt_si * derivative

        # Feedback per step: how much a nudge to v grows the drive on v.
        nudged = dict(before)
        delta = _FEEDBACK_PROBE
        nudged["v"] = before["v"] + delta
        feedback = (
            self.dynamics.derivative("v", nudged, context) - derivative
        ) / delta
        runaway = feedback * dt_si > RUNAWAY_FEEDBACK

        substepped = self._substep(before, context, dt_si)
        step["v"] = jnp.where(runaway, substepped, step["v"])
        return step, runaway

    def _substep(self, before: dict, context: dict, dt_si: float):
        """One Jaxley step of the voltage equation, in explicit sub-steps.

        The integration stops at the spike threshold rather than running to
        the end of the step.  That is not an optimisation: the equation being
        integrated diverges to infinity in finite time — AdEx's exponential
        overflows to ``inf`` and then to ``nan`` within a few sub-steps — and
        NeuroML has nothing more to say about the voltage once it is above
        threshold anyway, because the reset takes over there.
        """
        threshold = (
            context.get(self._threshold_parameter)
            if self._threshold_parameter
            else None
        )
        fine = dt_si / RUNAWAY_SUBSTEPS
        state = dict(before)
        voltage = before["v"]
        for _ in range(RUNAWAY_SUBSTEPS):
            if threshold is None:
                state["v"] = voltage
                voltage = voltage + fine * self.dynamics.derivative(
                    "v", state, context
                )
                continue
            done = voltage > threshold
            # Evaluated below threshold so the divergence cannot overflow.
            state["v"] = jnp.minimum(voltage, threshold)
            advanced = voltage + fine * self.dynamics.derivative(
                "v", state, context
            )
            voltage = jnp.where(done, voltage, advanced)
        return voltage

    def _reset(self, states, dt, v, params, predicted, context, updated):
        """Apply whichever reset the component declares, on the predicted voltage."""
        if self.regimes is not None:
            return self._apply_regimes(
                states, dt, v, params, predicted, context, updated
            )
        prefix = self._name
        assigned = self.dynamics.apply_conditions(predicted, context)
        voltage = from_si(assigned["v"], "voltage")
        # A condition fired if it moved the voltage away from where the step
        # was heading.
        spiked = jnp.asarray(assigned["v"] != predicted["v"], dtype=float)
        for name, dimension in self._state_dimensions.items():
            key = f"{prefix}_{name}"
            updated[key] = jnp.where(
                spiked > 0, from_si(assigned[name], dimension), updated[key]
            )
        # Where nothing fired, leave the voltage to Jaxley's own solve.
        return jnp.where(spiked > 0, voltage, v), spiked, updated

    def _apply_regimes(self, states, dt, v, params, predicted, context, updated):
        """The spike-and-refractory arrangement, as a countdown.

        NeuroML holds the cell in a refractory regime for a fixed time after a
        threshold crossing: the reset assignments fire on entry, and only the
        regime's own time derivatives advance while it lasts.  The exit is
        written against the simulation clock, which a Jaxley channel cannot
        see, so the remaining time is carried as a state instead.
        """
        prefix = self._name
        regimes = self.regimes
        values = self.dynamics.evaluate(predicted, context)
        # The integrating regime's exit test, on the voltage the step is
        # heading for; see `update_states`.
        crossed = self._spike_test(values) > 0

        if regimes.refract_parameter is not None:
            remaining = jnp.maximum(states[f"{prefix}_refractory"] - dt, 0.0)
            duration = params[f"{prefix}_{regimes.refract_parameter}"]
            remaining = jnp.where(crossed, duration, remaining)
            held = jnp.logical_or(crossed, remaining > 0.0)
            updated[f"{prefix}_refractory"] = remaining
        else:
            held = crossed

        # OnEntry assignments: the reset, and the adaptation kick. Where
        # nothing fires, the voltage is left to Jaxley's own solve.
        voltage = v
        for variable, function in self._entry:
            value_si = function(values)
            if variable == "v":
                voltage = jnp.where(
                    held, from_si(value_si, "voltage"), voltage
                )
            elif variable in self._state_dimensions:
                dimension = self._state_dimensions[variable]
                updated[f"{prefix}_{variable}"] = jnp.where(
                    crossed,
                    from_si(value_si, dimension),
                    updated[f"{prefix}_{variable}"],
                )

        # Variables the refractory regime does not advance stay where they were.
        for variable in regimes.held:
            if variable in self._state_dimensions:
                dimension = self._state_dimensions[variable]
                key = f"{prefix}_{variable}"
                updated[key] = jnp.where(
                    jnp.logical_and(held, jnp.logical_not(crossed)),
                    states[key],
                    updated[key],
                )
        return voltage, jnp.asarray(crossed, dtype=float), updated

    def init_state(
        self,
        states: dict[str, ArrayLike],
        v: ArrayLike,
        params: dict[str, ArrayLike],
        delta_t: float,
    ) -> dict[str, ArrayLike]:
        prefix = self._name
        initial = self.initial_states_si()
        started = {
            f"{prefix}_{name}": from_si(initial.get(name, 0.0), dimension)
            for name, dimension in self._state_dimensions.items()
        }
        started[f"{prefix}_spikes"] = 0.0
        started[f"{prefix}_runaway"] = 0.0
        if self.regimes is not None and self.regimes.refract_parameter is not None:
            started[f"{prefix}_refractory"] = 0.0
        return started


def make_lems_point_cell(
    point_cell: ir.PointCell,
    component_types: dict[str, ComponentType],
    area_um2: float,
) -> LemsPointCell:
    """Build a :class:`LemsPointCell` from a point cell component instance.

    Args:
        point_cell: The IR point cell.  Its ``custom`` attributes are read with
            the dimensions the ComponentType declares, which is more reliable
            than inferring them from the written units alone.
        component_types: ComponentTypes the document defines, which are merged
            over the built-in ones.
        area_um2: Surface area of the compartment the cell becomes.

    Raises:
        UnsupportedComponentError: If the type is unknown, or uses LEMS
            features tarjuman cannot interpret.
    """
    from .lems.builtins import builtin_registry
    from .lems.component_types import resolve_inheritance
    from .units import parse_quantity

    registry = builtin_registry()
    registry.update(component_types)
    declared = registry.get(point_cell.kind)
    if declared is None:
        raise UnsupportedComponentError(
            f"Point cell '{point_cell.id}' has type '{point_cell.kind}', which is "
            "neither a NeuroML abstract cell tarjuman knows nor a ComponentType "
            "defined in the document."
        )

    component = resolve_inheritance(declared, registry)
    component, regimes = flatten_spiking_regimes(component)

    attributes = point_cell.custom.attributes if point_cell.custom else {}
    context = f"{point_cell.kind} '{point_cell.id}'"
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

    return LemsPointCell(
        component,
        parameters_si,
        area_um2=area_um2,
        name=point_cell.id,
        regimes=regimes,
    )
