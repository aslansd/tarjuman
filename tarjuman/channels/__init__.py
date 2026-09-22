"""Turn a NeuroML ``ionChannel`` into a Jaxley :class:`~jaxley.channels.Channel`.

Rather than generating Python source, tarjuman builds one generic channel class
whose behaviour is driven by the IR.  Every numeric quantity in the NeuroML
description — maximal conductance, reversal potential, and the ``rate``,
``midpoint`` and ``scale`` of every gate's rate equations — is exposed as a
Jaxley channel parameter.  That means a converted NeuroML model is not merely
runnable in Jaxley, it is differentiable with respect to its own channel
kinetics::

    net.make_trainable("naChan_m_alpha_midpoint")

which is the thing Jaxley can do that ``jnml`` and NEURON cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxley.channels import Channel
from jaxley.solver_gate import solve_inf_gate_exponential

from .. import ir
from ..errors import UnsupportedComponentError
from ..lems.builtins import builtin_registry
from ..lems.component_types import ComponentType
from ..lems.runtime import CompiledDynamics
from ..units import _SI_TO_JAXLEY, convert as from_si, to_si
from .expressions import RATE_FUNCTIONS, VARIABLE_FUNCTIONS

__all__ = [
    "NeuroMLChannel",
    "make_channel",
    "DEFAULT_TEMPERATURE",
    "CALCIUM_STATE",
]

#: Jaxley's state name for intracellular calcium, which calcium-dependent
#: gates read as the LEMS requirement ``caConc``.
CALCIUM_STATE = "CaCon_i"

#: Temperature assumed when neither the network nor the caller specifies one.
#: 6.3 degC is the temperature of the original Hodgkin-Huxley squid axon data
#: and the value ``jnml`` falls back on for NeuroML models without one.
DEFAULT_TEMPERATURE = 279.45  # K

#: Below this, a time constant counts as instantaneous.
_MIN_TAU = 1e-9

_RATES_FROM_ALPHA_BETA = {"gateHHrates", "gateHHratesTau"}
_TAU_FROM_ALPHA_BETA = {"gateHHrates", "gateHHratesInf"}


def _has_exposure(dynamics: CompiledDynamics, name: str) -> bool:
    return any(
        derived.name == name for derived in dynamics.component.derived_variables
    ) or name in dynamics.component.exposures


def _sole_exposure(dynamics: CompiledDynamics, preferred: str) -> str:
    """The name a custom rate/variable exposes, if it is not the standard one."""
    names = [derived.name for derived in dynamics.component.derived_variables]
    if preferred in names:
        return preferred
    for candidate in ("x", "r", "t", "inf", "tau"):
        if candidate in names:
            return candidate
    if names:
        return names[-1]
    raise UnsupportedComponentError(
        f"ComponentType '{dynamics.component.name}' exposes nothing tarjuman can "
        "use as a rate, steady state or time course."
    )


@dataclass
class _GateSpec:
    """Everything the channel needs to evaluate one gate at run time."""

    id: str
    instances: int
    kind: str
    state_name: str
    is_instantaneous: bool
    alpha: Optional[Callable] = None
    beta: Optional[Callable] = None
    steady_state: Optional[Callable] = None
    alpha_params: tuple[str, ...] = ()
    beta_params: tuple[str, ...] = ()
    steady_state_params: tuple[str, ...] = ()
    tau_param: Optional[str] = None
    rate_scale_param: Optional[str] = None
    #: A custom LEMS ComponentType driving the whole gate.
    dynamics: Optional[CompiledDynamics] = None
    #: Custom components driving only part of a standard gate, by part name.
    part_dynamics: dict[str, CompiledDynamics] = field(default_factory=dict)
    #: Jaxley parameter names of the custom component's parameters, and the
    #: factor that converts each back to SI.
    custom_params: dict[str, tuple[str, float]] = field(default_factory=dict)
    part_params: dict[str, dict[str, tuple[str, float]]] = field(default_factory=dict)
    needs_calcium: bool = False


class NeuroMLChannel(Channel):
    """A Jaxley channel whose kinetics are defined by a NeuroML ``ionChannel``.

    Args:
        channel: The IR description of the ion channel.
        name: Name of the channel inside Jaxley; defaults to the NeuroML id.
            Jaxley prefixes every parameter and state with it, so give each
            ``channelDensity`` its own name when the same ion channel is used
            with different conductances on different segment groups.
        cond_density: Default maximal conductance density in S/cm2.
        erev: Default reversal potential in mV.
        temperature: Temperature in K, used to evaluate ``q10Settings``.
    """

    def __init__(
        self,
        channel: ir.IonChannel,
        name: Optional[str] = None,
        cond_density: float = 0.0,
        erev: float = 0.0,
        temperature: float = DEFAULT_TEMPERATURE,
        component_types: Optional[dict] = None,
        current_name: Optional[str] = None,
        initial_calcium: float = 5e-5,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name or channel.id)

        self.neuroml = channel
        self.temperature = temperature
        self.component_types = component_types or {}
        prefix = self._name

        self.channel_params = {
            f"{prefix}_gbar": float(cond_density),
            # Named after the NeuroML attribute rather than "_e": Jaxley keeps
            # parameters and states in one table, and gates are often called
            # "e" (the calcium activation gate of Boyle & Cohen, for one).
            f"{prefix}_erev": float(erev),
        }
        self.channel_states: dict[str, float] = {}
        #: Channels of the same ion share a current name so that concentration
        #: models can see the total current of that ion, as NeuroML's `iCa`
        #: requirement expects.
        self.current_name = current_name or f"i_{prefix}"

        self._conductance_scaling = 1.0
        if channel.q10_conductance_scaling is not None:
            self._conductance_scaling = channel.q10_conductance_scaling.factor(
                temperature
            )

        self._gates: list[_GateSpec] = [
            self._build_gate(gate, prefix) for gate in channel.gates
        ]
        #: Calcium-dependent gates read the pool's concentration.  Declaring
        #: the state here is how a Jaxley channel asks to be given it; it is
        #: never written back, the pump owns it.
        collisions = set(self.channel_params) & set(self.channel_states)
        if collisions:
            raise UnsupportedComponentError(
                f"Channel '{prefix}' would use {sorted(collisions)} as both a "
                "parameter and a state. Jaxley keeps them in one table, so the "
                "names have to differ; rename the gate in the NeuroML file."
            )
        self.needs_calcium = any(gate.needs_calcium for gate in self._gates)
        if self.needs_calcium:
            self.channel_states[CALCIUM_STATE] = float(initial_calcium)

    # -- construction ----------------------------------------------------- #
    def _build_gate(self, gate: ir.Gate, prefix: str) -> _GateSpec:
        base = f"{prefix}_{gate.id}"
        spec = _GateSpec(
            id=gate.id,
            instances=gate.instances,
            kind=gate.kind,
            state_name=base,
            is_instantaneous=gate.kind == "gateHHInstantaneous",
        )

        if gate.custom is not None:
            return self._build_custom_gate(gate, spec, base)
        for part, component in gate.custom_parts.items():
            dynamics, parameters, needs_calcium = self._compile_custom(
                component, f"{base}_{part}"
            )
            spec.part_dynamics[part] = dynamics
            spec.part_params[part] = parameters
            spec.needs_calcium |= needs_calcium

        if gate.forward_rate is not None:
            spec.alpha = RATE_FUNCTIONS[gate.forward_rate.kind]
            spec.alpha_params = self._register_rate(
                f"{base}_alpha", gate.forward_rate
            )
        if gate.reverse_rate is not None:
            spec.beta = RATE_FUNCTIONS[gate.reverse_rate.kind]
            spec.beta_params = self._register_rate(f"{base}_beta", gate.reverse_rate)
        if gate.steady_state is not None:
            spec.steady_state = VARIABLE_FUNCTIONS[gate.steady_state.kind]
            spec.steady_state_params = self._register_rate(
                f"{base}_inf", gate.steady_state
            )
        if gate.time_course is not None:
            spec.tau_param = f"{base}_tau"
            self.channel_params[spec.tau_param] = float(gate.time_course.tau)

        rate_scale = gate.q10.factor(self.temperature) if gate.q10 is not None else 1.0
        spec.rate_scale_param = f"{base}_rateScale"
        self.channel_params[spec.rate_scale_param] = float(rate_scale)

        if not spec.is_instantaneous:
            self.channel_states[spec.state_name] = 0.0

        self._validate_gate(spec)
        return spec

    def _compile_custom(
        self, component: ir.CustomComponent, base: str
    ) -> tuple[CompiledDynamics, dict[str, tuple[str, float]], bool]:
        """Compile a custom LEMS component and register its parameters.

        Parameters are exposed in Jaxley units (so they can be inspected and
        made trainable like any other) and converted back to SI when the LEMS
        expressions are evaluated.
        """
        from ..units import parse_quantity

        registry = builtin_registry()
        registry.update(self.component_types)
        component_type = registry.get(component.component_type)
        if component_type is None:
            raise UnsupportedComponentError(
                f"Gate component '{component.component_type}' is not defined in the "
                "document and is not a NeuroML core type tarjuman implements."
            )

        context = f"{component.component_type} '{component.id}'"
        parameters_si: dict[str, float] = {}
        registered: dict[str, tuple[str, float]] = {}
        for parameter, dimension in component_type.parameters.items():
            text = component.attributes.get(parameter)
            if text is None:
                raise UnsupportedComponentError(
                    f"{context} is missing the parameter '{parameter}'."
                )
            value_si = parse_quantity(
                text, dimension if dimension != "none" else None, context
            )[0]
            parameters_si[parameter] = value_si
            factor = _SI_TO_JAXLEY.get(dimension, 1.0)
            parameter_name = f"{base}_{parameter}"
            self.channel_params[parameter_name] = float(value_si * factor)
            registered[parameter] = (parameter_name, 1.0 / factor)

        dynamics = CompiledDynamics(component_type, parameters_si)
        needs_calcium = "caConc" in component_type.requirements or any(
            "caConc" in expression
            for expression in list(component_type.time_derivatives.values())
            + [derived.value for derived in component_type.derived_variables]
        )
        return dynamics, registered, needs_calcium

    def _build_custom_gate(
        self, gate: ir.Gate, spec: _GateSpec, base: str
    ) -> _GateSpec:
        dynamics, parameters, needs_calcium = self._compile_custom(gate.custom, base)
        spec.dynamics = dynamics
        spec.custom_params = parameters
        spec.needs_calcium = needs_calcium
        spec.rate_scale_param = f"{base}_rateScale"
        self.channel_params[spec.rate_scale_param] = 1.0
        # A custom gate that exposes no time course is instantaneous; one with
        # a time course gets a state like any other gate.
        spec.is_instantaneous = "tau" not in {
            derived.name for derived in dynamics.component.derived_variables
        } and not dynamics.component.time_derivatives
        if not spec.is_instantaneous:
            self.channel_states[spec.state_name] = 0.0
        return spec

    def _custom_context(
        self,
        parameters: dict[str, tuple[str, float]],
        v: ArrayLike,
        states: dict,
        params: dict,
    ) -> dict:
        """SI context for a custom component: parameters plus requirements."""
        context = {
            name: params[parameter_name] * factor
            for name, (parameter_name, factor) in parameters.items()
        }
        context["v"] = to_si(v, "voltage")
        if CALCIUM_STATE in states:
            context["caConc"] = to_si(states[CALCIUM_STATE], "concentration")
        return context

    def _register_rate(self, base: str, rate) -> tuple[str, str, str]:
        names = (f"{base}_rate", f"{base}_midpoint", f"{base}_scale")
        self.channel_params[names[0]] = float(rate.rate)
        self.channel_params[names[1]] = float(rate.midpoint)
        self.channel_params[names[2]] = float(rate.scale)
        return names

    def _validate_gate(self, spec: _GateSpec) -> None:
        needs_alpha_beta = spec.kind in (
            "gateHHrates",
            "gateHHratesTau",
            "gateHHratesInf",
            "gateHHratesTauInf",
        )
        if needs_alpha_beta and (spec.alpha is None or spec.beta is None):
            raise UnsupportedComponentError(
                f"Gate '{spec.id}' of type {spec.kind} is missing forwardRate or "
                "reverseRate."
            )
        needs_steady_state = spec.kind in (
            "gateHHtauInf",
            "gateHHratesInf",
            "gateHHratesTauInf",
            "gateHHInstantaneous",
        )
        if needs_steady_state and spec.steady_state is None:
            raise UnsupportedComponentError(
                f"Gate '{spec.id}' of type {spec.kind} is missing steadyState."
            )
        needs_tau = spec.kind in ("gateHHtauInf", "gateHHratesTau", "gateHHratesTauInf")
        if needs_tau and spec.tau_param is None:
            raise UnsupportedComponentError(
                f"Gate '{spec.id}' of type {spec.kind} is missing timeCourse."
            )

    # -- dynamics --------------------------------------------------------- #
    def _inf_and_tau(
        self,
        spec: _GateSpec,
        v: ArrayLike,
        params: dict[str, Array],
        states: Optional[dict] = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Steady state and (temperature-scaled) time constant of one gate."""
        states = states or {}
        rate_scale = params[spec.rate_scale_param]

        if spec.dynamics is not None:
            return self._custom_inf_and_tau(spec, v, params, states, rate_scale)

        alpha = beta = None
        if spec.alpha is not None:
            alpha = spec.alpha(v, *(params[name] for name in spec.alpha_params))
        elif "forward_rate" in spec.part_dynamics:
            alpha = self._part_exposure(spec, "forward_rate", "r", "per_time", v, states, params)
        if spec.beta is not None:
            beta = spec.beta(v, *(params[name] for name in spec.beta_params))
        elif "reverse_rate" in spec.part_dynamics:
            beta = self._part_exposure(spec, "reverse_rate", "r", "per_time", v, states, params)

        if spec.steady_state is not None:
            inf = spec.steady_state(
                v, *(params[name] for name in spec.steady_state_params)
            )
        elif "steady_state" in spec.part_dynamics:
            inf = self._part_exposure(spec, "steady_state", "x", "none", v, states, params)
        else:
            inf = alpha / (alpha + beta)

        if "time_course" in spec.part_dynamics:
            tau = (
                self._part_exposure(spec, "time_course", "t", "time", v, states, params)
                / rate_scale
            )
        elif spec.kind in _TAU_FROM_ALPHA_BETA:
            tau = 1.0 / ((alpha + beta) * rate_scale)
        elif spec.tau_param is not None:
            tau = params[spec.tau_param] / rate_scale
        else:  # instantaneous gate
            tau = jnp.array(0.0)
        return inf, tau

    def _part_exposure(
        self,
        spec: _GateSpec,
        part: str,
        exposure: str,
        dimension: str,
        v: ArrayLike,
        states: dict,
        params: dict[str, Array],
    ):
        """Evaluate a custom rate/variable/timeCourse inside a standard gate."""
        dynamics = spec.part_dynamics[part]
        context = self._custom_context(spec.part_params[part], v, states, params)
        values = dynamics.evaluate({}, context)
        name = exposure if exposure in values else _sole_exposure(dynamics, exposure)
        return from_si(values[name], dimension)

    def _custom_inf_and_tau(
        self,
        spec: _GateSpec,
        v: ArrayLike,
        params: dict[str, Array],
        states: dict,
        rate_scale,
    ) -> tuple[ArrayLike, ArrayLike]:
        context = self._custom_context(spec.custom_params, v, states, params)
        values = spec.dynamics.evaluate({}, context)
        inf = values.get("inf", values.get("q", jnp.array(0.0)))
        tau_si = values.get("tau", 0.0)
        tau = from_si(tau_si, "time") / rate_scale
        return inf, tau

    def _open_fraction(
        self,
        spec: _GateSpec,
        q,
        v: ArrayLike,
        params: dict[str, Array],
        states: dict,
    ):
        """A gate's contribution to the open fraction: ``q^instances`` by default.

        A custom gate may redefine it: c302's ``customHGate`` exposes
        ``fcond = 1 + (q - 1) * alpha``, a partial block rather than a gate in
        series, and the conductance is wrong if that is ignored.
        """
        if spec.dynamics is not None and _has_exposure(spec.dynamics, "fcond"):
            context = self._custom_context(spec.custom_params, v, states, params)
            context["q"] = q
            values = spec.dynamics.evaluate({}, context)
            return values["fcond"]
        return q**spec.instances

    def update_states(
        self,
        states: dict[str, Array],
        dt: float,
        v: float,
        params: dict[str, Array],
    ) -> dict[str, Array]:
        """Advance every dynamic gate by ``dt`` with exponential Euler."""
        updated: dict[str, Array] = {}
        for spec in self._gates:
            if spec.is_instantaneous:
                continue
            inf, tau = self._inf_and_tau(spec, v, params, states)
            safe_tau = jnp.where(tau > _MIN_TAU, tau, _MIN_TAU)
            stepped = solve_inf_gate_exponential(
                states[spec.state_name], dt, inf, safe_tau
            )
            # A custom gate may have tau = 0, i.e. instantaneous.
            updated[spec.state_name] = jnp.where(tau > _MIN_TAU, stepped, inf)
        return updated

    def compute_current(
        self, states: dict[str, Array], v: float, params: dict[str, Array]
    ) -> Array:
        """Current density in mA/cm2 (``gbar * prod(q^instances) * (v - e)``)."""
        open_fraction = jnp.array(1.0)
        for spec in self._gates:
            if spec.is_instantaneous:
                q, _ = self._inf_and_tau(spec, v, params, states)
            else:
                q = states[spec.state_name]
            open_fraction = open_fraction * self._open_fraction(
                spec, q, v, params, states
            )

        conductance = (
            params[f"{self._name}_gbar"] * open_fraction * self._conductance_scaling
        )
        return conductance * (v - params[f"{self._name}_erev"])

    def init_state(
        self,
        states: dict[str, ArrayLike],
        v: ArrayLike,
        params: dict[str, ArrayLike],
        delta_t: float,
    ) -> dict[str, ArrayLike]:
        """Initialise each gate at its steady state, as NeuroML's ``OnStart`` does."""
        initial: dict[str, ArrayLike] = {}
        for spec in self._gates:
            if spec.is_instantaneous:
                continue
            inf, _ = self._inf_and_tau(spec, v, params, states)
            initial[spec.state_name] = inf
        return initial


def make_channel(
    channel: ir.IonChannel,
    density: Optional[ir.ChannelDensity] = None,
    name: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    erev_override: Optional[float] = None,
    component_types: Optional[dict] = None,
    current_name: Optional[str] = None,
    initial_calcium: float = 5e-5,
) -> NeuroMLChannel:
    """Build a Jaxley channel for one ``channelDensity`` of one ion channel.

    Args:
        channel: The ion channel description.
        density: The ``channelDensity`` that places it on the membrane; its id
            is used as the Jaxley channel name and its ``condDensity``/``erev``
            as the default parameter values.
        name: Explicit Jaxley name, overriding the density id.
        temperature: Temperature in K for ``q10Settings``.
        erev_override: Reversal potential in mV to use when the NeuroML
            document does not give one (e.g. ``channelDensityNernst``).
        component_types: Custom LEMS ComponentTypes the document defines, for
            gates that are not one of the standard NeuroML forms.
        current_name: Jaxley state the channel's current is added to. Channels
            of the same ion share one so that concentration models can see the
            total current of that ion.
        initial_calcium: Initial intracellular calcium in mM, for gates that
            depend on it.

    Returns:
        A :class:`NeuroMLChannel` ready to be passed to ``Module.insert()``.
    """
    cond_density = density.cond_density if density is not None else 0.0
    erev = density.erev if density is not None and density.erev is not None else None
    if erev is None:
        erev = erev_override if erev_override is not None else 0.0
    channel_name = name or (density.id if density is not None else channel.id)
    return NeuroMLChannel(
        channel,
        name=channel_name,
        cond_density=cond_density,
        erev=erev,
        temperature=temperature,
        component_types=component_types,
        current_name=current_name,
        initial_calcium=initial_calcium,
    )
