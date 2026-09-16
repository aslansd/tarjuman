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

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxley.channels import Channel
from jaxley.solver_gate import solve_inf_gate_exponential

from .. import ir
from ..errors import UnsupportedComponentError
from .expressions import RATE_FUNCTIONS, VARIABLE_FUNCTIONS

__all__ = ["NeuroMLChannel", "make_channel", "DEFAULT_TEMPERATURE"]

#: Temperature assumed when neither the network nor the caller specifies one.
#: 6.3 degC is the temperature of the original Hodgkin-Huxley squid axon data
#: and the value ``jnml`` falls back on for NeuroML models without one.
DEFAULT_TEMPERATURE = 279.45  # K

_RATES_FROM_ALPHA_BETA = {"gateHHrates", "gateHHratesTau"}
_TAU_FROM_ALPHA_BETA = {"gateHHrates", "gateHHratesInf"}


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
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name or channel.id)

        self.neuroml = channel
        self.temperature = temperature
        prefix = self._name

        self.channel_params = {
            f"{prefix}_gbar": float(cond_density),
            f"{prefix}_e": float(erev),
        }
        self.channel_states: dict[str, float] = {}
        self.current_name = f"i_{prefix}"

        self._conductance_scaling = 1.0
        if channel.q10_conductance_scaling is not None:
            self._conductance_scaling = channel.q10_conductance_scaling.factor(
                temperature
            )

        self._gates: list[_GateSpec] = [
            self._build_gate(gate, prefix) for gate in channel.gates
        ]

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
        self, spec: _GateSpec, v: ArrayLike, params: dict[str, Array]
    ) -> tuple[ArrayLike, ArrayLike]:
        """Steady state and (temperature-scaled) time constant of one gate."""
        alpha = beta = None
        if spec.alpha is not None:
            alpha = spec.alpha(v, *(params[name] for name in spec.alpha_params))
        if spec.beta is not None:
            beta = spec.beta(v, *(params[name] for name in spec.beta_params))

        rate_scale = params[spec.rate_scale_param]

        if spec.steady_state is not None:
            inf = spec.steady_state(
                v, *(params[name] for name in spec.steady_state_params)
            )
        else:
            inf = alpha / (alpha + beta)

        if spec.kind in _TAU_FROM_ALPHA_BETA:
            tau = 1.0 / ((alpha + beta) * rate_scale)
        elif spec.tau_param is not None:
            tau = params[spec.tau_param] / rate_scale
        else:  # instantaneous gate
            tau = jnp.array(0.0)
        return inf, tau

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
            inf, tau = self._inf_and_tau(spec, v, params)
            updated[spec.state_name] = solve_inf_gate_exponential(
                states[spec.state_name], dt, inf, tau
            )
        return updated

    def compute_current(
        self, states: dict[str, Array], v: float, params: dict[str, Array]
    ) -> Array:
        """Current density in mA/cm2 (``gbar * prod(q^instances) * (v - e)``)."""
        open_fraction = jnp.array(1.0)
        for spec in self._gates:
            if spec.is_instantaneous:
                q, _ = self._inf_and_tau(spec, v, params)
            else:
                q = states[spec.state_name]
            open_fraction = open_fraction * q**spec.instances

        conductance = (
            params[f"{self._name}_gbar"] * open_fraction * self._conductance_scaling
        )
        return conductance * (v - params[f"{self._name}_e"])

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
            inf, _ = self._inf_and_tau(spec, v, params)
            initial[spec.state_name] = inf
        return initial


def make_channel(
    channel: ir.IonChannel,
    density: Optional[ir.ChannelDensity] = None,
    name: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    erev_override: Optional[float] = None,
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
    )
