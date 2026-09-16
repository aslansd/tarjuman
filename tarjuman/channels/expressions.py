"""JAX implementations of the NeuroML 2 HH rate and variable forms.

Each function here is a literal transcription of the ``<Dynamics>`` block of the
corresponding ``ComponentType`` in ``NeuroML2CoreTypes/Channels.xml``, with two
differences forced by running on JAX:

* voltages are in mV and rates in 1/ms (NeuroML/LEMS works in SI; tarjuman
  converts once, at read time), and
* the ``x .eq. 0`` branch of ``HHExpLinearRate`` is replaced by a numerically
  safe expansion, because a ``jnp.where`` alone would still propagate ``NaN``
  through the gradient of the unselected branch.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "exp_rate",
    "sigmoid_rate",
    "exp_linear_rate",
    "RATE_FUNCTIONS",
    "VARIABLE_FUNCTIONS",
]

#: Below this |x| the linear expansion of ``x / (1 - exp(-x))`` is used.
_LINEAR_EPSILON = 1e-6


def _safe_exp(x: ArrayLike, max_value: float = 30.0) -> ArrayLike:
    """``exp`` with the argument clipped, mirroring ``jaxley.solver_gate.save_exp``."""
    return jnp.exp(jnp.minimum(x, max_value))


def exp_rate(
    v: ArrayLike, rate: ArrayLike, midpoint: ArrayLike, scale: ArrayLike
) -> ArrayLike:
    """``HHExpRate`` / ``HHExpVariable``: ``rate * exp((v - midpoint) / scale)``."""
    return rate * _safe_exp((v - midpoint) / scale)


def sigmoid_rate(
    v: ArrayLike, rate: ArrayLike, midpoint: ArrayLike, scale: ArrayLike
) -> ArrayLike:
    """``HHSigmoidRate``: ``rate / (1 + exp(-(v - midpoint) / scale))``."""
    return rate / (1.0 + _safe_exp(-(v - midpoint) / scale))


def exp_linear_rate(
    v: ArrayLike, rate: ArrayLike, midpoint: ArrayLike, scale: ArrayLike
) -> ArrayLike:
    """``HHExpLinearRate``: ``rate * x / (1 - exp(-x))`` with ``x = (v - m) / s``.

    The removable singularity at ``x = 0`` is handled with the double-``where``
    idiom so that both the value and its gradient stay finite.
    """
    x = (v - midpoint) / scale
    is_small = jnp.abs(x) < _LINEAR_EPSILON
    safe_x = jnp.where(is_small, _LINEAR_EPSILON, x)
    regular = rate * safe_x / (1.0 - _safe_exp(-safe_x))
    # Series expansion of x / (1 - exp(-x)) around 0: 1 + x/2 + x^2/12 + ...
    expansion = rate * (1.0 + x / 2.0)
    return jnp.where(is_small, expansion, regular)


#: LEMS component type -> callable ``(v, rate, midpoint, scale)``.
RATE_FUNCTIONS = {
    "HHExpRate": exp_rate,
    "HHSigmoidRate": sigmoid_rate,
    "HHExpLinearRate": exp_linear_rate,
}

#: The variable forms share the same algebra; only the dimension of ``rate`` differs.
VARIABLE_FUNCTIONS = {
    "HHExpVariable": exp_rate,
    "HHSigmoidVariable": sigmoid_rate,
    "HHExpLinearVariable": exp_linear_rate,
}
