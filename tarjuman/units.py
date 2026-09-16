"""Parsing of NeuroML/LEMS quantity strings and conversion to Jaxley units.

NeuroML quantities are written as a number followed by a LEMS unit symbol,
optionally separated by whitespace: ``"-65mV"``, ``"120.0 mS_per_cm2"``,
``"3.0 S_per_m2"``, ``"1e-3 s"``.  LEMS itself is unit-agnostic and works in SI;
Jaxley, like NEURON, works in physiological units.  This module is the single
place where that translation happens.

Jaxley's unit conventions (see ``jaxley.channels.Channel`` and
``jaxley.synapses.Synapse``):

===================  ====================
quantity             Jaxley unit
===================  ====================
time                 ms
voltage              mV
length, radius       um
conductance density  S/cm2
point conductance    uS
current (point)      nA
current density      mA/cm2
specific capacitance uF/cm2
axial resistivity    ohm cm
concentration        mM
temperature          K
===================  ====================
"""

from __future__ import annotations

import re
from typing import Optional

from ._units_table import DIMENSIONS, UNITS
from .errors import UnitError

__all__ = ["parse_quantity", "to_jaxley", "convert", "JAXLEY_UNITS"]

# SI value -> Jaxley value, per LEMS dimension name.
_SI_TO_JAXLEY: dict[str, float] = {
    "none": 1.0,
    "time": 1e3,  # s -> ms
    "per_time": 1e-3,  # per_s -> per_ms
    "voltage": 1e3,  # V -> mV
    "per_voltage": 1e-3,  # per_V -> per_mV
    "length": 1e6,  # m -> um
    "area": 1e12,  # m2 -> um2
    "volume": 1e18,  # m3 -> um3
    "conductance": 1e6,  # S -> uS
    "conductanceDensity": 1e-4,  # S/m2 -> S/cm2
    "capacitance": 1e12,  # F -> pF
    "specificCapacitance": 1e2,  # F/m2 -> uF/cm2
    "resistance": 1e-6,  # ohm -> Mohm
    "resistivity": 1e2,  # ohm m -> ohm cm
    "current": 1e9,  # A -> nA
    "currentDensity": 1e-1,  # A/m2 -> mA/cm2
    "concentration": 1.0,  # mol/m3 == mM
    "temperature": 1.0,  # K
    "charge": 1.0,
    "charge_per_mole": 1.0,
    "substance": 1.0,
    "permeability": 1e3,  # m/s -> mm/s
    "rho_factor": 1.0,
    "idealGasConstantDims": 1.0,
    "conductance_per_voltage": 1.0,
}

#: Human-readable name of the Jaxley unit for each dimension, for messages.
JAXLEY_UNITS: dict[str, str] = {
    "none": "",
    "time": "ms",
    "per_time": "1/ms",
    "voltage": "mV",
    "length": "um",
    "area": "um2",
    "volume": "um3",
    "conductance": "uS",
    "conductanceDensity": "S/cm2",
    "capacitance": "pF",
    "specificCapacitance": "uF/cm2",
    "resistivity": "ohm cm",
    "current": "nA",
    "currentDensity": "mA/cm2",
    "concentration": "mM",
    "temperature": "K",
}

_QUANTITY_RE = re.compile(
    r"^\s*(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>[a-zA-Z_][a-zA-Z_0-9]*)?\s*$"
)


def parse_quantity(
    text: str | float | int | None,
    expected_dimension: Optional[str] = None,
    context: str = "",
) -> tuple[float, str]:
    """Parse a LEMS quantity string into ``(si_value, dimension)``.

    Args:
        text: e.g. ``"-65mV"``, ``"3.0 S_per_m2"``, ``"0.5"``, or a bare number.
        expected_dimension: If given, the parsed unit must have this dimension.
        context: Free-text used in error messages (e.g. ``"channelDensity 'naChans'"``).

    Returns:
        The value converted to SI, and the LEMS dimension name (``"none"`` for a
        dimensionless number without a unit symbol).
    """
    if text is None:
        raise UnitError(f"Missing quantity{_ctx(context)}.")
    if isinstance(text, (int, float)):
        return float(text), expected_dimension or "none"

    match = _QUANTITY_RE.match(str(text))
    if match is None:
        raise UnitError(f"Cannot parse quantity {text!r}{_ctx(context)}.")

    value = float(match.group("value"))
    symbol = match.group("unit")

    if symbol is None:
        dimension = expected_dimension or "none"
        return value, dimension

    if symbol not in UNITS:
        raise UnitError(f"Unknown LEMS unit symbol {symbol!r}{_ctx(context)}.")

    dimension, scale, offset = UNITS[symbol]
    if expected_dimension is not None and dimension != expected_dimension:
        raise UnitError(
            f"Unit {symbol!r} has dimension {dimension!r} but "
            f"{expected_dimension!r} was expected{_ctx(context)}."
        )
    return value * scale + offset, dimension


def to_jaxley(
    text: str | float | int | None,
    dimension: str,
    context: str = "",
    default: Optional[float] = None,
) -> float:
    """Parse a LEMS quantity and return it in Jaxley units.

    Args:
        text: The quantity string, or ``None`` to fall back on ``default``.
        dimension: The LEMS dimension the quantity must have.
        context: Free-text used in error messages.
        default: Value (already in Jaxley units) returned when ``text`` is ``None``.
    """
    if text is None:
        if default is None:
            raise UnitError(f"Missing required quantity{_ctx(context)}.")
        return float(default)
    si_value, parsed_dimension = parse_quantity(text, dimension, context)
    return convert(si_value, parsed_dimension)


def convert(si_value: float, dimension: str) -> float:
    """Convert an SI value of the given LEMS dimension into Jaxley units."""
    if dimension not in _SI_TO_JAXLEY:
        if dimension in DIMENSIONS:
            raise UnitError(
                f"No Jaxley unit defined for LEMS dimension {dimension!r}. "
                "Please open an issue."
            )
        raise UnitError(f"Unknown LEMS dimension {dimension!r}.")
    return si_value * _SI_TO_JAXLEY[dimension]


def _ctx(context: str) -> str:
    return f" in {context}" if context else ""
