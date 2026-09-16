"""Exception and warning types raised by tarjuman."""

from __future__ import annotations

__all__ = [
    "TarjumanError",
    "UnitError",
    "ParseError",
    "UnsupportedComponentError",
    "MorphologyError",
    "ConversionWarning",
]


class TarjumanError(Exception):
    """Base class for all errors raised by tarjuman."""


class UnitError(TarjumanError):
    """A LEMS quantity could not be parsed or has the wrong dimension."""


class ParseError(TarjumanError):
    """A NeuroML or LEMS document is malformed or references a missing component."""


class UnsupportedComponentError(TarjumanError):
    """A NeuroML component type has no Jaxley equivalent.

    Raised only in strict mode; otherwise the component is recorded in the
    :class:`~tarjuman.report.ConversionReport` and skipped.
    """


class MorphologyError(TarjumanError):
    """A NeuroML morphology cannot be mapped onto Jaxley's branch structure."""


class ConversionWarning(UserWarning):
    """Emitted when a model is converted with a documented approximation."""
