"""Reporting of everything tarjuman did, approximated, or refused to convert.

A silent converter is a dangerous converter: a NeuroML model that loses a
calcium pool or a gap junction on the way into Jaxley will still run, and will
still produce plausible-looking voltage traces.  Every conversion therefore
returns a :class:`ConversionReport` alongside the Jaxley module, and every
approximation is recorded in it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

from .errors import ConversionWarning, UnsupportedComponentError

__all__ = ["Note", "ConversionReport"]

Severity = Literal["info", "approximation", "unsupported"]


@dataclass
class Note:
    severity: Severity
    component: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.component}: {self.message}"


@dataclass
class ConversionReport:
    """Structured log of a NeuroML -> Jaxley conversion."""

    source: str | None = None
    strict: bool = False
    notes: list[Note] = field(default_factory=list)
    #: Mapping ``(population, cell_index) -> jaxley cell index``.
    cell_index_map: dict[tuple[str, int], int] = field(default_factory=dict)
    #: Mapping ``(cell_id, segment_id) -> (branch_index, comp_index_range)``.
    segment_map: dict[tuple[str, int], tuple[int, int, int]] = field(
        default_factory=dict
    )
    #: Includes that could not be found on disk, as ``(href, searched path)``.
    #: A LEMS or NeuroML file is only a pointer to the files it includes, so an
    #: unresolved one usually means the model was run away from its siblings.
    unresolved_includes: list[tuple[str, str]] = field(default_factory=list)
    n_cells: int = 0
    n_branches: int = 0
    n_compartments: int = 0
    n_synapses: int = 0

    # -- recording -------------------------------------------------------- #
    def info(self, component: str, message: str) -> None:
        self.notes.append(Note("info", component, message))

    def approximation(self, component: str, message: str) -> None:
        note = Note("approximation", component, message)
        self.notes.append(note)
        warnings.warn(str(note), ConversionWarning, stacklevel=2)

    def missing_include(self, href: str, searched: str) -> None:
        """Record an ``<include>``/``<Include>`` that could not be resolved."""
        self.unresolved_includes.append((href, searched))
        self.approximation(
            "include",
            f"could not find '{href}' (looked in {searched}); anything it defines "
            "is missing from the model",
        )

    def missing_includes_hint(self) -> str:
        """A sentence naming the unresolved includes, for error messages."""
        if not self.unresolved_includes:
            return ""
        names = ", ".join(f"'{href}'" for href, _ in self.unresolved_includes)
        directory = self.unresolved_includes[0][1]
        return (
            f" The file includes {names}, which could not be found in {directory}. "
            "A LEMS or NeuroML file only points at the files it includes, and "
            "those paths are relative to it, so run it where it lives (or give "
            "the full path to it) rather than copying it somewhere else."
        )

    def unsupported(self, component: str, message: str) -> None:
        note = Note("unsupported", component, message)
        self.notes.append(note)
        if self.strict:
            raise UnsupportedComponentError(str(note))
        warnings.warn(str(note), ConversionWarning, stacklevel=2)

    # -- inspection ------------------------------------------------------- #
    @property
    def approximations(self) -> list[Note]:
        return [n for n in self.notes if n.severity == "approximation"]

    @property
    def unsupported_components(self) -> list[Note]:
        return [n for n in self.notes if n.severity == "unsupported"]

    @property
    def is_faithful(self) -> bool:
        """True if nothing was approximated or dropped."""
        return not self.approximations and not self.unsupported_components

    def summary(self) -> str:
        lines = [f"tarjuman conversion report for {self.source or '<in-memory model>'}"]
        lines.append(
            f"  {self.n_cells} cells, {self.n_branches} branches, "
            f"{self.n_compartments} compartments, {self.n_synapses} synapses"
        )
        if self.is_faithful:
            lines.append("  no approximations, nothing dropped")
        else:
            for note in self.notes:
                if note.severity != "info":
                    lines.append(f"  {note}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()
