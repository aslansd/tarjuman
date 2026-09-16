"""Readers that turn NeuroML 2 and LEMS documents into the tarjuman IR."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from .. import ir
from ..report import ConversionReport
from .lems import CORE_DEFINITION_FILES, parse_quantity_path, read_lems
from .nml import read_neuroml, read_neuroml_string

__all__ = [
    "read_neuroml",
    "read_neuroml_string",
    "read_lems",
    "parse_quantity_path",
    "from_libneuroml",
    "CORE_DEFINITION_FILES",
]


def from_libneuroml(
    nml_document, report: Optional[ConversionReport] = None
) -> ir.Document:
    """Read a ``libNeuroML`` ``NeuroMLDocument`` object into the tarjuman IR.

    This lets tarjuman accept models that were built programmatically with
    ``libNeuroML`` (as ``pyNeuroML`` and most model-generation pipelines do)
    without going through a file on disk.  ``libNeuroML`` is an optional
    dependency; install it with ``pip install tarjuman[neuroml]``.

    Args:
        nml_document: A ``neuroml.NeuroMLDocument`` instance.
        report: Report to record unsupported components in.

    Returns:
        The same :class:`tarjuman.ir.Document` the XML reader would produce.
    """
    try:
        import neuroml.writers  # noqa: F401
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "from_libneuroml() requires libNeuroML: pip install tarjuman[neuroml]"
        ) from error

    from neuroml.writers import NeuroMLWriter

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{nml_document.id or 'model'}.nml"
        NeuroMLWriter.write(nml_document, str(path))
        return read_neuroml(path, report=report, follow_includes=False)
