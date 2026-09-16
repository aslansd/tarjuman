"""Read LEMS simulation files (``LEMS_*.xml``).

A NeuroML model on its own says what exists; the LEMS file says what to run:
the duration, the time step, the target network, and which quantities to
record.  ``tarjuman`` reads that file so that ``tarjuman run LEMS_Sim.xml``
reproduces the same experiment that ``jnml`` or ``pyneuroml`` would run.

``<Include file=...>`` elements pointing at NeuroML documents are followed;
those pointing at the core LEMS type definitions (``Cells.xml``,
``Channels.xml``, ``Simulation.xml``, ...) are ignored, since tarjuman
implements those types natively.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .. import ir
from ..errors import ParseError
from ..report import ConversionReport
from ..units import to_jaxley
from .nml import _tag, read_neuroml

__all__ = ["read_lems", "parse_quantity_path"]

#: Core LEMS/NeuroML definition files that tarjuman implements natively.
CORE_DEFINITION_FILES = {
    "NeuroML2CoreTypes.xml",
    "NeuroMLCoreCompTypes.xml",
    "NeuroMLCoreDimensions.xml",
    "Cells.xml",
    "Channels.xml",
    "Inputs.xml",
    "Networks.xml",
    "PyNN.xml",
    "Simulation.xml",
    "Synapses.xml",
}


def read_lems(
    path: str | os.PathLike, report: Optional[ConversionReport] = None
) -> tuple[ir.Document, ir.SimulationSpec]:
    """Read a LEMS file and return its model and its simulation spec.

    Args:
        path: Path to a ``LEMS_*.xml`` file.
        report: Report to record unsupported components in.

    Returns:
        ``(document, simulation)`` where ``document`` holds every component
        defined in the LEMS file and in the NeuroML files it includes.
    """
    path = Path(path).resolve()
    report = report or ConversionReport(source=str(path))
    root = ET.parse(path).getroot()

    document = ir.Document(id=root.get("id"), source=str(path))
    for element in root.iter():
        if _tag(element) != "Include":
            continue
        href = element.get("file") or element.get("href")
        if href is None or Path(href).name in CORE_DEFINITION_FILES:
            continue
        included = (path.parent / href).resolve()
        if not included.exists():
            report.info("Include", f"skipping unresolved include '{href}'")
            continue
        document.merge(read_neuroml(included, report))

    # Some LEMS files define NeuroML components inline.
    from .nml import _read_root  # local import to avoid a cycle at import time

    document.merge(_read_root(root, report))

    target = None
    for element in root.iter():
        if _tag(element) == "Target":
            target = element.get("component")
            break

    simulation_element = None
    for element in root.iter():
        if _tag(element) == "Simulation" and (
            target is None or element.get("id") == target
        ):
            simulation_element = element
            break
    if simulation_element is None:
        raise ParseError(f"No <Simulation> element found in {path}.")

    simulation = ir.SimulationSpec(
        id=simulation_element.get("id"),
        target=simulation_element.get("target"),
        length=to_jaxley(simulation_element.get("length"), "time", "Simulation"),
        step=to_jaxley(simulation_element.get("step"), "time", "Simulation"),
        seed=(
            int(simulation_element.get("seed"))
            if simulation_element.get("seed")
            else None
        ),
    )

    seen: set[str] = set()
    for output_file in simulation_element:
        if _tag(output_file) == "OutputFile":
            for column in output_file:
                if _tag(column) != "OutputColumn":
                    continue
                quantity = column.get("quantity")
                if quantity not in seen:
                    seen.add(quantity)
                    simulation.outputs.append((column.get("id"), quantity))
        elif _tag(output_file) == "EventOutputFile":
            for selection in output_file:
                if _tag(selection) != "EventSelection":
                    continue
                simulation.event_outputs.append(
                    (
                        selection.get("id"),
                        selection.get("select"),
                        0.0,
                    )
                )

    # Fall back on <Display><Line/> when no OutputFile is present.
    if not simulation.outputs:
        for display in simulation_element:
            if _tag(display) != "Display":
                continue
            for line in display:
                if _tag(line) != "Line":
                    continue
                quantity = line.get("quantity")
                if quantity not in seen:
                    seen.add(quantity)
                    simulation.outputs.append((line.get("id"), quantity))

    return document, simulation


def parse_quantity_path(quantity: str) -> dict:
    """Split a LEMS quantity path into the parts tarjuman needs to record it.

    ``"hhpop[0]/v"`` becomes
    ``{"population": "hhpop", "cell": 0, "segment": None, "state": "v"}`` and
    ``"pop0/1/MultiCompCell/2/v"`` becomes
    ``{"population": "pop0", "cell": 1, "segment": 2, "state": "v"}``.

    Channel-state paths such as
    ``"hhpop[0]/bioPhys1/membraneProperties/naChans/naChan/m/q"`` resolve to
    ``{"population": "hhpop", "cell": 0, "density": "naChans",
    "channel": "naChan", "gate": "m", "state": "q"}``.  The ``density`` entry
    is the one tarjuman needs: Jaxley names a channel's states after the
    ``channelDensity`` that placed it, since the same ion channel may sit on
    several segment groups with different conductances.
    """
    text = quantity.strip()
    result: dict = {
        "population": None,
        "cell": 0,
        "segment": None,
        "state": None,
        "density": None,
        "channel": None,
        "gate": None,
        "raw": quantity,
    }

    parts = [part for part in text.split("/") if part not in ("", "..")]
    if not parts:
        raise ParseError(f"Empty quantity path {quantity!r}.")

    head = parts[0]
    if "[" in head and head.endswith("]"):
        result["population"] = head[: head.index("[")]
        result["cell"] = int(head[head.index("[") + 1 : -1])
        rest = parts[1:]
    else:
        result["population"] = head
        rest = parts[1:]
        if rest and rest[0].isdigit():
            result["cell"] = int(rest[0])
            rest = rest[1:]
            if rest:  # skip the component name in populationList paths
                rest = rest[1:]

    if not rest:
        result["state"] = "v"
        return result

    result["state"] = rest[-1]
    if rest[0].isdigit():
        result["segment"] = int(rest[0])
        rest = rest[1:]
        if len(rest) == 1:
            return result

    if "membraneProperties" in rest:
        index = rest.index("membraneProperties")
        tail = rest[index + 1 :]
        if tail:
            result["density"] = tail[0]
        if len(tail) >= 2:
            result["channel"] = tail[1]
        if len(tail) >= 3:
            result["gate"] = tail[2]
    return result
