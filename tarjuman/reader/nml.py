"""Read NeuroML 2 documents into the tarjuman IR.

The reader uses only the standard library (:mod:`xml.etree.ElementTree`), so
``tarjuman`` has no hard dependency on ``libNeuroML``.  If ``libNeuroML`` is
installed it can be used instead via :func:`tarjuman.reader.from_libneuroml`,
which is useful when a model is generated programmatically rather than read
from disk.

Namespaces are stripped on the way in: NeuroML files in the wild declare the
schema namespace inconsistently (and LEMS files not at all), and no NeuroML
element name collides across namespaces.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

from .. import ir
from ..errors import ParseError
from ..lems.builtins import builtin_registry
from ..lems.component_types import base_kind_of, read_component_type, resolve_inheritance
from ..report import ConversionReport
from ..units import to_jaxley

__all__ = ["read_neuroml", "read_neuroml_string"]

# Component types we understand, grouped by the IR object they produce.
_ION_CHANNEL_TAGS = {
    "ionChannel",
    "ionChannelHH",
    "ionChannelPassive",
    "ionChannelVShift",
}
_GATE_TAGS = {
    "gate",
    "gateHHrates",
    "gateHHtauInf",
    "gateHHratesTau",
    "gateHHratesInf",
    "gateHHratesTauInf",
    "gateHHInstantaneous",
    "gateFractional",
    "gateKS",
}
_SYNAPSE_TAGS = {
    "alphaSynapse",
    "expOneSynapse",
    "expTwoSynapse",
    "expThreeSynapse",
    "blockingPlasticSynapse",
    "alphaCurrentSynapse",
    "gapJunction",
    "silentSynapse",
    "linearGradedSynapse",
    "gradedSynapse",
}
_INPUT_TAGS = {
    "pulseGenerator",
    "pulseGeneratorDL",
    "sineGenerator",
    "rampGenerator",
    "voltageClamp",
    "spikeArray",
    "spikeGenerator",
    "spikeGeneratorPoisson",
    "poissonFiringSynapse",
    "transientPoissonFiringSynapse",
}
_CONCENTRATION_MODEL_TAGS = {
    "fixedFactorConcentrationModel",
    "decayingPoolConcentrationModel",
    "concentrationModel",
}
_POINT_CELL_TAGS = {
    "iafCell",
    "iafTauCell",
    "iafRefCell",
    "iafTauRefCell",
    "izhikevichCell",
    "izhikevich2007Cell",
    "adExIaFCell",
    "fitzHughNagumoCell",
    "hindmarshRose1984Cell",
    "pinskyRinzelCA3Cell",
    "IF_curr_alpha",
    "IF_curr_exp",
    "IF_cond_alpha",
    "IF_cond_exp",
    "EIF_cond_exp_isfa_ista",
    "EIF_cond_alpha_isfa_ista",
    "HH_cond_exp",
    "pointCellCondBased",
    "pointCellCondBasedCa",
}


# --------------------------------------------------------------------------- #
# XML helpers
# --------------------------------------------------------------------------- #
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _tag(element: ET.Element) -> str:
    return _strip_ns(element.tag)


def _children(element: ET.Element, *names: str) -> Iterable[ET.Element]:
    wanted = set(names)
    for child in element:
        if _tag(child) in wanted:
            yield child


def _first(element: ET.Element, *names: str) -> Optional[ET.Element]:
    for child in _children(element, *names):
        return child
    return None


def _component_type(element: ET.Element) -> str:
    """The effective LEMS component type of an element.

    NeuroML allows both ``<gateHHrates id="m"/>`` and
    ``<gate id="m" type="gateHHrates"/>``; ``xsi:type`` is used in some files
    too.  All three spellings resolve to the same thing here.
    """
    explicit = element.get("type")
    if explicit:
        return explicit
    xsi_type = element.get("{http://www.w3.org/2001/XMLSchema-instance}type")
    if xsi_type:
        return xsi_type.split(":")[-1]
    return _tag(element)


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
def _read_rate(element: ET.Element, context: str) -> ir.Rate:
    kind = _component_type(element)
    if kind not in ("HHExpRate", "HHSigmoidRate", "HHExpLinearRate"):
        raise ParseError(
            f"Unsupported rate type {kind!r} in {context}. tarjuman implements the "
            "standard NeuroML rate forms (HHExpRate, HHSigmoidRate, "
            "HHExpLinearRate); a rate defined by a custom LEMS ComponentType has "
            "to be added to tarjuman.channels.expressions first."
        )
    return ir.Rate(
        kind=kind,
        rate=to_jaxley(element.get("rate"), "per_time", context),
        midpoint=to_jaxley(element.get("midpoint"), "voltage", context),
        scale=to_jaxley(element.get("scale"), "voltage", context),
    )


def _read_variable(element: ET.Element, context: str) -> ir.Variable:
    kind = _component_type(element)
    if kind not in ("HHExpVariable", "HHSigmoidVariable", "HHExpLinearVariable"):
        raise ParseError(f"Unsupported steadyState type {kind!r} in {context}.")
    return ir.Variable(
        kind=kind,
        rate=to_jaxley(element.get("rate"), "none", context),
        midpoint=to_jaxley(element.get("midpoint"), "voltage", context),
        scale=to_jaxley(element.get("scale"), "voltage", context),
    )


def _read_time_course(element: ET.Element, context: str) -> ir.TimeCourse:
    kind = _component_type(element)
    if kind != "fixedTimeCourse":
        raise ParseError(
            f"Unsupported timeCourse type {kind!r} in {context}; only "
            "'fixedTimeCourse' is supported."
        )
    return ir.TimeCourse(kind=kind, tau=to_jaxley(element.get("tau"), "time", context))


def _read_q10(element: ET.Element, context: str) -> ir.Q10:
    kind = _component_type(element)
    if kind == "q10Fixed":
        return ir.Q10(kind=kind, fixed_q10=float(element.get("fixedQ10")))
    if kind == "q10ExpTemp":
        return ir.Q10(
            kind=kind,
            q10_factor=float(element.get("q10Factor")),
            experimental_temp=to_jaxley(
                element.get("experimentalTemp"), "temperature", context
            ),
        )
    raise ParseError(f"Unsupported q10Settings type {kind!r} in {context}.")


def _read_gate(
    element: ET.Element, context: str, custom_types: Optional[dict] = None
) -> ir.Gate:
    """Read a gate, whether standard or defined by a custom ComponentType."""
    custom_types = custom_types or {}
    kind = _component_type(element)
    gate_id = element.get("id", "q")
    context = f"{context} gate '{gate_id}'"
    instances = int(float(element.get("instances", "1")))

    if kind in custom_types:
        # The whole gate is a custom ComponentType (c302's customHGate).
        return ir.Gate(
            id=gate_id,
            instances=instances,
            kind=kind,
            custom=ir.CustomComponent(
                id=gate_id, component_type=kind, attributes=dict(element.attrib)
            ),
        )

    forward = _first(element, "forwardRate")
    reverse = _first(element, "reverseRate")
    steady_state = _first(element, "steadyState")
    time_course = _first(element, "timeCourse")
    q10 = _first(element, "q10Settings")

    gate = ir.Gate(id=gate_id, instances=instances, kind=kind)
    for part, part_element, reader in (
        ("forward_rate", forward, _read_rate),
        ("reverse_rate", reverse, _read_rate),
        ("steady_state", steady_state, _read_variable),
        ("time_course", time_course, _read_time_course),
    ):
        if part_element is None:
            continue
        part_kind = _component_type(part_element)
        if part_kind in custom_types:
            # Only this part of the gate is custom (e.g. MuscleSigmoidVariable).
            gate.custom_parts[part] = ir.CustomComponent(
                id=part_element.get("id"),
                component_type=part_kind,
                attributes=dict(part_element.attrib),
            )
        else:
            setattr(gate, part, reader(part_element, context))
    gate.q10 = _read_q10(q10, context) if q10 is not None else None
    return gate


def _read_ion_channel(
    element: ET.Element,
    report: ConversionReport,
    custom_types: Optional[dict] = None,
) -> ir.IonChannel:
    custom_types = custom_types or {}
    channel_id = element.get("id")
    context = f"ionChannel '{channel_id}'"
    gate_tags = set(_GATE_TAGS) | {
        name
        for name, component in custom_types.items()
        if component[1] == "gate"
    }
    gates: list[ir.Gate] = []
    for gate_element in _children(element, *gate_tags):
        gate_kind = _component_type(gate_element)
        if gate_kind in ("gateKS", "gateFractional"):
            report.unsupported(
                context,
                f"gate '{gate_element.get('id')}' is a {gate_kind}; kinetic-scheme "
                "and fractional gates are not yet translated, the gate is skipped",
            )
            continue
        gates.append(_read_gate(gate_element, context, custom_types))

    conductance = element.get("conductance")
    scaling = _first(element, "q10ConductanceScaling")

    return ir.IonChannel(
        id=channel_id,
        kind=_tag(element),
        gates=gates,
        species=element.get("species"),
        conductance=(
            to_jaxley(conductance, "conductance", context) if conductance else None
        ),
        q10_conductance_scaling=(
            _read_q10(scaling, context) if scaling is not None else None
        ),
        notes=(element.findtext("{*}notes") or "").strip() or None,
    )


# --------------------------------------------------------------------------- #
# Morphology and biophysics
# --------------------------------------------------------------------------- #
def _read_point(element: Optional[ET.Element]) -> Optional[tuple[float, ...]]:
    if element is None:
        return None
    return (
        float(element.get("x", 0.0)),
        float(element.get("y", 0.0)),
        float(element.get("z", 0.0)),
        float(element.get("diameter", 1.0)),
    )


def _read_morphology(element: ET.Element) -> ir.Morphology:
    morphology = ir.Morphology(id=element.get("id", "morphology"))

    for segment_element in _children(element, "segment"):
        segment_id = int(segment_element.get("id"))
        parent_element = _first(segment_element, "parent")
        parent = None
        fraction_along = 1.0
        if parent_element is not None:
            parent = int(parent_element.get("segment"))
            fraction_along = float(parent_element.get("fractionAlong", 1.0))
        distal = _read_point(_first(segment_element, "distal"))
        if distal is None:
            raise ParseError(f"Segment {segment_id} has no <distal> point.")
        morphology.segments[segment_id] = ir.Segment(
            id=segment_id,
            name=segment_element.get("name"),
            parent=parent,
            fraction_along=fraction_along,
            proximal=_read_point(_first(segment_element, "proximal")),
            distal=distal,
        )

    for group_element in _children(element, "segmentGroup"):
        group = ir.SegmentGroup(
            id=group_element.get("id"),
            neuro_lex_id=group_element.get("neuroLexId"),
        )
        for child in group_element:
            tag = _tag(child)
            if tag == "member":
                group.members.append(int(child.get("segment")))
            elif tag == "include":
                group.includes.append(child.get("segmentGroup"))
            elif tag == "path" or tag == "subTree":
                start = _first(child, "from")
                end = _first(child, "to")
                group.paths.append(
                    (
                        int(start.get("segment")) if start is not None else None,
                        int(end.get("segment")) if end is not None else None,
                    )
                )
            elif tag == "property":
                group.properties[child.get("tag")] = child.get("value")
        morphology.groups[group.id] = group

    # Segments whose <proximal> is omitted inherit the parent's distal point.
    for segment in morphology.segments.values():
        if segment.proximal is None and segment.parent is not None:
            parent_segment = morphology.segments.get(segment.parent)
            if parent_segment is not None and parent_segment.distal is not None:
                segment.proximal = parent_segment.distal
    return morphology


def _read_value_group_pairs(
    element: ET.Element, tag: str, dimension: str, context: str
) -> list[tuple[float, str]]:
    pairs = []
    for child in _children(element, tag):
        pairs.append(
            (
                to_jaxley(child.get("value"), dimension, f"{context} {tag}"),
                child.get("segmentGroup", "all") or "all",
            )
        )
    return pairs


def _read_biophysics(
    element: ET.Element, report: ConversionReport
) -> ir.BiophysicalProperties:
    properties_id = element.get("id", "biophysicalProperties")
    context = f"biophysicalProperties '{properties_id}'"
    biophysics = ir.BiophysicalProperties(id=properties_id)

    membrane = _first(element, "membraneProperties", "membraneProperties2CaPools")
    if membrane is not None:
        for density_element in membrane:
            tag = _tag(density_element)
            if not tag.startswith("channelDensity"):
                continue
            density_id = density_element.get("id")
            density_context = f"{context} {tag} '{density_id}'"
            if tag in ("channelDensityGHK", "channelDensityGHK2"):
                report.unsupported(
                    density_context,
                    "Goldman-Hodgkin-Katz permeability is not supported; "
                    "the channel is skipped",
                )
                continue
            cond_density = density_element.get("condDensity")
            if cond_density is None:
                report.unsupported(
                    density_context,
                    "no condDensity given (non-uniform density); the channel is skipped",
                )
                continue
            erev = density_element.get("erev")
            if erev is None and tag == "channelDensityNernst":
                report.approximation(
                    density_context,
                    "Nernst reversal potential depends on the calcium pool, which "
                    "Jaxley does not integrate here; erev must be supplied via "
                    "`erev_overrides` or the channel current will be wrong",
                )
            biophysics.channel_densities.append(
                ir.ChannelDensity(
                    id=density_id,
                    ion_channel=density_element.get("ionChannel"),
                    cond_density=to_jaxley(
                        cond_density, "conductanceDensity", density_context
                    ),
                    erev=(
                        to_jaxley(erev, "voltage", density_context)
                        if erev is not None
                        else None
                    ),
                    segment_group=density_element.get("segmentGroup", "all") or "all",
                    ion=density_element.get("ion"),
                    kind=tag,
                )
            )
        biophysics.specific_capacitance = _read_value_group_pairs(
            membrane, "specificCapacitance", "specificCapacitance", context
        )
        biophysics.init_memb_potential = _read_value_group_pairs(
            membrane, "initMembPotential", "voltage", context
        )
        biophysics.spike_thresh = _read_value_group_pairs(
            membrane, "spikeThresh", "voltage", context
        )

    intracellular = _first(
        element, "intracellularProperties", "intracellularProperties2CaPools"
    )
    if intracellular is not None:
        biophysics.resistivity = _read_value_group_pairs(
            intracellular, "resistivity", "resistivity", context
        )
        for species in _children(intracellular, "species"):
            species_context = f"{context} species '{species.get('id')}'"
            biophysics.species.append(
                ir.Species(
                    id=species.get("id"),
                    concentration_model=species.get("concentrationModel"),
                    ion=species.get("ion", "ca"),
                    initial_concentration=to_jaxley(
                        species.get("initialConcentration"),
                        "concentration",
                        species_context,
                        default=0.0,
                    ),
                    initial_ext_concentration=to_jaxley(
                        species.get("initialExtConcentration"),
                        "concentration",
                        species_context,
                        default=2.0,
                    ),
                    segment_group=species.get("segmentGroup", "all") or "all",
                )
            )
    return biophysics


def _read_cell(
    element: ET.Element,
    document: ir.Document,
    root: ET.Element,
    report: ConversionReport,
) -> ir.Cell:
    cell_id = element.get("id")
    morphology_element = _first(element, "morphology")
    if morphology_element is None:
        reference = element.get("morphology")
        morphology_element = _find_by_id(root, "morphology", reference)
        if morphology_element is None:
            raise ParseError(f"Cell '{cell_id}' has no morphology.")
    biophysics_element = _first(element, "biophysicalProperties")
    if biophysics_element is None and element.get("biophysicalProperties"):
        biophysics_element = _find_by_id(
            root, "biophysicalProperties", element.get("biophysicalProperties")
        )

    return ir.Cell(
        id=cell_id,
        morphology=_read_morphology(morphology_element),
        biophysical_properties=(
            _read_biophysics(biophysics_element, report)
            if biophysics_element is not None
            else None
        ),
        notes=(element.findtext("{*}notes") or "").strip() or None,
    )


def _find_by_id(root: ET.Element, tag: str, element_id: Optional[str]):
    if element_id is None:
        return None
    for element in root.iter():
        if _tag(element) == tag and element.get("id") == element_id:
            return element
    return None


# --------------------------------------------------------------------------- #
# Synapses, inputs, networks
# --------------------------------------------------------------------------- #
_SYNAPSE_PARAM_DIMENSIONS = {
    "gbase": "conductance",
    "gbase1": "conductance",
    "gbase2": "conductance",
    "erev": "voltage",
    "tau": "time",
    "tauRise": "time",
    "tauDecay": "time",
    "tauDecay1": "time",
    "tauDecay2": "time",
    "conductance": "conductance",
    "delay": "time",
    "duration": "time",
    "amplitude": "current",
    "period": "time",
    "phase": "none",
    "startAmplitude": "current",
    "finishAmplitude": "current",
    "baselineAmplitude": "current",
    "averageRate": "per_time",
    "weight": "none",
    # point cells
    "C": "capacitance",
    "leakConductance": "conductance",
    "leakReversal": "voltage",
    "thresh": "voltage",
    "reset": "voltage",
    "refract": "time",
    "tau": "time",
    # gradedSynapse / gapJunction
    "delta": "voltage",
    "k": "per_time",
    "Vth": "voltage",
}


def _read_params(element: ET.Element, context: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, value in element.attrib.items():
        if name in ("id", "type", "species", "neuroLexId"):
            continue
        dimension = _SYNAPSE_PARAM_DIMENSIONS.get(name)
        if dimension is None:
            continue
        params[name] = to_jaxley(value, dimension, context)
    return params


def _read_network(element: ET.Element, report: ConversionReport) -> ir.Network:
    network_id = element.get("id")
    context = f"network '{network_id}'"
    network = ir.Network(id=network_id)

    temperature = element.get("temperature")
    if temperature is not None:
        network.temperature = to_jaxley(temperature, "temperature", context)

    for population_element in _children(element, "population"):
        population = ir.Population(
            id=population_element.get("id"),
            component=population_element.get("component"),
            size=int(float(population_element.get("size", "1"))),
        )
        for instance_element in _children(population_element, "instance"):
            location = _first(instance_element, "location")
            population.instances.append(
                ir.Instance(
                    id=int(instance_element.get("id")),
                    x=float(location.get("x", 0.0)) if location is not None else 0.0,
                    y=float(location.get("y", 0.0)) if location is not None else 0.0,
                    z=float(location.get("z", 0.0)) if location is not None else 0.0,
                )
            )
        if population.instances:
            population.size = len(population.instances)
        network.populations.append(population)

    for projection_element in _children(
        element, "projection", "electricalProjection", "continuousProjection"
    ):
        kind = _tag(projection_element)
        projection_id = projection_element.get("id")
        projection = ir.Projection(
            id=projection_id,
            pre_population=projection_element.get("presynapticPopulation"),
            post_population=projection_element.get("postsynapticPopulation"),
            synapse=projection_element.get("synapse"),
            kind=kind,
        )
        for connection_element in projection_element:
            connection_tag = _tag(connection_element)
            if not connection_tag.lower().endswith(
                ("connection", "connectionwd", "connectioninstance", "connectioninstancew")
            ):
                continue
            # Chemical projections name the cells "preCellId"/"postCellId";
            # electrical and continuous ones use "preCell"/"postCell".
            pre = connection_element.get("preCellId") or connection_element.get(
                "preCell"
            )
            post = connection_element.get("postCellId") or connection_element.get(
                "postCell"
            )
            projection.connections.append(
                ir.Connection(
                    pre_cell=_cell_index(pre),
                    post_cell=_cell_index(post),
                    pre_segment=int(
                        float(
                            connection_element.get("preSegmentId")
                            or connection_element.get("preSegment")
                            or 0
                        )
                    ),
                    pre_fraction_along=float(
                        connection_element.get("preFractionAlong", 0.5)
                    ),
                    post_segment=int(
                        float(
                            connection_element.get("postSegmentId")
                            or connection_element.get("postSegment")
                            or 0
                        )
                    ),
                    post_fraction_along=float(
                        connection_element.get("postFractionAlong", 0.5)
                    ),
                    weight=float(connection_element.get("weight", 1.0)),
                    delay=to_jaxley(
                        connection_element.get("delay"),
                        "time",
                        f"projection '{projection_id}'",
                        default=0.0,
                    ),
                    synapse=connection_element.get("synapse"),
                    pre_component=connection_element.get("preComponent"),
                    post_component=connection_element.get("postComponent"),
                )
            )
        network.projections.append(projection)

    for input_list_element in _children(element, "inputList"):
        input_list = ir.InputList(
            id=input_list_element.get("id"),
            component=input_list_element.get("component"),
            population=input_list_element.get("population"),
        )
        for input_element in _children(input_list_element, "input", "inputW"):
            input_list.inputs.append(
                ir.InputPlacement(
                    target_cell=_cell_index(input_element.get("target")),
                    segment=int(float(input_element.get("segmentId", 0))),
                    fraction_along=float(input_element.get("fractionAlong", 0.5)),
                    weight=float(input_element.get("weight", 1.0)),
                )
            )
        network.input_lists.append(input_list)

    for explicit_element in _children(element, "explicitInput"):
        target = explicit_element.get("target")
        population, index = _population_and_index(target)
        network.explicit_inputs.append(
            ir.ExplicitInput(
                target_population=population,
                target_cell=index,
                input=explicit_element.get("input"),
            )
        )
    return network


def _cell_index(path: Optional[str]) -> int:
    """Extract the instance index from a NeuroML target path.

    Accepts ``"../pop0/3/MultiCompCell"``, ``"pop0[3]"`` and ``"3"``.
    """
    if path is None:
        return 0
    text = path.strip()
    if "[" in text and text.endswith("]"):
        return int(text[text.index("[") + 1 : -1])
    parts = [part for part in text.split("/") if part not in ("", "..")]
    for part in parts[1:]:
        if part.isdigit():
            return int(part)
    if text.isdigit():
        return int(text)
    raise ParseError(f"Cannot extract a cell index from target path {path!r}.")


def _population_and_index(path: str) -> tuple[str, int]:
    text = path.strip()
    if "[" in text and text.endswith("]"):
        return text[: text.index("[")], int(text[text.index("[") + 1 : -1])
    parts = [part for part in text.split("/") if part not in ("", "..")]
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return parts[0], 0


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def read_neuroml(
    path: str | os.PathLike,
    report: Optional[ConversionReport] = None,
    follow_includes: bool = True,
    _seen: Optional[set] = None,
) -> ir.Document:
    """Read a ``.nml`` file (and, by default, everything it ``<include>``s).

    Args:
        path: Path to the NeuroML 2 document.
        report: Report to record unsupported components in; a fresh one is
            created if omitted.
        follow_includes: Whether to resolve ``<include href=...>`` elements
            relative to the including file.

    Returns:
        A :class:`tarjuman.ir.Document` with every component the file defines.
    """
    path = Path(path).resolve()
    report = report or ConversionReport(source=str(path))
    seen = _seen if _seen is not None else set()
    if path in seen:
        return ir.Document(source=str(path))
    seen.add(path)

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:  # pragma: no cover - malformed input
        raise ParseError(f"Could not parse {path}: {error}") from error

    # Included files are read first: they may define the ComponentTypes this
    # file uses, and a custom gate cannot be read without its definition.
    included_documents: list[ir.Document] = []
    if follow_includes:
        for href in _include_hrefs(root):
            included = (path.parent / href).resolve()
            if not included.exists():
                # Core NeuroML type definitions (Cells.xml, Channels.xml, ...)
                # are built into tarjuman and need not be resolved.
                report.info("include", f"skipping unresolved include '{href}'")
                continue
            included_documents.append(
                read_neuroml(included, report, follow_includes, _seen=seen)
            )

    inherited_types: dict = {}
    for included_document in included_documents:
        inherited_types.update(included_document.component_types)

    document = _read_root(root, report, inherited_types)
    document.source = str(path)
    for included_document in included_documents:
        merged = ir.Document()
        merged.merge(included_document)
        merged.merge(document)
        document.merge(included_document)
    return document


def _include_hrefs(root: ET.Element) -> list[str]:
    hrefs = []
    for element in root:
        if _tag(element) == "include":
            href = element.get("href") or element.get("file")
            if href:
                hrefs.append(href)
    return hrefs


def read_neuroml_string(
    text: str, report: Optional[ConversionReport] = None
) -> ir.Document:
    """Read a NeuroML 2 document from a string (``<include>``s are ignored)."""
    report = report or ConversionReport(source="<string>")
    return _read_root(ET.fromstring(text), report)


def collect_component_types(
    root: ET.Element, inherited: Optional[dict] = None
) -> tuple[dict, dict]:
    """Read every ``<ComponentType>`` in a document.

    Returns ``(definitions, usable)`` where ``definitions`` maps name ->
    :class:`~tarjuman.lems.component_types.ComponentType` (inheritance
    resolved) and ``usable`` maps name -> ``(component, base kind)`` for those
    whose base type tarjuman can attach to.
    """
    registry = builtin_registry()
    registry.update(inherited or {})

    raw: dict = {}
    for element in root.iter():
        if _tag(element) == "ComponentType":
            component = read_component_type(element)
            raw[component.name] = component
    registry.update(raw)

    definitions: dict = {}
    usable: dict = {}
    for name in raw:
        resolved = resolve_inheritance(registry[name], registry)
        definitions[name] = resolved
        kind = base_kind_of(registry[name], registry)
        if kind is not None:
            usable[name] = (resolved, kind)
    return definitions, usable


def _read_root(
    root: ET.Element,
    report: ConversionReport,
    inherited_types: Optional[dict] = None,
) -> ir.Document:
    document = ir.Document(id=root.get("id"))

    definitions, custom_types = collect_component_types(root, inherited_types)
    document.component_types.update(definitions)
    # Custom types defined in files included earlier are usable here too.
    for name, component in (inherited_types or {}).items():
        kind = base_kind_of(component, {**builtin_registry(), **(inherited_types or {})})
        if kind is not None and name not in custom_types:
            custom_types[name] = (component, kind)

    for element in root:
        tag = _tag(element)
        if tag == "ComponentType":
            continue
        if tag in _CONCENTRATION_MODEL_TAGS or (
            tag in custom_types and custom_types[tag][1] == "concentration_model"
        ):
            document.concentration_models[element.get("id")] = ir.ConcentrationModel(
                id=element.get("id"),
                kind=tag,
                ion=element.get("ion", "ca"),
                attributes=dict(element.attrib),
            )
            continue
        if tag in custom_types and custom_types[tag][1] == "synapse":
            synapse_id = element.get("id")
            document.synapses[synapse_id] = ir.Synapse(
                id=synapse_id,
                kind=tag,
                custom=ir.CustomComponent(
                    id=synapse_id, component_type=tag, attributes=dict(element.attrib)
                ),
            )
            continue
        if tag == "include":
            href = element.get("href") or element.get("file")
            if href:
                document.includes.append(href)
        elif tag in _ION_CHANNEL_TAGS:
            channel = _read_ion_channel(element, report, custom_types)
            if tag == "ionChannelVShift":
                report.approximation(
                    f"ionChannelVShift '{channel.id}'",
                    "the vShift attribute is ignored; rates are used unshifted",
                )
            document.ion_channels[channel.id] = channel
        elif tag == "ionChannelKS":
            report.unsupported(
                f"ionChannelKS '{element.get('id')}'",
                "kinetic-scheme channels are not yet translated",
            )
        elif tag == "cell":
            cell = _read_cell(element, document, root, report)
            document.cells[cell.id] = cell
        elif tag in _POINT_CELL_TAGS:
            document.point_cells[element.get("id")] = ir.PointCell(
                id=element.get("id"),
                kind=tag,
                params=_read_params(element, f"{tag} '{element.get('id')}'"),
            )
        elif tag in _SYNAPSE_TAGS:
            synapse_id = element.get("id")
            synapse = ir.Synapse(
                id=synapse_id,
                kind=tag,
                params=_read_params(element, f"{tag} '{synapse_id}'"),
            )
            if _first(element, "plasticityMechanism") is not None:
                report.approximation(
                    f"{tag} '{synapse_id}'",
                    "short-term plasticity (tsodyksMarkram) is ignored; the synapse "
                    "is converted without it",
                )
            if _first(element, "blockMechanism") is not None:
                synapse.kind = "blockingPlasticSynapse"
                block = _first(element, "blockMechanism")
                synapse.params["blockConcentration"] = to_jaxley(
                    block.get("blockConcentration"), "concentration", synapse_id
                )
                synapse.params["scalingConc"] = to_jaxley(
                    block.get("scalingConc"), "concentration", synapse_id
                )
                synapse.params["scalingVolt"] = to_jaxley(
                    block.get("scalingVolt"), "voltage", synapse_id
                )
            document.synapses[synapse_id] = synapse
        elif tag in _INPUT_TAGS:
            input_id = element.get("id")
            document.input_sources[input_id] = ir.InputSource(
                id=input_id,
                kind=tag,
                params=_read_params(element, f"{tag} '{input_id}'"),
            )
        elif tag == "network":
            network = _read_network(element, report)
            document.networks[network.id] = network
    return document
