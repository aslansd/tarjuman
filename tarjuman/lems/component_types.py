"""Parsing of LEMS ``<ComponentType>`` definitions.

NeuroML is a LEMS dialect: the standard elements (``ionChannelHH``,
``expTwoSynapse``, ``fixedFactorConcentrationModel``) are themselves
``ComponentType`` definitions in ``NeuroML2CoreTypes``.  A model file may
define new ones inline, and real models do it constantly — c302 defines
``customHGate``, ``gradedSynapse2``, ``delayedGapJunction``,
``muscleConcentrationModel`` and ``iafActivityCell`` this way.

tarjuman implements the standard types natively (they are fixed and
well-tested) and *interprets* custom ones: this module reads the definition,
:mod:`tarjuman.lems.runtime` compiles its ``<Dynamics>`` into JAX.

Everything here stays in **SI units**, which is the unit system LEMS itself
works in.  A dimensioned expression is only guaranteed to be correct in a
coherent unit system, and Jaxley's (mV, ms, µm, nA, mM, S/cm²) is not one;
conversion happens at the boundary, in :mod:`tarjuman.lems.runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from xml.etree.ElementTree import Element

from ..errors import ParseError
from ..units import parse_quantity

__all__ = [
    "ComponentType",
    "StateVariable",
    "DerivedVariable",
    "ConditionalDerivedVariable",
    "EventAction",
    "read_component_type",
    "resolve_inheritance",
    "base_kind_of",
    "BASE_KINDS",
]


@dataclass
class StateVariable:
    name: str
    dimension: str = "none"
    exposure: Optional[str] = None


@dataclass
class DerivedVariable:
    name: str
    value: Optional[str]
    dimension: str = "none"
    exposure: Optional[str] = None
    #: ``<DerivedVariable select="..." reduce="add"/>`` gathers values from
    #: attached child components (e.g. every synapse on a cell).  tarjuman
    #: records it but cannot evaluate it; a component that uses one is only
    #: rejected if the model actually instantiates it.
    select: Optional[str] = None
    reduce: Optional[str] = None

    @property
    def is_gathered(self) -> bool:
        return self.value is None


@dataclass
class ConditionalDerivedVariable:
    name: str
    #: ``(condition or None for the default case, value)`` in order.
    cases: list[tuple[Optional[str], str]]
    dimension: str = "none"
    exposure: Optional[str] = None


@dataclass
class EventAction:
    """An ``OnStart``, ``OnCondition`` or ``OnEvent`` block."""

    kind: str  # start | condition | event
    test: Optional[str] = None  # OnCondition
    port: Optional[str] = None  # OnEvent
    #: ``(state variable, expression)`` assignments to apply.
    assignments: list[tuple[str, str]] = field(default_factory=list)
    emits_event: bool = False


@dataclass
class ComponentType:
    """A LEMS ``<ComponentType>``, with its dynamics unparsed but structured."""

    name: str
    extends: Optional[str] = None
    description: Optional[str] = None
    parameters: dict[str, str] = field(default_factory=dict)  # name -> dimension
    #: name -> (SI value, dimension)
    constants: dict[str, tuple[float, str]] = field(default_factory=dict)
    derived_parameters: dict[str, tuple[str, str]] = field(default_factory=dict)
    requirements: dict[str, str] = field(default_factory=dict)
    exposures: dict[str, str] = field(default_factory=dict)
    texts: list[str] = field(default_factory=list)
    state_variables: dict[str, StateVariable] = field(default_factory=dict)
    derived_variables: list[DerivedVariable] = field(default_factory=list)
    conditional_derived: list[ConditionalDerivedVariable] = field(default_factory=list)
    time_derivatives: dict[str, str] = field(default_factory=dict)
    events: list[EventAction] = field(default_factory=list)
    #: Features of this ComponentType that tarjuman cannot interpret. Empty
    #: for every type it can instantiate.
    unsupported: list[str] = field(default_factory=list)

    @property
    def gathered_variables(self) -> list[str]:
        """Derived variables defined by ``select`` rather than an expression."""
        return [
            derived.name for derived in self.derived_variables if derived.is_gathered
        ]

    @property
    def is_interpretable(self) -> bool:
        return not self.unsupported and not self.gathered_variables

    @property
    def has_dynamics(self) -> bool:
        return bool(
            self.state_variables
            or self.derived_variables
            or self.conditional_derived
            or self.time_derivatives
            or self.events
        )

    def dimension_of(self, name: str) -> str:
        """Dimension of a parameter, constant, exposure or variable."""
        if name in self.parameters:
            return self.parameters[name]
        if name in self.constants:
            return self.constants[name][1]
        if name in self.exposures:
            return self.exposures[name]
        if name in self.requirements:
            return self.requirements[name]
        if name in self.state_variables:
            return self.state_variables[name].dimension
        for derived in self.derived_variables:
            if derived.name == name:
                return derived.dimension
        for conditional in self.conditional_derived:
            if conditional.name == name:
                return conditional.dimension
        return "none"


#: NeuroML base types tarjuman knows how to attach a custom component to.
BASE_KINDS: dict[str, str] = {
    # gates
    "gate": "gate",
    "gateHHrates": "gate",
    "gateHHtauInf": "gate",
    "gateHHratesTau": "gate",
    "gateHHratesInf": "gate",
    "gateHHratesTauInf": "gate",
    "gateHHInstantaneous": "gate",
    "baseGate": "gate",
    # rates and variables inside a gate
    "baseVoltageDepRate": "rate",
    "baseVoltageConcDepRate": "rate",
    "baseHHRate": "rate",
    "baseVoltageDepVariable": "variable",
    "baseVoltageConcDepVariable": "variable",
    "baseHHVariable": "variable",
    "baseVoltageDepTime": "time_course",
    "baseVoltageConcDepTime": "time_course",
    "baseHHTime": "time_course",
    # concentration models
    "concentrationModel": "concentration_model",
    "baseIonConcentrationModel": "concentration_model",
    "fixedFactorConcentrationModel": "concentration_model",
    "decayingPoolConcentrationModel": "concentration_model",
    # synapses
    "baseSynapse": "synapse",
    "baseVoltageDepSynapse": "synapse",
    "baseGradedSynapse": "synapse",
    "baseGradedSynapseDL": "synapse",
    "gapJunction": "synapse",
    "gradedSynapse": "synapse",
    "linearGradedSynapse": "synapse",
    "expTwoSynapse": "synapse",
    "expOneSynapse": "synapse",
    # cells
    "baseCell": "point_cell",
    "baseSpikingCell": "point_cell",
    "baseCellMembPotCap": "point_cell",
    "iafCell": "point_cell",
    "iafTauCell": "point_cell",
    "baseIaf": "point_cell",
    # channels
    "baseIonChannel": "ion_channel",
    "ionChannelHH": "ion_channel",
}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _si_value(text: Optional[str], dimension: str, context: str) -> float:
    value, _ = parse_quantity(text, dimension if dimension != "none" else None, context)
    return value


def read_component_type(element: Element) -> ComponentType:
    """Read one ``<ComponentType>`` element."""
    component = ComponentType(
        name=element.get("name"),
        extends=element.get("extends"),
        description=element.get("description"),
    )
    context = f"ComponentType '{component.name}'"

    for child in element:
        tag = _strip_ns(child.tag)
        if tag == "Parameter":
            component.parameters[child.get("name")] = child.get("dimension", "none")
        elif tag == "Constant":
            dimension = child.get("dimension", "none")
            component.constants[child.get("name")] = (
                _si_value(child.get("value"), dimension, context),
                dimension,
            )
        elif tag == "DerivedParameter":
            component.derived_parameters[child.get("name")] = (
                child.get("value"),
                child.get("dimension", "none"),
            )
        elif tag == "Requirement":
            component.requirements[child.get("name")] = child.get("dimension", "none")
        elif tag == "Exposure":
            component.exposures[child.get("name")] = child.get("dimension", "none")
        elif tag == "Text":
            component.texts.append(child.get("name"))
        elif tag == "Dynamics":
            _read_dynamics(child, component, context)
        elif tag in ("Child", "Children", "Attachments", "EventPort", "Property",
                     "Fixed", "ComponentReference", "Structure", "Simulation"):
            # Structural elements: harmless to ignore for the component types
            # tarjuman interprets (gates, concentration models, synapses).
            continue
    return component


def _read_dynamics(element: Element, component: ComponentType, context: str) -> None:
    for child in element:
        tag = _strip_ns(child.tag)
        if tag == "StateVariable":
            component.state_variables[child.get("name")] = StateVariable(
                name=child.get("name"),
                dimension=child.get("dimension", "none"),
                exposure=child.get("exposure"),
            )
        elif tag == "DerivedVariable":
            component.derived_variables.append(
                DerivedVariable(
                    name=child.get("name"),
                    value=child.get("value"),
                    dimension=child.get("dimension", "none"),
                    exposure=child.get("exposure"),
                    select=child.get("select"),
                    reduce=child.get("reduce"),
                )
            )
        elif tag == "ConditionalDerivedVariable":
            cases: list[tuple[Optional[str], str]] = []
            for case in child:
                if _strip_ns(case.tag) != "Case":
                    continue
                cases.append((case.get("condition"), case.get("value")))
            component.conditional_derived.append(
                ConditionalDerivedVariable(
                    name=child.get("name"),
                    cases=cases,
                    dimension=child.get("dimension", "none"),
                    exposure=child.get("exposure"),
                )
            )
        elif tag == "TimeDerivative":
            component.time_derivatives[child.get("variable")] = child.get("value")
        elif tag in ("OnStart", "OnCondition", "OnEvent", "OnEntry"):
            action = EventAction(
                kind={
                    "OnStart": "start",
                    "OnEntry": "start",
                    "OnCondition": "condition",
                    "OnEvent": "event",
                }[tag],
                test=child.get("test"),
                port=child.get("port"),
            )
            for assignment in child:
                assignment_tag = _strip_ns(assignment.tag)
                if assignment_tag == "StateAssignment":
                    action.assignments.append(
                        (assignment.get("variable"), assignment.get("value"))
                    )
                elif assignment_tag == "EventOut":
                    action.emits_event = True
            component.events.append(action)
        elif tag == "Regime":
            # Multi-regime dynamics (e.g. a refractory regime).  Recorded so
            # that instantiating the component fails with a clear message,
            # rather than the whole document failing to parse.
            component.unsupported.append("Regime (multi-regime dynamics)")


def resolve_inheritance(
    component: ComponentType, registry: dict[str, ComponentType]
) -> ComponentType:
    """Merge a component type with the custom types it extends.

    Standard NeuroML base types are not merged — they are implemented natively,
    and :func:`base_kind_of` reports which family the component belongs to.
    """
    chain: list[ComponentType] = []
    current: Optional[ComponentType] = component
    seen = set()
    while current is not None:
        if current.name in seen:
            raise ParseError(f"Cyclic 'extends' at ComponentType '{current.name}'.")
        seen.add(current.name)
        chain.append(current)
        current = registry.get(current.extends) if current.extends else None

    merged = ComponentType(
        name=component.name, extends=component.extends, description=component.description
    )
    for ancestor in reversed(chain):  # base first, so the child can override
        merged.parameters.update(ancestor.parameters)
        merged.constants.update(ancestor.constants)
        merged.derived_parameters.update(ancestor.derived_parameters)
        merged.requirements.update(ancestor.requirements)
        merged.exposures.update(ancestor.exposures)
        merged.texts.extend(t for t in ancestor.texts if t not in merged.texts)
        merged.state_variables.update(ancestor.state_variables)
        merged.unsupported.extend(ancestor.unsupported)
        for derived in ancestor.derived_variables:
            merged.derived_variables = [
                existing
                for existing in merged.derived_variables
                if existing.name != derived.name
            ] + [derived]
        merged.conditional_derived.extend(ancestor.conditional_derived)
        merged.time_derivatives.update(ancestor.time_derivatives)
        merged.events.extend(ancestor.events)
    return merged


def base_kind_of(
    component: ComponentType, registry: dict[str, ComponentType]
) -> Optional[str]:
    """Which NeuroML family a custom component ultimately extends.

    Returns one of ``gate``, ``rate``, ``variable``, ``time_course``,
    ``concentration_model``, ``synapse``, ``point_cell``, ``ion_channel``, or
    ``None`` if the chain ends somewhere tarjuman does not recognise.
    """
    current: Optional[ComponentType] = component
    seen = set()
    while current is not None:
        if current.extends in BASE_KINDS:
            return BASE_KINDS[current.extends]
        if current.extends is None or current.extends in seen:
            return None
        seen.add(current.extends)
        current = registry.get(current.extends)
    return None
