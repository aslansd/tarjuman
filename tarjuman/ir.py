"""The tarjuman intermediate representation (IR).

The IR is a small set of dataclasses that sit between the NeuroML XML and the
Jaxley model.  It exists so that the reader does not need to know anything about
Jaxley and the builder does not need to know anything about XML.

**All numeric fields in the IR are already in Jaxley units** (mV, ms, um,
S/cm2, uF/cm2, ohm cm, uS, nA, mM, K).  Unit conversion happens exactly once,
in :mod:`tarjuman.reader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

__all__ = [
    "Rate",
    "Variable",
    "TimeCourse",
    "Q10",
    "Gate",
    "IonChannel",
    "VariableParameter",
    "ChannelDensity",
    "Segment",
    "SegmentGroup",
    "Morphology",
    "BiophysicalProperties",
    "Cell",
    "PointCell",
    "Synapse",
    "InputSource",
    "Instance",
    "Population",
    "Connection",
    "Projection",
    "InputPlacement",
    "InputList",
    "ExplicitInput",
    "Network",
    "Document",
    "SimulationSpec",
]


# --------------------------------------------------------------------------- #
# Channel-level components
# --------------------------------------------------------------------------- #
@dataclass
class Rate:
    """A voltage-dependent rate, ``r(v)``, in 1/ms.

    ``kind`` is one of the LEMS types ``HHExpRate``, ``HHSigmoidRate`` or
    ``HHExpLinearRate``.
    """

    kind: str
    rate: float  # 1/ms
    midpoint: float  # mV
    scale: float  # mV


@dataclass
class Variable:
    """A dimensionless voltage-dependent variable, ``x(v)`` (e.g. ``inf``)."""

    kind: str  # HHExpVariable | HHSigmoidVariable | HHExpLinearVariable
    rate: float  # dimensionless
    midpoint: float  # mV
    scale: float  # mV


@dataclass
class TimeCourse:
    """A voltage-dependent time course, ``t(v)``, in ms."""

    kind: str  # fixedTimeCourse
    tau: float  # ms


@dataclass
class Q10:
    """Temperature scaling of a gate's rates."""

    kind: str  # q10Fixed | q10ExpTemp
    fixed_q10: Optional[float] = None
    q10_factor: Optional[float] = None
    experimental_temp: Optional[float] = None  # K

    def factor(self, temperature: float) -> float:
        """Return the rate-scaling factor at ``temperature`` (K)."""
        if self.kind == "q10Fixed":
            return float(self.fixed_q10)
        return float(self.q10_factor) ** (
            (temperature - float(self.experimental_temp)) / 10.0
        )


@dataclass
class Gate:
    """One gating variable of an ion channel, raised to ``instances``."""

    id: str
    instances: int
    kind: str  # gateHHrates | gateHHtauInf | gateHHratesTau | gateHHratesInf
    # | gateHHratesTauInf | gateHHInstantaneous
    forward_rate: Optional[Rate] = None
    reverse_rate: Optional[Rate] = None
    steady_state: Optional[Variable] = None
    time_course: Optional[TimeCourse] = None
    q10: Optional[Q10] = None


@dataclass
class IonChannel:
    """A NeuroML ``ionChannel`` / ``ionChannelHH`` / ``ionChannelPassive``."""

    id: str
    kind: str
    gates: list[Gate] = field(default_factory=list)
    species: Optional[str] = None
    conductance: Optional[float] = None  # uS, single-channel conductance
    q10_conductance_scaling: Optional[Q10] = None
    notes: Optional[str] = None

    @property
    def is_passive(self) -> bool:
        return len(self.gates) == 0


# --------------------------------------------------------------------------- #
# Cell-level components
# --------------------------------------------------------------------------- #
@dataclass
class VariableParameter:
    """An inhomogeneous parameter on a ``channelDensityNonUniform``."""

    parameter: str
    segment_group: str
    value_expression: str


@dataclass
class ChannelDensity:
    """A ``channelDensity``-family element from ``membraneProperties``."""

    id: str
    ion_channel: str
    cond_density: float  # S/cm2
    erev: Optional[float] = None  # mV
    segment_group: str = "all"
    ion: Optional[str] = None
    kind: str = "channelDensity"
    variable_parameters: list[VariableParameter] = field(default_factory=list)


@dataclass
class Segment:
    """A NeuroML morphology segment (a truncated cone)."""

    id: int
    name: Optional[str] = None
    parent: Optional[int] = None
    fraction_along: float = 1.0
    proximal: Optional[tuple[float, float, float, float]] = None  # x, y, z, diameter
    distal: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @property
    def length(self) -> float:
        """Euclidean length in um (0 for spherical/point segments)."""
        if self.proximal is None:
            return 0.0
        px, py, pz, _ = self.proximal
        dx, dy, dz, _ = self.distal
        return float(((dx - px) ** 2 + (dy - py) ** 2 + (dz - pz) ** 2) ** 0.5)

    @property
    def radius(self) -> float:
        """Mean radius in um."""
        d_distal = self.distal[3]
        d_proximal = self.proximal[3] if self.proximal is not None else d_distal
        return float((d_proximal + d_distal) / 4.0)


@dataclass
class SegmentGroup:
    """A NeuroML ``segmentGroup``."""

    id: str
    members: list[int] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    paths: list[tuple[Optional[int], Optional[int]]] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    neuro_lex_id: Optional[str] = None

    @property
    def is_section(self) -> bool:
        """True for ``sao864921383`` groups, i.e. unbranched cables."""
        return self.neuro_lex_id == "sao864921383"

    @property
    def n_internal_divisions(self) -> Optional[int]:
        value = self.properties.get("numberInternalDivisions")
        return int(float(value)) if value is not None else None


@dataclass
class Morphology:
    id: str
    segments: dict[int, Segment] = field(default_factory=dict)
    groups: dict[str, SegmentGroup] = field(default_factory=dict)


@dataclass
class BiophysicalProperties:
    id: str
    channel_densities: list[ChannelDensity] = field(default_factory=list)
    specific_capacitance: list[tuple[float, str]] = field(default_factory=list)
    init_memb_potential: list[tuple[float, str]] = field(default_factory=list)
    resistivity: list[tuple[float, str]] = field(default_factory=list)
    spike_thresh: list[tuple[float, str]] = field(default_factory=list)


@dataclass
class Cell:
    """A multicompartmental NeuroML ``cell``."""

    id: str
    morphology: Morphology
    biophysical_properties: Optional[BiophysicalProperties] = None
    notes: Optional[str] = None


@dataclass
class PointCell:
    """An abstract (point) cell such as ``izhikevich2007Cell`` or ``iafCell``."""

    id: str
    kind: str
    params: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Network-level components
# --------------------------------------------------------------------------- #
@dataclass
class Synapse:
    """A NeuroML synapse component (``expTwoSynapse``, ``alphaSynapse``, ...)."""

    id: str
    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class InputSource:
    """A current source such as ``pulseGenerator``."""

    id: str
    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class Instance:
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Population:
    id: str
    component: str
    size: int = 1
    instances: list[Instance] = field(default_factory=list)


@dataclass
class Connection:
    pre_cell: int
    post_cell: int
    pre_segment: int = 0
    pre_fraction_along: float = 0.5
    post_segment: int = 0
    post_fraction_along: float = 0.5
    weight: float = 1.0
    delay: float = 0.0  # ms
    #: Synapse component of this connection, for electrical projections where
    #: it is given per connection rather than once for the whole projection.
    synapse: Optional[str] = None
    #: ``continuousConnection`` endpoints: the component on each side.
    pre_component: Optional[str] = None
    post_component: Optional[str] = None


@dataclass
class Projection:
    id: str
    pre_population: str
    post_population: str
    synapse: str
    connections: list[Connection] = field(default_factory=list)
    kind: str = "projection"  # projection | electricalProjection | continuousProjection


@dataclass
class InputPlacement:
    target_cell: int
    segment: int = 0
    fraction_along: float = 0.5
    weight: float = 1.0


@dataclass
class InputList:
    id: str
    component: str
    population: str
    inputs: list[InputPlacement] = field(default_factory=list)


@dataclass
class ExplicitInput:
    target_population: str
    target_cell: int
    input: str


@dataclass
class Network:
    id: str
    populations: list[Population] = field(default_factory=list)
    projections: list[Projection] = field(default_factory=list)
    input_lists: list[InputList] = field(default_factory=list)
    explicit_inputs: list[ExplicitInput] = field(default_factory=list)
    temperature: Optional[float] = None  # K


@dataclass
class Document:
    """Everything read out of one NeuroML file (plus its ``<include>``s)."""

    id: Optional[str] = None
    source: Optional[str] = None
    ion_channels: dict[str, IonChannel] = field(default_factory=dict)
    cells: dict[str, Cell] = field(default_factory=dict)
    point_cells: dict[str, PointCell] = field(default_factory=dict)
    synapses: dict[str, Synapse] = field(default_factory=dict)
    input_sources: dict[str, InputSource] = field(default_factory=dict)
    networks: dict[str, Network] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)

    def merge(self, other: "Document") -> "Document":
        """Merge components of ``other`` into this document (in place)."""
        for attr in (
            "ion_channels",
            "cells",
            "point_cells",
            "synapses",
            "input_sources",
            "networks",
        ):
            getattr(self, attr).update(getattr(other, attr))
        self.includes.extend(other.includes)
        return self


@dataclass
class SimulationSpec:
    """A LEMS ``<Simulation>`` element: what to run and what to record."""

    id: str
    target: str  # id of the network or component to simulate
    length: float  # ms
    step: float  # ms
    seed: Optional[int] = None
    #: (column id, LEMS quantity path) pairs from ``OutputFile``/``Display``.
    outputs: list[tuple[str, str]] = field(default_factory=list)
    #: (event id, LEMS quantity path, threshold) from ``EventOutputFile``.
    event_outputs: list[tuple[str, str, float]] = field(default_factory=list)
