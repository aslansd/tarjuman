"""Build Jaxley modules from the tarjuman IR.

This is where NeuroML's description of a model meets Jaxley's data structures:

* a NeuroML ``cell`` becomes a :class:`jaxley.Cell` whose branches are the
  cell's unbranched cables,
* a NeuroML ``network`` becomes a :class:`jaxley.Network` whose cells are the
  instances of every population, in population order,
* ``channelDensity`` elements become inserted channels,
* ``projection``/``connection`` elements become Jaxley synapses, and
* ``inputList``/``explicitInput`` elements are collected as stimuli and applied
  when the simulation's time step is known.

The result is a :class:`ConvertedModel`, which keeps the maps needed to talk
about the Jaxley module in NeuroML terms — "population ``pop0``, cell 2,
segment 3, 0.7 along" — long after the conversion is done.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import jaxley as jx
import numpy as np
from jaxley.connect import connect

from . import ir
from .channels import DEFAULT_TEMPERATURE, NeuroMLChannel, make_channel
from .errors import ParseError
from .morphology import CellMorphology, build_morphology
from .report import ConversionReport
from .synapses import DEFAULT_SPIKE_THRESHOLD, make_synapse

__all__ = ["ConvertedModel", "Stimulus", "build_cell", "build_network", "build"]


@dataclass
class Stimulus:
    """A current injection, kept until the time step is known."""

    source: ir.InputSource
    cell_index: int
    branch: int
    comp: int
    weight: float = 1.0
    population: Optional[str] = None
    population_cell: int = 0


@dataclass
class ConvertedModel:
    """A Jaxley module plus everything needed to address it in NeuroML terms."""

    module: jx.Module
    report: ConversionReport
    document: ir.Document
    morphologies: dict[str, CellMorphology] = field(default_factory=dict)
    network: Optional[ir.Network] = None
    #: ``(population id, instance index) -> Jaxley cell index``
    cell_index: dict[tuple[str, int], int] = field(default_factory=dict)
    #: NeuroML cell id of each Jaxley cell, by Jaxley cell index.
    cell_types: list[str] = field(default_factory=list)
    stimuli: list[Stimulus] = field(default_factory=list)
    temperature: float = DEFAULT_TEMPERATURE

    # -- addressing ------------------------------------------------------- #
    def locate(
        self,
        population: Optional[str],
        cell: int = 0,
        segment: int = 0,
        fraction_along: float = 0.5,
    ) -> tuple[int, int, int]:
        """Resolve a NeuroML location to ``(cell index, branch, comp)`` in Jaxley."""
        if population is None:
            cell_index = cell
        else:
            key = (population, cell)
            if key not in self.cell_index:
                raise ParseError(
                    f"No cell {cell} in population '{population}'. Known populations: "
                    f"{sorted({p for p, _ in self.cell_index})}."
                )
            cell_index = self.cell_index[key]
        morphology = self.morphologies[self.cell_types[cell_index]]
        branch, comp = morphology.locate(segment, fraction_along)
        return cell_index, branch, comp

    def view(
        self,
        population: Optional[str],
        cell: int = 0,
        segment: int = 0,
        fraction_along: float = 0.5,
    ):
        """A single-compartment Jaxley view of a NeuroML location."""
        cell_index, branch, comp = self.locate(
            population, cell, segment, fraction_along
        )
        module = self.module
        if isinstance(module, jx.Network):
            return module.cell(cell_index).branch(branch).comp(comp)
        return module.branch(branch).comp(comp)

    # -- stimuli ---------------------------------------------------------- #
    def apply_stimuli(self, delta_t: float, t_max: float) -> None:
        """Turn the collected NeuroML input sources into Jaxley stimuli."""
        for stimulus in self.stimuli:
            current = self._current_of(stimulus, delta_t, t_max)
            if current is None:
                continue
            module = self.module
            view = (
                module.cell(stimulus.cell_index)
                .branch(stimulus.branch)
                .comp(stimulus.comp)
                if isinstance(module, jx.Network)
                else module.branch(stimulus.branch).comp(stimulus.comp)
            )
            view.stimulate(current, verbose=False)

    def _current_of(self, stimulus: Stimulus, delta_t: float, t_max: float):
        source = stimulus.source
        if source.kind in ("pulseGenerator", "pulseGeneratorDL"):
            return jx.step_current(
                i_delay=source.params.get("delay", 0.0),
                i_dur=source.params.get("duration", t_max),
                i_amp=source.params.get("amplitude", 0.0) * stimulus.weight,
                delta_t=delta_t,
                t_max=t_max,
            )
        if source.kind == "rampGenerator":
            time = np.arange(int(t_max // delta_t) + 2) * delta_t
            delay = source.params.get("delay", 0.0)
            duration = source.params.get("duration", t_max)
            start = source.params.get("startAmplitude", 0.0)
            finish = source.params.get("finishAmplitude", 0.0)
            baseline = source.params.get("baselineAmplitude", 0.0)
            fraction = np.clip((time - delay) / max(duration, 1e-12), 0.0, 1.0)
            inside = (time >= delay) & (time <= delay + duration)
            ramp = start + (finish - start) * fraction
            return np.where(inside, ramp, baseline) * stimulus.weight
        if source.kind == "sineGenerator":
            time = np.arange(int(t_max // delta_t) + 2) * delta_t
            delay = source.params.get("delay", 0.0)
            duration = source.params.get("duration", t_max)
            amplitude = source.params.get("amplitude", 0.0)
            period = source.params.get("period", 1.0)
            phase = source.params.get("phase", 0.0)
            inside = (time >= delay) & (time <= delay + duration)
            wave = amplitude * np.sin(phase + 2 * np.pi * (time - delay) / period)
            return np.where(inside, wave, 0.0) * stimulus.weight
        self.report.unsupported(
            f"{source.kind} '{source.id}'",
            "this input type is not translated; no current is injected",
        )
        return None


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def build_cell(
    document: ir.Document,
    cell_id: str,
    report: Optional[ConversionReport] = None,
    ncomp: int = 1,
    max_comp_length: Optional[float] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    erev_overrides: Optional[dict[str, float]] = None,
) -> tuple[jx.Cell, CellMorphology]:
    """Build one Jaxley cell from a NeuroML ``cell``.

    Args:
        document: The document the cell lives in (its ion channels are needed).
        cell_id: Id of the NeuroML cell.
        report: Conversion report.
        ncomp: Default compartments per branch.
        max_comp_length: Cap on compartment length in um (adds compartments).
        temperature: Temperature in K, for ``q10Settings``.
        erev_overrides: Reversal potentials in mV keyed by ``channelDensity``
            id, used where NeuroML leaves ``erev`` implicit.

    Returns:
        The Jaxley cell and the morphology map used to build it.
    """
    report = report or ConversionReport()
    erev_overrides = erev_overrides or {}
    if cell_id not in document.cells:
        if cell_id in document.point_cells:
            raise ParseError(
                f"'{cell_id}' is an abstract point cell "
                f"({document.point_cells[cell_id].kind}); tarjuman converts "
                "multicompartmental NeuroML cells. See the README for the status "
                "of point-cell support."
            )
        raise ParseError(f"No cell '{cell_id}' in the document.")

    cell_ir = document.cells[cell_id]
    morphology = build_morphology(
        cell_ir, report, ncomp=ncomp, max_comp_length=max_comp_length
    )

    branches = [
        jx.Branch(jx.Compartment(), ncomp=branch.ncomp)
        for branch in morphology.branches
    ]
    cell = jx.Cell(
        branches,
        parents=morphology.parents,
        xyzr=[branch.xyzr for branch in morphology.branches],
    )

    # Geometry: Jaxley stores one length and one radius per compartment, and
    # derives the membrane area from them.  We pass the traced points as `xyzr`
    # for plotting, but the NeuroML segment lengths and diameters are what
    # define the electrotonic structure, so they are set explicitly.  Jaxley
    # warns about exactly this when `xyzr` is present (it assumes an SWC
    # tracing); the warning does not apply here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="You are modifying the", category=UserWarning
        )
        for branch in morphology.branches:
            view = cell.branch(branch.index)
            view.set("length", branch.length / branch.ncomp)
            for comp, radius in enumerate(_compartment_radii(branch)):
                view.comp(comp).set("radius", radius)

    biophysics = cell_ir.biophysical_properties
    if biophysics is None:
        report.approximation(
            f"cell '{cell_id}'",
            "has no biophysicalProperties; the cell is passive with Jaxley defaults",
        )
        return cell, morphology

    _apply_passive_properties(cell, morphology, biophysics, report, cell_id)
    _insert_channels(
        cell,
        morphology,
        biophysics,
        document,
        report,
        temperature,
        erev_overrides,
        cell_id,
    )
    return cell, morphology


def _compartment_radii(branch) -> list[float]:
    """Radius of each compartment, interpolated along the traced segments."""
    if branch.ncomp == 1 or branch.length <= 0:
        return [branch.radius] * branch.ncomp
    edges = np.cumsum([0.0] + branch.lengths)
    comp_length = branch.length / branch.ncomp
    radii = []
    for comp in range(branch.ncomp):
        centre = (comp + 0.5) * comp_length
        index = int(np.clip(np.searchsorted(edges, centre) - 1, 0, len(branch.radii) - 1))
        radii.append(branch.radii[index])
    return radii


def _views_for_group(cell, morphology: CellMorphology, group: str):
    """Yield Jaxley views covering a NeuroML segment group."""
    if group in ("all", "", None):
        yield cell
        return
    by_branch: dict[int, list[int]] = {}
    for branch, comp in morphology.compartments_of_group(group):
        by_branch.setdefault(branch, []).append(comp)
    for branch, comps in by_branch.items():
        yield cell.branch(branch).comp(comps)


def _apply_passive_properties(
    cell,
    morphology: CellMorphology,
    biophysics: ir.BiophysicalProperties,
    report: ConversionReport,
    cell_id: str,
) -> None:
    for value, group in biophysics.specific_capacitance:
        for view in _views_for_group(cell, morphology, group):
            view.set("capacitance", value)
    for value, group in biophysics.resistivity:
        for view in _views_for_group(cell, morphology, group):
            view.set("axial_resistivity", value)
    for value, group in biophysics.init_memb_potential:
        for view in _views_for_group(cell, morphology, group):
            view.set("v", value)


def _insert_channels(
    cell,
    morphology: CellMorphology,
    biophysics: ir.BiophysicalProperties,
    document: ir.Document,
    report: ConversionReport,
    temperature: float,
    erev_overrides: dict[str, float],
    cell_id: str,
) -> None:
    for density in biophysics.channel_densities:
        channel_ir = document.ion_channels.get(density.ion_channel)
        if channel_ir is None:
            report.unsupported(
                f"channelDensity '{density.id}'",
                f"references unknown ionChannel '{density.ion_channel}'; "
                "the channel is skipped",
            )
            continue
        channel = make_channel(
            channel_ir,
            density,
            temperature=temperature,
            erev_override=erev_overrides.get(density.id),
        )
        for view in _views_for_group(cell, morphology, density.segment_group):
            view.insert(channel)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def build_network(
    document: ir.Document,
    network_id: Optional[str] = None,
    report: Optional[ConversionReport] = None,
    ncomp: int = 1,
    max_comp_length: Optional[float] = None,
    temperature: Optional[float] = None,
    erev_overrides: Optional[dict[str, float]] = None,
) -> ConvertedModel:
    """Build a Jaxley network from a NeuroML ``network``.

    Args:
        document: The document holding the network and its components.
        network_id: Which network to build; defaults to the only one present.
        report: Conversion report.
        ncomp: Default compartments per branch.
        max_comp_length: Cap on compartment length in um.
        temperature: Temperature in K; defaults to the network's own
            ``temperature`` attribute, or 6.3 degC if it has none.
        erev_overrides: Reversal potentials keyed by ``channelDensity`` id.

    Returns:
        A :class:`ConvertedModel` wrapping a :class:`jaxley.Network`.
    """
    report = report or ConversionReport(source=document.source)
    if network_id is None:
        if len(document.networks) != 1:
            raise ParseError(
                "The document defines "
                f"{len(document.networks)} networks ({sorted(document.networks)}); "
                "pass network_id to choose one."
            )
        network_id = next(iter(document.networks))
    network_ir = document.networks[network_id]

    if temperature is None:
        temperature = (
            network_ir.temperature
            if network_ir.temperature is not None
            else DEFAULT_TEMPERATURE
        )

    # One Jaxley cell per instance, in population order.
    cells: list[jx.Cell] = []
    cell_types: list[str] = []
    cell_index: dict[tuple[str, int], int] = {}
    morphologies: dict[str, CellMorphology] = {}
    prototypes: dict[str, jx.Cell] = {}

    for population in network_ir.populations:
        component = population.component
        if component in document.point_cells:
            report.unsupported(
                f"population '{population.id}'",
                f"component '{component}' is an abstract point cell "
                f"({document.point_cells[component].kind}); the population is skipped",
            )
            continue
        if component in document.input_sources:
            report.unsupported(
                f"population '{population.id}'",
                f"component '{component}' is a spike source "
                f"({document.input_sources[component].kind}); populations of spike "
                "sources are not translated, so anything they drive will be silent",
            )
            continue
        if component not in prototypes:
            prototype, morphology = build_cell(
                document,
                component,
                report=report,
                ncomp=ncomp,
                max_comp_length=max_comp_length,
                temperature=temperature,
                erev_overrides=erev_overrides,
            )
            prototypes[component] = prototype
            morphologies[component] = morphology
        indices = (
            [instance.id for instance in population.instances]
            if population.instances
            else list(range(population.size))
        )
        for instance_id in indices:
            cell_index[(population.id, instance_id)] = len(cells)
            cells.append(prototypes[component])
            cell_types.append(component)

    if not cells:
        described = ", ".join(
            f"'{population.id}' (component '{population.component}'"
            + (
                f", {document.point_cells[population.component].kind})"
                if population.component in document.point_cells
                else ")"
            )
            for population in network_ir.populations
        )
        raise ParseError(
            f"Network '{network_id}' has no populations tarjuman can convert: "
            f"{described or 'the network has no populations'}. tarjuman converts "
            "multicompartmental NeuroML cells; abstract point cells and spike "
            "sources are not supported yet."
        )

    net = jx.Network(cells)
    _place_cells(net, network_ir, cell_index)

    model = ConvertedModel(
        module=net,
        report=report,
        document=document,
        morphologies=morphologies,
        network=network_ir,
        cell_index=cell_index,
        cell_types=cell_types,
        temperature=temperature,
    )

    _connect_projections(model, network_ir, document, report)
    _collect_inputs(model, network_ir, document, report)

    report.n_cells = len(cells)
    report.n_branches = int(net.total_nbranches)
    report.n_compartments = int(net.nodes.shape[0])
    report.n_synapses = int(net.edges.shape[0]) if hasattr(net, "edges") else 0
    report.cell_index_map = cell_index
    return model


def _place_cells(net, network_ir: ir.Network, cell_index) -> None:
    """Move each cell to the location its ``<instance>`` gives, if any."""
    for population in network_ir.populations:
        for instance in population.instances:
            key = (population.id, instance.id)
            if key not in cell_index:
                continue
            if (instance.x, instance.y, instance.z) == (0.0, 0.0, 0.0):
                continue
            try:
                net.cell(cell_index[key]).move(instance.x, instance.y, instance.z)
            except Exception:  # pragma: no cover - visualisation only
                pass


def _spike_threshold(document: ir.Document, cell_id: Optional[str]) -> float:
    if cell_id is None or cell_id not in document.cells:
        return DEFAULT_SPIKE_THRESHOLD
    biophysics = document.cells[cell_id].biophysical_properties
    if biophysics is None or not biophysics.spike_thresh:
        return DEFAULT_SPIKE_THRESHOLD
    return biophysics.spike_thresh[0][0]


def _synapse_for(
    document: ir.Document,
    component_id: Optional[str],
    threshold: float,
    cache: dict[str, object],
    report: ConversionReport,
    context: str,
):
    """Return (and cache) the Jaxley synapse for a NeuroML synapse component."""
    if component_id is None:
        report.unsupported(context, "the connection names no synapse component")
        return None
    if component_id in cache:
        return cache[component_id]
    synapse_ir = document.synapses.get(component_id)
    if synapse_ir is None:
        report.unsupported(
            context, f"references unknown synapse '{component_id}'; skipped"
        )
        return None
    try:
        synapse = make_synapse(synapse_ir, name=component_id, threshold=threshold)
    except Exception as error:
        report.unsupported(context, str(error))
        return None
    cache[component_id] = synapse
    return synapse


def _connect_projections(
    model: ConvertedModel,
    network_ir: ir.Network,
    document: ir.Document,
    report: ConversionReport,
) -> None:
    """Translate every projection into Jaxley synapses.

    Chemical ``projection``s become one Jaxley synapse per connection.
    ``electricalProjection``s are symmetric in NeuroML but Jaxley synapses are
    one-directional, so each gap junction becomes a pair.  In a
    ``continuousProjection`` each side has its own component, and a
    ``silentSynapse`` side contributes no current and is skipped.
    """
    net = model.module
    cache: dict[str, object] = {}

    for projection in network_ir.projections:
        pre_population = next(
            (p for p in network_ir.populations if p.id == projection.pre_population),
            None,
        )
        threshold = _spike_threshold(
            document, pre_population.component if pre_population else None
        )
        context = f"{projection.kind} '{projection.id}'"

        if any(connection.delay > 0 for connection in projection.connections):
            report.approximation(
                context, "synaptic delays are dropped; Jaxley has no delay line"
            )

        for connection in projection.connections:
            try:
                pre = model.view(
                    projection.pre_population,
                    connection.pre_cell,
                    connection.pre_segment,
                    connection.pre_fraction_along,
                )
                post = model.view(
                    projection.post_population,
                    connection.post_cell,
                    connection.post_segment,
                    connection.post_fraction_along,
                )
            except Exception as error:
                report.unsupported(context, str(error))
                continue

            if projection.kind == "projection":
                pairs = [
                    (
                        connection.synapse or projection.synapse,
                        pre,
                        post,
                        connection.weight,
                    )
                ]
            elif projection.kind == "electricalProjection":
                component = connection.synapse or projection.synapse
                # A NeuroML gap junction passes current both ways.
                pairs = [
                    (component, pre, post, connection.weight),
                    (component, post, pre, connection.weight),
                ]
            else:  # continuousProjection
                pairs = []
                if _is_active(document, connection.post_component):
                    pairs.append(
                        (connection.post_component, pre, post, connection.weight)
                    )
                if _is_active(document, connection.pre_component):
                    pairs.append(
                        (connection.pre_component, post, pre, connection.weight)
                    )

            for component, source, target, weight in pairs:
                synapse = _synapse_for(
                    document, component, threshold, cache, report, context
                )
                if synapse is None:
                    continue
                edge_index = int(net.edges.shape[0])
                connect(source, target, synapse)
                if weight != 1.0:
                    # NeuroML puts the weight on the connection, Jaxley on the edge.
                    net.select(edges=[edge_index]).set(
                        f"{synapse.name}_weight", weight
                    )


def _is_active(document: ir.Document, component_id: Optional[str]) -> bool:
    """False for endpoints that carry no current (``silentSynapse``)."""
    if component_id is None:
        return False
    synapse = document.synapses.get(component_id)
    return synapse is None or synapse.kind != "silentSynapse"


def _collect_inputs(
    model: ConvertedModel,
    network_ir: ir.Network,
    document: ir.Document,
    report: ConversionReport,
) -> None:
    for input_list in network_ir.input_lists:
        source = document.input_sources.get(input_list.component)
        if source is None:
            report.unsupported(
                f"inputList '{input_list.id}'",
                f"references unknown input '{input_list.component}'; skipped",
            )
            continue
        for placement in input_list.inputs:
            try:
                cell_index, branch, comp = model.locate(
                    input_list.population,
                    placement.target_cell,
                    placement.segment,
                    placement.fraction_along,
                )
            except Exception as error:
                report.unsupported(f"inputList '{input_list.id}'", str(error))
                continue
            model.stimuli.append(
                Stimulus(
                    source=source,
                    cell_index=cell_index,
                    branch=branch,
                    comp=comp,
                    weight=placement.weight,
                    population=input_list.population,
                    population_cell=placement.target_cell,
                )
            )

    for explicit in network_ir.explicit_inputs:
        source = document.input_sources.get(explicit.input)
        if source is None:
            report.unsupported(
                f"explicitInput '{explicit.input}'",
                "references an unknown input component; skipped",
            )
            continue
        try:
            cell_index, branch, comp = model.locate(
                explicit.target_population, explicit.target_cell, 0, 0.5
            )
        except Exception as error:
            report.unsupported("explicitInput", str(error))
            continue
        model.stimuli.append(
            Stimulus(
                source=source,
                cell_index=cell_index,
                branch=branch,
                comp=comp,
                population=explicit.target_population,
                population_cell=explicit.target_cell,
            )
        )


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def build(
    document: ir.Document,
    network_id: Optional[str] = None,
    cell_id: Optional[str] = None,
    **kwargs,
) -> ConvertedModel:
    """Build whatever the document describes: a network if it has one, else a cell.

    Args:
        document: The IR document.
        network_id: Network to build, if the document has more than one.
        cell_id: Build this cell on its own instead of a network.
        **kwargs: Forwarded to :func:`build_network` or :func:`build_cell`.

    Returns:
        A :class:`ConvertedModel`.
    """
    if cell_id is not None or not document.networks:
        if cell_id is None:
            if len(document.cells) != 1:
                raise ParseError(
                    "The document defines no network and "
                    f"{len(document.cells)} cells; pass cell_id to choose one."
                )
            cell_id = next(iter(document.cells))
        report = kwargs.pop("report", None) or ConversionReport(source=document.source)
        temperature = kwargs.pop("temperature", None) or DEFAULT_TEMPERATURE
        cell, morphology = build_cell(
            document, cell_id, report=report, temperature=temperature, **kwargs
        )
        report.n_cells = 1
        report.n_branches = len(morphology.branches)
        report.n_compartments = morphology.n_compartments
        return ConvertedModel(
            module=cell,
            report=report,
            document=document,
            morphologies={cell_id: morphology},
            cell_types=[cell_id],
            temperature=temperature,
        )
    return build_network(document, network_id=network_id, **kwargs)
