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
from .cells import build_point_cell, point_cell_threshold
from .channels import DEFAULT_TEMPERATURE, NeuroMLChannel, make_channel
from .errors import ParseError
from .morphology import CellMorphology, build_morphology
from .pumps import ion_current_name, make_concentration_model
from .report import ConversionReport
from .synapses import DEFAULT_SPIKE_THRESHOLD, DelayedSynapse, make_synapse

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
    #: Integration step in ms, when it is known at build time.  Synaptic delays
    #: are quantised to whole steps, so they can only be built into the model
    #: once the step is fixed; a LEMS file supplies it, a bare ``.nml`` file
    #: does not unless the caller passes ``delta_t``.
    delta_t: Optional[float] = None

    def delay_steps(self, delay: float) -> int:
        """Whole integration steps corresponding to a delay in ms."""
        if not delay or self.delta_t is None or self.delta_t <= 0:
            return 0
        return int(round(delay / self.delta_t))

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
    pools = _insert_concentration_models(
        cell, morphology, biophysics, document, report, cell_id
    )
    _insert_channels(
        cell,
        morphology,
        biophysics,
        document,
        report,
        temperature,
        erev_overrides,
        cell_id,
        pools,
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


def _insert_concentration_models(
    cell,
    morphology: CellMorphology,
    biophysics: ir.BiophysicalProperties,
    document: ir.Document,
    report: ConversionReport,
    cell_id: str,
) -> dict[str, ir.Species]:
    """Insert a Jaxley pump for every ``<species>`` on the cell.

    Returns the species by ion, so that channels of that ion can be given a
    shared current name for the pool to read.
    """
    pools: dict[str, ir.Species] = {}
    for species in biophysics.species:
        model = document.concentration_models.get(species.concentration_model)
        if model is None:
            report.unsupported(
                f"cell '{cell_id}' species '{species.id}'",
                f"references unknown concentration model "
                f"'{species.concentration_model}'; the pool is skipped and any "
                "calcium-dependent gate will see a constant concentration",
            )
            continue
        try:
            pump = make_concentration_model(model, document.component_types, species)
        except Exception as error:
            report.unsupported(f"cell '{cell_id}' species '{species.id}'", str(error))
            continue
        for view in _views_for_group(cell, morphology, species.segment_group):
            view.insert(pump)
        pools[species.ion] = species
        report.info(
            f"cell '{cell_id}'",
            f"concentration model '{model.id}' ({model.kind}) tracks {species.ion}",
        )
    return pools


def _insert_channels(
    cell,
    morphology: CellMorphology,
    biophysics: ir.BiophysicalProperties,
    document: ir.Document,
    report: ConversionReport,
    temperature: float,
    erev_overrides: dict[str, float],
    cell_id: str,
    pools: Optional[dict[str, ir.Species]] = None,
) -> None:
    pools = pools or {}
    for density in biophysics.channel_densities:
        channel_ir = document.ion_channels.get(density.ion_channel)
        if channel_ir is None:
            report.unsupported(
                f"channelDensity '{density.id}'",
                f"references unknown ionChannel '{density.ion_channel}'; "
                "the channel is skipped",
            )
            continue
        ion = (density.ion or channel_ir.species or "").lower()
        species = pools.get(ion)
        try:
            channel = make_channel(
                channel_ir,
                density,
                temperature=temperature,
                erev_override=erev_overrides.get(density.id),
                component_types=document.component_types,
                # Channels of a pooled ion share a current name so the pool
                # sees the total current of that ion, as NeuroML's `iCa` does.
                current_name=ion_current_name(ion) if species is not None else None,
                initial_calcium=(
                    pools["ca"].initial_concentration if "ca" in pools else 5e-5
                ),
            )
        except Exception as error:
            report.unsupported(f"channelDensity '{density.id}'", str(error))
            continue
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
    delta_t: Optional[float] = None,
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
        delta_t: Integration step in ms. Needed to build synaptic delays, which
            are quantised to whole steps; without it they are dropped.

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
        if component in document.point_cells and component not in prototypes:
            try:
                prototype, morphology = build_point_cell(
                    document.point_cells[component], report
                )
            except Exception as error:
                report.unsupported(f"population '{population.id}'", str(error))
                continue
            prototypes[component] = prototype
            morphologies[component] = morphology
        if component in document.input_sources:
            report.unsupported(
                f"population '{population.id}'",
                f"component '{component}' is a spike source "
                f"({document.input_sources[component].kind}); populations of spike "
                "sources are not translated, so anything they drive will be silent",
            )
            continue
        if component not in prototypes and component in document.cells:
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
        if component not in prototypes:
            continue
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
    _reregister_pumped_ions(net)
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
        delta_t=delta_t,
    )

    _connect_projections(model, network_ir, document, report)
    _collect_inputs(model, network_ir, document, report)

    report.n_cells = len(cells)
    report.n_branches = int(net.total_nbranches)
    report.n_compartments = int(net.nodes.shape[0])
    report.n_synapses = int(net.edges.shape[0]) if hasattr(net, "edges") else 0
    report.cell_index_map = cell_index
    return model


def _reregister_pumped_ions(net) -> None:
    """Re-register pumped ions on a network built from cells.

    Jaxley collects the pumps of the cells a ``Network`` is built from, but not
    the list of ion concentrations those pumps modify, so the integrator does
    not know it has to solve for them (jaxleyverse/jaxley#811, confirmed as a
    bug and being fixed). This restores the list, and becomes a no-op once the
    upstream fix is in.
    """
    for pump in net.base.pumps:
        ion_name = getattr(pump, "ion_name", None)
        if ion_name is not None and ion_name not in net.base.pumped_ions:
            net.base.pumped_ions.append(ion_name)


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
    if cell_id in document.point_cells:
        # An integrate-and-fire cell never overshoots: the only voltage that
        # marks a spike is its own threshold.
        threshold = point_cell_threshold(document.point_cells[cell_id])
        if threshold is not None:
            return threshold
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
    cache: dict,
    report: ConversionReport,
    context: str,
    delay_steps: int = 0,
    initial_voltage: float = -70.0,
):
    """Return (and cache) the Jaxley synapse for a NeuroML synapse component.

    A connection with a delay gets its own synapse type, because the length of
    the shift register is part of the type rather than a parameter.
    """
    if component_id is None:
        report.unsupported(context, "the connection names no synapse component")
        return None
    key = (component_id, delay_steps)
    if key in cache:
        return cache[key]
    name = component_id if delay_steps == 0 else f"{component_id}_delay{delay_steps}"
    synapse_ir = document.synapses.get(component_id)
    if synapse_ir is None:
        report.unsupported(
            context, f"references unknown synapse '{component_id}'; skipped"
        )
        return None
    try:
        synapse = make_synapse(
            synapse_ir,
            name=name,
            threshold=threshold,
            component_types=document.component_types,
        )
        if delay_steps > 0:
            synapse = DelayedSynapse(synapse, delay_steps, initial_voltage)
    except Exception as error:
        report.unsupported(context, str(error))
        return None
    cache[key] = synapse
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

    Connections are grouped by synapse component and made in one ``connect``
    call each: ``connect`` pairs two equal-length views element-wise, so a
    whole projection costs one call rather than one per connection.  On c302's
    full connectome that is the difference between minutes and seconds.
    """
    net = model.module
    cache: dict = {}
    # (component id, delay steps) -> (pre comps, post comps, weights)
    grouped: dict[tuple, tuple[list[int], list[int], list[float]]] = {}
    index_of = _compartment_index_map(net)
    resting = _resting_potential(net)

    def add(
        component: Optional[str],
        source,
        target,
        weight: float,
        context: str,
        delay_steps: int = 0,
    ):
        synapse = _synapse_for(
            document,
            component,
            threshold,
            cache,
            report,
            context,
            delay_steps,
            resting,
        )
        if synapse is None:
            return
        pre_comps, post_comps, weights = grouped.setdefault(
            (component, delay_steps), ([], [], [])
        )
        pre_comps.append(index_of[source])
        post_comps.append(index_of[target])
        weights.append(weight)

    for projection in network_ir.projections:
        pre_population = next(
            (p for p in network_ir.populations if p.id == projection.pre_population),
            None,
        )
        threshold = _spike_threshold(
            document, pre_population.component if pre_population else None
        )
        context = f"{projection.kind} '{projection.id}'"

        delays = [connection.delay for connection in projection.connections]
        if any(delay > 0 for delay in delays):
            _report_delays(model, report, context, delays)

        for connection in projection.connections:
            try:
                source = model.locate(
                    projection.pre_population,
                    connection.pre_cell,
                    connection.pre_segment,
                    connection.pre_fraction_along,
                )
                target = model.locate(
                    projection.post_population,
                    connection.post_cell,
                    connection.post_segment,
                    connection.post_fraction_along,
                )
            except Exception as error:
                report.unsupported(context, str(error))
                continue

            delay_steps = model.delay_steps(connection.delay)
            if projection.kind == "projection":
                add(
                    connection.synapse or projection.synapse,
                    source,
                    target,
                    connection.weight,
                    context,
                    delay_steps,
                )
            elif projection.kind == "electricalProjection":
                # A NeuroML gap junction passes current both ways.
                component = connection.synapse or projection.synapse
                add(component, source, target, connection.weight, context, delay_steps)
                add(component, target, source, connection.weight, context, delay_steps)
            else:  # continuousProjection
                if _is_active(document, connection.post_component):
                    add(
                        connection.post_component,
                        source,
                        target,
                        connection.weight,
                        context,
                        delay_steps,
                    )
                if _is_active(document, connection.pre_component):
                    add(
                        connection.pre_component,
                        target,
                        source,
                        connection.weight,
                        context,
                        delay_steps,
                    )

    for key, (pre_comps, post_comps, weights) in grouped.items():
        synapse = cache[key]
        first_edge = int(net.edges.shape[0])
        connect(
            net.select(nodes=pre_comps), net.select(nodes=post_comps), synapse
        )
        # NeuroML puts the weight on the connection, Jaxley on the edge.  Edges
        # that share a weight are set together, again to keep this off the
        # per-connection path.
        by_weight: dict[float, list[int]] = {}
        for offset, weight in enumerate(weights):
            if weight != 1.0:
                by_weight.setdefault(weight, []).append(first_edge + offset)
        for weight, edges in by_weight.items():
            net.select(edges=edges).set(f"{synapse.name}_weight", weight)


def _report_delays(
    model: ConvertedModel,
    report: ConversionReport,
    context: str,
    delays: list[float],
) -> None:
    """Say what happened to a projection's synaptic delays."""
    delayed = [delay for delay in delays if delay > 0]
    if model.delta_t is None:
        report.approximation(
            context,
            f"{len(delayed)} connections carry a synaptic delay, which is dropped: "
            "delays are quantised to whole integration steps, so the step has to "
            "be known when the model is built. Pass delta_t to from_neuroml(), or "
            "use run_lems(), which takes it from the LEMS file",
        )
        return
    steps = {model.delay_steps(delay) for delay in delayed}
    if steps == {0}:
        report.approximation(
            context,
            f"synaptic delays are shorter than one integration step "
            f"({model.delta_t:g} ms) and are dropped",
        )
        return
    report.info(
        context,
        f"{len(delayed)} synaptic delays implemented as shift registers of "
        f"{sorted(steps)} steps at delta_t = {model.delta_t:g} ms",
    )


def _resting_potential(net) -> float:
    """A sensible value to prefill a delay register with."""
    try:
        return float(net.nodes["v"].median())
    except Exception:  # pragma: no cover - defensive
        return -70.0


def _compartment_index_map(net) -> dict[tuple[int, int, int], int]:
    """``(cell, branch, comp)`` -> global compartment index, built once."""
    nodes = net.nodes
    cells = (
        nodes["global_cell_index"]
        if "global_cell_index" in nodes.columns
        else nodes["local_cell_index"]
    )
    return {
        (int(cell), int(branch), int(comp)): int(index)
        for index, cell, branch, comp in zip(
            nodes.index,
            cells,
            nodes["local_branch_index"],
            nodes["local_comp_index"],
        )
    }


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
    report = kwargs.pop("report", None) or ConversionReport(source=document.source)
    if cell_id is not None or not document.networks:
        if cell_id is None:
            if len(document.cells) != 1:
                raise ParseError(
                    "The document defines no network and "
                    f"{len(document.cells)} cells"
                    + ("; pass cell_id to choose one." if document.cells else ".")
                    + report.missing_includes_hint()
                )
            cell_id = next(iter(document.cells))
        temperature = kwargs.pop("temperature", None) or DEFAULT_TEMPERATURE
        delta_t = kwargs.pop("delta_t", None)
        cell, morphology = build_cell(
            document, cell_id, report=report, temperature=temperature, **kwargs
        )
        kwargs["delta_t"] = delta_t
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
            delta_t=kwargs.get("delta_t"),
        )
    return build_network(document, network_id=network_id, report=report, **kwargs)
