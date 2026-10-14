"""Run converted models, reproducing what the LEMS file asked for.

``tarjuman.simulate`` is deliberately thin: it sets up recordings and stimuli on
the Jaxley module and calls :func:`jaxley.integrate`.  Anything Jaxley can do
to a module — making parameters trainable, taking gradients, ``vmap``-ing over
parameter sets — still applies, because what tarjuman hands back is an ordinary
Jaxley module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import jax.numpy as jnp
import jaxley as jx
import numpy as np

from . import ir
from .builder import ConvertedModel, build
from .channels import NeuroMLChannel
from .errors import ParseError
from .reader import parse_quantity_path, read_lems, read_neuroml
from .report import ConversionReport

__all__ = ["SimulationResult", "simulate", "run_lems", "from_neuroml"]

#: LEMS cell exposures for ion concentrations, and the Jaxley state they map to.
_ION_CONCENTRATION_STATES = {
    "caConc": "CaCon_i",
    "caConcExt": "CaCon_e",
    "naConc": "NaCon_i",
    "kConc": "KCon_i",
}


@dataclass
class SimulationResult:
    """Recorded traces, addressed the way the LEMS file addressed them."""

    time: np.ndarray  # ms
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    quantities: dict[str, str] = field(default_factory=dict)
    model: Optional[ConvertedModel] = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.traces[key]

    def to_dataframe(self):
        """Return the traces as a pandas DataFrame with a ``t`` column in ms."""
        import pandas as pd

        return pd.DataFrame({"t": self.time, **self.traces})

    def save(self, path: str | os.PathLike, in_si_units: bool = True) -> Path:
        """Write the traces in the column format ``jnml`` writes.

        Args:
            path: Destination file.
            in_si_units: Write seconds and volts, as LEMS ``OutputFile`` does,
                rather than the ms and mV Jaxley works in.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        time = self.time / 1e3 if in_si_units else self.time
        columns = [time]
        for key, trace in self.traces.items():
            state = self.quantities.get(key, "").rsplit("/", 1)[-1]
            scale = (
                {"v": 1e-3, "iDensity": -10.0, "gDensity": 1e4}.get(state, 1.0)
                if in_si_units else 1.0
            )
            columns.append(trace * scale)
        np.savetxt(path, np.column_stack(columns), fmt="%g", delimiter="\t")
        return path


def from_neuroml(
    path: str | os.PathLike,
    network_id: Optional[str] = None,
    cell_id: Optional[str] = None,
    strict: bool = False,
    **kwargs,
) -> ConvertedModel:
    """Read a NeuroML file and build the corresponding Jaxley module.

    Args:
        path: Path to a ``.nml`` document.
        network_id: Which network to build, if the file defines several.
        cell_id: Build this cell alone instead of a network.
        strict: Raise instead of warning when something cannot be converted.
        **kwargs: Passed to the builder (``ncomp``, ``max_comp_length``,
            ``temperature``, ``erev_overrides``, ``delta_t``). Pass ``delta_t``
            to build synaptic delays, which are quantised to whole steps.

    Returns:
        A :class:`~tarjuman.builder.ConvertedModel`; ``model.module`` is the
        Jaxley ``Cell`` or ``Network``.
    """
    report = ConversionReport(source=str(path), strict=strict)
    document = read_neuroml(path, report=report)
    return build(
        document, network_id=network_id, cell_id=cell_id, report=report, **kwargs
    )


def simulate(
    model: ConvertedModel,
    t_max: float,
    delta_t: float = 0.025,
    records: Optional[Sequence[str]] = None,
    solver: str = "bwd_euler",
    voltage_solver: str = "jaxley.dhs",
    apply_stimuli: bool = True,
    init_states: bool = True,
) -> SimulationResult:
    """Run a converted model.

    Args:
        model: The converted model.
        t_max: Duration in ms.
        delta_t: Time step in ms.
        records: LEMS quantity paths to record, e.g. ``["hhpop[0]/v"]``.
            Defaults to the voltage of the first compartment of every cell.
        solver: Jaxley ODE solver.
        voltage_solver: Jaxley voltage solver.
        apply_stimuli: Whether to apply the NeuroML inputs (set False if you
            have already stimulated the module yourself).
        init_states: Whether to put every gate at its steady state for the
            initial membrane potential before integrating.  NeuroML gates do
            exactly this in their ``OnStart`` block, so this defaults to True;
            Jaxley on its own would start from the channel's default state.

    Returns:
        A :class:`SimulationResult`.
    """
    module = model.module
    module.delete_recordings()
    if apply_stimuli:
        # Only clear stimuli we are about to replace; a caller who passes
        # apply_stimuli=False has set up their own and expects them to survive.
        module.delete_stimuli()

    if records is None:
        records = _default_records(model)

    recordings = [_add_recording(model, quantity) for quantity in records]

    if apply_stimuli:
        model.apply_stimuli(delta_t=delta_t, t_max=t_max)

    if init_states:
        module.init_states(delta_t=delta_t)

    recorded = jx.integrate(
        module,
        delta_t=delta_t,
        t_max=t_max,
        solver=solver,
        voltage_solver=voltage_solver,
    )
    # Jaxley deduplicates recordings. Several LEMS paths, and the dependencies
    # of derived quantities, can therefore share the same recorded row.
    rows = {
        (row.rec_index, row.state): index
        for index, row in enumerate(module.recordings.itertuples())
    }
    time = np.arange(recorded.shape[1]) * delta_t

    return SimulationResult(
        time=time,
        traces={
            recording.quantity: recording.values(recorded, rows, module.nodes)
            for recording in recordings
        },
        quantities={recording.quantity: recording.quantity for recording in recordings},
        model=model,
    )


def run_lems(
    path: str | os.PathLike,
    strict: bool = False,
    delta_t: Optional[float] = None,
    **kwargs,
) -> tuple[SimulationResult, ConvertedModel]:
    """Run a LEMS simulation file through Jaxley.

    This is the tarjuman equivalent of ``pynml <LEMS file> -jnml``: it reads
    the model, converts it, applies the inputs, records the quantities the
    ``OutputFile`` elements ask for, and integrates for the requested duration.

    Args:
        path: Path to a ``LEMS_*.xml`` file.
        strict: Raise instead of warning on anything that cannot be converted.
        delta_t: Override the LEMS ``step``.
        **kwargs: Passed to the builder (``ncomp``, ``max_comp_length``, ...).

    Returns:
        ``(result, model)``.
    """
    report = ConversionReport(source=str(path), strict=strict)
    document, simulation = read_lems(path, report=report)

    network_id = simulation.target if simulation.target in document.networks else None
    cell_id = simulation.target if simulation.target in document.cells else None
    step = delta_t if delta_t is not None else simulation.step
    model = build(
        document,
        network_id=network_id,
        cell_id=cell_id,
        report=report,
        delta_t=kwargs.pop("delta_t", step),
        **kwargs,
    )

    records = [quantity for _, _, quantity in simulation.outputs] or None
    result = simulate(
        model,
        t_max=simulation.length,
        delta_t=delta_t if delta_t is not None else simulation.step,
        records=records,
    )
    # Present the traces under the ids the LEMS file gave them.  Column ids are
    # only unique within one OutputFile, so a repeated id is qualified with the
    # file it came from rather than silently overwriting the earlier trace.
    if simulation.outputs:
        counts: dict[str, int] = {}
        for _, column_id, _ in simulation.outputs:
            counts[column_id] = counts.get(column_id, 0) + 1

        renamed, quantities = {}, {}
        for (file_id, column_id, quantity), key in zip(
            simulation.outputs, result.traces
        ):
            name = column_id if counts[column_id] == 1 else f"{file_id}.{column_id}"
            renamed[name] = result.traces[key]
            quantities[name] = quantity
        result.traces, result.quantities = renamed, quantities
    return result, model


# --------------------------------------------------------------------------- #
# Recording helpers
# --------------------------------------------------------------------------- #
@dataclass
class _Recording:
    """A LEMS output and the compartment states needed to evaluate it."""

    quantity: str
    index: int
    states: tuple[str, ...]
    channel: Optional[NeuroMLChannel] = None
    exposure: Optional[str] = None

    def values(self, recorded, rows: dict, nodes) -> np.ndarray:
        states = {state: recorded[rows[self.index, state]] for state in self.states}
        if self.channel is None:
            return np.asarray(states[self.states[0]])

        channel = self.channel
        params = {
            name: jnp.asarray(nodes.loc[self.index, name])
            for name in channel.channel_params
        }
        v = states["v"]
        value = channel.compute_open_fraction(states, v, params)
        if self.exposure in ("iDensity", "gDensity"):
            value = params[f"{channel.name}_gbar"] * value
        if self.exposure == "iDensity":
            value = value * (v - params[f"{channel.name}_erev"])
        return np.asarray(jnp.broadcast_to(value, v.shape))


def _default_records(model: ConvertedModel) -> list[str]:
    if model.network is None:
        return ["0/0/v"]
    records = []
    for population in model.network.populations:
        for cell in range(population.size):
            if (population.id, cell) in model.cell_index:
                records.append(f"{population.id}[{cell}]/v")
    return records


def _recording_view(model: ConvertedModel, parsed: dict):
    """Validate each component in an output path against the source model."""
    quantity = parsed["raw"]
    population = parsed["population"]
    if model.network is None:
        cell_id = model.cell_types[0]
        if population not in ("0", cell_id) or parsed["cell"] != 0:
            raise ParseError(f"No cell matching output path {quantity!r}.")
        population = None
    else:
        index = model.cell_index.get((population, parsed["cell"]))
        if index is None:
            raise ParseError(f"No population instance matching output path {quantity!r}.")
        cell_id = model.cell_types[index]

    if parsed["component"] is not None and parsed["component"] != cell_id:
        raise ParseError(f"Output path {quantity!r} does not refer to cell {cell_id!r}.")

    if parsed["density"] is not None:
        cell = model.document.cells.get(cell_id)
        properties = cell.biophysical_properties if cell is not None else None
        if properties is None or parsed["biophysics"] != properties.id:
            raise ParseError(
                f"Unknown biophysical properties in output path {quantity!r}."
            )
        density = next(
            (d for d in properties.channel_densities if d.id == parsed["density"]),
            None,
        )
        if density is None:
            raise ParseError(f"Unknown channel density in output path {quantity!r}.")
        if parsed["channel"] is not None and parsed["channel"] != density.ion_channel:
            raise ParseError(f"Unknown ion channel in output path {quantity!r}.")
        if parsed["gate"] is not None:
            channel = model.document.ion_channels[density.ion_channel]
            if not any(gate.id == parsed["gate"] for gate in channel.gates):
                raise ParseError(f"Unknown gate in output path {quantity!r}.")

    return model.view(
        population,
        parsed["cell"],
        parsed["segment"] if parsed["segment"] is not None else 0,
        0.5,
    )


def _add_recording(model: ConvertedModel, quantity: str) -> _Recording:
    """Resolve one LEMS output and record the states it depends on."""
    parsed = parse_quantity_path(quantity)
    view = _recording_view(model, parsed)
    state = parsed["state"]
    if parsed["density"] is not None:
        name = parsed["density"]
        channel = next(
            (channel for channel in model.module.channels if channel.name == name),
            None,
        )
        if not isinstance(channel, NeuroMLChannel) or not bool(
            view.nodes.iloc[0].get(name, False)
        ):
            raise ParseError(
                f"Cannot record {quantity!r}: channel density {parsed['density']!r} "
                "was not converted on this compartment; see `model.report.summary()`."
            )
        if parsed["gate"] is None:
            supported = (
                ("iDensity", "gDensity") if parsed["channel"] is None else ("fopen",)
            )
            if state not in supported:
                raise ParseError(f"Unsupported channel output path {quantity!r}.")
            # Shared ion currents and cached pre-solve currents cannot supply
            # a channel's output at the actual sampling time. Use its states.
            states = tuple(dict.fromkeys(["v", *channel.channel_states]))
            for dependency in states:
                view.record(dependency, verbose=False)
            return _Recording(quantity, view.nodes.index[0], states, channel, state)
        if state != "q":
            raise ParseError(f"Unsupported gate output {quantity!r}; only q is recorded.")
        # Jaxley names gate states "<channelDensity id>_<gate id>".
        jaxley_state = f"{parsed['density']}_{parsed['gate']}"

    elif state == "v":
        jaxley_state = "v"
    elif state in _ION_CONCENTRATION_STATES:
        # A cell's exposed ion concentration, e.g. "AVBL/0/GenericNeuronCell/caConc".
        jaxley_state = _ION_CONCENTRATION_STATES[state]
    elif "synapses:" in quantity:
        raise ParseError(
            f"tarjuman cannot record the synapse quantity {quantity!r} yet; record "
            "the postsynaptic voltage instead, or read the synapse state from "
            "`model.module.edges` after the run."
        )
    else:
        raise ParseError(
            f"tarjuman does not know how to record the LEMS quantity {quantity!r} yet."
        )

    try:
        view.record(jaxley_state, verbose=False)
    except KeyError as error:
        available = [
            column
            for column in model.module.nodes.columns
            if column == "v" or column.startswith(("i_", f"{parsed['density']}_"))
        ]
        raise ParseError(
            f"Cannot record {quantity!r}: Jaxley has no state '{jaxley_state}'. "
            "This usually means the channel it belongs to was not converted - see "
            "`model.report.summary()`. States available on this compartment: "
            f"{', '.join(sorted(available)) or 'v'}."
        ) from error
    return _Recording(quantity, view.nodes.index[0], (jaxley_state,))
