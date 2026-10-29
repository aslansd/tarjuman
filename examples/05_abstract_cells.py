"""Run NeuroML's abstract cells, and see where one needs explicit integration.

    python examples/05_abstract_cells.py

`izhikevich2007Cell` and `adExIaFCell` are LEMS ComponentTypes carrying a
membrane equation and a reset rule, and tarjuman drives them with the same
interpreter it uses for a model's own custom types. Both read their synaptic
input with `<DerivedVariable select="synapses[*]/i" reduce="add"/>`, which
Jaxley's solver supplies: the cell contributes the current its equations
describe, the solver adds the stimulus and the synapses.

The second half of this script is about the one place that is delicate. An AdEx
cell's `exp((v - VT) / delT)` is built to diverge — the divergence is the
spike — so a few millivolts above `VT` the membrane conductance goes negative
and grows without bound. An implicit step inverts `C/dt + dI/dv`, and once the
feedback passes `C/dt` the step runs backwards. The cell detects that and
integrates those steps itself.
"""

from pathlib import Path

import numpy as np

import tarjuman
from tarjuman.cells_lems import LemsPointCell

DATA = Path(__file__).parent.parent / "tests" / "data"


def resets(time: np.ndarray, voltage: np.ndarray, drop: float = 3.0) -> np.ndarray:
    """Times of one-sample falls bigger than `drop` mV: the reset of a spike."""
    return time[np.where(np.diff(voltage) < -drop)[0]]


def izhikevich() -> None:
    print("izhikevich2007Cell — tonic and bursting")
    result, model = tarjuman.run_lems(DATA / "LEMS_izhikevich2007.xml")
    for name in ("Tonic", "Burst"):
        voltage = np.asarray(result[f"v{name}"])
        recovery = np.asarray(result[f"u{name}"])
        spikes = resets(result.time, voltage, drop=20.0)
        print(
            f"  {name:6s}  {len(spikes):2d} spikes, "
            f"v {voltage.min():7.2f}..{voltage.max():6.2f} mV, "
            f"u {recovery.min():6.3f}..{recovery.max():5.3f} nA"
        )
    cell = next(c for c in model.module.channels if isinstance(c, LemsPointCell))
    print(f"  the interpreter integrates {sorted(cell._state_dimensions)}")
    print("  Jaxley integrates the voltage, so the stimulus reaches the cell\n")


def adex() -> None:
    print("adExIaFCell — NeuroML's adExRebound parameters")
    print("  a 50 ms hyperpolarising pulse, then a rebound burst")
    print("  jnml puts three spikes in that burst at every step from 0.025 ms down")
    for delta_t in (0.025, 0.005, 0.001):
        result, _ = tarjuman.run_lems(DATA / "LEMS_adex_rebound.xml", delta_t=delta_t)
        voltage = np.asarray(result["v"])
        spikes = resets(result.time, voltage)
        times = ", ".join(f"{t:.2f}" for t in spikes)
        print(
            f"  dt={delta_t:<6} {len(spikes)} spikes at {times} ms"
            f"   (v stays within {voltage.min():.1f}..{voltage.max():.1f} mV)"
        )
    print()


def where_the_equation_diverges() -> None:
    """Show the band of voltages the cell has to integrate itself."""
    import jax.numpy as jnp

    from tarjuman import ir
    from tarjuman.cells_lems import RUNAWAY_FEEDBACK, make_lems_point_cell

    rebound = {
        "C": "281pF", "gL": "30nS", "EL": "-60mV", "reset": "-51mV",
        "VT": "-54mV", "thresh": "-30mV", "delT": "2mV", "tauw": "150ms",
        "a": "200nS", "b": "0.1nA", "refract": "0ms",
    }
    cell = make_lems_point_cell(
        ir.PointCell(
            id="adExRebound",
            kind="adExIaFCell",
            custom=ir.CustomComponent(
                id="adExRebound", component_type="adExIaFCell", attributes=rebound
            ),
        ),
        {},
        area_um2=100.0,
    )
    states = {k: jnp.array(v) for k, v in cell.channel_states.items()}
    params = {k: jnp.array(v) for k, v in cell.channel_params.items()}
    params["capacitance"] = jnp.array(2.81)
    delta_t, delta_t_si = 0.025, 2.5e-5

    print("How much positive feedback the membrane equation has, per step")
    print(f"  VT = -54 mV, thresh = -30 mV; the cell takes over above {RUNAWAY_FEEDBACK}")
    print(f"  {'v (mV)':>8}  {'feedback*dt':>12}  {'integrated by':>14}  {'v out':>8}")
    for v in (-58.0, -50.0, -45.0, -43.0, -42.0, -40.0, -35.0):
        voltage = jnp.array(v)
        context = cell._context(voltage, params)
        before = cell._states_si(states, voltage)
        _, runaway = cell._predict(before, context, delta_t_si)
        stepped = cell.update_states(states, delta_t, voltage, params)
        nudged = dict(before)
        nudged["v"] = before["v"] + 1e-6
        feedback = (
            cell.dynamics.derivative("v", nudged, context)
            - cell.dynamics.derivative("v", before, context)
        ) / 1e-6
        who = "the cell" if bool(runaway) else "Jaxley"
        print(
            f"  {v:8.1f}  {float(feedback) * delta_t_si:12.4f}  {who:>14}  "
            f"{float(stepped['v']):8.2f}"
            + ("  <- reset" if float(stepped[f"{cell.name}_spikes"]) else "")
        )
    print(
        "\n  Below the band Jaxley solves implicitly, with the stimulus and\n"
        "  synapses included. Inside it the equation's own feedback would\n"
        "  invert the implicit step, so the cell integrates explicitly until\n"
        "  it reaches the threshold and resets."
    )


def main() -> None:
    izhikevich()
    adex()
    where_the_equation_diverges()


if __name__ == "__main__":
    main()
