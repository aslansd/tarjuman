"""Abstract point cells as single-compartment Jaxley cells.

NeuroML's ``iafCell`` family is defined by absolute quantities — a capacitance
in pF, a leak conductance in nS — and by a *discontinuity*: when the voltage
crosses threshold it is assigned the reset value.  Jaxley has neither absolute
membrane properties (it works in densities) nor any mechanism for assigning a
state, so both need translating:

* **Densities.** A point cell becomes one cylindrical compartment of fixed
  surface area, and the absolute values are divided by that area.  Since every
  term in the membrane equation scales with area, the dynamics are unchanged:
  ``C/g`` and therefore the membrane time constant come out exactly right, and
  injected and synaptic currents stay absolute in both simulators.

* **Reset.** Threshold crossing switches on a very large conductance towards
  the reset potential for one time step (and for the refractory period, where
  the cell type has one).  With Jaxley's implicit step the voltage lands on the
  reset value to within a fraction of a millivolt, rather than being assigned
  it.  This is an approximation, and tarjuman records it as one.

What this does *not* give you is a faithful ``izhikevich2007Cell`` or the
custom integrate-and-fire types some models define: those carry extra state
variables with their own reset rules.  They are reported as unsupported.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import jaxley as jx
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxley.channels import Channel

from . import ir
from .errors import UnsupportedComponentError
from .morphology import BranchSpec, CellMorphology
from .report import ConversionReport

__all__ = [
    "IntegrateAndFire",
    "build_point_cell",
    "point_cell_threshold",
    "SUPPORTED_POINT_CELLS",
    "POINT_CELL_AREA",
]

#: Surface area of the compartment a point cell becomes, in um2.  Arbitrary:
#: every membrane property is divided by it, so it cancels out.
POINT_CELL_AREA = 100.0
#: Radius of that compartment in um; the length follows from the area.
POINT_CELL_RADIUS = 5.0
#: NeuroML point cell types tarjuman can build.
SUPPORTED_POINT_CELLS = {
    "iafCell",
    "iafRefCell",
    "iafTauCell",
    "iafTauRefCell",
}

# uS / um2 -> S/cm2 (Jaxley reports point conductances in uS)
_US_PER_UM2_TO_S_PER_CM2 = 100.0
# pF / um2 -> uF/cm2
_PF_PER_UM2_TO_UF_PER_CM2 = 100.0


class IntegrateAndFire(Channel):
    """Leak, threshold and reset: a NeuroML integrate-and-fire cell.

    NeuroML specifies the reset as a *state assignment* — ``OnCondition
    test="v .gt. thresh"`` with ``StateAssignment variable="v" value="reset"``
    — and Jaxley supports exactly that: a non-capacitive channel may return
    ``"v"`` from ``update_states`` and the integrator takes it. (Jaxley's own
    ``Leak`` + ``Fire`` pair does the same thing; this keeps them as one
    channel so the refractory period, which ``Fire`` has no notion of, can be
    handled in the same place.) The reset is therefore exact rather than
    approximated.

    One consequence is worth knowing: a reset writes the voltage, and a
    threshold crossing is a step function, so the gradient through a spike is
    zero. Sub-threshold parameters stay differentiable; spike timing does not.
    Jaxley's own ``Fire`` warns about the same thing, for the same reason.

    Args:
        name: Channel name.
        g_leak: Leak conductance density in S/cm2.
        e_leak: Leak reversal (and, for ``iafTauCell``, the resting value) in mV.
        threshold: Spike threshold in mV.
        reset: Reset potential in mV.
        refractory: Refractory period in ms (0 for cells without one).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        g_leak: float = 1e-4,
        e_leak: float = -50.0,
        threshold: float = -30.0,
        reset: float = -50.0,
        refractory: float = 0.0,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": float(g_leak),
            f"{prefix}_eLeak": float(e_leak),
            f"{prefix}_thresh": float(threshold),
            f"{prefix}_reset": float(reset),
            f"{prefix}_refract": float(refractory),
        }
        self.channel_states = {
            f"{prefix}_spikes": 0.0,
            f"{prefix}_refractory": 0.0,  # ms remaining
        }
        self.current_name = f"i_{prefix}"
        self.META = {"mechanism": "NeuroML integrate-and-fire cell"}

    def update_states(
        self, states: dict[str, Array], dt: float, v: float, params: dict[str, Array]
    ) -> dict[str, Array]:
        """Assign the reset voltage on a threshold crossing, as NeuroML does."""
        prefix = self._name
        threshold = params[f"{prefix}_thresh"]
        reset = params[f"{prefix}_reset"]

        # NeuroML's condition is strictly greater than the threshold.
        crossed = v > threshold
        remaining = jnp.maximum(states[f"{prefix}_refractory"] - dt, 0.0)
        remaining = jnp.where(crossed, params[f"{prefix}_refract"], remaining)
        # The cell is held at the reset potential for the refractory period,
        # which is what iafRefCell and iafTauRefCell mean by one.
        held = jnp.logical_or(crossed, remaining > 0.0)

        return {
            "v": jnp.where(held, reset, v),
            f"{prefix}_spikes": crossed.astype(jnp.result_type(float)),
            f"{prefix}_refractory": remaining,
        }

    def compute_current(
        self, states: dict[str, Array], v: float, params: dict[str, Array]
    ) -> Array:
        """The leak current; the reset is an assignment, not a current."""
        prefix = self._name
        return params[f"{prefix}_gLeak"] * (v - params[f"{prefix}_eLeak"])

    def init_state(
        self,
        states: dict[str, ArrayLike],
        v: ArrayLike,
        params: dict[str, ArrayLike],
        delta_t: float,
    ) -> dict[str, ArrayLike]:
        prefix = self._name
        return {f"{prefix}_spikes": 0.0, f"{prefix}_refractory": 0.0}


def point_cell_threshold(point_cell: ir.PointCell) -> Optional[float]:
    """The spike threshold of a point cell in mV, for presynaptic detection."""
    return point_cell.params.get("thresh")


def build_point_cell(
    point_cell: ir.PointCell, report: ConversionReport
) -> tuple[jx.Cell, CellMorphology]:
    """Build a single-compartment Jaxley cell from a NeuroML point cell.

    Args:
        point_cell: The IR point cell.
        report: Conversion report; the reset approximation is recorded here.

    Returns:
        The Jaxley cell and a one-segment morphology map, so that the rest of
        tarjuman can address it exactly like a multicompartmental cell.
    """
    if point_cell.kind not in SUPPORTED_POINT_CELLS:
        raise UnsupportedComponentError(
            f"Point cell '{point_cell.id}' has type '{point_cell.kind}'. tarjuman "
            f"builds {', '.join(sorted(SUPPORTED_POINT_CELLS))}; other abstract "
            "cells carry extra state variables with their own reset rules."
        )

    parameters = point_cell.params
    context = f"{point_cell.kind} '{point_cell.id}'"

    capacitance_pf = parameters.get("C")  # pF
    leak_conductance_us = parameters.get("leakConductance")  # uS
    e_leak = parameters.get("leakReversal", parameters.get("reset", -65.0))

    if point_cell.kind in ("iafTauCell", "iafTauRefCell"):
        # Defined by a time constant rather than C and g: pick a capacitance
        # and derive the conductance that gives the right tau.
        tau = parameters.get("tau")
        if tau is None:
            raise UnsupportedComponentError(f"{context} has no 'tau'.")
        # NeuroML's iafTauCell extends baseIaf, not baseIafCapCell: it has no
        # capacitance and no current term, so `dv/dt = (leakReversal - v)/tau`
        # and an injected current does nothing to it at all. Jaxley has no way
        # to express a compartment that ignores its own input current, so the
        # cell is given a capacitance and the conductance that reproduces tau.
        # The free dynamics are then exact; the response to injected current is
        # not, because NeuroML specifies none.
        capacitance_pf = 10.0
        # pF / ms == nS; Jaxley conductances are in uS.
        leak_conductance_us = capacitance_pf / tau * 1e-3
        report.approximation(
            context,
            "NeuroML's iafTauCell family has no current term - injected "
            "current has no effect on it - but a Jaxley compartment cannot "
            "ignore its input current. The free decay to leakReversal with "
            f"time constant {tau:g} ms is exact; any current injected into "
            "this cell will move it when in NeuroML it would not",
        )
        e_leak = parameters.get("leakReversal", parameters.get("reset", -65.0))
    if capacitance_pf is None or leak_conductance_us is None:
        raise UnsupportedComponentError(
            f"{context} is missing 'C' or 'leakConductance'."
        )

    length = POINT_CELL_AREA / (2.0 * np.pi * POINT_CELL_RADIUS)
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.set("radius", POINT_CELL_RADIUS)
    cell.set("length", length)
    cell.set("axial_resistivity", 100.0)
    cell.set(
        "capacitance", capacitance_pf / POINT_CELL_AREA * _PF_PER_UM2_TO_UF_PER_CM2
    )

    threshold = parameters.get("thresh", 0.0)
    reset = parameters.get("reset", e_leak)
    channel = IntegrateAndFire(
        name=point_cell.id,
        g_leak=leak_conductance_us / POINT_CELL_AREA * _US_PER_UM2_TO_S_PER_CM2,
        e_leak=e_leak,
        threshold=threshold,
        reset=reset,
        refractory=parameters.get("refract", 0.0),
    )
    cell.insert(channel)
    cell.set("v", e_leak)

    report.info(
        context,
        f"integrate-and-fire cell: threshold {threshold:g} mV, reset {reset:g} mV"
        + (f", refractory {parameters['refract']:g} ms" if parameters.get("refract") else "")
        + ". The reset is a state assignment, as NeuroML specifies; the "
        "gradient through a spike is therefore zero, though sub-threshold "
        "parameters stay differentiable",
    )

    branch = BranchSpec(
        index=0,
        segments=[0],
        lengths=[length],
        radii=[POINT_CELL_RADIUS],
        parent=-1,
        ncomp=1,
        xyzr=np.array([[0.0, 0.0, 0.0, POINT_CELL_RADIUS]] * 2),
    )
    morphology = CellMorphology(
        cell_id=point_cell.id,
        branches=[branch],
        segment_to_branch={0: 0},
        segment_offset={0: 0.0},
        groups={"all": [0], "soma_group": [0]},
    )
    return cell, morphology
