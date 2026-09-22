"""Run a C. elegans connectome model (c302) on Jaxley.

    git clone https://github.com/openworm/c302
    python examples/03_c302_connectome.py path/to/c302/examples/LEMS_c302_C_Oscillator.xml

Parameter set C follows Boyle & Cohen (2008): each cell carries a calcium pool,
and the calcium channel's third gate is a ``customHGate`` — a LEMS
ComponentType defined inside the model file — whose inactivation depends on the
calcium concentration.  All of that is interpreted at load time; nothing about
c302 is special-cased in tarjuman.

Without an argument this falls back on the bundled fixture, a single cell
modelled on the same parameter set.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tarjuman

FIXTURE = Path(__file__).parent.parent / "tests" / "data" / "calcium_custom_gate.nml"


def run_lems_file(path: str):
    result, model = tarjuman.run_lems(path)
    print(model.report.summary())
    return result, model


def run_fixture():
    model = tarjuman.from_neuroml(FIXTURE)
    print(model.report.summary())
    result = tarjuman.simulate(
        model,
        t_max=400.0,
        delta_t=0.05,
        records=["pop[0]/v", "pop[0]/caConc"],
    )
    return result, model


def main() -> None:
    if len(sys.argv) > 1:
        result, model = run_lems_file(sys.argv[1])
    else:
        print(f"No LEMS file given; running the bundled fixture {FIXTURE.name}.\n")
        result, model = run_fixture()

    voltages = {
        name: trace
        for name, trace in result.traces.items()
        if result.quantities.get(name, "").endswith("/v")
    }
    calcium = {
        name: trace
        for name, trace in result.traces.items()
        if result.quantities.get(name, "").endswith("caConc")
    }

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for name, trace in list(voltages.items())[:8]:
        axes[0].plot(result.time, trace, linewidth=1, label=name)
    axes[0].set_ylabel("v (mV)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title("c302-style model, simulated by Jaxley via tarjuman")

    for name, trace in list(calcium.items())[:8]:
        axes[1].plot(result.time, trace, linewidth=1, label=name)
    axes[1].set_ylabel("[Ca] (mM)")
    axes[1].set_xlabel("time (ms)")

    figure.tight_layout()
    figure.savefig("c302.png", dpi=150)
    print(f"\nwrote c302.png  ({len(result.traces)} traces)")

    # The converted network is an ordinary Jaxley module, so the calcium
    # decay constant of every cell is a parameter you can fit.
    pumps = [pump.name for pump in model.module.base.pumps]
    if pumps:
        print(f"calcium pools: {pumps}")
        print(
            "e.g. model.module.make_trainable("
            f"'{pumps[0]}_decayConstant') to fit the pool to data"
        )


if __name__ == "__main__":
    main()
