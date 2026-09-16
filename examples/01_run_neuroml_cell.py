"""Convert a NeuroML cell and run it on Jaxley.

    python examples/01_run_neuroml_cell.py

Uses the test fixture copied from the NeuroML2 repository, so it works without
downloading anything.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import tarjuman

MODEL = Path(__file__).parent.parent / "tests" / "data" / "hh_single_comp.nml"


def main() -> None:
    model = tarjuman.from_neuroml(MODEL, network_id="net1")
    print(model.report.summary())

    # `model.module` is an ordinary Jaxley network.
    print(model.module)
    print(model.module.nodes[["length", "radius", "capacitance"]].head())

    result = tarjuman.simulate(
        model,
        t_max=300.0,
        delta_t=0.01,
        records=[
            "hhpop[0]/v",
            "hhpop[0]/bioPhys1/membraneProperties/naChans/naChan/m/q",
            "hhpop[0]/bioPhys1/membraneProperties/kChans/kChan/n/q",
        ],
    )

    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    top.plot(result.time, result["hhpop[0]/v"], color="black", linewidth=1)
    top.set_ylabel("v (mV)")
    top.set_title("NeuroML HH cell, simulated by Jaxley via tarjuman")
    for quantity, label in [
        ("hhpop[0]/bioPhys1/membraneProperties/naChans/naChan/m/q", "m"),
        ("hhpop[0]/bioPhys1/membraneProperties/kChans/kChan/n/q", "n"),
    ]:
        bottom.plot(result.time, result[quantity], label=label, linewidth=1)
    bottom.set_xlabel("time (ms)")
    bottom.set_ylabel("gate")
    bottom.legend()
    figure.tight_layout()
    figure.savefig("hh_cell.png", dpi=150)
    print("wrote hh_cell.png")


if __name__ == "__main__":
    main()
