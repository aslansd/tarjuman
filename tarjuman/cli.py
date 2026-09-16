"""``tarjuman`` command-line interface.

Three commands:

``tarjuman info model.nml``
    Read a NeuroML file and say what is in it and what tarjuman would do with
    it, without building anything.

``tarjuman convert model.nml``
    Build the Jaxley model, print the conversion report, and optionally write a
    runnable Python script that rebuilds it.

``tarjuman run LEMS_Sim.xml``
    Convert and simulate, writing the traces the LEMS file asks for.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from . import __version__
from .builder import build
from .errors import TarjumanError
from .reader import read_lems, read_neuroml
from .report import ConversionReport
from .simulate import run_lems, simulate

__all__ = ["main"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ncomp",
        type=int,
        default=1,
        help="default compartments per branch (default: 1, or the NeuroML "
        "numberInternalDivisions where present)",
    )
    parser.add_argument(
        "--max-comp-length",
        type=float,
        default=None,
        metavar="UM",
        help="split branches so that no compartment is longer than this",
    )
    parser.add_argument(
        "--network", default=None, help="id of the network to build"
    )
    parser.add_argument("--cell", default=None, help="id of a single cell to build")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of warning when something cannot be converted",
    )


def _build_from_args(args) -> tuple:
    report = ConversionReport(source=str(args.model), strict=args.strict)
    document = read_neuroml(args.model, report=report)
    model = build(
        document,
        network_id=args.network,
        cell_id=args.cell,
        report=report,
        ncomp=args.ncomp,
        max_comp_length=args.max_comp_length,
    )
    return document, model


def _cmd_info(args) -> int:
    document = read_neuroml(args.model)
    print(f"NeuroML document: {document.id or '<no id>'}  ({document.source})")

    def show(title, items):
        if items:
            print(f"  {title} ({len(items)}): {', '.join(sorted(items))}")

    show("ion channels", document.ion_channels)
    show("cells", document.cells)
    show("point cells", document.point_cells)
    show("synapses", document.synapses)
    show("inputs", document.input_sources)
    show("networks", document.networks)
    for network in document.networks.values():
        print(f"  network '{network.id}':")
        for population in network.populations:
            print(
                f"    population '{population.id}': {population.size} x "
                f"'{population.component}'"
            )
        for projection in network.projections:
            print(
                f"    {projection.kind} '{projection.id}': "
                f"{len(projection.connections)} connections "
                f"({projection.pre_population} -> {projection.post_population})"
            )
    for cell in document.cells.values():
        n_segments = len(cell.morphology.segments)
        n_densities = (
            len(cell.biophysical_properties.channel_densities)
            if cell.biophysical_properties
            else 0
        )
        print(
            f"  cell '{cell.id}': {n_segments} segments, "
            f"{n_densities} channel densities"
        )
    return 0


def _cmd_convert(args) -> int:
    document, model = _build_from_args(args)
    print(model.report.summary())
    if args.script:
        path = Path(args.script)
        path.write_text(_script_for(args, model))
        print(f"wrote {path}")
    return 0


def _cmd_run(args) -> int:
    if args.model.suffix == ".xml":
        result, model = run_lems(
            args.model,
            strict=args.strict,
            delta_t=args.dt,
            ncomp=args.ncomp,
            max_comp_length=args.max_comp_length,
        )
    else:
        _, model = _build_from_args(args)
        if args.duration is None:
            raise TarjumanError(
                "Give --duration (ms) when running a .nml file; a LEMS file "
                "carries its own duration."
            )
        result = simulate(model, t_max=args.duration, delta_t=args.dt or 0.025)
    print(model.report.summary())
    print(
        f"simulated {result.time[-1]:.1f} ms, {len(result.traces)} traces recorded"
    )
    if args.output:
        path = result.save(args.output)
        print(f"wrote {path}")
    if args.plot:
        _plot(result, args.plot)
        print(f"wrote {args.plot}")
    return 0


def _plot(result, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    for key, trace in result.traces.items():
        axis.plot(result.time, trace, label=key, linewidth=1)
    axis.set_xlabel("time (ms)")
    axis.set_ylabel("recorded quantity")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def _script_for(args, model) -> str:
    return f'''"""Rebuild and run this model with tarjuman + Jaxley.

Generated by tarjuman {__version__} from {args.model}.
"""

import jaxley as jx
import matplotlib.pyplot as plt
import tarjuman

model = tarjuman.from_neuroml(
    {str(args.model)!r},
    network_id={args.network!r},
    cell_id={args.cell!r},
    ncomp={args.ncomp},
)
print(model.report.summary())

# `model.module` is an ordinary Jaxley module: make parameters trainable,
# take gradients through it, or run it on a GPU.
result = tarjuman.simulate(model, t_max=300.0, delta_t=0.025)

for name, trace in result.traces.items():
    plt.plot(result.time, trace, label=name)
plt.xlabel("time (ms)")
plt.ylabel("v (mV)")
plt.legend()
plt.show()
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tarjuman",
        description="Run NeuroML 2 models on the Jaxley simulator.",
    )
    parser.add_argument("--version", action="version", version=f"tarjuman {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="describe a NeuroML document")
    info.add_argument("model", type=Path)
    info.set_defaults(func=_cmd_info)

    convert = subparsers.add_parser(
        "convert", help="convert a NeuroML document and report what happened"
    )
    convert.add_argument("model", type=Path)
    convert.add_argument(
        "--script", default=None, metavar="FILE", help="write a runnable Python script"
    )
    _add_common(convert)
    convert.set_defaults(func=_cmd_convert)

    run = subparsers.add_parser("run", help="convert and simulate")
    run.add_argument("model", type=Path, help="a LEMS_*.xml or a .nml file")
    run.add_argument("--dt", type=float, default=None, help="time step in ms")
    run.add_argument(
        "--duration", type=float, default=None, help="duration in ms (.nml files)"
    )
    run.add_argument("--output", default=None, metavar="FILE", help="write traces")
    run.add_argument("--plot", default=None, metavar="FILE.png", help="write a plot")
    _add_common(run)
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TarjumanError as error:
        print(f"tarjuman: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
