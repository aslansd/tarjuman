# SPDX-License-Identifier: Apache-2.0
"""Compare NeuroML conversions with handwritten Jaxley models.

Install: python -m pip install -r tests/neuroml/requirements.txt -e .
Run:     python tests/neuroml/run.py
One:     python tests/neuroml/run.py --model ex5

Official models are downloaded on first use. Their sources and licenses are
listed in models.json. Each *_jaxley.py contains its own model equations.
"""

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import jax

jax.config.update("jax_enable_x64", True)

import jaxley as jx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
MODELS = ("ex5", "ex10", "ex15", "ex20a", "ex24", "ex25", "c302_c_oscillator")


def download_models(destination):
    manifest = json.loads((HERE / "models.json").read_text())
    for name, project in manifest.items():
        base = f"https://raw.githubusercontent.com/{project['repository']}/{project['commit']}/"
        for relative in project["files"]:
            path = destination / name / relative
            if not path.exists():
                with urlopen(base + relative, timeout=30) as response:
                    data = response.read()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)


def simulate(net, dt, duration):
    net.record("v", verbose=False)
    net.init_states(delta_t=dt)
    voltage = np.asarray(
        jx.integrate(
            net,
            delta_t=dt,
            t_max=duration,
            solver="bwd_euler",
            voltage_solver="jaxley.dhs",
        )
    )
    return np.arange(voltage.shape[1]) * dt, voltage


def compare(name, models, output):
    reference = importlib.import_module(name + "_jaxley")
    net, thresholds = reference.build_model()
    expected_time, expected = simulate(net, reference.DT, reference.DURATION)
    converted = importlib.import_module(name + "_neuroml")
    model = converted.build_model(models, reference.DT, reference.DURATION)
    _, actual = simulate(model, reference.DT, reference.DURATION)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-9, equal_nan=False)
    for index, cell in enumerate(net.nodes.global_cell_index):
        threshold = thresholds[int(cell)]
        a, b = actual[index], expected[index]
        actual_spikes = np.flatnonzero((a[:-1] <= threshold) & (a[1:] > threshold))
        expected_spikes = np.flatnonzero((b[:-1] <= threshold) & (b[1:] > threshold))
        np.testing.assert_array_equal(actual_spikes, expected_spikes)
    error = float(np.max(np.abs(actual - expected)))
    np.savez_compressed(
        output / (name + ".npz"), time=expected_time, neuroml=actual, jaxley=expected
    )
    print(
        f"{name}: PASS; max voltage error {error:.4g} mV; spike deviation 0 ms",
        flush=True,
    )
    return {"model": name, "max_voltage_error_mV": error, "spike_deviation_ms": 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, help="Run one model; default: all.")
    parser.add_argument(
        "--models", type=Path, help="Use existing models without downloading."
    )
    parser.add_argument(
        "--output", type=Path, help="Directory for voltage arrays and summary."
    )
    args = parser.parse_args()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tarjuman"
    models = (args.models or cache / "validation/models").resolve()
    if args.models is None:
        download_models(models)
    cache.mkdir(parents=True, exist_ok=True)
    output = args.output or Path(tempfile.mkdtemp(prefix="results-", dir=cache))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    for name in [args.model] if args.model else MODELS:
        print(f"Running {name} ...", flush=True)
        results.append(compare(name, models, output))
        jax.clear_caches()
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"{len(results)}/{len(results)} passed in {time.monotonic() - started:.1f}s; {output}"
    )


if __name__ == "__main__":
    main()
