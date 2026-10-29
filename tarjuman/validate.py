"""Compare tarjuman's traces against a reference NeuroML simulator.

Agreement with analytic references says the equations were implemented
correctly; agreement with ``jnml`` says the *model* was translated correctly,
which is the thing a NeuroML user actually needs to know.  This module runs
both simulators on the same LEMS file and compares what comes out.

**Spiking models cannot be compared point by point.**  Two simulators that
agree perfectly on the equations still drift apart in spike *timing*, because
they discretise differently; at the rising edge of an action potential a
sub-millisecond shift shows up as a 70 mV difference.  A point-wise RMS over a
spiking trace therefore measures timing jitter, not correctness.  The
comparison here separates the two:

* **spike count** — the coarsest check, and the one that fails loudest when a
  channel is wrong;
* **spike times** — compared pairwise, reported as a mean shift and a drift
  per inter-spike interval;
* **sub-threshold RMS** — the voltage away from spikes, where point-wise
  comparison does mean something.

A systematic timing difference is usually *discretisation*, not error.
:func:`convergence_study` settles which by running both simulators at a
sequence of time steps: a difference that shrinks as ``dt`` falls is numerical,
one that persists is a real disagreement.  ``jnml`` integrates explicitly,
Jaxley implicitly, so at any fixed ``dt`` they sit on opposite sides of the
converged answer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .errors import TarjumanError
from .reader import read_lems
from .simulate import run_lems
from .units import convert as from_si

__all__ = [
    "ReferenceUnavailable",
    "TraceComparison",
    "ValidationResult",
    "reference_available",
    "reference_problem",
    "neuron_home",
    "neuron_problem",
    "run_reference",
    "compare_traces",
    "auto_threshold",
    "model_spike_thresholds",
    "model_reset_drops",
    "validate_lems",
    "convergence_study",
    "spike_times",
    "reset_times",
    "detection_mode",
]

#: Default tolerances. Deliberately loose on absolute voltage and tight on
#: spike count: the first is dominated by timing jitter, the second is not.
DEFAULT_TOLERANCES = {
    "subthreshold_rms_mv": 1.0,
    "spike_time_shift_ms": 1.0,
    "spike_count": 0,  # exact
}

#: How far either side of a reference spike counts as "spiking", in ms.
SPIKE_WINDOW_MS = 3.0

#: A trace whose voltage spans less than this is treated as non-spiking, in mV.
MIN_SPIKING_RANGE_MV = 5.0


class ReferenceUnavailable(TarjumanError):
    """The reference simulator (``jnml``) is not installed or failed to run."""


# --------------------------------------------------------------------------- #
# Spike detection
# --------------------------------------------------------------------------- #
def auto_threshold(voltage: np.ndarray) -> Optional[float]:
    """Pick a spike-detection threshold from the trace itself.

    A fixed 0 mV threshold suits Hodgkin-Huxley overshoots and nothing else:
    an integrate-and-fire cell resetting between -70 and -50 mV never reaches
    it, and would be reported as silent. The threshold is instead placed
    between the trace's resting level (its 10th percentile) and its maximum,
    which lands sensibly for both. A trace with too small a range is treated as
    non-spiking and compared point-wise.

    Args:
        voltage: Voltages in mV, from the reference simulator.

    Returns:
        A threshold in mV, or ``None`` if the trace does not look like it
        spikes.
    """
    voltage = np.asarray(voltage, dtype=float)
    if voltage.size < 3:
        return None
    # Skip the initial transient: a model often starts at a voltage it never
    # returns to (NeuroML's iafTauCell starts at its own threshold), and a
    # single such sample would otherwise set the threshold above the whole
    # trace and report a spiking cell as silent.
    start = min(max(3, voltage.size // 50), voltage.size - 2)
    settled = voltage[start:]
    rest = float(np.percentile(settled, 10))
    peak = float(settled.max())
    if peak - rest < MIN_SPIKING_RANGE_MV:
        return None
    return rest + 0.8 * (peak - rest)


def spike_times(
    time: np.ndarray,
    voltage: np.ndarray,
    threshold: float = 0.0,
    mode: str = "auto",
    min_drop: Optional[float] = None,
) -> np.ndarray:
    """Spike times of a trace, interpolated to sub-sample accuracy.

    Taking the crossing index alone quantises every spike to one time step,
    which is the size of the differences being measured; interpolating between
    the two bracketing samples removes that floor.

    Args:
        time: Times in ms.
        voltage: Voltages in mV.
        threshold: Crossing threshold in mV.
        mode: ``"crossing"`` for upward threshold crossings, ``"reset"`` for
            reset discontinuities, or ``"auto"`` to use crossings where there
            are any and resets otherwise. **Comparing two traces requires the
            same mode for both** — see :func:`detection_mode`.
        min_drop: Smallest fall counted as a reset, in mV; see
            :func:`reset_times`.

    Returns:
        Spike times in ms.
    """
    if mode == "reset":
        return reset_times(time, voltage, min_drop)
    crossings = _crossings(time, voltage, threshold)
    if crossings.size or mode == "crossing":
        return crossings
    # An integrate-and-fire cell is clamped at its threshold and reset from it,
    # so whether it registers as crossing turns on floating-point equality at
    # exactly that value. Where nothing rises above the threshold, fall back on
    # the unambiguous marker: the reset itself.
    return reset_times(time, voltage, min_drop)


def detection_mode(
    time: np.ndarray,
    voltage: np.ndarray,
    threshold: float,
    min_drop: Optional[float] = None,
) -> str:
    """Which detection mode a trace calls for: ``"crossing"`` or ``"reset"``.

    Decided once, from the reference trace, and then applied to both traces
    being compared. Letting each trace choose for itself is a bug: a model
    clamped exactly at its threshold may register crossings in one simulator
    and only resets in the other, and the two spike counts are then not
    measuring the same thing.
    """
    if _crossings(time, voltage, threshold).size:
        return "crossing"
    return "reset"


def _crossings(
    time: np.ndarray, voltage: np.ndarray, threshold: float
) -> np.ndarray:
    """Upward threshold crossings, interpolated between samples."""
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    below = voltage[:-1] < threshold
    above = voltage[1:] >= threshold
    indices = np.where(below & above)[0]
    if indices.size == 0:
        return np.empty(0)
    v0, v1 = voltage[indices], voltage[indices + 1]
    t0, t1 = time[indices], time[indices + 1]
    fraction = np.where(v1 != v0, (threshold - v0) / (v1 - v0), 0.0)
    return t0 + fraction * (t1 - t0)


#: A one-sample drop larger than this fraction of the trace's range is a reset.
RESET_FRACTION = 0.5


#: Fraction of a model's declared ``thresh - reset`` that still counts as a
#: reset.  Well below one because the two traces are resampled onto a common
#: time base before detection, and interpolation splits a one-step fall across
#: two samples whenever the simulators' sample times are offset — halving the
#: apparent drop, or worse.  A quarter of the declared fall is still some fifty
#: times the sub-threshold agreement between the two simulators, so nothing
#: short of a real reset comes close.
DECLARED_DROP_FRACTION = 0.25


def reset_times(
    time: np.ndarray, voltage: np.ndarray, min_drop: Optional[float] = None
) -> np.ndarray:
    """Times of reset discontinuities — a spiking cell's other signature.

    A reset is a single-sample fall of more than ``min_drop``, or, without one,
    of more than :data:`RESET_FRACTION` of the trace's whole range. An action
    potential repolarises over about a millisecond and so is not mistaken for
    one; an integrate-and-fire reset happens in one step and is.

    Taking the size from the trace fails on an adaptive cell: ``adExIaFCell``
    resets by ``thresh - reset`` — 8 mV for NeuroML's bursting parameters —
    while the trace as a whole spans 30 mV, so a fraction of the range sets
    the bar above every real reset and finds none. The model states the drop it
    makes, so ``min_drop`` is the better answer wherever the caller knows it.

    Args:
        time: Times in ms.
        voltage: Voltages in mV.
        min_drop: Smallest fall counted as a reset, in mV.

    Returns:
        Reset times in ms.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    if voltage.size < 3:
        return np.empty(0)
    if min_drop is not None and min_drop > 0:
        limit = float(min_drop)
    else:
        span = float(voltage.max() - voltage.min())
        if span <= 0:
            return np.empty(0)
        limit = RESET_FRACTION * span
    drops = np.diff(voltage)
    indices = np.where(drops < -limit)[0]
    return time[indices]


# --------------------------------------------------------------------------- #
# Running the reference simulator
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def reference_available() -> bool:
    """True if the reference simulator is installed **and can run**.

    Checking the PATH is not enough. ``pip install pyneuroml`` puts a ``jnml``
    script there whether or not a Java runtime exists, and on macOS the
    system's own ``java`` is a stub whose only purpose is to report that Java
    is missing. Taking the script's presence as readiness turns "Java is not
    installed" into two confusing test failures instead of two skips, so this
    actually invokes ``jnml``.
    """
    if shutil.which("jnml") is None:
        return False
    try:
        process = subprocess.run(
            ["jnml", "-v"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return process.returncode == 0


def reference_problem() -> Optional[str]:
    """Why the reference simulator is unusable, or ``None`` if it is fine."""
    if shutil.which("jnml") is None:
        return (
            "jnml was not found on the PATH. Install it with "
            "`pip install pyneuroml`."
        )
    if reference_available():
        return None
    if shutil.which("java") is None:
        return (
            "jnml is installed but there is no Java runtime. Install one with "
            "`brew install --cask temurin` on macOS, or your platform's JDK "
            "package."
        )
    return (
        "jnml is installed but will not run. On macOS the system `java` is a "
        "stub that only reports Java as missing; install a real JDK with "
        "`brew install --cask temurin`. Check with `java -version`."
    )


#: Reference engines, as the full ``jnml`` argument list for each. ``-nogui``
#: belongs only to the jLEMS run; combining it with an export flag is a usage
#: error.
ENGINES = {
    "jlems": ["-nogui"],            # jLEMS itself: single-compartment only
    "neuron": ["-neuron", "-run"],  # export to NEURON and run it
}


def neuron_home() -> Optional[Path]:
    """Where ``jnml`` should look for NEURON, or ``None`` if it is not found.

    ``jnml`` does not search the PATH for NEURON. It reads ``NEURON_HOME`` and
    otherwise tries a fixed list of install prefixes from the days of source
    builds (``/usr/local/nrn/x86_64`` and friends), expecting ``bin/nriv``
    under each. A pip-installed NEURON matches none of them, which is how
    ``pip install neuron`` ends in the bare Java error "Could not find NEURON
    home directory!" while ``nrniv`` sits happily on the PATH. So find it here
    and tell ``jnml`` where it is.

    Returns:
        A directory containing ``bin/nrniv``, or ``None``.
    """
    existing = os.environ.get("NEURON_HOME")
    if existing and (Path(existing) / "bin" / "nrniv").exists():
        return Path(existing)

    candidates: list[Path] = []
    try:  # The wheel keeps its binaries in the package, under `.data`.
        import neuron  # noqa: PLC0415  (optional dependency)

        candidates.append(Path(neuron.__file__).resolve().parent / ".data")
    except Exception:  # pragma: no cover - NEURON is optional
        pass

    executable = shutil.which("nrniv")
    if executable:
        # `<prefix>/bin/nrniv`, or a venv shim pointing into the real prefix.
        resolved = Path(executable).resolve()
        candidates.extend([Path(executable).parent.parent, resolved.parent.parent])

    for candidate in candidates:
        if (candidate / "bin" / "nrniv").exists():
            return candidate
    return None


def neuron_problem() -> Optional[str]:
    """Why the NEURON engine is unusable, or ``None`` if it is fine."""
    if neuron_home() is not None:
        return None
    return (
        "jnml needs NEURON for multicompartmental models and cannot find it. "
        "Install it with `pip install neuron`, or point jnml at an existing "
        "install by setting NEURON_HOME to the directory holding `bin/nrniv`. "
        "Note that jnml ignores the PATH, so `nrniv` being runnable is not "
        "enough on its own."
    )


def _compiler(variable: str, fallbacks: Sequence[str]) -> Optional[str]:
    """A compiler that exists, for ``nrnivmodl`` to compile mod files with.

    The wheel's makefile hardcodes the compiler its manylinux image was built
    with (``/opt/rh/gcc-toolset-10/root/usr/bin/g++``), which exists on nobody
    else's machine. NEURON's own PATH wrapper repairs that by exporting ``CC``
    and ``CXX`` before handing over; ``jnml`` calls the makefile's script
    directly and so misses it.
    """
    import sysconfig

    candidates = [sysconfig.get_config_var(variable) or "", *fallbacks]
    for candidate in candidates:
        name = candidate.split()[0] if candidate else ""
        if name and shutil.which(name):
            return candidate
    return None


def _c_compiler() -> Optional[str]:
    return _compiler("CC", ("cc", "gcc", "clang"))


def _cxx_compiler() -> Optional[str]:
    return _compiler("CXX", ("c++", "g++", "clang++"))


def _reference_environment(engine: str) -> dict[str, str]:
    """The environment to run ``jnml`` in, with NEURON located for it.

    ``jnml`` calls ``$NEURON_HOME/bin/nrngui`` and ``nrnivmodl`` directly.
    In a pip-installed NEURON those two are shell scripts that still carry the
    temporary prefix the wheel was built under (``/tmp/tmp…/wheel/platlib``),
    and they fall back to it unless ``NRNHOME`` is set — the PATH versions are
    Python wrappers that set it, which is why ``nrniv`` works from a shell but
    not when ``jnml`` reaches past the wrapper. Setting it here is what makes
    ``pip install neuron`` enough.
    """
    environment = dict(os.environ)
    if engine != "neuron":
        return environment

    home = neuron_home()
    if home is None:
        raise ReferenceUnavailable(neuron_problem())
    environment["NEURON_HOME"] = str(home)
    environment.setdefault("NRNHOME", str(home))
    environment.setdefault("CORENRNHOME", str(home))
    environment.setdefault("NEURONHOME", str(home / "share" / "nrn"))
    environment.setdefault("NMODLHOME", str(home))
    environment.setdefault("NRN_PYTHONEXE", sys.executable)
    environment.setdefault("CORENRN_PYTHONEXE", sys.executable)
    for variable, compiler in (("CC", _c_compiler()), ("CXX", _cxx_compiler())):
        if compiler:
            environment.setdefault(variable, compiler)
    # nrnivmodl compiles the exported mod files, and it is beside nrniv.
    binaries = str(home / "bin")
    if binaries not in environment.get("PATH", "").split(os.pathsep):
        environment["PATH"] = binaries + os.pathsep + environment.get("PATH", "")
    return environment


def run_reference(
    lems_path: str | os.PathLike,
    timeout: int = 900,
    work_dir: Optional[str | os.PathLike] = None,
    engine: str = "jlems",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run a reference NeuroML simulator on a LEMS file and read its output.

    Two engines are available through ``jnml``. ``jlems`` is its own LEMS
    interpreter, which is the most direct reading of the NeuroML specification
    but **cannot simulate multicompartmental cells** — it refuses them
    outright. ``neuron`` exports the model to NEURON and runs that, which
    handles morphology; it needs NEURON installed (``pip install neuron``).

    The LEMS file is copied, with everything beside it, into a scratch
    directory: ``jnml`` writes its results relative to the file, and a
    validation run should not litter the model's own directory.

    Args:
        lems_path: Path to a ``LEMS_*.xml`` file.
        timeout: Seconds to allow the reference simulator.
        work_dir: Where to run; a temporary directory by default.
        engine: ``"jlems"`` or ``"neuron"``.

    Returns:
        ``{column id: (time in ms, values)}``, in the units the LEMS file's
        ``OutputFile`` columns are written in (SI), converted to tarjuman's
        units for voltage.

    Raises:
        ReferenceUnavailable: If ``jnml`` is missing or exits non-zero.
    """
    if not reference_available():
        raise ReferenceUnavailable(reference_problem())

    lems_path = Path(lems_path).resolve()
    _, simulation = read_lems(lems_path)

    created = work_dir is None
    directory = (
        Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="tarjuman-ref-"))
    )
    try:
        _stage_model(lems_path, directory)
        (directory / "results").mkdir(exist_ok=True)

        if engine not in ENGINES:
            raise ReferenceUnavailable(
                f"Unknown reference engine {engine!r}; "
                f"choose one of {sorted(ENGINES)}."
            )
        process = subprocess.run(
            ["jnml", lems_path.name, *ENGINES[engine]],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_reference_environment(engine),
        )
        if process.returncode != 0:
            message = _reference_error(process)
            output = (process.stdout or "") + (process.stderr or "")
            if "MULTICOMPARTMENTAL_CELL_MODEL" in output:
                message = (
                    "jLEMS cannot simulate multicompartmental cells (it reports "
                    "'Level of support: NONE'). Use engine='neuron' for this "
                    "model."
                )
            elif "NEURON home directory" in output:
                # NEURON_HOME was set from neuron_home() and jnml still refused,
                # so the directory exists but is not the one it wants.
                message = (
                    f"jnml rejected the NEURON install at {neuron_home()}. Set "
                    "NEURON_HOME to a directory holding `bin/nrniv` from a "
                    "NEURON build jnml accepts. Original error: " + message
                )
            elif "Unsuccessful compilation of NEURON mod files" in output:
                message = (
                    "NEURON could not compile the exported mod files "
                    "(`nrnivmodl`). A C++ compiler is needed, plus the ncurses "
                    "and readline development packages. The compiler error is "
                    "in the lines marked 'NMODL Error'. Original error: "
                    + message
                )
            raise ReferenceUnavailable(
                f"jnml failed on {lems_path.name} (exit {process.returncode}): "
                + message
            )

        return _read_reference_outputs(directory, simulation)
    finally:
        if created and work_dir is None:
            shutil.rmtree(directory, ignore_errors=True)


def _stage_model(lems_path: Path, directory: Path) -> None:
    """Copy a LEMS file and everything it includes into a scratch directory.

    Includes are resolved relative to the file that declares them and may point
    outside its directory (``../examples/cell.nml`` is common in the NeuroML
    examples), so each one is copied to the same relative position rather than
    flattened.
    """
    base = lems_path.parent
    staged: set[Path] = set()

    def stage(path: Path) -> None:
        path = path.resolve()
        if path in staged or not path.exists():
            return
        staged.add(path)
        try:
            relative = path.relative_to(base)
        except ValueError:
            # Outside the model directory: keep the same relative path, which
            # may climb out of it, by mirroring it under the scratch root.
            relative = Path(os.path.relpath(path, base))
        target = (directory / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        for href in _includes_of(path):
            stage((path.parent / href))

    stage(lems_path)
    # Data files the model may read are not declared as includes; copy the
    # LEMS file's own directory as well, minus previous results.
    for entry in base.iterdir():
        if entry.is_dir() or entry.name == "results":
            continue
        target = directory / entry.name
        if not target.exists():
            shutil.copy2(entry, target)


def _includes_of(path: Path) -> list[str]:
    """``href``/``file`` attributes of a NeuroML or LEMS document."""
    import xml.etree.ElementTree as ET

    if path.suffix.lower() not in (".xml", ".nml"):
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    hrefs = []
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag.lower() != "include":
            continue
        href = element.get("href") or element.get("file")
        if href:
            hrefs.append(href)
    return hrefs


def _reference_error(process) -> str:
    """The most informative line from a failed reference run."""
    text = (process.stderr or "") + "\n" + (process.stdout or "")
    interesting = [
        line.strip()
        for line in text.splitlines()
        if any(
            marker in line
            for marker in ("ContentError", "Exception", "ERROR", "Can't find", "Unable")
        )
    ]
    if interesting:
        return interesting[0][:300]
    tail = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return (tail[-1] if tail else "no output")[:300]


def _read_reference_outputs(directory: Path, simulation) -> dict:
    """Match the reference simulator's ``.dat`` files to the LEMS columns."""
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    columns_by_file: dict[str, list[tuple[str, str]]] = {}
    for file_id, column_id, quantity in simulation.outputs:
        columns_by_file.setdefault(file_id, []).append((column_id, quantity))

    counts = {}
    for _, column_id, _ in simulation.outputs:
        counts[column_id] = counts.get(column_id, 0) + 1

    for file_id, columns in columns_by_file.items():
        file_name = simulation.output_files.get(file_id)
        if file_name is None:
            continue
        path = directory / file_name
        if not path.exists():
            raise ReferenceUnavailable(
                f"jnml did not write {file_name}, which OutputFile '{file_id}' "
                "declares. The reference run may have failed silently."
            )
        data = np.loadtxt(path)
        if data.ndim == 1:  # a single row
            data = data.reshape(1, -1)
        # LEMS OutputFiles are written in SI: seconds, volts.
        time_ms = data[:, 0] * 1e3
        for index, (column_id, quantity) in enumerate(columns):
            values = data[:, index + 1]
            if _is_voltage(quantity):
                values = values * 1e3  # V -> mV
            name = column_id if counts[column_id] == 1 else f"{file_id}.{column_id}"
            traces[name] = (time_ms, values)
    return traces


def model_reset_drops(model) -> dict[str, float]:
    """How far each population's cells fall on a reset, in mV, by population.

    A cell that declares both a threshold and a reset potential states the size
    of its own discontinuity, which is a far better detector than a fraction of
    the trace's range — see :func:`reset_times`. Only a fraction
    (:data:`DECLARED_DROP_FRACTION`) of the declared drop is required, so that a
    reset split across two samples still counts.

    Args:
        model: A :class:`~tarjuman.builder.ConvertedModel`.

    Returns:
        ``{population id: minimum drop in mV}``, omitting populations whose
        cells declare no reset.
    """
    drops: dict[str, float] = {}
    if model.network is None:
        return drops
    for population in model.network.populations:
        point_cell = model.document.point_cells.get(population.component)
        if point_cell is None:
            continue
        threshold = point_cell.params.get("thresh")
        reset = point_cell.params.get("reset")
        if threshold is None or reset is None or threshold <= reset:
            continue
        drops[population.id] = DECLARED_DROP_FRACTION * (threshold - reset)
    return drops


def model_spike_thresholds(model) -> dict[str, float]:
    """Spike thresholds in mV, by population, as the NeuroML model declares them.

    A NeuroML cell states its own ``spikeThresh``, and an integrate-and-fire
    cell its ``thresh``. Using those is strictly better than inferring a
    threshold from the trace: a compartment receiving only sub-threshold EPSPs
    has no spikes to find, but a trace-derived threshold will happily place
    itself among the EPSP peaks and count them.

    Args:
        model: A :class:`~tarjuman.builder.ConvertedModel`.

    Returns:
        ``{population id: threshold in mV}``; empty for a model without
        populations.
    """
    from .builder import _spike_threshold

    thresholds: dict[str, float] = {}
    if model.network is None:
        return thresholds
    for population in model.network.populations:
        thresholds[population.id] = _spike_threshold(
            model.document, population.component
        )
    return thresholds


def _is_voltage(quantity: str) -> bool:
    return quantity.rstrip("/").endswith("/v") or quantity.endswith("v")


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass
class TraceComparison:
    """How one recorded quantity compares between the two simulators."""

    name: str
    quantity: str = ""
    is_voltage: bool = True
    n_points: int = 0
    max_abs_diff: float = 0.0
    rms_diff: float = 0.0
    subthreshold_rms: float = 0.0
    subthreshold_max: float = 0.0
    n_spikes_reference: int = 0
    n_spikes_tarjuman: int = 0
    spike_time_shift: float = 0.0  # mean, ms
    spike_time_drift: float = 0.0  # per inter-spike interval, ms
    correlation: float = 1.0
    #: Threshold used for spike detection, or ``None`` for a non-spiking trace.
    spike_threshold: Optional[float] = None
    #: ``"crossing"`` or ``"reset"``; see :func:`detection_mode`.
    detection: str = "crossing"

    @property
    def spikes_match(self) -> bool:
        return self.n_spikes_reference == self.n_spikes_tarjuman

    def passed(self, tolerances: Optional[dict] = None) -> bool:
        """Whether this trace is within tolerance."""
        tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
        if abs(self.n_spikes_reference - self.n_spikes_tarjuman) > tolerances[
            "spike_count"
        ]:
            return False
        if self.is_voltage:
            if self.subthreshold_rms > tolerances["subthreshold_rms_mv"]:
                return False
            if abs(self.spike_time_shift) > tolerances["spike_time_shift_ms"]:
                return False
        return True

    def summary(self) -> str:
        if not self.is_voltage:
            return (
                f"{self.name}: max |diff| {self.max_abs_diff:.4g}, "
                f"RMS {self.rms_diff:.4g}, r={self.correlation:.5f}"
            )
        if self.spike_threshold is None:
            return (
                f"{self.name}: non-spiking, RMS {self.rms_diff:.3f} mV, "
                f"max {self.max_abs_diff:.3f} mV"
            )
        parts = [
            f"{self.name}: spikes {self.n_spikes_reference} vs "
            f"{self.n_spikes_tarjuman}",
            f"sub-threshold RMS {self.subthreshold_rms:.3f} mV",
        ]
        if self.n_spikes_reference:
            parts.append(
                f"spike shift {self.spike_time_shift:+.3f} ms "
                f"(drift {self.spike_time_drift:+.4f} ms/ISI)"
            )
        return ", ".join(parts)


def compare_traces(
    reference: tuple[np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray],
    name: str = "trace",
    quantity: str = "",
    is_voltage: bool = True,
    threshold: Optional[float] = None,
    reset_drop: Optional[float] = None,
) -> TraceComparison:
    """Compare one reference trace with one tarjuman trace.

    The two are resampled onto the reference's time base, since the simulators
    need not have used the same step.

    Args:
        reference: ``(time in ms, values)`` from the reference simulator.
        candidate: ``(time in ms, values)`` from tarjuman.
        name: Column name, for reporting.
        quantity: The LEMS quantity path, for reporting.
        is_voltage: Whether to apply spike-aware comparison.
        threshold: Spike threshold in mV; chosen from the reference trace by
            :func:`auto_threshold` when omitted.
        reset_drop: Smallest voltage fall counted as a reset, in mV — the drop
            the model itself declares, where the caller knows it. See
            :func:`reset_times`.

    Returns:
        A :class:`TraceComparison`.
    """
    t_ref, v_ref = np.asarray(reference[0]), np.asarray(reference[1])
    t_cand, v_cand = np.asarray(candidate[0]), np.asarray(candidate[1])

    # Compare only over the span both simulators covered.
    end = min(t_ref[-1], t_cand[-1])
    mask = t_ref <= end
    t_ref, v_ref = t_ref[mask], v_ref[mask]
    v_cand = np.interp(t_ref, t_cand, v_cand)

    difference = np.abs(v_cand - v_ref)
    comparison = TraceComparison(
        name=name,
        quantity=quantity,
        is_voltage=is_voltage,
        n_points=int(t_ref.size),
        max_abs_diff=float(difference.max()) if difference.size else 0.0,
        rms_diff=float(np.sqrt((difference**2).mean())) if difference.size else 0.0,
    )
    if v_ref.size > 1 and np.std(v_ref) > 0 and np.std(v_cand) > 0:
        comparison.correlation = float(np.corrcoef(v_ref, v_cand)[0, 1])

    if not is_voltage:
        comparison.subthreshold_rms = comparison.rms_diff
        comparison.subthreshold_max = comparison.max_abs_diff
        return comparison

    if threshold is None:
        threshold = auto_threshold(v_ref)
    if threshold is None:
        # Not a spiking trace: the point-wise comparison is the whole story.
        comparison.spike_threshold = None
        comparison.subthreshold_rms = comparison.rms_diff
        comparison.subthreshold_max = comparison.max_abs_diff
        return comparison
    comparison.spike_threshold = float(threshold)

    # One detection mode, chosen from the reference and used for both.
    mode = detection_mode(t_ref, v_ref, threshold, reset_drop)
    comparison.detection = mode
    reference_spikes = spike_times(
        t_ref, v_ref, threshold, mode=mode, min_drop=reset_drop
    )
    candidate_spikes = spike_times(
        t_ref, v_cand, threshold, mode=mode, min_drop=reset_drop
    )
    comparison.n_spikes_reference = int(reference_spikes.size)
    comparison.n_spikes_tarjuman = int(candidate_spikes.size)

    # Sub-threshold: everything outside a window around any reference spike.
    quiet = np.ones(t_ref.shape, dtype=bool)
    for spike in reference_spikes:
        quiet &= np.abs(t_ref - spike) > SPIKE_WINDOW_MS
    if quiet.any():
        quiet_difference = difference[quiet]
        comparison.subthreshold_rms = float(
            np.sqrt((quiet_difference**2).mean())
        )
        comparison.subthreshold_max = float(quiet_difference.max())
    else:  # the whole trace is spiking
        comparison.subthreshold_rms = comparison.rms_diff
        comparison.subthreshold_max = comparison.max_abs_diff

    paired = min(reference_spikes.size, candidate_spikes.size)
    if paired:
        shifts = candidate_spikes[:paired] - reference_spikes[:paired]
        comparison.spike_time_shift = float(shifts.mean())
        if paired > 1:
            # Slope of the shift against spike number: a constant offset is
            # harmless, a growing one is a frequency difference.
            comparison.spike_time_drift = float(
                np.polyfit(np.arange(paired), shifts, 1)[0]
            )
    return comparison


# --------------------------------------------------------------------------- #
# Whole-model validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    """The outcome of validating one LEMS file against the reference."""

    model: str
    comparisons: list[TraceComparison] = field(default_factory=list)
    tolerances: dict = field(default_factory=lambda: dict(DEFAULT_TOLERANCES))
    reference_seconds: float = 0.0
    tarjuman_seconds: float = 0.0
    #: Which reference engine produced the comparison.
    engine: str = "jlems"
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        return bool(self.comparisons) and all(
            comparison.passed(self.tolerances) for comparison in self.comparisons
        )

    @property
    def speedup(self) -> Optional[float]:
        if self.tarjuman_seconds <= 0:
            return None
        return self.reference_seconds / self.tarjuman_seconds

    def summary(self) -> str:
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}  {self.model}",
        ]
        if self.error:
            lines.append(f"  error: {self.error}")
            return "\n".join(lines)
        for comparison in self.comparisons:
            mark = "ok  " if comparison.passed(self.tolerances) else "FAIL"
            lines.append(f"  {mark} {comparison.summary()}")
        if self.speedup:
            lines.append(
                f"  {self.engine} {self.reference_seconds:.1f}s, "
                f"tarjuman {self.tarjuman_seconds:.1f}s "
                f"({self.speedup:.1f}x)"
            )
        return "\n".join(lines)


def validate_lems(
    lems_path: str | os.PathLike,
    tolerances: Optional[dict] = None,
    delta_t: Optional[float] = None,
    timeout: int = 900,
    engine: str = "auto",
    **kwargs,
) -> ValidationResult:
    """Run a LEMS file through both simulators and compare the traces.

    Args:
        lems_path: Path to a ``LEMS_*.xml`` file.
        tolerances: Overrides for :data:`DEFAULT_TOLERANCES`.
        delta_t: Time step for both runs; the LEMS file's own step by default.
        timeout: Seconds to allow the reference simulator.
        engine: Reference engine, or ``"auto"`` to try jLEMS and fall back to
            NEURON for the multicompartmental models jLEMS refuses.
        **kwargs: Passed to :func:`tarjuman.run_lems`.

    Returns:
        A :class:`ValidationResult`. A failure to run either simulator is
        recorded in ``error`` rather than raised, so a sweep over many models
        reports on all of them.
    """
    import time as _time

    lems_path = Path(lems_path)
    result = ValidationResult(
        model=lems_path.name, tolerances={**DEFAULT_TOLERANCES, **(tolerances or {})}
    )

    if not lems_path.exists():
        # Report a missing file once, as itself, rather than letting every
        # engine rediscover it and reporting the same thing twice.
        result.error = (
            f"no such file: {lems_path}. Give the path to a LEMS file, "
            "e.g. tests/data/LEMS_hh_single_comp.xml"
        )
        return result

    engines = ["jlems", "neuron"] if engine == "auto" else [engine]
    failures = []
    for candidate_engine in engines:
        try:
            started = _time.time()
            reference = run_reference(
                lems_path, timeout=timeout, engine=candidate_engine
            )
            result.reference_seconds = _time.time() - started
            result.engine = candidate_engine
            break
        except Exception as error:
            failures.append(f"{candidate_engine}: {error}")
    else:
        result.error = "reference: " + " | ".join(failures)
        return result

    try:
        started = _time.time()
        traces, model = run_lems(lems_path, delta_t=delta_t, **kwargs)
        result.tarjuman_seconds = _time.time() - started
    except Exception as error:
        result.error = f"tarjuman: {error}"
        return result

    thresholds = model_spike_thresholds(model)
    reset_drops = model_reset_drops(model)

    for name, (t_ref, v_ref) in reference.items():
        if name not in traces.traces:
            continue
        quantity = traces.quantities.get(name, "")
        dimension = traces.dimensions.get(name)
        if dimension is not None and dimension not in ("none", "voltage"):
            # The reference writes SI; tarjuman's traces are in its own units.
            # Voltage is already handled when the reference is read, but an
            # abstract cell's own exposures — an adaptation current in nA, say
            # — would otherwise be compared against amps.
            v_ref = np.asarray(from_si(np.asarray(v_ref), dimension))
        result.comparisons.append(
            compare_traces(
                (t_ref, v_ref),
                (traces.time, np.asarray(traces[name])),
                name=name,
                quantity=quantity,
                is_voltage=_is_voltage(quantity),
                threshold=_threshold_for(quantity, thresholds),
                reset_drop=_threshold_for(quantity, reset_drops),
            )
        )
    if not result.comparisons:
        result.error = (
            "no columns in common between the reference output and tarjuman's"
        )
    return result


def convergence_study(
    lems_path: str | os.PathLike,
    steps: Sequence[float] = (0.01, 0.005, 0.001),
    threshold: Optional[float] = None,
    timeout: int = 900,
) -> dict:
    """Measure how both simulators converge as the time step falls.

    This is what distinguishes a discretisation artefact from a real
    disagreement.  ``jnml`` integrates explicitly and Jaxley implicitly, so the
    two approach the exact solution from opposite directions; if their mean
    inter-spike intervals converge on each other as ``dt`` falls, the models
    agree and the difference at any fixed step is numerical.

    Args:
        lems_path: Path to a ``LEMS_*.xml`` file.
        steps: Time steps in ms, largest first.
        threshold: Spike threshold in mV.
        timeout: Seconds to allow each reference run.

    Returns:
        ``{"steps": [...], "reference_isi": [...], "tarjuman_isi": [...]}``,
        with ``None`` where a run produced too few spikes to measure.
    """
    import re

    lems_path = Path(lems_path).resolve()
    original = lems_path.read_text()
    study: dict = {"steps": list(steps), "reference_isi": [], "tarjuman_isi": []}

    for step in steps:
        # The reference reads its step from the file, so write a variant.
        with tempfile.TemporaryDirectory(prefix="tarjuman-conv-") as directory:
            work = Path(directory)
            for entry in lems_path.parent.iterdir():
                if entry.is_dir():
                    if entry.name != "results":
                        shutil.copytree(entry, work / entry.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, work / entry.name)
            variant = work / lems_path.name
            variant.write_text(
                re.sub(r'step="[^"]*"', f'step="{step}ms"', original, count=1)
            )
            (work / "results").mkdir(exist_ok=True)
            try:
                reference = run_reference(variant, timeout=timeout, work_dir=work)
                study["reference_isi"].append(_mean_isi(reference, threshold))
            except Exception:
                study["reference_isi"].append(None)

        try:
            traces, _ = run_lems(lems_path, delta_t=step)
            candidate = {
                name: (traces.time, np.asarray(traces[name]))
                for name in traces.traces
                if _is_voltage(traces.quantities.get(name, ""))
            }
            study["tarjuman_isi"].append(_mean_isi(candidate, threshold))
        except Exception:
            study["tarjuman_isi"].append(None)

    return study


def _threshold_for(quantity: str, thresholds: dict[str, float]) -> Optional[float]:
    """The entry for the population a quantity belongs to, from any table."""
    if not quantity or not thresholds:
        return None
    try:
        from .reader import parse_quantity_path

        population = parse_quantity_path(quantity)["population"]
    except Exception:
        return None
    return thresholds.get(population)


def _mean_isi(traces: dict, threshold: Optional[float]) -> Optional[float]:
    """Mean inter-spike interval of the first trace with at least two spikes."""
    for time, values in traces.values():
        level = threshold if threshold is not None else auto_threshold(values)
        if level is None:
            continue
        spikes = spike_times(time, values, level)
        if spikes.size >= 2:
            return float(np.diff(spikes).mean())
    return None
