# tarjuman

**Run NeuroML 2 models on [Jaxley](https://jaxley.readthedocs.io).**

`tarjuman` (ترجمان / *tərcüman* / *tercüman* — translator, interpreter) reads
[NeuroML 2](https://docs.neuroml.org) and LEMS documents and builds the
equivalent Jaxley model. Morphologies become branches, `channelDensity`
elements become inserted channels, projections become synapses, ion pools
become pumps, inputs become stimuli.

The point is not only portability. Jaxley is written in [JAX](https://docs.jax.dev),
so a NeuroML model converted with `tarjuman` runs on a GPU, vectorises over
parameter sets with `vmap`, and is **differentiable with respect to its own
channel kinetics** — every `rate`, `midpoint` and `scale` in every gate of
every ion channel becomes a Jaxley parameter you can take gradients through and
fit to data.

```python
import tarjuman

model = tarjuman.from_neuroml("NML2_SingleCompHHCell.nml")
result = tarjuman.simulate(model, t_max=300.0, delta_t=0.01)
print(model.report.summary())

model.module.make_trainable("naChans_m_alpha_midpoint")   # now fit it to data
```

## Install

```bash
pip install tarjuman
```

Optional extras:

| extra | pulls in | for |
|---|---|---|
| `tarjuman[neuroml]` | `libNeuroML` | reading models straight from `libNeuroML` objects |
| `tarjuman[plot]` | `matplotlib` | the CLI's `--plot` |
| `tarjuman[examples]` | `matplotlib`, `optax` | the four example scripts |
| `tarjuman[validate]` | `pyneuroml`, `neuron` | `tarjuman validate` and `tarjuman converge` |
| `tarjuman[dev]` | the above plus `pytest`, `ruff` | running the test suite |

```bash
pip install "tarjuman[dev]"     # everything needed to reproduce this README
```

## Command line

Five subcommands. Each one below is a real command with its real output.

```bash
tarjuman info NeuroML2/examples/NML2_SingleCompHHCell.nml
```
```
NeuroML document: NML2_SingleCompHHCell  (.../NML2_SingleCompHHCell.nml)
  ion channels (3): kChan, naChan, passiveChan
  cells (1): hhcell
  inputs (1): pulseGen1
  networks (1): net1
  network 'net1':
    population 'hhpop': 1 x 'hhcell'
  cell 'hhcell': 1 segments, 3 channel densities
```

```bash
tarjuman convert NeuroML2/examples/NML2_SingleCompHHCell.nml --script hh.py
```
```
tarjuman conversion report for .../NML2_SingleCompHHCell.nml
  1 cells, 1 branches, 1 compartments, 0 synapses
  [approximation] cell 'hhcell' segment 0: zero-length (spherical) segment
    converted to a cylinder of length = diameter = 17.8412 um, preserving surface area
wrote hh.py
```

`--script` writes a standalone Jaxley script, so you can take the conversion
and leave tarjuman behind.

```bash
tarjuman run tests/data/LEMS_hh_single_comp.xml --output v.dat
```
```
simulated 300.0 ms, 3 traces recorded
wrote v.dat
```

`tarjuman run` on a LEMS file reproduces the experiment the file describes —
duration, time step, inputs, and the quantities its `OutputFile` elements ask
for — and writes them in the same SI column format `jnml` writes, so the two
outputs are diffable:

```
0        -0.065      0.0529325   0.317677
1e-05    -0.0649997  0.0529325   0.317677
```

Add `--plot out.png` for a figure instead, and `--ncomp N` /
`--max-comp-length L` to control compartmentalisation.

`tarjuman validate` and `tarjuman converge` are the two that call the reference
simulator; both are described under *Validation against the reference
simulator* below.

## The C. elegans connectome

The models `tarjuman` is built against are the OpenWorm ones. From a checkout
of [c302](https://github.com/openworm/c302):

```bash
tarjuman run c302/examples/LEMS_c302_C_Oscillator.xml --plot oscillator.png
```

Point at the LEMS file where it lives: it includes the NeuroML files beside it
(`c302_C_Oscillator.net.nml`, `cell_C.xml`) by relative path, so copying it
elsewhere on its own leaves it with nothing to simulate. tarjuman says which
includes it could not find if that happens.

| c302 parameter set | what it needs | status |
|---|---|---|
| **A** | `iafCell` populations, `expTwoSynapse` | ✅ runs |
| **B** | `iafActivityCell`, a custom integrate-and-fire cell with an activity trace, gathering synaptic input with `select="synapses[*]/i"` | ✅ runs |
| **C** | `gateHHtauInf`, a `customHGate` reading `caConc`, `fixedFactorConcentrationModel`, `expTwoSynapse`, `gapJunction` | ✅ runs |
| **C0 / C1 / C2** | as C, plus Kunert-style `gradedSynapse2` defined in the model | ✅ runs |
| **D / W2D** | `custom_muscle_components.xml` (`MuscleSigmoidVariable`, `muscleConcentrationModel`) | ⚠️ converts; muscle models not yet validated |

`c302_C_Oscillator` — 14 cells, 79 synapses, a calcium pool per cell and a
calcium-dependent gate — runs 1000 ms in about ten seconds on a CPU.

**Every parameter set now runs.** Set B was the last holdout, and what blocked
it was its own cell type: `iafActivityCell`, defined in `cell_B.xml`, extends
`iafCell` with an activity trace and reads its synaptic current with
`<DerivedVariable select="synapses[*]/i" reduce="add"/>`. See *Abstract cells*
for how that is resolved.

```bash
tarjuman run      c302/examples/LEMS_c302_B_Oscillator.xml
tarjuman validate c302/examples/LEMS_c302_B_Oscillator.xml
tarjuman converge c302/examples/LEMS_c302_B_Oscillator.xml --steps 0.05 0.02 0.005
```

Against jnml, 26 of its 28 traces pass and all 14 activity traces agree with
r ≥ 0.991. The two that do not are the pacemaker cells driving the oscillator,
where the inter-spike interval difference falls from 0.185 ms at `dt = 0.05` to
0.017 ms at `dt = 0.005` — discretisation, converging away.

## Python API

```python
import tarjuman

# A whole network, or a single cell.
model = tarjuman.from_neuroml("network.nml", network_id="net1")
model = tarjuman.from_neuroml("cell.nml", cell_id="pyr", ncomp=4)

# `model.module` is an ordinary jaxley.Network / jaxley.Cell.
net = model.module

# Address it in NeuroML terms: population, cell, segment, fraction along.
model.view("pop0", cell=2, segment=7, fraction_along=0.3).record("v")

# Reproduce a LEMS simulation.
result, model = tarjuman.run_lems("tests/data/LEMS_hh_single_comp.xml")
result.to_dataframe().head()
```

### Compartmentalisation

NeuroML segments are grouped into unbranched cables (segment groups carrying
neuroLex id `sao864921383`, i.e. NEURON sections); each becomes one Jaxley
branch. Compartments per branch come from the group's
`numberInternalDivisions` where the file gives one, otherwise from `ncomp`
(default 1). `max_comp_length=10.0` adds compartments so no compartment
exceeds 10 µm — useful when a NeuroML file was written for a simulator that
does its own spatial discretisation.

## Custom LEMS component types

NeuroML is a LEMS dialect, and real models define their own component types
constantly. `tarjuman` implements the standard ones natively and *interprets*
the rest: it parses the `<ComponentType>`, compiles its `<Dynamics>` to JAX,
and attaches it wherever its base type says it belongs.

```xml
<ComponentType name="customHGate" extends="gateHHtauInf">
    <Parameter name="alpha" dimension="none"/>
    <Requirement name="caConc" dimension="concentration"/>
    <Dynamics>
        <DerivedVariable name="inf" value="1 / (1 + (exp((ca_half - caConc) / k)))"/>
        <DerivedVariable name="fcond" exposure="fcond" value="1 +((q-1) * alpha)"/>
    </Dynamics>
</ComponentType>
```

Supported inside `<Dynamics>`: `StateVariable`, `DerivedVariable`,
`ConditionalDerivedVariable`, `TimeDerivative`, `OnStart`, `OnCondition`,
`OnEvent`. Expressions support `+ - * / ^`, the comparison and boolean word
operators (`.gt.`, `.and.`, …) and the LEMS function set (`exp ln log sin cos
tan asin acos atan sinh cosh tanh sqrt abs ceil floor H`).

Custom types can be attached as gates (and as the rate, steady state or time
course inside a standard gate), as concentration models, and as synapses. A
synapse's `<DerivedVariable select="peer/v">` resolves to the presynaptic
voltage Jaxley hands it.

**Expressions are evaluated in SI.** A dimensioned LEMS expression is only
correct in a coherent unit system and Jaxley's (mV, ms, µm, nA, S/cm²) is not
one, so parameters are converted to SI on the way in and exposures converted
back on the way out, using the dimensions the `ComponentType` declares.

## Abstract cells

`izhikevich2007Cell`, `adExIaFCell` and the integrate-and-fire cells models
define for themselves are LEMS ComponentTypes carrying a membrane equation, a
few state variables and a reset rule. tarjuman does not reimplement them:
`tarjuman/_cell_types.py` is generated from the `Cells.xml` shipped with
NeuroML2 and holds those ComponentTypes verbatim, so the same interpreter that
runs a model's own custom types runs the standard cells too.

```bash
tarjuman validate tests/data/LEMS_izhikevich2007.xml
tarjuman validate tests/data/LEMS_adex_rebound.xml
```

The work splits with Jaxley along one line. **Jaxley integrates the voltage**,
because only it knows the stimulus and synaptic currents arriving at the
compartment; the cell contributes the current its equations describe, which is
the LEMS `dv/dt` multiplied by `C`. **The interpreter integrates everything
else** — an adaptation current, a recovery variable, an activity trace — and
applies the `OnCondition` assignments, including the one that writes `v` on a
spike, so the reset is exact rather than approximated.

That split is what resolves `<DerivedVariable select="synapses[*]/i"
reduce="add"/>`, the construct every NeuroML abstract cell uses to read its
total synaptic current, and which earlier releases refused. The cell does not
need to *read* that sum; it needs the sum to reach its membrane equation, and
Jaxley's solver puts it there already. Evaluating the LEMS term as zero and
letting the solver supply the real one gives the same equation. This is what
unblocked c302 parameter set B.

### Where a spiking equation needs explicit integration

An AdEx cell's `exp((v - VT) / delT)` is designed to diverge: the divergence
*is* the spike, cut short by the reset. That makes the membrane conductance
regenerative — negative, and growing without bound — about 12 mV above `VT`.
Jaxley's implicit step infers that conductance from a finite difference and
inverts `C/dt + dI/dv`, so once the feedback passes `C/dt` the step changes
sign and the voltage flies off the wrong way: NeuroML's own `adExRebound`
parameters reached **-156 mV**, resetting on seven consecutive steps.

The region cannot be stepped over either — `adExRebound` turns regenerative
around -42 mV while its detection threshold sits at -30 mV, several steps
away. So the cell measures the feedback in its own equation each step, and
where one step of it would exceed half of `C/dt`, integrates the voltage
itself, explicitly and in sub-steps, stopping at the threshold. Everywhere
else, which is all of the sub-threshold trace, Jaxley solves as before with
the stimulus and synapses included.

The reset is tested against the voltage the step is *about* to produce rather
than the one it starts from. That is not an approximation: jLEMS integrates
explicitly, forms `v + dt dv/dt`, finds it above threshold and stores the
reset, so the overshoot appears in neither simulator's output.

Measured against jnml, the rebound burst comes out at three spikes for every
time step from 0.025 ms down — 211.95 / 214.44 / 218.64 ms against jnml's
211.94 / 214.45 / 218.66 ms.

## What gets converted

| NeuroML | Status |
|---|---|
| `ionChannel`, `ionChannelHH`, `ionChannelPassive` | ✅ |
| all six standard gate types, the three rate/variable forms, `q10` scaling | ✅ |
| gates, rates, steady states and time courses defined as custom ComponentTypes, including `fcond` | ✅ |
| `morphology`: segments, `fractionAlong`, segment groups, paths, spherical somata | ✅ |
| `channelDensity`, `specificCapacitance`, `initMembPotential`, `resistivity`, `spikeThresh` | ✅ |
| `species` + `fixedFactorConcentrationModel` / `decayingPoolConcentrationModel` / custom pools | ✅ |
| calcium-dependent gates (`caConc`) | ✅ |
| `expOneSynapse`, `expTwoSynapse`, `alphaSynapse`, NMDA Mg block | ✅ |
| `gapJunction`, `linearGradedSynapse`, `gradedSynapse`, `silentSynapse` | ✅ |
| synapses defined as custom ComponentTypes (`gradedSynapse2`, …) | ✅ |
| `iafCell`, `iafRefCell`, `iafTauCell`, `iafTauRefCell` | ✅ exact reset (state assignment) |
| Validation against `jnml` / NEURON (`tarjuman validate`) | ✅ |
| `population`, `populationList`, `projection`, `electricalProjection`, `continuousProjection`, weights | ✅ |
| `pulseGenerator`, `rampGenerator`, `sineGenerator`, `inputList`, `explicitInput` | ✅ |
| LEMS `Simulation`, `OutputFile`, `Display` | ✅ |
| `channelDensityNernst` | ⚠️ needs an explicit `erev` (`erev_overrides`) |
| Synaptic `delay` | ✅ shift register; needs `delta_t` at build time |
| `tsodyksMarkram` short-term plasticity | ⚠️ ignored, synapse converted without it |
| `gateFractional` and `subGate` (weighted subgate kinetics) | ✅ |
| Custom `timeCourse`/`steadyState` reading the parent gate's `alpha`/`beta` | ✅ |
| Recording `iDensity`, `gDensity`, `fopen` | ✅ |
| `ionChannelKS`, `gateKS` | ❌ not yet |
| `channelDensityGHK` | ❌ not yet |
| `izhikevich2007Cell`, `adExIaFCell`, `hindmarshRose1984Cell` | ✅ interpreted from their LEMS dynamics |
| abstract cells a model defines itself, including via `<iafCell type="...">` | ✅ |
| `<DerivedVariable select="synapses[*]/i">` in a cell | ✅ supplied by Jaxley's solver |
| LEMS `<Regime>` — the spike-and-refractory arrangement | ✅ |
| dimensionless cells (`izhikevichCell`, `fitzHughNagumoCell`, `baseCellMembPotDL`) | ❌ not yet |
| `pointCellCondBased`, PyNN cells | ❌ not yet |
| Spike sources (`spikeArray`, `spikeGenerator*`, `poissonFiringSynapse`) | ❌ not yet |
| LEMS `<Regime>` — any other arrangement | ❌ refused with a reason |

Nothing on the ⚠️ or ❌ rows is ever dropped quietly. Every conversion returns
a report:

```python
model.report.is_faithful          # True if nothing was approximated or dropped
model.report.summary()            # human-readable
model.report.unsupported_components
```

and each approximation also raises a `ConversionWarning` where it happens.
`strict=True` turns them into exceptions instead.

## Known limits

**Integrate-and-fire spikes are not differentiable.** The reset is a state
assignment, exactly as NeuroML specifies (measured ISI 20.800 ms against an
analytic 20.794 ms), but writing the voltage on a threshold crossing is a step
function, so the gradient through a spike is zero. Sub-threshold parameters
stay differentiable. Jaxley's own `Fire` mechanism carries the same caveat.

**A cell's own `iSyn` and `iMemb` cannot be recorded.** A LEMS cell reads its
synaptic current with `<DerivedVariable select="synapses[*]/i" reduce="add"/>`,
and tarjuman resolves that by evaluating the term as zero and letting Jaxley's
solver supply the real sum, which it assembles anyway — see *Abstract cells*
below. The equation is the same, but the cell never computes the sum itself, so
an exposure derived from it has no value here. Recording one is refused rather
than answered with the membrane current minus the synapses. Record `v`, or the
cell's own state variables, which are available.

**Dimensionless cells are not supported.** `izhikevichCell` (the 2003 form),
`fitzHughNagumoCell` and anything else extending `baseCellMembPotDL` carry no
capacitance and are driven by dimensionless currents (`basePointCurrentDL`).
tarjuman drives abstract cells through Jaxley's own membrane equation, which
needs a capacitance. Use `izhikevich2007Cell`, which is the dimensioned form of
the same model and is NeuroML's recommendation. The conversion reports which
cell was rejected and why.

**`iafTauCell` has no current term.** It extends `baseIaf`, not
`baseIafCapCell`, so in NeuroML an injected current does nothing to it at all;
a Jaxley compartment cannot ignore its own input current. The free decay to
`leakReversal` with the stated time constant is exact, but current injected
into such a cell moves it when in NeuroML it would not. The conversion report
says so.

**Synaptic delays** are quantised to whole integration steps. A NeuroML
connection `delay` becomes a shift register over the presynaptic voltage
(`DelayedSynapse`), which costs `delay / delta_t` extra states per synapse — a
5 ms delay at `delta_t = 0.025 ms` is 200 of them. Because the register length
is fixed when the model is built, the step must be known then: `run_lems()`
takes it from the LEMS file, `from_neuroml()` accepts `delta_t=`. Without it
the delay is dropped and the report says so.

## Independent validation

tarjuman has been validated by **Hengye Zhu** (@HengyeZhu), a NeuroML editor,
using a two-level approach that separates translation correctness from
integrator differences — Jaxley, jnml and jnml-neuron differ in numerical
integration and update ordering, so a single point-wise comparison conflates
the two:

- **channels** against jnml, via LEMS tests generated with NeuroML's
  `NML2ChannelAnalysis`;
- **cells and networks** against independently handwritten Jaxley models built
  from the original NeuroML files.

```bash
python tests/neuroml/run.py                 # all seven models
python tests/neuroml/run.py --model ex25     # just one
python tests/neuroml/run.py --output results # keep the voltage arrays
```
```
ex5:                PASS; max voltage error 3.237e-11 mV; spike deviation 0 ms
ex10:               PASS; max voltage error 1.680e-11 mV; spike deviation 0 ms
ex15:               PASS; max voltage error 1.092e-09 mV; spike deviation 0 ms
ex20a:              PASS; max voltage error 6.055e-11 mV; spike deviation 0 ms
ex24:               PASS; max voltage error 1.375e-11 mV; spike deviation 0 ms
ex25:               PASS; max voltage error 4.015e-11 mV; spike deviation 0 ms
c302_c_oscillator:  PASS; max voltage error 2.548e-12 mV; spike deviation 0 ms
7/7 passed
```

The models are downloaded on first use at pinned upstream commits, with their
licences recorded in `tests/neuroml/models.json`, and the harness enables
float64 — which is what makes agreement at 1e-11 mV meaningful rather than a
float32 artefact. Several real bugs were found and fixed this way; see the
changelog for 0.5.0.

The pass tolerance is 1 nanovolt (`TOLERANCE = 1e-6` mV), relative and
absolute, and a run above 1e-8 mV still passes but says so. That is not
slack — it is portability. The converted and handwritten models are the *same*
computation, so the only difference between them is the order the
floating-point operations end up in, and that is not reproducible across
machines: Apple Silicon contracts fused multiply-adds differently from x86 and
links a different BLAS, so over tens of thousands of integration steps the two
separate by a few parts in 10⁹ on the steep part of a spike. The `ex25` figure
above is 4.0e-11 on x86_64 and 2.2e-9 on an M-series Mac, with the traces
agreeing to nine significant figures either way. Any genuine mistranslation —
a wrong unit, a dropped channel, a misapplied Q10 — moves a trace by 0.1 mV or
more, five orders of magnitude above the bar.

## Validation against the reference simulator

Agreement with analytic references says the equations were implemented
correctly. Agreement with `jnml` says the **model was translated correctly**,
which is what a NeuroML user needs to know.

```bash
pip install "tarjuman[validate]"      # pyneuroml + neuron
tarjuman validate NeuroML2/LEMSexamples/LEMS_NML2_Ex5_DetCell.xml
```
```
PASS  LEMS_NML2_Ex5_DetCell.xml
  ok   v: spikes 7 vs 7, sub-threshold RMS 0.382 mV, spike shift -0.415 ms (drift -0.1139 ms/ISI)
  ok   m: max |diff| 0.739,  RMS 0.08445, r=0.89930
  ok   h: max |diff| 0.1793, RMS 0.02195, r=0.98828
  ok   n: max |diff| 0.2177, RMS 0.02268, r=0.97959
  jlems 4.0s, tarjuman 3.9s (1.0x)
```

It takes several files at once, and the tolerances and the engine are flags:

```bash
tarjuman validate NeuroML2/LEMSexamples/LEMS_NML2_Ex*.xml
tarjuman validate LEMS_NML2_Ex25_MultiComp.xml --engine neuron
tarjuman validate LEMS_hh_single_comp.xml --rms 1.0 --spike-shift 0.5 --dt 0.001
```

Every path in this README is a real file, either bundled with tarjuman or in a
checkout of [NeuroML2](https://github.com/NeuroML/NeuroML2) or
[c302](https://github.com/openworm/c302) beside it.

The reference simulator needs a Java runtime. `pip install pyneuroml` gives you
the `jnml` script but not Java itself, and macOS ships a `java` stub that only
reports Java as missing, so install a real JDK:

```bash
brew install --cask temurin     # macOS
java -version                   # should print a version, not an error
```

The tests that use the reference simulator skip when it is unavailable, and
`tarjuman validate` says what is missing rather than failing obscurely:

```python
>>> from tarjuman.validate import reference_available, reference_problem
>>> reference_available()          # actually invokes jnml, not just a PATH check
True
>>> reference_problem() is None    # None means jLEMS will run
True
```

Measured on the NeuroML example suite:

| model | reference | result |
|---|---|---|
| `Ex5_DetCell` — HH cell, voltage and three gates | jLEMS | spikes 7 vs 7, sub-threshold RMS **0.38 mV** |
| `Ex25_MultiComp` — 3 cells, 12 compartments, AMPA + NMDA | NEURON | all 12 traces, spike counts equal, RMS **0.05–0.48 mV** |
| `Ex19_GapJunctions` — electrical coupling | jLEMS | spikes 3 vs 3, RMS **0.02 mV** |
| `Ex20_AnalogSynapses` — graded transmission | jLEMS | spike counts equal |
| `Ex10_Q10` — temperature-scaled gates | jLEMS | spikes 3 vs 3, RMS **0.93 mV** |
| `Ex0_IaF` — four integrate-and-fire variants | jLEMS | spike counts equal, RMS **0.005–0.022 mV** |
| `Ex8_AdEx` — four AdEx cells, voltage and adaptation | jLEMS | 7 of 8 traces; RMS **0.06–0.24 mV** on the bursting cells, **0.11 mV** on the rebound cell |
| `izhikevich2007Cell` — tonic and bursting | jLEMS | spike counts equal, RMS **0.06** and **0.40 mV**, recovery variable r > 0.995 |
| `c302 parameter set B` — 14 cells, 79 synapses, activity traces | jLEMS | 26 of 28 traces; all 14 activity traces r ≥ **0.991** |

The eighth trace of `Ex8_AdEx` is `adExBurstChaos`, and the name is the
finding. Run jnml against *itself* at two different time steps and it
disagrees by **more** than tarjuman disagrees with it — sub-threshold RMS
1.104 mV and 0.555 ms/ISI drift, against tarjuman's 1.149 mV and
0.517 ms/ISI, both giving 20 spikes. That trace measures the model's
sensitivity to initial conditions, not a translation error.

The two failing traces of `c302 parameter set B` are the pacemaker cells
driving the oscillator, and a convergence study says why: the mean
inter-spike interval difference falls from 0.185 ms at `dt = 0.05` to
**0.017 ms at `dt = 0.005`** — 0.65% down to 0.06%, the same
explicit-versus-implicit difference measured for the HH cell below.

Across the whole suite: of the 31 models, 15 use NeuroML features tarjuman
cannot yet convert and one jLEMS cannot run. **Of the 14 that both simulators
can run, 9 match the reference.** The five that do not are each explained by a
limitation this README already lists — spike sources, short-term plasticity,
and the `iafTauCell` current term — and `tarjuman validate` prints which.

### Why the comparison is spike-aware

Two simulators that agree perfectly still drift apart in spike *timing*,
because they discretise differently. At the rising edge of an action potential
a sub-millisecond shift shows up as a 70 mV difference, so a point-wise RMS
over a spiking trace measures timing jitter rather than correctness. The
comparison therefore separates:

- **spike count** — exact by default, and the check that fails loudest when a
  channel is wrong;
- **spike timing** — a mean shift and a drift per inter-spike interval, which
  distinguishes a harmless constant offset from a frequency difference;
- **sub-threshold RMS** — the voltage away from spikes, where a point-wise
  comparison does mean something.

Spike thresholds come from the model's own `spikeThresh` rather than being
inferred from the trace, and the detection mode is chosen once from the
reference and applied to both traces. Both matter: a compartment receiving only
sub-threshold EPSPs has no spikes, but a trace-derived threshold will sit among
the EPSP peaks and count them; and a cell clamped exactly at its threshold may
register crossings in one simulator and none in the other.

### Is a difference numerical or real?

`tarjuman converge` runs both simulators at a sequence of time steps. jLEMS
integrates explicitly and Jaxley implicitly, so the two approach the exact
solution from opposite sides:

```bash
tarjuman converge tests/data/LEMS_hh_single_comp.xml
tarjuman converge tests/data/LEMS_hh_single_comp.xml --steps 0.01 0.001 0.0005
```
```
convergence of mean inter-spike interval: LEMS_hh_single_comp.xml
   dt (ms)     reference      tarjuman        diff
      0.01      16.16728      16.05303    -0.11425
     0.005      16.09751      16.03911    -0.05840
     0.001      16.04001      16.02810    -0.01191
```

| `dt` (ms) | jLEMS mean ISI | tarjuman |
|---|---|---|
| 0.01 | 16.167 | 16.053 |
| 0.001 | 16.040 | 16.028 |
| 0.0005 | **16.029** | **16.024** |

The 0.7% discrepancy at `dt = 0.01` is entirely jLEMS's discretisation error.
In the converged limit the two agree to **0.03%** — and at any given step
tarjuman is the closer of the two to the converged answer, because Jaxley
solves implicitly with exponential-Euler gates.

### What the reference cannot do

**jLEMS cannot simulate multicompartmental cells at all** — it refuses them
with "Level of support: NONE". For those models `engine="auto"` falls back to
exporting the model to NEURON and running that, which is how the `Ex25` row
above was measured.

That fallback needs NEURON (`pip install neuron`, or `pip install
"tarjuman[validate]"`), and a C++ compiler for it to compile the exported mod
files with. `jnml` does not look on the PATH for NEURON — it reads
`NEURON_HOME` and otherwise tries a fixed list of install prefixes from the
days of source builds — so tarjuman locates NEURON itself and tells `jnml`
where it is. If it cannot, it says so and how to fix it:

```python
>>> from tarjuman.validate import neuron_home, neuron_problem
>>> neuron_home()
PosixPath('.../site-packages/neuron/.data')
>>> neuron_problem() is None       # None means the NEURON engine will run
True
```

Setting `NEURON_HOME` yourself overrides the search. On macOS the compiler
comes from the Xcode command-line tools (`xcode-select --install`); without
them `tarjuman validate` reports the mod-file compilation failure with the
compiler's own error, rather than a Java stack trace.

With that in place a multicompartmental model validates end to end:

```bash
tarjuman validate NeuroML2/LEMSexamples/LEMS_NML2_Ex25_MultiComp.xml
```
```
PASS  LEMS_NML2_Ex25_MultiComp.xml
  ok   of0.0: spikes 12 vs 12, sub-threshold RMS 0.278 mV, spike shift -0.112 ms (drift -0.0207 ms/ISI)
  ok   of0.3: spikes 12 vs 12, sub-threshold RMS 0.067 mV, spike shift -0.046 ms (drift -0.0197 ms/ISI)
  ok   of1.0: spikes  7 vs  7, sub-threshold RMS 0.366 mV, spike shift -0.319 ms (drift +0.0002 ms/ISI)
  ...
  neuron 9.2s, tarjuman 11.2s (0.8x)
```

Three segments, Na/K channels, AMPA and NMDA synapses, 12 recorded traces,
spike counts equal in every one. That is the one check the handwritten-Jaxley
harness cannot make — it compares tarjuman against a model written by hand from
the same NeuroML file, so a misreading shared by both would pass. NEURON read
the NeuroML independently.

## Correctness

The package is also checked against things that do not depend on either
simulator:

- **Hodgkin-Huxley equivalence.** `NML2_SingleCompHHCell.nml` uses the textbook
  HH equations, which `jaxley.channels.HH` implements independently. Converted
  and built-in agree to **< 0.05 mV over 300 ms** of spiking.
- **Analytic membrane time constant** for a passive compartment, within 2%.
- **Integrate-and-fire rate** against `tau * ln((v∞ - reset)/(v∞ - thresh))`,
  within 2%; and `tau = C/g` survives the translation into densities exactly.
- **Calcium pool decay** matches the model's own `decayConstant`, and the pool
  never goes negative.
- **Graded synapse steady state** against `ar·φ / (ar·φ + ad)`.
- **Custom gate `fcond`** against the closed form, including the partial-block
  floor at `1 - alpha`.
- **Synaptic waveform**: one presynaptic spike through `expTwoSynapse` peaks at
  exactly `gbase * weight`.
- Plus gap-junction symmetry, threshold-crossing counting, segment-group
  targeting, and `(segment, fractionAlong)` → `(branch, compartment)` mapping.

## Reproducing everything in this README

Everything above, in order, from a fresh clone. `$NML` is a
[NeuroML2](https://github.com/NeuroML/NeuroML2) checkout and `$C302` a
[c302](https://github.com/openworm/c302) one.

```bash
git clone https://github.com/aslansd/tarjuman && cd tarjuman
pip install -e ".[dev]"
brew install --cask temurin          # macOS: a real JDK for jnml
xcode-select --install               # macOS: a compiler for NEURON's mod files

pytest -q                                              # 145 tests
python tests/neuroml/run.py                            # 7 conversions

python examples/01_run_neuroml_cell.py
python examples/02_fit_a_channel_parameter.py
python examples/03_c302_connectome.py $C302/examples/LEMS_c302_C_Oscillator.xml
python examples/04_validate_against_jnml.py $NML/LEMSexamples/LEMS_NML2_Ex5_DetCell.xml
python examples/05_abstract_cells.py

tarjuman info     $NML/examples/NML2_SingleCompHHCell.nml
tarjuman convert  $NML/examples/NML2_SingleCompHHCell.nml --script hh.py
tarjuman run      tests/data/LEMS_hh_single_comp.xml --output v.dat
tarjuman run      $C302/examples/LEMS_c302_C_Oscillator.xml --plot oscillator.png
tarjuman validate tests/data/LEMS_hh_single_comp.xml
tarjuman validate $NML/LEMSexamples/LEMS_NML2_Ex25_MultiComp.xml   # via NEURON
tarjuman converge tests/data/LEMS_hh_single_comp.xml
tarjuman validate tests/data/LEMS_izhikevich2007.xml
tarjuman validate tests/data/LEMS_adex_rebound.xml
tarjuman validate $NML/LEMSexamples/LEMS_NML2_Ex8_AdEx.xml
tarjuman validate $C302/examples/LEMS_c302_B_Oscillator.xml   # parameter set B
```

The rest of this section is what each of those prints.

### The examples

Four scripts, each self-contained and each printing what it checked.
`03` and `04` take a path; `01` and `02` use bundled fixtures.

```bash

python examples/01_run_neuroml_cell.py
python examples/02_fit_a_channel_parameter.py
python examples/03_c302_connectome.py  c302/examples/LEMS_c302_C_Oscillator.xml
python examples/04_validate_against_jnml.py \
    NeuroML2/LEMSexamples/LEMS_NML2_Ex5_DetCell.xml
python examples/04_validate_against_jnml.py LEMS_NML2_Ex5_DetCell.xml --converge
python examples/05_abstract_cells.py
```

`01` converts a Hodgkin-Huxley cell, prints the branch geometry and writes
`hh_cell.png`:

```
jx.Network
      length    radius  capacitance
0  17.841242  8.920621          1.0
wrote hh_cell.png
```

`02` is the point of the package: it takes a NeuroML sodium conductance,
perturbs it, and recovers it by gradient descent through the converted model.

```
Number of newly added trainable parameters: 1. Total number of trainable parameters: 1
step   0  loss   491.4646  naChans_gbar 0.0600
step  20  loss   184.6497  naChans_gbar 0.1211
step  40  loss    38.3713  naChans_gbar 0.1204
step  50  loss     2.2555  naChans_gbar 0.1205
recovered 0.1196 S/cm2 (NeuroML file says 0.12)
```

`05` runs NeuroML's abstract cells and prints the table that shows where an
AdEx cell's own equation stops being safe for an implicit solver:

```
    v (mV)   feedback*dt   integrated by     v out
     -45.0        0.2373          Jaxley    -45.00
     -43.0        0.6496        the cell    -41.08
     -40.0        2.9205        the cell    -51.00  <- reset
```

`03` converts the C. elegans connectome and reports what is now trainable:

```
tarjuman conversion report for .../LEMS_c302_C_Oscillator.xml
  14 cells, 14 branches, 14 compartments, 79 synapses
wrote c302.png  (28 traces)
calcium pools: ['CaPool']
e.g. model.module.make_trainable('CaPool_decayConstant') to fit the pool to data
```

### The test suites

Two of them, answering different questions.

```bash
pytest -q                           # 145 tests: the package's own behaviour
python tests/neuroml/run.py         # 7 models: conversions against NeuroML
```

```
145 passed
7/7 passed
```

`pytest` needs nothing beyond `[dev]`; the reference-simulator tests inside it
skip cleanly when `jnml` or Java is absent, rather than failing. The NeuroML
harness is Hengye Zhu's and downloads its models on first run.

```bash
pytest -q tests/test_validate.py     # 30 tests: the comparison machinery itself
pytest -q tests/test_lems_and_c302.py
pytest -q -k iaf -v                  # one topic, verbosely
ruff check tarjuman tests            # lint
```

## Design notes

**Units are converted once.** `tarjuman/_units_table.py` is generated from the
canonical `NeuroMLCoreDimensions.xml` by `tools/generate_units_table.py`. The
IR in `tarjuman/ir.py` is already in Jaxley units; readers convert, builders
never do. The one exception is the LEMS interpreter, which works in SI by
design (above).

**Channels are interpreted, not code-generated.** One `NeuroMLChannel` class is
driven by the IR, and every numeric quantity in the NeuroML description is a
Jaxley channel parameter. No generated source to read, and the whole channel is
trainable.

**Events without an event system.** NeuroML synapses are event driven; Jaxley
synapses see only the presynaptic voltage. Each converted synapse keeps the
previous presynaptic voltage as a state and treats an upward crossing of
`spikeThresh` as one event. For an integrate-and-fire presynaptic cell the
threshold used is the cell's own, since it never overshoots.

**Gate initialisation.** NeuroML gates start at steady state (`OnStart`);
`jx.integrate` on its own does not initialise them. `tarjuman.simulate` calls
`init_states` for you — worth knowing if you drive the module yourself.

**Name collisions.** A channel's reversal potential is `{channel}_erev`, not
`{channel}_e`, because Jaxley keeps parameters and states in one table and
NeuroML gates are often called `e` — Boyle & Cohen's calcium activation gate,
for one. Any remaining collision raises rather than silently overwriting.

## Upstream

Two things belong in Jaxley rather than here, and `tarjuman` works around both:

1. `jx.Network(cells)` collects the pumps of its cells but not the list of ion
   concentrations they modify, so the integrator does not solve for them —
   confirmed as a bug in
   [jaxleyverse/jaxley#811](https://github.com/jaxleyverse/jaxley/issues/811)
   and being fixed; tarjuman restores the list meanwhile.
2. ~~`connect()` appends one row per edge~~ — resolved: `connect()` accepts
   views of many compartments and pairs them element-wise, which tarjuman now
   uses (48x faster at 1000 synapses).

## Roadmap

1. Kinetic-scheme channels (`ionChannelKS`).
2. Spike sources (`spikeArray`, `spikeGeneratorPoisson`) as presynaptic drivers.
3. Dimensionless cells: `baseCellMembPotDL` and `basePointCurrentDL`, which
   between them cover `izhikevichCell`, `fitzHughNagumoCell` and the
   dimensionless inputs that drive them.
4. `channelDensityNernst` computed from the pool, and `channelDensityGHK`.
5. A CI job comparing traces against `pynml`/`jnml` across the NeuroML example
   suite and the c302 parameter sets.
6. Validated muscle models, and the rest of the OpenWorm stack.

## Relation to other tools

`tarjuman` does for Jaxley what the NeuroML export machinery does for NEURON,
NetPyNE, EDEN, Brian and MOOSE: it takes the simulator-independent description
and makes it run. The difference is what you get on the other side — a JAX
program you can differentiate, batch and put on an accelerator.

## Credits

tarjuman is written by Aslan Satary Dizaji.

[Hengye Zhu](https://github.com/HengyeZhu) (@HengyeZhu), a NeuroML editor,
contributed the independent validation of the package and the fixes it turned
up — most of release 0.5.0:

- the NeuroML validation suite in `tests/neuroml/`, including the
  `NML2ChannelAnalysis`-generated channel tests and the handwritten Jaxley
  reference models;
- `gateFractional` and `subGate` support (weighted subgate kinetics, inherited
  Q10 scaling, `fopen` recording);
- custom `timeCourse`/`steadyState` components reading the parent gate's
  `alpha`/`beta`, which had silently blocked any such channel;
- the calcium-pool constants fix, without which every shell-based pool raised;
- recording rows resolved against the model structure rather than by position,
  and `iDensity`/`gDensity` with the right scaling and sign;
- NeuroML's synaptic threshold semantics (latched while above threshold,
  rearmed only below, strictly-greater comparison);
- and the observation that a Jaxley channel may return `"v"` from
  `update_states`, which is what made the exact integrate-and-fire reset
  possible.

See the changelog for 0.5.0, where each of these is described as the bug it
fixed.

## License

Apache-2.0. NeuroML 2 is © the NeuroML contributors (LGPL-3.0); Jaxley is ©
the Jaxley developers (Apache-2.0). Test fixtures derived from the NeuroML2 and
c302 repositories are marked as such.
