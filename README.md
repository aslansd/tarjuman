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

Optional extras: `tarjuman[neuroml]` to read models straight from `libNeuroML`
objects, `tarjuman[plot]` for the CLI's `--plot`, `tarjuman[examples]` for the
example scripts.

## Command line

```bash
tarjuman info    model.nml                    # what is in this file?
tarjuman convert model.nml --script run.py    # convert, report, emit a script
tarjuman run     tests/data/LEMS_hh_single_comp.xml --output v.dat
```

`tarjuman run` on a LEMS file reproduces the experiment the file describes —
duration, time step, inputs, and the quantities its `OutputFile` elements ask
for — and writes them in the same SI column format `jnml` writes.

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
| **B** | a custom integrate-and-fire cell gathering synaptic input with `select="synapses[*]/i"` | ❌ see *Known limits* |
| **C** | `gateHHtauInf`, a `customHGate` reading `caConc`, `fixedFactorConcentrationModel`, `expTwoSynapse`, `gapJunction` | ✅ runs |
| **C0 / C1 / C2** | as C, plus Kunert-style `gradedSynapse2` defined in the model | ✅ runs |
| **D / W2D** | `custom_muscle_components.xml` (`MuscleSigmoidVariable`, `muscleConcentrationModel`) | ⚠️ converts; muscle models not yet validated |

`c302_C_Oscillator` — 14 cells, 79 synapses, a calcium pool per cell and a
calcium-dependent gate — runs 1000 ms in about ten seconds on a CPU.

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
| `izhikevich2007Cell`, `adExIaFCell`, `pointCellCondBased`, PyNN cells | ❌ not yet |
| Spike sources (`spikeArray`, `spikeGenerator*`, `poissonFiringSynapse`) | ❌ not yet |
| `<DerivedVariable select="synapses[*]/i">` in a cell | ❌ see *Known limits* |
| LEMS `<Regime>` (multi-regime dynamics) | ❌ not yet |

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

**Cells that gather their own synaptic input.** A LEMS cell reads its synaptic
current with `<DerivedVariable select="synapses[*]/i" reduce="add"/>`. In
Jaxley that sum is assembled by the network, not read by the cell, so such a
cell type cannot be interpreted as written. This is what blocks c302 parameter
set B. Parsing does not fail — only instantiating that component does, with an
explanation.

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

| examples | passed | max absolute voltage error | spike timing deviation |
| --- | ---: | ---: | ---: |
| Ex5, Ex10, Ex15, Ex20a, Ex24, Ex25 | 6/6 | 1.092e-9 mV | 0 ms |
| c302 C Oscillator | 1/1 | 2.548e-12 mV | 0 ms |

```bash
python tests/neuroml/run.py
```

That harness pins the upstream model commits, records their licences, and runs
in float64 — which is what makes agreement at 1e-9 mV meaningful rather than a
float32 artefact. Several real bugs were found and fixed this way; see the
changelog for 0.5.0.

## Validation against the reference simulator

Agreement with analytic references says the equations were implemented
correctly. Agreement with `jnml` says the **model was translated correctly**,
which is what a NeuroML user needs to know.

```bash
pip install "tarjuman[validate]"      # pyneuroml + neuron
tarjuman validate tests/data/LEMS_hh_single_comp.xml
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
`tarjuman validate` says what is missing.

```
PASS  LEMS_NML2_Ex5_DetCell.xml
  ok   v: spikes 7 vs 7, sub-threshold RMS 0.382 mV, spike shift -0.415 ms (drift -0.1139 ms/ISI)
  ok   m: max |diff| 0.739, RMS 0.08445, r=0.89930
  jlems 2.2s, tarjuman 1.7s (1.3x)
```

Measured on the NeuroML example suite:

| model | reference | result |
|---|---|---|
| `Ex5_DetCell` — HH cell, voltage and three gates | jLEMS | spikes 7 vs 7, sub-threshold RMS **0.38 mV** |
| `Ex25_MultiComp` — 3 cells, 12 compartments, AMPA + NMDA | NEURON | all 12 traces, spike counts equal, RMS **0.05–0.48 mV** |
| `Ex19_GapJunctions` — electrical coupling | jLEMS | spikes 3 vs 3, RMS **0.02 mV** |
| `Ex20_AnalogSynapses` — graded transmission | jLEMS | spike counts equal |
| `Ex10_Q10` — temperature-scaled gates | jLEMS | spikes 3 vs 3, RMS **0.93 mV** |

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

| `dt` (ms) | jLEMS mean ISI | tarjuman |
|---|---|---|
| 0.01 | 16.163 | 16.049 |
| 0.001 | 16.036 | 16.024 |
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

```bash
pytest          # 95 tests
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
3. The remaining point cells. Jaxley already ships `AdEx` and `Izhikevich`
   mechanisms in `jaxley.channels.non_capacitive`, so `adExIaFCell` and
   `izhikevich2007Cell` should map onto them directly rather than needing a
   bespoke implementation.
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

tarjuman is written by Aslan Satary Dizaji. The NeuroML validation suite in
`tests/neuroml/`, the fractional-gate and parent-requirement support, the
calcium-pool and recording fixes, and the synaptic threshold semantics are the
work of [Hengye Zhu](https://github.com/HengyeZhu).

## License

Apache-2.0. NeuroML 2 is © the NeuroML contributors (LGPL-3.0); Jaxley is ©
the Jaxley developers (Apache-2.0). Test fixtures derived from the NeuroML2 and
c302 repositories are marked as such.
