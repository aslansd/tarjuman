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
tarjuman run     LEMS_Sim.xml --output v.dat  # convert and simulate
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
result, model = tarjuman.run_lems("LEMS_Sim.xml")
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
| `iafCell`, `iafRefCell`, `iafTauCell`, `iafTauRefCell` | ⚠️ reset approximated, see below |
| `population`, `populationList`, `projection`, `electricalProjection`, `continuousProjection`, weights | ✅ |
| `pulseGenerator`, `rampGenerator`, `sineGenerator`, `inputList`, `explicitInput` | ✅ |
| LEMS `Simulation`, `OutputFile`, `Display` | ✅ |
| `channelDensityNernst` | ⚠️ needs an explicit `erev` (`erev_overrides`) |
| Synaptic `delay` | ⚠️ dropped — Jaxley has no delay line |
| `tsodyksMarkram` short-term plasticity | ⚠️ ignored, synapse converted without it |
| `ionChannelKS`, `gateKS`, `gateFractional` | ❌ not yet |
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

**Integrate-and-fire reset.** NeuroML assigns the voltage on threshold
crossing; Jaxley has no state-assignment mechanism. `tarjuman` switches on a
large conductance towards the reset potential for one step (and any refractory
period). The firing rate comes out right — measured ISI 20.77 ms against an
analytic 20.79 ms — but the reset is an approximation and is reported as one.

**Cells that gather their own synaptic input.** A LEMS cell reads its synaptic
current with `<DerivedVariable select="synapses[*]/i" reduce="add"/>`. In
Jaxley that sum is assembled by the network, not read by the cell, so such a
cell type cannot be interpreted as written. This is what blocks c302 parameter
set B. Parsing does not fail — only instantiating that component does, with an
explanation.

**Synaptic delays** have no representation in Jaxley at all.

## Correctness

The package is checked against things that do not depend on it:

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
pytest          # 88 tests
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
   concentrations they modify, so the integrator does not solve for them.
2. `connect()` appends one row per edge, which is slow for connectome-scale
   networks (c302's full model has ~5000 synapses).

## Roadmap

1. Kinetic-scheme channels (`ionChannelKS`).
2. Spike sources (`spikeArray`, `spikeGeneratorPoisson`) as presynaptic drivers.
3. The remaining point cells (`izhikevich2007Cell`, `adExIaFCell`), which need
   the same reset treatment plus their own recovery variables.
4. `channelDensityNernst` computed from the pool, and `channelDensityGHK`.
5. A CI job comparing traces against `pynml`/`jnml` across the NeuroML example
   suite and the c302 parameter sets.
6. Validated muscle models, and the rest of the OpenWorm stack.

## Relation to other tools

`tarjuman` does for Jaxley what the NeuroML export machinery does for NEURON,
NetPyNE, EDEN, Brian and MOOSE: it takes the simulator-independent description
and makes it run. The difference is what you get on the other side — a JAX
program you can differentiate, batch and put on an accelerator.

## License

Apache-2.0. NeuroML 2 is © the NeuroML contributors (LGPL-3.0); Jaxley is ©
the Jaxley developers (Apache-2.0). Test fixtures derived from the NeuroML2 and
c302 repositories are marked as such.
