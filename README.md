# tarjuman

**Run NeuroML 2 models on [Jaxley](https://jaxley.readthedocs.io).**

`tarjuman` (ترجمان / *tərcüman* / *tercüman* — translator, interpreter) reads
[NeuroML 2](https://docs.neuroml.org) and LEMS documents and builds the
equivalent Jaxley model. Morphologies become branches, `channelDensity`
elements become inserted channels, projections become synapses, inputs become
stimuli.

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

Optional extras: `pip install tarjuman[neuroml]` to read models straight from
`libNeuroML` objects, `tarjuman[plot]` for the CLI's `--plot`.

## Command line

```bash
tarjuman info    model.nml                    # what is in this file?
tarjuman convert model.nml --script run.py    # convert, report, emit a script
tarjuman run     LEMS_Sim.xml --output v.dat  # convert and simulate
```

`tarjuman run` on a LEMS file reproduces the experiment the file describes —
duration, time step, inputs, and the quantities its `OutputFile` elements ask
for — and writes them in the same SI column format `jnml` writes.

## Python API

```python
import tarjuman

# A whole network, or a single cell.
model = tarjuman.from_neuroml("network.nml", network_id="net1")
model = tarjuman.from_neuroml("cell.nml", cell_id="pyr", ncomp=4)

# `model.module` is an ordinary jaxley.Network / jaxley.Cell.
net = model.module

# Address it in NeuroML terms: population, cell, segment, fraction along.
net_view = model.view("pop0", cell=2, segment=7, fraction_along=0.3)
net_view.record("v")

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

## What gets converted

| NeuroML | Status |
|---|---|
| `ionChannel`, `ionChannelHH`, `ionChannelPassive` | ✅ |
| `gateHHrates`, `gateHHtauInf`, `gateHHratesTau`, `gateHHratesInf`, `gateHHratesTauInf`, `gateHHInstantaneous` | ✅ |
| `HHExpRate`, `HHSigmoidRate`, `HHExpLinearRate` and the matching `*Variable` forms | ✅ |
| `q10Fixed`, `q10ExpTemp`, `q10ConductanceScaling` | ✅ |
| `morphology`: segments, `fractionAlong`, segment groups, paths, spherical somata | ✅ |
| `channelDensity`, `specificCapacitance`, `initMembPotential`, `resistivity`, `spikeThresh` | ✅ |
| `expOneSynapse`, `expTwoSynapse`, `alphaSynapse` | ✅ |
| `blockingPlasticSynapse` with `voltageConcDepBlockMechanism` (NMDA Mg block) | ✅ |
| `gapJunction`, `linearGradedSynapse`, `gradedSynapse`, `silentSynapse` | ✅ |
| `population`, `populationList`, `projection`, `electricalProjection`, `continuousProjection`, `connectionWD` weights | ✅ |
| `pulseGenerator`, `rampGenerator`, `sineGenerator`, `inputList`, `explicitInput` | ✅ |
| LEMS `Simulation`, `OutputFile`, `Display` | ✅ |
| `channelDensityNernst` | ⚠️ needs an explicit `erev` (see `erev_overrides`) |
| Synaptic `delay` | ⚠️ dropped — Jaxley has no delay line |
| `tsodyksMarkram` short-term plasticity | ⚠️ ignored, synapse converted without it |
| `ionChannelKS`, `gateKS`, `gateFractional` | ❌ not yet |
| `channelDensityGHK`, `species` / calcium pools | ❌ not yet |
| Point cells (`iafCell`, `izhikevich2007Cell`, `adExIaFCell`, `pointCellCondBased`, PyNN cells) | ❌ see below |
| Spike sources (`spikeArray`, `spikeGenerator*`, `poissonFiringSynapse`) | ❌ not yet |
| Rates or time courses defined by custom LEMS `ComponentType`s | ❌ needs a LEMS expression evaluator |

Nothing on the ⚠️ or ❌ rows is ever dropped quietly. Every conversion returns
a report:

```python
model.report.is_faithful          # True if nothing was approximated or skipped
model.report.summary()            # human-readable
model.report.unsupported_components
```

and each approximation also raises a `ConversionWarning` at the moment it
happens. `strict=True` turns them into exceptions instead.

## Correctness

The package is checked against things that do not depend on it:

- **Hodgkin-Huxley equivalence.** `NML2_SingleCompHHCell.nml` uses the textbook
  HH rate equations, which are also what `jaxley.channels.HH` implements
  independently. Converted and built-in agree to **< 0.05 mV over 300 ms** of
  spiking.
- **Analytic membrane time constant.** A converted passive compartment relaxes
  with τ = c<sub>m</sub>/g<sub>leak</sub> to within 2%.
- **Synaptic waveform.** One presynaptic spike through a converted
  `expTwoSynapse` produces a peak conductance of exactly `gbase * weight`,
  which is what the NeuroML `waveformFactor` normalisation promises.
- **Gap junction symmetry**, **threshold-crossing counting**, **segment-group
  targeting of channel densities**, and **`(segment, fractionAlong)` →
  `(branch, compartment)` mapping** all have their own tests.

```bash
pytest          # 57 tests
```

## Design notes

**Units are converted once.** `tarjuman/_units_table.py` is generated from the
canonical `NeuroMLCoreDimensions.xml` by `tools/generate_units_table.py`, so
`120.0 mS_per_cm2`, `3.0 S_per_m2` and `0.03 kohm_cm` all arrive correctly in
Jaxley's physiological units (mV, ms, µm, S/cm², µF/cm², Ω·cm, µS, nA). The IR
in `tarjuman/ir.py` is already in Jaxley units; readers convert, builders never
do.

**Channels are interpreted, not code-generated.** One `NeuroMLChannel` class is
driven by the IR, and every numeric quantity in the NeuroML description is a
Jaxley channel parameter. No generated source to read, and the whole channel is
trainable.

**Events without an event system.** NeuroML synapses are event driven; Jaxley
synapses see only the presynaptic voltage. Each converted synapse keeps the
previous presynaptic voltage as a state and treats an upward crossing of
`spikeThresh` as one event, so the conductance waveform matches NeuroML up to
the discretisation of the crossing. A spike narrower than `dt` can be missed;
delays are not representable at all and are reported.

**Gate initialisation.** NeuroML gates start at steady state (`OnStart`);
`jx.integrate` on its own does not initialise them. `tarjuman.simulate` calls
`init_states` for you — worth knowing if you drive the module yourself.

## Roadmap

1. **A small LEMS expression evaluator.** Many real models (ChannelML
   conversions, most of the Open Source Brain library) define rates and time
   courses as custom `ComponentType`s with arbitrary expressions. Parsing those
   expressions into JAX closures is the single largest coverage win available
   and is the next thing to build.
2. **Point cells.** `iafCell` and friends need a state reset on threshold
   crossing, which Jaxley has no mechanism for today. Options: a `Channel` that
   approximates reset with a strong conductance pulse, or a small upstream
   addition to Jaxley. Worth discussing with the Jaxley developers rather than
   hacking around.
3. **Calcium pools and `channelDensityNernst`**, which need a concentration
   state and a Nernst reversal recomputed each step — Jaxley pumps are the
   natural home for this.
4. **Spike sources** (`spikeArray`, `spikeGeneratorPoisson`) as presynaptic
   drivers.
5. **Kinetic-scheme channels** (`ionChannelKS`).
6. **A round-trip test harness** comparing traces against `pynml`/`jnml` on the
   NeuroML example suite, as a CI job.

## Relation to other tools

`tarjuman` does for Jaxley what the NeuroML export machinery does for NEURON,
NetPyNE, EDEN, Brian and MOOSE: it takes the simulator-independent description
and makes it run. The difference is what you get on the other side — a
JAX program you can differentiate, batch and put on an accelerator.

## License

Apache-2.0. NeuroML 2 is © the NeuroML contributors (LGPL-3.0); Jaxley is ©
the Jaxley developers (Apache-2.0). Test fixtures derived from the NeuroML2
repository are marked as such.
