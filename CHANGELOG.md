# Changelog

## 0.3.0 — unreleased

Acts on the Jaxley developers' answers in
[jaxleyverse/jaxley#810](https://github.com/jaxleyverse/jaxley/issues/810).

### Synaptic delays are now implemented

NeuroML connection `delay`s used to be dropped. They are now built as shift
registers over the presynaptic voltage, following the approach Jonas Beck
suggested upstream: `DelayedSynapse` wraps any synapse and hands it the
voltage from `delay_steps` integration steps ago, so threshold detection and
graded transmission are delayed with it.

Delays are quantised to whole steps, so the step has to be known when the model
is built: `run_lems()` takes it from the LEMS file, and `from_neuroml()` accepts
`delta_t`. Without it the delay is still dropped, and the report now says why
and how to fix it.

### Connections are made in batches

`connect()` accepts views of many compartments and pairs them element-wise, so
a whole projection is now one call instead of one per connection. Measured on a
synthetic network, 1000 synapses went from 21.7 s to 0.45 s — a 48x speedup,
and the gap widens with size, since the one-by-one path is quadratic.
Converting c302's full connectome (397 cells, 4999 synapses) now takes about
13 s.

Connections are grouped by synapse component and delay; per-connection weights
are applied in groups too.

### Other

- The `pumped_ions` workaround is annotated with jaxleyverse/jaxley#811, where
  it is confirmed as a bug; it becomes a no-op once the fix lands.

## 0.2.1 — unreleased

- Unresolved `<include>` / `<Include>` elements are now reported instead of
  skipped quietly. A LEMS file only points at the NeuroML files it includes,
  and those paths are relative to it, so running one away from its siblings
  used to produce a baffling "the document defines no network and 0 cells".
  The error now names the files it could not find, says where it looked, and
  explains that the LEMS file should be run where it lives. Core NeuroML type
  definitions (`Cells.xml`, `Channels.xml`, …) are still skipped silently,
  since tarjuman implements them natively.

## 0.2.0 — unreleased

Built against the OpenWorm models: c302 parameter sets A, C, C0, C1 and C2 now
convert and run.

### A LEMS ComponentType interpreter

Real NeuroML models define their own component types; every c302 parameter set
does. `tarjuman.lems` now parses `<ComponentType>` definitions, resolves
inheritance, and compiles `<Dynamics>` into JAX:

- an expression parser for the LEMS infix syntax (`^` right-associative, the
  `.gt.`/`.and.` word operators, the LEMS function set), compiled to closures;
- `StateVariable`, `DerivedVariable`, `ConditionalDerivedVariable`,
  `TimeDerivative`, `OnStart`, `OnCondition`, `OnEvent`;
- state variables advanced by exponential Euler with local linearisation;
- evaluation in SI, converted at the boundary from the dimensions the
  ComponentType declares.

Custom types can act as gates (including a custom `fcond`), as the rate,
steady state or time course inside a standard gate, as concentration models,
and as synapses, where `select="peer/v"` resolves to the presynaptic voltage.

### Ion concentrations

- `<species>` plus `fixedFactorConcentrationModel` /
  `decayingPoolConcentrationModel` (and custom pools) become Jaxley pumps.
- Channels of a pooled ion share a current name, so a pool sees the total
  current of that ion as NeuroML's `iCa` requirement expects.
- Gates can depend on `caConc`; `caConc` can be recorded from LEMS.

### Point cells

- `iafCell`, `iafRefCell`, `iafTauCell` and `iafTauRefCell` become
  single-compartment cells with area-scaled densities; the threshold reset is
  approximated with a large conductance towards the reset potential and
  reported as an approximation.
- Presynaptic spike detection uses a point cell's own threshold.

### Fixes

- A channel's reversal potential is now `{channel}_erev`. It was `{channel}_e`,
  which collided with a gate named `e` (Boyle & Cohen's calcium activation
  gate) — Jaxley keeps parameters and states in one table, so the two
  overwrote each other and flipped the sign of the calcium current. Any
  remaining collision now raises.
- Point-cell conductances were read as nS rather than µS, a 1000x error.
- Duplicate LEMS `OutputColumn` ids (c302 names both the voltage and the
  calcium column of a cell `AVBL_v`) no longer overwrite each other.
- `<DerivedVariable select=...>` and `<Regime>` no longer fail at parse time;
  they fail only if the component is instantiated, with an explanation.
- Included files are read before the including file, so a channel can use a
  ComponentType defined in an include.
- `simulate(..., apply_stimuli=False)` no longer deletes stimuli the caller set.

### Workarounds for upstream Jaxley gaps

- `jx.Network(cells)` does not carry over the pumped-ion list from its cells;
  tarjuman restores it so the integrator solves for the concentrations.

## 0.1.0 — unreleased

First working version.

- NeuroML 2 and LEMS readers (standard library only; optional `libNeuroML` bridge).
- Units generated from the canonical `NeuroMLCoreDimensions.xml`.
- `ionChannel`/`ionChannelHH`/`ionChannelPassive` with all six HH gate types,
  the three rate/variable forms, and `q10` scaling, translated into Jaxley
  channels whose every rate parameter is trainable.
- Morphology mapping: segment trees and NeuroML sections to Jaxley branches,
  `numberInternalDivisions`, `max_comp_length`, `(segment, fractionAlong)` to
  `(branch, compartment)`.
- Chemical, electrical (gap junction) and continuous (graded) projections;
  `expOne`/`expTwo`/`alpha` synapses and the NMDA Mg block.
- `pulseGenerator`, `rampGenerator`, `sineGenerator`, `inputList`, `explicitInput`.
- `tarjuman info | convert | run` command line, and LEMS `OutputFile` replay.
- Conversion reports: nothing is approximated or dropped silently.
