# Changelog

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
