# Changelog

## 0.5.1

Two problems a user hit on a Mac that neither test suite could see from Linux.

### `tests/neuroml/run.py` compared float64 traces to an absolute 2 pV

The converted and handwritten models are the same computation, so the only
difference between them is the order the floating-point operations end up in,
and that is not reproducible across machines: Apple Silicon contracts fused
multiply-adds differently from x86 and links a different BLAS. Over tens of
thousands of integration steps the two traces separate by a few parts in 10^9
on the steep part of a spike. `Ex25` therefore failed on macOS at 2.16e-9 mV
against a 2e-9 tolerance, while the same run on x86_64 gives 4.0e-11 — and the
19 samples it failed on agreed to nine significant figures.

The tolerance is now 1e-6 mV, relative and absolute: one nanovolt. Any genuine
mistranslation — a wrong unit, a dropped channel, a misapplied Q10 — moves a
trace by 0.1 mV or more, so this keeps five orders of magnitude of margin for
catching real regressions. Runs above 1e-8 mV still pass but say so, because a
jump from 1e-11 to 1e-7 means something changed even if it is not yet wrong.

### `validate --engine neuron` could not find a pip-installed NEURON

`jnml` does not search the PATH for NEURON. It reads `NEURON_HOME` and
otherwise tries a fixed list of prefixes from the days of source builds
(`/usr/local/nrn/x86_64` and friends). A pip-installed NEURON matches none of
them, so `pip install neuron` ended in the bare Java error `Could not find
NEURON home directory!` while `nrniv` sat on the PATH all along — which is
every multicompartmental model, since jLEMS refuses those outright.

`neuron_home()` now finds it, in the wheel (`neuron/.data`) or from the PATH,
and `run_reference` passes it to `jnml`. Two further wheel quirks had to be
handled to get an actual run, both of which NEURON's own PATH wrappers paper
over and `jnml` bypasses by calling into `$NEURON_HOME/bin` directly:

- the wheel's `nrngui` and `nrnivmodl` shell scripts still carry the temporary
  prefix they were built under and fall back to it unless `NRNHOME` is set;
- the wheel's makefile hardcodes the compiler of the build image
  (`/opt/rh/gcc-toolset-10/.../g++`), so `CC` and `CXX` are now set to a
  compiler that exists.

With all three, `LEMS_NML2_Ex25_MultiComp` validates against NEURON and
passes: 12 recorded traces across a three-segment cell with Na/K channels and
AMPA/NMDA synapses, spike counts matching exactly, sub-threshold RMS 0.05–0.48
mV and spike shifts under 0.35 ms. That is the first multicompartmental
conversion checked against NEURON rather than against a handwritten Jaxley
model.

A NEURON that still will not run is now explained — `neuron_problem()`, in the
shape of `reference_problem()` — along with a mod-file compilation failure,
rather than being passed through as a Java stack trace.


## 0.5.0

Most of this release is the work of **Hengye Zhu** (@HengyeZhu), a NeuroML
editor, who validated tarjuman against the NeuroML reference tooling, found
several real bugs, and fixed them. Their commit is merged here essentially as
written; the validation harness they wrote lives in `tests/neuroml/`.

### Validation (Hengye Zhu)

A two-level approach, which is the right one given that Jaxley, jnml and
jnml-neuron differ in numerical integration and update ordering:

- **channels** against jnml, via LEMS tests generated with NeuroML's
  `NML2ChannelAnalysis`;
- **cells and networks** against independently handwritten Jaxley models built
  from the original NeuroML files, which isolates *translation* correctness
  from integrator differences.

| examples | passed | max absolute voltage error | spike timing deviation |
| --- | ---: | ---: | ---: |
| Ex5, Ex10, Ex15, Ex20a, Ex24, Ex25 | 6/6 | 1.092e-9 mV | 0 ms |
| c302 C Oscillator | 1/1 | 2.548e-12 mV | 0 ms |

Run with `python tests/neuroml/run.py`. The harness pins the upstream model
commits, records their licences, and enables float64 — which is what makes
agreement at 1e-9 mV meaningful rather than a float32 artefact.

### Fixed (Hengye Zhu)

- **Fractional conductance gates.** `gateFractional` and its `subGate`
  children were not implemented: subgate kinetics, their weighted combination,
  the Q10 scale inherited from the parent gate, and `fopen` recording. This is
  `Ex24`.
- **Parent-gate requirements.** A custom `timeCourse` or `steadyState` may
  declare `alpha`/`beta` as LEMS requirements and read them from its parent
  gate. Those were never supplied, so any such channel failed to convert. With
  the unit conversion for custom time constants, this is `Ex15`.
- **Calcium-pool constants.** `NeuroMLConcentrationModel.compute_current`
  rebuilt its evaluation context by hand and left out the ComponentType's
  constants, so `decayingPoolConcentrationModel` — which needs `Faraday`,
  `AREA_SCALE` and `LENGTH_SCALE` — raised `ParseError` on every call. Any
  shell-based calcium pool was unusable.
- **Recording rows.** Jaxley deduplicates recordings, so two LEMS output
  columns that resolve to the same compartment and state share one row.
  Mapping traces by position therefore misassigned them. Output paths are now
  resolved against the model structure, and `iDensity`/`gDensity` are
  recordable with the right unit scaling and sign.
- **Synaptic threshold boundaries.** Spike detection now latches while the
  presynaptic voltage is above threshold and rearms only once it falls back
  below, with NeuroML's strictly-greater comparison. The previous rule
  compared against the previous sample, which is not what
  `OnCondition test="v .gt. thresh"` means.

### Integrate-and-fire reset is now exact

Earlier releases approximated NeuroML's reset with a large conductance towards
the reset potential, on the stated premise that Jaxley has no state-assignment
mechanism. That premise was wrong, as @HengyeZhu pointed out: a non-capacitive
channel may return `"v"` from `update_states` and Jaxley takes it, which is how
Jaxley's own `Leak` + `Fire` pair works.

The reset is therefore a state assignment now, exactly as NeuroML specifies.
The measured inter-spike interval moves from 0.1% to **0.027%** of the analytic
value, the voltage lands exactly on the reset potential instead of converging
towards it, and the conversion no longer reports an approximation. The
refractory period is handled in the same channel, which is why this is not
simply `Leak` + `Fire`: `Fire` has no notion of one.

The gradient through a spike is zero, since a reset writes the voltage and a
threshold crossing is a step. Sub-threshold parameters remain differentiable.
Jaxley's `Fire` carries the same caveat for the same reason.

## 0.4.2

- Every path in the README and the examples is now a real file, either bundled
  with tarjuman or in a NeuroML2 / c302 checkout beside it. The old
  `LEMS_Sim.xml` was a placeholder that looked like something to paste.
- A missing LEMS file is reported once, as a missing file, instead of being
  rediscovered by each reference engine and reported twice.

## 0.4.1

- `reference_available()` now checks that `jnml` actually runs, not merely that
  the script is on the PATH. `pip install pyneuroml` installs that script with
  or without a Java runtime, and macOS ships a `java` stub whose only purpose
  is to report that Java is missing — so a machine without a JDK got two
  confusing test failures where it should have got two skips. A new
  `reference_problem()` says what is wrong and how to fix it, and
  `tarjuman validate` prints it.

## 0.4.0

Validation against the reference NeuroML simulator, answering the question the
test suite could not: agreement with analytic references says the equations
were implemented correctly, agreement with `jnml` says the *model* was
translated correctly.

### `tarjuman validate`

A new module, `tarjuman.validate`, runs both simulators on the same LEMS file
and compares the traces; `tarjuman validate` and `tarjuman converge` expose it,
and a GitHub Actions workflow runs it on every push.

Comparison is spike-aware rather than point-wise. Two simulators that agree
perfectly still drift in spike *timing* because they discretise differently,
and at the rising edge of an action potential a sub-millisecond shift reads as
a 70 mV error, so a plain RMS measures jitter rather than correctness. The
comparison separates spike count, spike timing (as a mean shift and a drift per
inter-spike interval) and the sub-threshold voltage.

Two engines are supported. jLEMS is the most direct reading of the
specification but **cannot simulate multicompartmental cells** — it refuses
them outright — so `engine="auto"` falls back to NEURON for those.

`convergence_study` settles whether a difference is numerical: jLEMS integrates
explicitly and Jaxley implicitly, so they approach the exact solution from
opposite sides, and a difference that shrinks as `dt` falls is discretisation
rather than disagreement.

### Fixed

- A parameter's dimension now comes from the unit symbol the model wrote, not
  from a table keyed on the attribute's name. NeuroML reuses short names across
  component types with different dimensions — `k` is a rate in `gradedSynapse`
  and a conductance slope in `gradedSynapse2` — so the old table rejected valid
  models (`Unit 'nS_per_mV' has dimension 'conductance_per_voltage' but
  'per_time' was expected`). The table is kept only for attributes written as
  bare numbers, where there is no unit to read.
- Spike thresholds for validation are taken from the model's own `spikeThresh`
  (or a point cell's `thresh`) rather than inferred from the trace. A
  compartment receiving only sub-threshold EPSPs has no spikes, but a
  trace-derived threshold lands among the EPSP peaks and counts them.
- A gate part supplied as a custom ComponentType is no longer treated as
  missing. A `gateHHrates` whose `forwardRate` was model-defined was rejected
  as "missing forwardRate", so the whole channel was dropped and the cell ran
  without that current. Found by the reference comparison, which reported the
  wrong spike count on `LEMS_NML2_Ex10_Q10`; no unit test could have seen it,
  because the conversion reported the channel as unsupported and carried on.
- `iafTauCell` and `iafTauRefCell` now report that NeuroML gives them no
  current term at all: they extend `baseIaf`, not `baseIafCapCell`, so an
  injected current does nothing to them in NeuroML, while a Jaxley compartment
  cannot ignore its input. The free decay is exact; the response to injected
  current is not, and the conversion report says so.

## 0.3.0

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

## 0.2.1

- Unresolved `<include>` / `<Include>` elements are now reported instead of
  skipped quietly. A LEMS file only points at the NeuroML files it includes,
  and those paths are relative to it, so running one away from its siblings
  used to produce a baffling "the document defines no network and 0 cells".
  The error now names the files it could not find, says where it looked, and
  explains that the LEMS file should be run where it lives. Core NeuroML type
  definitions (`Cells.xml`, `Channels.xml`, …) are still skipped silently,
  since tarjuman implements them natively.

## 0.2.0

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

## 0.1.0

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
