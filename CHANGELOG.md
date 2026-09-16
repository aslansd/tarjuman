# Changelog

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
