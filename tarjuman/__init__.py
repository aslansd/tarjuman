"""tarjuman — run NeuroML 2 models on Jaxley.

``tarjuman`` (ترجمان / *tərcüman* / *tercüman*: translator, interpreter) reads
NeuroML 2 and LEMS documents and builds the equivalent
`Jaxley <https://jaxley.readthedocs.io>`_ model: morphologies become branches, ``channelDensity`` elements become
inserted channels, projections become synapses, and inputs become stimuli.

Because the result is an ordinary Jaxley module, a NeuroML model converted with
tarjuman is GPU-ready and differentiable: every rate equation parameter of every
channel is exposed as a Jaxley parameter that can be made trainable.

Quick start::

    import tarjuman

    model = tarjuman.from_neuroml("NML2_SingleCompHHCell.nml")
    result = tarjuman.simulate(model, t_max=300.0, delta_t=0.01)
    print(model.report.summary())

or, to reproduce a LEMS simulation exactly as ``pynml`` would run it::

    result, model = tarjuman.run_lems("LEMS_NML2_Ex5_DetCell.xml")
"""

from __future__ import annotations

from ._version import __version__
from .builder import ConvertedModel, build, build_cell, build_network
from .channels import NeuroMLChannel, make_channel
from .errors import (
    ConversionWarning,
    MorphologyError,
    ParseError,
    TarjumanError,
    UnitError,
    UnsupportedComponentError,
)
from .morphology import CellMorphology, build_morphology
from .reader import read_lems, read_neuroml, read_neuroml_string
from .report import ConversionReport
from .simulate import SimulationResult, from_neuroml, run_lems, simulate
from .synapses import make_synapse

__all__ = [
    "__version__",
    # high-level
    "from_neuroml",
    "run_lems",
    "simulate",
    "SimulationResult",
    "ConvertedModel",
    "ConversionReport",
    # reading
    "read_neuroml",
    "read_neuroml_string",
    "read_lems",
    # building
    "build",
    "build_cell",
    "build_network",
    "build_morphology",
    "CellMorphology",
    "make_channel",
    "make_synapse",
    "NeuroMLChannel",
    # errors
    "TarjumanError",
    "ParseError",
    "UnitError",
    "MorphologyError",
    "UnsupportedComponentError",
    "ConversionWarning",
]
