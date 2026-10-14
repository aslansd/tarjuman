# SPDX-License-Identifier: Apache-2.0
"""Build c302_C_Oscillator from its original NeuroML files using tarjuman."""

import tarjuman


def build_model(models, dt, duration):
    document, simulation = tarjuman.read_lems(
        models / "c302/examples/LEMS_c302_C_Oscillator.xml"
    )
    model = tarjuman.build(document, network_id=simulation.target, delta_t=dt)
    model.apply_stimuli(delta_t=dt, t_max=duration)
    return model.module
