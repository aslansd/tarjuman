# SPDX-License-Identifier: Apache-2.0
"""Build Ex25_MultiComp from its original NeuroML files using tarjuman."""

import tarjuman


def build_model(models, dt, duration):
    document, simulation = tarjuman.read_lems(
        models / "NeuroML2/LEMSexamples/LEMS_NML2_Ex25_MultiComp.xml"
    )
    model = tarjuman.build(document, network_id=simulation.target, delta_t=dt)
    model.apply_stimuli(delta_t=dt, t_max=duration)
    return model.module
