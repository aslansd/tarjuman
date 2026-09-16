"""Fit a NeuroML channel parameter by gradient descent.

This is the thing that is awkward with jnml or NEURON and easy on Jaxley: the
converted model is a JAX program, so the sodium conductance of a NeuroML
``channelDensity`` — or the midpoint of its activation curve — can be
optimised against a target trace with ordinary gradient descent.

    python examples/02_fit_a_channel_parameter.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
import numpy as np
import optax

import tarjuman

MODEL = Path(__file__).parent.parent / "tests" / "data" / "hh_single_comp.nml"
DT, T_MAX = 0.025, 100.0
TRUE_GBAR = 0.12  # S/cm2, the value in the NeuroML file


def build():
    model = tarjuman.from_neuroml(MODEL, network_id="net1")
    module = model.module
    module.delete_recordings()
    module.delete_stimuli()
    module.cell(0).branch(0).comp(0).record("v", verbose=False)
    module.cell(0).branch(0).comp(0).stimulate(
        jx.step_current(10.0, 80.0, 0.1, DT, T_MAX), verbose=False
    )
    module.init_states(delta_t=DT)
    return model


def main() -> None:
    model = build()
    module = model.module

    # A target trace from the model as the NeuroML file specifies it.
    target = jnp.asarray(jx.integrate(module, delta_t=DT, t_max=T_MAX))

    # Start from a wrong sodium conductance and recover it.
    module.make_trainable("naChans_gbar", 0.05)
    parameters = module.get_parameters()

    def loss(parameters):
        voltages = jx.integrate(module, params=parameters, delta_t=DT, t_max=T_MAX)
        return jnp.mean((voltages - target) ** 2)

    gradient_fn = jax.jit(jax.value_and_grad(loss))
    optimiser = optax.adam(learning_rate=0.01)
    state = optimiser.init(parameters)

    history = []
    for step in range(60):
        value, gradients = gradient_fn(parameters)
        updates, state = optimiser.update(gradients, state)
        parameters = optax.apply_updates(parameters, updates)
        gbar = float(list(parameters[0].values())[0].ravel()[0])
        history.append((step, float(value), gbar))
        if step % 10 == 0:
            print(f"step {step:3d}  loss {value:10.4f}  naChans_gbar {gbar:.4f}")

    steps, losses, gbars = zip(*history)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
    left.plot(steps, losses, color="black")
    left.set_xlabel("step")
    left.set_ylabel("MSE (mV$^2$)")
    left.set_yscale("log")
    right.plot(steps, gbars, color="black")
    right.axhline(TRUE_GBAR, linestyle="--", color="tab:red", label="NeuroML value")
    right.set_xlabel("step")
    right.set_ylabel("naChans_gbar (S/cm$^2$)")
    right.legend()
    figure.suptitle("Fitting a NeuroML channelDensity through Jaxley")
    figure.tight_layout()
    figure.savefig("fit_channel.png", dpi=150)
    print(f"recovered {gbars[-1]:.4f} S/cm2 (NeuroML file says {TRUE_GBAR})")
    print("wrote fit_channel.png")


if __name__ == "__main__":
    main()
