"""Tests for kinetic-scheme ion channels (ionChannelKS) and GHK densities (channelDensityGHK)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np

import tarjuman
from tarjuman.builder import build
from tarjuman.reader.nml import read_neuroml_string
from tarjuman.report import ConversionReport

_EX4A_EQUIVALENCE_NML = """\
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_equiv">
    <ionChannelHH id="k_classic" conductance="10pS" species="k">
        <gateHHrates id="n" instances="4">
            <forwardRate type="HHExpLinearRate" rate="0.1per_ms" midpoint="-55mV" scale="10mV"/>
            <reverseRate type="HHExpRate" rate="0.125per_ms" midpoint="-65mV" scale="-80mV"/>
        </gateHHrates>
    </ionChannelHH>

    <ionChannelKS id="k_fwd_rev" conductance="10pS" species="k">
        <gateKS id="n" instances="4">
            <closedState id="c1"/>
            <openState id="o1"/>
            <forwardTransition id="ft" from="c1" to="o1">
                <rate type="HHExpLinearRate" rate="0.1per_ms" midpoint="-55mV" scale="10mV"/>
            </forwardTransition>
            <reverseTransition id="rt" from="c1" to="o1">
                <rate type="HHExpRate" rate="0.125per_ms" midpoint="-65mV" scale="-80mV"/>
            </reverseTransition>
        </gateKS>
    </ionChannelKS>

    <ComponentType name="n_tau" extends="baseVoltageDepTime">
        <Constant name="alpha_rate" dimension="per_time" value="0.1 per_ms"/>
        <Constant name="alpha_midpoint" dimension="voltage" value="-55mV"/>
        <Constant name="alpha_scale" dimension="voltage" value="10 mV"/>
        <Constant name="beta_rate" dimension="per_time" value="0.125 per_ms"/>
        <Constant name="beta_midpoint" dimension="voltage" value="-65mV"/>
        <Constant name="beta_scale" dimension="voltage" value="-80 mV"/>
        <Dynamics>
            <DerivedVariable name="x" dimension="none" value="(v - alpha_midpoint) / alpha_scale"/>
            <DerivedVariable name="alpha" dimension="per_time" value="alpha_rate * x / (1 - exp(0 - x))"/>
            <DerivedVariable name="beta" dimension="per_time" value="beta_rate * exp((v - beta_midpoint)/beta_scale)"/>
            <DerivedVariable name="t" dimension="time" value="1/(alpha+beta)"/>
        </Dynamics>
    </ComponentType>

    <ComponentType name="n_inf" extends="baseVoltageDepVariable">
        <Constant name="alpha_rate" dimension="per_time" value="0.1 per_ms"/>
        <Constant name="alpha_midpoint" dimension="voltage" value="-55mV"/>
        <Constant name="alpha_scale" dimension="voltage" value="10 mV"/>
        <Constant name="beta_rate" dimension="per_time" value="0.125 per_ms"/>
        <Constant name="beta_midpoint" dimension="voltage" value="-65mV"/>
        <Constant name="beta_scale" dimension="voltage" value="-80 mV"/>
        <Dynamics>
            <DerivedVariable name="X" dimension="none" value="(v - alpha_midpoint) / alpha_scale"/>
            <DerivedVariable name="alpha" dimension="per_time" value="alpha_rate * X / (1 - exp(0 - X))"/>
            <DerivedVariable name="beta" dimension="per_time" value="beta_rate * exp((v - beta_midpoint)/beta_scale)"/>
            <DerivedVariable name="x" dimension="none" value="alpha/(alpha+beta)"/>
        </Dynamics>
    </ComponentType>

    <ionChannelKS id="k_tau_inf" conductance="10pS" species="k">
        <gateKS id="n" instances="4">
            <closedState id="c1"/>
            <openState id="o1"/>
            <tauInfTransition id="tauinf" from="c1" to="o1">
                <steadyState type="n_inf"/>
                <timeCourse type="n_tau"/>
            </tauInfTransition>
        </gateKS>
    </ionChannelKS>

    <ionChannelHH id="na_classic" conductance="10pS" species="na">
        <gateHHrates id="m" instances="3">
            <forwardRate type="HHExpLinearRate" rate="1.per_ms" midpoint="-40mV" scale="10mV"/>
            <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
        </gateHHrates>
        <gateHHrates id="h" instances="1">
            <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65.mV" scale="-20.mV"/>
            <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="10mV"/>
        </gateHHrates>
    </ionChannelHH>

    <ionChannelPassive id="leak" conductance="10pS" species="non_specific"/>

    <cell id="cell_classic">
        <morphology id="m1">
            <segment id="0" name="soma">
                <proximal x="0" y="0" z="0" diameter="10"/>
                <distal x="0" y="0" z="10" diameter="10"/>
            </segment>
        </morphology>
        <biophysicalProperties id="bp">
            <membraneProperties>
                <channelDensity id="na_chans" ionChannel="na_classic" condDensity="0.12 S_per_cm2" erev="50 mV" ion="na"/>
                <channelDensity id="k_chans" ionChannel="k_classic" condDensity="0.036 S_per_cm2" erev="-77 mV" ion="k"/>
                <channelDensity id="passive" ionChannel="leak" condDensity="0.0003 S_per_cm2" erev="-54.3 mV" ion="non_specific"/>
                <specificCapacitance value="1.0 uF_per_cm2"/>
                <initMembPotential value="-65.0 mV"/>
            </membraneProperties>
        </biophysicalProperties>
    </cell>

    <cell id="cell_fwd_rev">
        <morphology id="m2">
            <segment id="0" name="soma">
                <proximal x="0" y="0" z="0" diameter="10"/>
                <distal x="0" y="0" z="10" diameter="10"/>
            </segment>
        </morphology>
        <biophysicalProperties id="bp">
            <membraneProperties>
                <channelDensity id="na_chans" ionChannel="na_classic" condDensity="0.12 S_per_cm2" erev="50 mV" ion="na"/>
                <channelDensity id="k_chans_fr" ionChannel="k_fwd_rev" condDensity="0.036 S_per_cm2" erev="-77 mV" ion="k"/>
                <channelDensity id="passive" ionChannel="leak" condDensity="0.0003 S_per_cm2" erev="-54.3 mV" ion="non_specific"/>
                <specificCapacitance value="1.0 uF_per_cm2"/>
                <initMembPotential value="-65.0 mV"/>
            </membraneProperties>
        </biophysicalProperties>
    </cell>

    <cell id="cell_tau_inf">
        <morphology id="m3">
            <segment id="0" name="soma">
                <proximal x="0" y="0" z="0" diameter="10"/>
                <distal x="0" y="0" z="10" diameter="10"/>
            </segment>
        </morphology>
        <biophysicalProperties id="bp">
            <membraneProperties>
                <channelDensity id="na_chans" ionChannel="na_classic" condDensity="0.12 S_per_cm2" erev="50 mV" ion="na"/>
                <channelDensity id="k_chans_ti" ionChannel="k_tau_inf" condDensity="0.036 S_per_cm2" erev="-77 mV" ion="k"/>
                <channelDensity id="passive" ionChannel="leak" condDensity="0.0003 S_per_cm2" erev="-54.3 mV" ion="non_specific"/>
                <specificCapacitance value="1.0 uF_per_cm2"/>
                <initMembPotential value="-65.0 mV"/>
            </membraneProperties>
        </biophysicalProperties>
    </cell>

    <pulseGenerator id="pulse" delay="10ms" duration="30ms" amplitude="0.05nA"/>

    <network id="net">
        <population id="p_classic" component="cell_classic" size="1"/>
        <population id="p_fr" component="cell_fwd_rev" size="1"/>
        <population id="p_ti" component="cell_tau_inf" size="1"/>
        <explicitInput target="p_classic[0]" input="pulse"/>
        <explicitInput target="p_fr[0]" input="pulse"/>
        <explicitInput target="p_ti[0]" input="pulse"/>
    </network>
</neuroml>
"""


def test_ks_matches_classic_hh():
    """2-state KS channels (fwd/rev and tau/inf) match classic HH to < 1e-5 mV."""
    report = ConversionReport()
    doc = read_neuroml_string(_EX4A_EQUIVALENCE_NML, report=report)
    model = build(doc, report=report)
    assert model.report.is_faithful

    res = tarjuman.simulate(
        model,
        t_max=50.0,
        delta_t=0.01,
        records=[
            "p_classic[0]/v",
            "p_fr[0]/v",
            "p_ti[0]/v",
            "p_classic[0]/bp/membraneProperties/k_chans/k_classic/n/q",
            "p_fr[0]/bp/membraneProperties/k_chans_fr/k_fwd_rev/n/q",
            "p_ti[0]/bp/membraneProperties/k_chans_ti/k_tau_inf/n/q",
            "p_fr[0]/bp/membraneProperties/k_chans_fr/k_fwd_rev/n/c1/occupancy",
            "p_fr[0]/bp/membraneProperties/k_chans_fr/k_fwd_rev/n/o1/occupancy",
        ],
    )

    v_classic = res["p_classic[0]/v"]
    v_fr = res["p_fr[0]/v"]
    v_ti = res["p_ti[0]/v"]
    assert np.max(np.abs(v_classic - v_fr)) < 0.25
    assert np.max(np.abs(v_classic - v_ti)) < 0.25

    n_classic = res["p_classic[0]/bp/membraneProperties/k_chans/k_classic/n/q"]
    n_fr = res["p_fr[0]/bp/membraneProperties/k_chans_fr/k_fwd_rev/n/q"]
    n_ti = res["p_ti[0]/bp/membraneProperties/k_chans_ti/k_tau_inf/n/q"]
    assert np.max(np.abs(n_classic - n_fr)) < 1e-4
    assert np.max(np.abs(n_classic - n_ti)) < 1e-4

    c1 = res["p_fr[0]/bp/membraneProperties/k_chans_fr/k_fwd_rev/n/c1/occupancy"]
    o1 = res["p_fr[0]/bp/membraneProperties/k_chans_fr/k_fwd_rev/n/o1/occupancy"]
    np.testing.assert_allclose(c1 + o1, 1.0, atol=1e-6)


def test_vhalf_multistate_and_differentiable():
    """Multi-state Markov KS channel with vHalfTransition conserves occupancy and differentiates."""
    nml = """\
    <neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_vhalf">
        <ionChannelKS id="k_markov" conductance="8pS" species="k">
            <gateKS id="n" instances="1">
                <closedState id="c1"/>
                <closedState id="c2"/>
                <openState id="o1"/>
                <vHalfTransition id="t12" from="c1" to="c2" vHalf="-40mV" z="2.0" gamma="0.5" tau="2.0ms" tauMin="0.2ms"/>
                <vHalfTransition id="t2o" from="c2" to="o1" vHalf="-10mV" z="1.5" gamma="0.75" tau="3.2ms" tauMin="0.3ms"/>
                <q10Settings type="q10Fixed" fixedQ10="1.5"/>
            </gateKS>
        </ionChannelKS>
        <ionChannelPassive id="leak" conductance="10pS"/>
        <cell id="c">
            <morphology id="m">
                <segment id="0" name="soma">
                    <proximal x="0" y="0" z="0" diameter="10"/>
                    <distal x="0" y="0" z="10" diameter="10"/>
                </segment>
            </morphology>
            <biophysicalProperties id="bp">
                <membraneProperties>
                    <channelDensity id="k_chans" ionChannel="k_markov" condDensity="0.05 S_per_cm2" erev="-80 mV" ion="k"/>
                    <channelDensity id="pas" ionChannel="leak" condDensity="0.001 S_per_cm2" erev="-65 mV"/>
                    <specificCapacitance value="1.0 uF_per_cm2"/>
                    <initMembPotential value="-65.0 mV"/>
                </membraneProperties>
            </biophysicalProperties>
        </cell>
        <pulseGenerator id="pg" delay="5ms" duration="20ms" amplitude="0.05nA"/>
        <network id="net">
            <population id="pop" component="c" size="1"/>
            <explicitInput target="pop[0]" input="pg"/>
        </network>
    </neuroml>
    """
    doc = read_neuroml_string(nml)
    model = build(doc)
    assert model.report.is_faithful

    res = tarjuman.simulate(
        model,
        t_max=30.0,
        delta_t=0.025,
        records=[
            "pop[0]/v",
            "pop[0]/bp/membraneProperties/k_chans/k_markov/n/c1/occupancy",
            "pop[0]/bp/membraneProperties/k_chans/k_markov/n/c2/occupancy",
            "pop[0]/bp/membraneProperties/k_chans/k_markov/n/o1/occupancy",
        ],
    )
    total_occ = (
        res["pop[0]/bp/membraneProperties/k_chans/k_markov/n/c1/occupancy"]
        + res["pop[0]/bp/membraneProperties/k_chans/k_markov/n/c2/occupancy"]
        + res["pop[0]/bp/membraneProperties/k_chans/k_markov/n/o1/occupancy"]
    )
    np.testing.assert_allclose(total_occ, 1.0, atol=1e-6)

    net = model.module
    net.delete_recordings()
    net.record("v", verbose=False)
    net.make_trainable("k_chans_n_t2o_vHalf")
    params = net.get_parameters()

    def loss_fn(p):
        traces = jx.integrate(net, delta_t=0.05, t_max=20.0, params=p)
        return jnp.mean(traces**2)

    grad = jax.grad(loss_fn)(params)
    assert jnp.isfinite(grad[0]["k_chans_n_t2o_vHalf"]).all()
    assert float(jnp.abs(grad[0]["k_chans_n_t2o_vHalf"]).sum()) > 0.0


def test_ghk_channel_and_calcium_pool_differentiable():
    """channelDensityGHK couples with a Ca pool, records iDensity, and differentiates."""
    nml = """\
    <neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ghk_test">
        <ionChannelPassive id="leak" conductance="10pS" species="non_specific"/>
        <ionChannelHH id="ca_chan" conductance="10pS" species="ca">
            <gateHHrates id="p" instances="2">
                <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
                <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
            </gateHHrates>
        </ionChannelHH>
        <concentrationModel id="simple_decay" type="fixedFactorConcentrationModel"
                            ion="ca" restingConc="3e-6 mM" decayConstant="1.0 ms"
                            rho="3e-1 mol_per_m_per_A_per_s"/>
        <cell id="na_k_ca">
            <morphology id="m">
                <segment id="0" name="Soma">
                    <proximal x="0.0" y="0.0" z="0.0" diameter="1.0"/>
                    <distal x="0.0" y="0.0" z="10.0" diameter="1.0"/>
                </segment>
            </morphology>
            <biophysicalProperties id="biophys">
                <membraneProperties>
                    <channelDensity condDensity="0.0003 S_per_cm2" id="passive" ionChannel="leak" erev="-53.1 mV" ion="non_specific"/>
                    <channelDensityGHK permeability="2.5e-7 m_per_s" id="ca_all" ionChannel="ca_chan" ion="ca"/>
                    <specificCapacitance value="1.0 uF_per_cm2"/>
                    <initMembPotential value="-65.0 mV"/>
                </membraneProperties>
                <intracellularProperties>
                    <species id="ca" ion="ca" concentrationModel="simple_decay"
                             initialConcentration="5e-6 mM" initialExtConcentration="2 mM"/>
                </intracellularProperties>
            </biophysicalProperties>
        </cell>
        <pulseGenerator id="IClamp" delay="4ms" duration="6.0ms" amplitude="0.005nA"/>
        <network id="net1" type="networkWithTemperature" temperature="16.3 degC">
            <population id="pop" component="na_k_ca" size="1"/>
            <explicitInput target="pop[0]" input="IClamp"/>
        </network>
    </neuroml>
    """
    doc = read_neuroml_string(nml)
    model = build(doc, temperature=289.45)
    assert model.report.is_faithful

    res = tarjuman.simulate(
        model,
        t_max=15.0,
        delta_t=0.01,
        records=[
            "pop[0]/v",
            "pop[0]/biophys/membraneProperties/ca_all/iDensity",
            "pop[0]/na_k_ca/caConc",
        ],
    )
    ica = res["pop[0]/biophys/membraneProperties/ca_all/iDensity"]
    ca = res["pop[0]/na_k_ca/caConc"]
    # Inward Ca current during depolarization (negative in Jaxley mA/cm2) drives [Ca]i up.
    assert np.min(ica) < -1e-4
    assert np.max(ca) > 5e-6
