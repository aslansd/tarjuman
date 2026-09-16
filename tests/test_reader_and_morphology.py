"""Reading NeuroML: units, components, and morphology mapping."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tarjuman import ir, read_neuroml
from tarjuman.errors import MorphologyError, UnitError
from tarjuman.morphology import build_morphology, resolve_groups
from tarjuman.report import ConversionReport
from tarjuman.units import parse_quantity, to_jaxley

DATA = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, dimension, expected",
    [
        ("-65mV", "voltage", -65.0),
        ("-65 mV", "voltage", -65.0),
        ("0.065V", "voltage", 65.0),
        ("120.0 mS_per_cm2", "conductanceDensity", 0.12),
        ("3.0 S_per_m2", "conductanceDensity", 3e-4),
        ("360 S_per_m2", "conductanceDensity", 0.036),
        ("1.0 uF_per_cm2", "specificCapacitance", 1.0),
        ("0.03 kohm_cm", "resistivity", 30.0),
        ("100 kohm_cm", "resistivity", 100000.0),
        ("0.08nA", "current", 0.08),
        ("0.5nS", "conductance", 5e-4),
        ("3e-5s", "time", 0.03),
        ("1per_ms", "per_time", 1.0),
        ("1000per_s", "per_time", 1.0),
        ("1.2mM", "concentration", 1.2),
    ],
)
def test_quantities_convert_to_jaxley_units(text, dimension, expected):
    assert to_jaxley(text, dimension) == pytest.approx(expected, rel=1e-9)


def test_temperature_offsets_are_applied():
    assert to_jaxley("6.3degC", "temperature") == pytest.approx(279.45)
    assert to_jaxley("279.45K", "temperature") == pytest.approx(279.45)


def test_dimensionless_numbers_take_the_expected_dimension():
    value, dimension = parse_quantity("0.5", "none")
    assert (value, dimension) == (0.5, "none")


def test_wrong_dimension_is_rejected():
    with pytest.raises(UnitError, match="dimension"):
        to_jaxley("-65mV", "time")


def test_unknown_unit_is_rejected():
    with pytest.raises(UnitError, match="Unknown LEMS unit"):
        to_jaxley("5furlongs", "length")


def test_missing_quantity_uses_the_default():
    assert to_jaxley(None, "time", default=0.0) == 0.0
    with pytest.raises(UnitError):
        to_jaxley(None, "time")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hh_document():
    return read_neuroml(DATA / "hh_single_comp.nml")


def test_document_lists_every_component(hh_document):
    assert set(hh_document.ion_channels) == {"passiveChan", "naChan", "kChan"}
    assert set(hh_document.cells) == {"hhcell"}
    assert set(hh_document.input_sources) == {"pulseGen1"}
    assert set(hh_document.networks) == {"net1"}


def test_gates_and_rates_are_read_in_jaxley_units(hh_document):
    na = hh_document.ion_channels["naChan"]
    assert [gate.id for gate in na.gates] == ["m", "h"]
    m_gate = na.gates[0]
    assert m_gate.instances == 3
    assert m_gate.kind == "gateHHrates"
    assert m_gate.forward_rate.kind == "HHExpLinearRate"
    assert m_gate.forward_rate.rate == pytest.approx(1.0)  # 1/ms
    assert m_gate.forward_rate.midpoint == pytest.approx(-40.0)  # mV
    assert m_gate.reverse_rate.scale == pytest.approx(-18.0)


def test_passive_channel_has_no_gates(hh_document):
    assert hh_document.ion_channels["passiveChan"].is_passive


def test_biophysics_are_read(hh_document):
    biophysics = hh_document.cells["hhcell"].biophysical_properties
    densities = {d.id: d for d in biophysics.channel_densities}
    assert densities["naChans"].cond_density == pytest.approx(0.12)  # S/cm2
    assert densities["naChans"].erev == pytest.approx(50.0)  # mV
    assert densities["leak"].cond_density == pytest.approx(3e-4)
    assert biophysics.specific_capacitance[0][0] == pytest.approx(1.0)
    assert biophysics.init_memb_potential[0][0] == pytest.approx(-65.0)
    assert biophysics.spike_thresh[0][0] == pytest.approx(-20.0)


def test_network_populations_and_inputs(hh_document):
    network = hh_document.networks["net1"]
    assert network.populations[0].component == "hhcell"
    assert network.explicit_inputs[0].target_population == "hhpop"
    assert network.explicit_inputs[0].input == "pulseGen1"


def test_electrical_and_continuous_projections_are_read():
    document = read_neuroml(DATA / "two_cell_analog.nml")
    kinds = {p.id: p.kind for p in document.networks["TwoCellNet"].projections}
    assert kinds == {
        "gjProj": "electricalProjection",
        "gradedProj": "continuousProjection",
    }
    graded = document.networks["TwoCellNet"].projections[1].connections[0]
    assert graded.pre_component == "silent1"
    assert graded.post_component == "gs1"


def test_unsupported_components_are_reported_not_raised():
    text = """
    <neuroml id="doc">
      <ionChannelKS id="ks1"/>
      <cell id="c1">
        <morphology id="m">
          <segment id="0"><distal x="0" y="10" z="0" diameter="2"/>
            <proximal x="0" y="0" z="0" diameter="2"/></segment>
        </morphology>
      </cell>
    </neuroml>
    """
    from tarjuman import read_neuroml_string

    report = ConversionReport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        document = read_neuroml_string(text, report=report)
    assert "c1" in document.cells
    assert any("ionChannelKS" in note.component for note in report.notes)


# --------------------------------------------------------------------------- #
# Morphology
# --------------------------------------------------------------------------- #
def _cell(segments, groups=()):
    morphology = ir.Morphology(id="m")
    for segment in segments:
        morphology.segments[segment.id] = segment
    for group in groups:
        morphology.groups[group.id] = group
    return ir.Cell(id="c", morphology=morphology)


def _segment(index, parent, start, end, diameter=2.0, fraction_along=1.0):
    return ir.Segment(
        id=index,
        parent=parent,
        fraction_along=fraction_along,
        proximal=(0.0, start, 0.0, diameter),
        distal=(0.0, end, 0.0, diameter),
    )


def test_unbranched_chain_becomes_one_branch():
    cell = _cell([_segment(0, None, 0, 10), _segment(1, 0, 10, 20)])
    morphology = build_morphology(cell, ConversionReport())
    assert len(morphology.branches) == 1
    assert morphology.branches[0].segments == [0, 1]
    assert morphology.branches[0].length == pytest.approx(20.0)


def test_tree_splits_at_branch_points():
    # 0 -> {1, 2}: three branches, parents [-1, 0, 0].
    cell = _cell(
        [
            _segment(0, None, 0, 10),
            _segment(1, 0, 10, 20),
            _segment(2, 0, 10, 30),
        ]
    )
    morphology = build_morphology(cell, ConversionReport())
    assert morphology.parents == [-1, 0, 0]


def test_parents_always_precede_children():
    cell = _cell(
        [
            _segment(0, None, 0, 10),
            _segment(1, 0, 10, 20),
            _segment(2, 1, 20, 30),
            _segment(3, 1, 20, 40),
        ]
    )
    morphology = build_morphology(cell, ConversionReport())
    for branch in morphology.branches:
        assert branch.parent < branch.index


def test_declared_sections_are_honoured_with_their_divisions():
    segments = [_segment(0, None, 0, 10), _segment(1, 0, 10, 40)]
    groups = [
        ir.SegmentGroup(id="soma", members=[0], neuro_lex_id="sao864921383"),
        ir.SegmentGroup(
            id="dend",
            members=[1],
            neuro_lex_id="sao864921383",
            properties={"numberInternalDivisions": "5"},
        ),
    ]
    morphology = build_morphology(_cell(segments, groups), ConversionReport())
    assert [branch.ncomp for branch in morphology.branches] == [1, 5]
    assert morphology.n_compartments == 6


def test_max_comp_length_adds_compartments():
    cell = _cell([_segment(0, None, 0, 100)])
    morphology = build_morphology(cell, ConversionReport(), max_comp_length=10.0)
    assert morphology.branches[0].ncomp == 10


def test_spherical_soma_becomes_an_area_preserving_cylinder():
    sphere = ir.Segment(
        id=0, proximal=(0.0, 0.0, 0.0, 17.841242), distal=(0.0, 0.0, 0.0, 17.841242)
    )
    report = ConversionReport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        morphology = build_morphology(_cell([sphere]), report)
    branch = morphology.branches[0]
    assert branch.length == pytest.approx(17.841242)
    assert branch.radius == pytest.approx(17.841242 / 2)
    assert report.approximations


def test_locate_maps_fraction_along_to_a_compartment():
    segments = [_segment(0, None, 0, 100)]
    groups = [
        ir.SegmentGroup(
            id="sec",
            members=[0],
            neuro_lex_id="sao864921383",
            properties={"numberInternalDivisions": "10"},
        )
    ]
    morphology = build_morphology(_cell(segments, groups), ConversionReport())
    assert morphology.locate(0, 0.0) == (0, 0)
    assert morphology.locate(0, 0.45) == (0, 4)
    assert morphology.locate(0, 1.0) == (0, 9)  # clipped to the last compartment


def test_locate_across_two_segments_of_one_branch():
    cell = _cell([_segment(0, None, 0, 10), _segment(1, 0, 10, 20)])
    morphology = build_morphology(cell, ConversionReport(), ncomp=4)
    assert morphology.locate(0, 0.5) == (0, 1)  # 5 um of 20 -> comp 1
    assert morphology.locate(1, 0.5) == (0, 3)  # 15 um of 20 -> comp 3


def test_segment_groups_resolve_includes():
    groups = [
        ir.SegmentGroup(id="a", members=[0]),
        ir.SegmentGroup(id="b", members=[1]),
        ir.SegmentGroup(id="all_dend", includes=["a", "b"]),
    ]
    morphology = ir.Morphology(id="m", groups={g.id: g for g in groups})
    assert resolve_groups(morphology)["all_dend"] == [0, 1]


def test_unknown_segment_group_is_an_error():
    cell = _cell([_segment(0, None, 0, 10)])
    morphology = build_morphology(cell, ConversionReport())
    with pytest.raises(MorphologyError, match="Unknown segmentGroup"):
        morphology.compartments_of_group("nope")
