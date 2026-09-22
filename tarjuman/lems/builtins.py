"""Built-in NeuroML core component types, as tarjuman's interpreter sees them.

A model that defines ``muscleConcentrationModel extends
fixedFactorConcentrationModel`` expects the base type to exist.  Rather than
ship a copy of ``NeuroML2CoreTypes``, tarjuman constructs the handful of base
types its interpreter needs: their parameters, requirements and exposures, and
— where the base type carries real dynamics — a transcription of the
``<Dynamics>`` block from ``Cells.xml``.

These are used in two ways:

* as the base of a custom type, so inheritance resolves, and
* directly, so ``<fixedFactorConcentrationModel>`` in a model file becomes a
  running calcium pool without any special-casing elsewhere.

Transcribed from the NeuroML 2 core types
(https://github.com/NeuroML/NeuroML2, LGPL-3.0).
"""

from __future__ import annotations

from .component_types import ComponentType, EventAction, StateVariable

__all__ = ["BUILTIN_COMPONENT_TYPES", "builtin_registry"]


def _concentration_model() -> ComponentType:
    return ComponentType(
        name="concentrationModel",
        description="Base for any model of an ion concentration changing with time.",
        requirements={
            "surfaceArea": "area",
            "initialConcentration": "concentration",
            "initialExtConcentration": "concentration",
        },
        exposures={
            "concentration": "concentration",
            "extConcentration": "concentration",
        },
        texts=["ion"],
        state_variables={
            "concentration": StateVariable(
                "concentration", "concentration", "concentration"
            ),
            "extConcentration": StateVariable(
                "extConcentration", "concentration", "extConcentration"
            ),
        },
        events=[
            EventAction(
                kind="start",
                assignments=[
                    ("concentration", "initialConcentration"),
                    ("extConcentration", "initialExtConcentration"),
                ],
            )
        ],
    )


def _fixed_factor_concentration_model() -> ComponentType:
    component = _concentration_model()
    component.name = "fixedFactorConcentrationModel"
    component.extends = "concentrationModel"
    component.description = (
        "Buffering of an ion concentration with a baseline restingConc, decaying "
        "with decayConstant; the incoming current is scaled by a fixed factor rho "
        "independently of compartment size."
    )
    component.parameters = {
        "restingConc": "concentration",
        "decayConstant": "time",
        "rho": "rho_factor",
    }
    component.requirements["iCa"] = "current"
    component.time_derivatives = {
        "concentration": (
            "(iCa/surfaceArea) * rho - ((concentration - restingConc) / decayConstant)"
        )
    }
    component.events.append(
        EventAction(
            kind="condition",
            test="concentration .lt. 0",
            assignments=[("concentration", "0")],
        )
    )
    return component


def _decaying_pool_concentration_model() -> ComponentType:
    component = _concentration_model()
    component.name = "decayingPoolConcentrationModel"
    component.extends = "concentrationModel"
    component.description = (
        "Intracellular buffering in a shell of thickness shellThickness just "
        "inside the membrane."
    )
    component.parameters = {
        "restingConc": "concentration",
        "decayConstant": "time",
        "shellThickness": "length",
    }
    component.constants = {
        "Faraday": (96485.3, "charge_per_mole"),
        "AREA_SCALE": (1.0, "area"),
        "LENGTH_SCALE": (1.0, "length"),
    }
    component.requirements["iCa"] = "current"
    from .component_types import DerivedVariable

    component.derived_variables = [
        DerivedVariable(
            name="effectiveRadius",
            value="LENGTH_SCALE * sqrt(surfaceArea/(AREA_SCALE * (4 * 3.14159)))",
            dimension="length",
        ),
        DerivedVariable(
            name="innerRadius",
            value="effectiveRadius - shellThickness",
            dimension="length",
        ),
        DerivedVariable(
            name="shellVolume",
            value=(
                "(4 * (effectiveRadius * effectiveRadius * effectiveRadius) "
                "* 3.14159 / 3) - (4 * (innerRadius * innerRadius * innerRadius) "
                "* 3.14159 / 3)"
            ),
            dimension="volume",
        ),
    ]
    component.time_derivatives = {
        "concentration": (
            "iCa / (2 * Faraday * shellVolume) "
            "- ((concentration - restingConc) / decayConstant)"
        )
    }
    component.events.append(
        EventAction(
            kind="condition",
            test="concentration .lt. 0",
            assignments=[("concentration", "0")],
        )
    )
    return component


def _gate_base(name: str, extends: str | None = None) -> ComponentType:
    """A gate base type: what a custom gate inherits and must expose."""
    return ComponentType(
        name=name,
        extends=extends,
        parameters={},
        requirements={"v": "voltage"},
        exposures={
            "q": "none",
            "fcond": "none",
            "inf": "none",
            "tau": "time",
            "rateScale": "none",
            "instances": "none",
        },
    )


def _voltage_dep(name: str, exposure: str, dimension: str) -> ComponentType:
    return ComponentType(
        name=name,
        requirements={"v": "voltage"},
        exposures={exposure: dimension},
    )


def _voltage_conc_dep(name: str, exposure: str, dimension: str) -> ComponentType:
    return ComponentType(
        name=name,
        requirements={"v": "voltage", "caConc": "concentration"},
        exposures={exposure: dimension},
    )


def _synapse_base(name: str, extends: str | None = None) -> ComponentType:
    return ComponentType(
        name=name,
        extends=extends,
        requirements={"v": "voltage"},
        exposures={"i": "current"},
    )


def _graded_synapse_base(name: str) -> ComponentType:
    component = _synapse_base(name)
    component.requirements["vpeer"] = "voltage"
    return component


def builtin_registry() -> dict[str, ComponentType]:
    """A fresh registry of the built-in base types."""
    types = [
        _concentration_model(),
        _fixed_factor_concentration_model(),
        _decaying_pool_concentration_model(),
        _gate_base("baseGate"),
        _gate_base("gate", "baseGate"),
        _gate_base("gateHHrates", "baseGate"),
        _gate_base("gateHHtauInf", "baseGate"),
        _gate_base("gateHHratesTau", "baseGate"),
        _gate_base("gateHHratesInf", "baseGate"),
        _gate_base("gateHHratesTauInf", "baseGate"),
        _gate_base("gateHHInstantaneous", "baseGate"),
        _voltage_dep("baseVoltageDepRate", "r", "per_time"),
        _voltage_dep("baseVoltageDepVariable", "x", "none"),
        _voltage_dep("baseVoltageDepTime", "t", "time"),
        _voltage_conc_dep("baseVoltageConcDepRate", "r", "per_time"),
        _voltage_conc_dep("baseVoltageConcDepVariable", "x", "none"),
        _voltage_conc_dep("baseVoltageConcDepTime", "t", "time"),
        _synapse_base("baseSynapse"),
        _synapse_base("baseSynapseDL"),
        _synapse_base("baseVoltageDepSynapse", "baseSynapse"),
        _graded_synapse_base("baseGradedSynapse"),
        _graded_synapse_base("baseGradedSynapseDL"),
    ]
    return {component.name: component for component in types}


#: The built-in base types, by name.
BUILTIN_COMPONENT_TYPES = builtin_registry()
