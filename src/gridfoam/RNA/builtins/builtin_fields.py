from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta


def builtin_U() -> FieldMeta:
    return FieldMeta(
        name="U",
        label="Velocity",
        description="Cell-centered velocity field",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=3,
        unit="m/s",
    )

def builtin_rAU() -> FieldMeta:
    return FieldMeta(
        name="rAU",
        label="reciprocal diagonal components of momentum equation",
        description="reciprocal diagonal components of momentum equation",
        role=FieldRole.AUXILIARY,
        layout=FieldLayout.CELL,
        components=3,
        unit="1/s",
    )

def builtin_p() -> FieldMeta:
    return FieldMeta(
        name="p",
        label="Pressure",
        description="Cell-centered pressure field",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=1,
        unit="Pa",
    )

def builtin_T() -> FieldMeta:
    return FieldMeta(
        name="T",
        label="Temperature",
        description="Cell-centered temperature field",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=1,
        unit="K",
    )

def builtin_rho() -> FieldMeta:
    return FieldMeta(
        name="rho",
        label="Density",
        description="Cell-centered density field",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=1,
        unit="kg/m^3",
    )

def builtin_phi() -> FieldMeta:
    return FieldMeta(
        name="phi",
        label="Flux",
        description="Face-centered flux field",
        role=FieldRole.AUXILIARY,
        layout=FieldLayout.FACE,
        components=1,
        unit="m^3/s",
    )
