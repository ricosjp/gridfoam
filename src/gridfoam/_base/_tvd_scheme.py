from gridfoam._interface._tvd_scheme import ITVDScheme
from gridfoam.utils.enums import TVDScheme

from ._tvd_schemes import (
    BoundedLinear,
    Minmod,
    Superbee,
    Upwind,
    VanAlbada,
    VanLeer,
)


def tvd_scheme(scheme: TVDScheme) -> ITVDScheme:
    match scheme:
        case TVDScheme.SUPERBEE:
            return Superbee()
        case TVDScheme.MINMOD:
            return Minmod()
        case TVDScheme.BOUNDED_LINEAR:
            return BoundedLinear()
        case TVDScheme.VAN_LEER:
            return VanLeer()
        case TVDScheme.VAN_ALBADA:
            return VanAlbada()
        case TVDScheme.UPWIND:
            return Upwind()
        case _:
            raise ValueError(f"Invalid scheme: {scheme}")
