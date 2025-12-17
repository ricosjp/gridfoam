from enum import Enum

from gridfoam.DNA.scheme.tvd._interface import ITVDScheme
from gridfoam.DNA.scheme.tvd._limited_linear import LimitedLinear
from gridfoam.DNA.scheme.tvd._minmod import Minmod
from gridfoam.DNA.scheme.tvd._superbee import Superbee
from gridfoam.DNA.scheme.tvd._upwind import Upwind
from gridfoam.DNA.scheme.tvd._van_albada import VanAlbada
from gridfoam.DNA.scheme.tvd._van_leer import VanLeer


class TVDSchemeChoice(Enum):
    """
    Choice of the TVD scheme.
    """

    SUPERBEE = "Superbee"
    """
    Superbee scheme.
    """
    MINMOD = "Minmod"
    """
    Minmod scheme.
    """
    LIMITED_LINEAR = "LimitedLinear"
    """
    Limited linear scheme.
    """
    VAN_LEER = "VanLeer"
    """
    Van Leer scheme.
    """
    VAN_ALBADA = "VanAlbada"
    """
    Van Albada scheme.
    """
    UPWIND = "Upwind"
    """
    Upwind scheme.
    """


class TVDFactory:
    registry: dict[TVDSchemeChoice, ITVDScheme] = {
        TVDSchemeChoice.SUPERBEE: Superbee(),
        TVDSchemeChoice.MINMOD: Minmod(),
        TVDSchemeChoice.LIMITED_LINEAR: LimitedLinear(),
        TVDSchemeChoice.VAN_LEER: VanLeer(),
        TVDSchemeChoice.VAN_ALBADA: VanAlbada(),
        TVDSchemeChoice.UPWIND: Upwind(),
    }

    @classmethod
    def create(cls, choice: TVDSchemeChoice) -> ITVDScheme:
        if choice not in cls.registry:
            raise ValueError(f"Unknown TVD scheme choice: {choice.name}")
        return cls.registry[choice]

    @classmethod
    def register(cls, choice: TVDSchemeChoice, impl: ITVDScheme) -> None:
        cls.registry[choice] = impl
