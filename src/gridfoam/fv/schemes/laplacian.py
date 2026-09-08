"""Laplacian scheme policies, independent of matrix assembly."""

from dataclasses import dataclass

from gridfoam.meta.enums import LaplacianScheme


@dataclass(frozen=True)
class LaplacianPolicy:
    """Coefficient interpolation and explicit hanging-face correction."""

    harmonic: bool
    corrected: bool


LAPLACIAN_SCHEMES: dict[LaplacianScheme, LaplacianPolicy] = {
    LaplacianScheme.CORRECTED: LaplacianPolicy(harmonic=False, corrected=True),
    LaplacianScheme.UNCORRECTED: LaplacianPolicy(
        harmonic=False, corrected=False
    ),
    LaplacianScheme.GAUSS_HARMONIC_CORRECTED: LaplacianPolicy(
        harmonic=True, corrected=True
    ),
    LaplacianScheme.GAUSS_HARMONIC_UNCORRECTED: LaplacianPolicy(
        harmonic=True, corrected=False
    ),
}


def get_laplacian_scheme(scheme: LaplacianScheme) -> LaplacianPolicy:
    """Return the Laplacian policy for the given canonical enum."""
    return LAPLACIAN_SCHEMES[scheme]
