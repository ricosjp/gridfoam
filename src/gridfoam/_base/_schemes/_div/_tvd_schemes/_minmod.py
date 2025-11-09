import torch

from gridfoam._base._interface._tvd_scheme import ITVDScheme


class Minmod(ITVDScheme):
    def correction_term(
        self, delta_minus: torch.Tensor, delta_plus: torch.Tensor
    ) -> torch.Tensor:
        """Correction term (minmod)"""
        f = torch.zeros_like(delta_minus)
        phi = torch.zeros_like(delta_minus)

        mask = (delta_minus * delta_plus) > 0
        delta_t = delta_minus + delta_plus
        f[mask] = delta_minus[mask] / delta_t[mask]
        phi = torch.min(2 * f, 2 * (1 - f))
        delta_t = delta_minus + delta_plus
        correction = 0.25 * phi * delta_t
        return correction
