import torch

from gridfoam._base._interface._tvd_scheme import ITVDScheme


class Superbee(ITVDScheme):
    def correction_term(
        self, delta_minus: torch.Tensor, delta_plus: torch.Tensor
    ) -> torch.Tensor:
        """Correction term (superbee)"""
        f = torch.zeros_like(delta_minus)
        phi = torch.zeros_like(delta_minus)

        mask = (delta_minus * delta_plus) > 0
        delta_t = delta_minus + delta_plus
        f[mask] = delta_minus[mask] / delta_t[mask]
        mask1 = (f >= 0) & (f <= 1 / 3)
        mask2 = (f > 1 / 3) & (f <= 1 / 2)
        mask3 = (f > 1 / 2) & (f <= 2 / 3)
        mask4 = (f > 2 / 3) & (f <= 1)

        phi[mask1] = 4 * f[mask1]
        phi[mask2] = 2 * (1 - f[mask2])
        phi[mask3] = 2 * f[mask3]
        phi[mask4] = 4 * (1 - f[mask4])
        delta_t = delta_minus + delta_plus
        correction = 0.25 * phi * delta_t
        return correction
