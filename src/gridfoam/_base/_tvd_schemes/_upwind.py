import torch

from gridfoam._interface._tvd_scheme import ITVDScheme


class Upwind(ITVDScheme):
    def correction_term(
        self, delta_minus: torch.Tensor, delta_plus: torch.Tensor
    ) -> torch.Tensor:
        """Correction term (upwind)"""
        return torch.zeros_like(delta_minus)
