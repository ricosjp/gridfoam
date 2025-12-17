import abc

import torch


class ITVDScheme(abc.ABC):
    @abc.abstractmethod
    def correction_term(
        self, delta_minus: torch.Tensor, delta_plus: torch.Tensor
    ) -> torch.Tensor:
        pass
