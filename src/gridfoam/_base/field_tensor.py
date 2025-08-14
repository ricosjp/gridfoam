from dataclasses import dataclass

import torch


@dataclass
class FieldTensor:
    width: int
    bnd: int
    raw: torch.Tensor

    @property
    def interior(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ]

    @property
    def xm(self) -> torch.Tensor:
        return self.raw[
            0 : self.bnd, self.bnd : -self.bnd, self.bnd : -self.bnd, ...
        ]

    @property
    def xp(self) -> torch.Tensor:
        return self.raw[
            -self.bnd :, self.bnd : -self.bnd, self.bnd : -self.bnd, ...
        ]

    @property
    def ym(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, 0 : self.bnd, self.bnd : -self.bnd, ...
        ]

    @property
    def yp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, -self.bnd :, self.bnd : -self.bnd, ...
        ]

    @property
    def zm(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, self.bnd : -self.bnd, 0 : self.bnd, ...
        ]

    @property
    def zp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, self.bnd : -self.bnd, -self.bnd :, ...
        ]
