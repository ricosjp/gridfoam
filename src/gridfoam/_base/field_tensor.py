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

    @interior.setter
    def interior(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ] = value

    @property
    def xm(self) -> torch.Tensor:
        return self.raw[
            0 : self.bnd, self.bnd : -self.bnd, self.bnd : -self.bnd, ...
        ]

    @xm.setter
    def xm(self, value: torch.Tensor) -> None:
        self.raw[
            0 : self.bnd, self.bnd : -self.bnd, self.bnd : -self.bnd, ...
        ] = value

    @property
    def xp(self) -> torch.Tensor:
        return self.raw[
            -self.bnd :, self.bnd : -self.bnd, self.bnd : -self.bnd, ...
        ]

    @xp.setter
    def xp(self, value: torch.Tensor) -> None:
        self.raw[
            -self.bnd :, self.bnd : -self.bnd, self.bnd : -self.bnd, ...
        ] = value

    @property
    def ym(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, 0 : self.bnd, self.bnd : -self.bnd, ...
        ]

    @ym.setter
    def ym(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd, 0 : self.bnd, self.bnd : -self.bnd, ...
        ] = value

    @property
    def yp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, -self.bnd :, self.bnd : -self.bnd, ...
        ]

    @yp.setter
    def yp(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd, -self.bnd :, self.bnd : -self.bnd, ...
        ] = value

    @property
    def zm(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, self.bnd : -self.bnd, 0 : self.bnd, ...
        ]

    @zm.setter
    def zm(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd, self.bnd : -self.bnd, 0 : self.bnd, ...
        ] = value

    @property
    def zp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd, self.bnd : -self.bnd, -self.bnd :, ...
        ]

    @zp.setter
    def zp(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd, self.bnd : -self.bnd, -self.bnd :, ...
        ] = value
