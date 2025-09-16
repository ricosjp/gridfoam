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
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            0 : self.bnd,  # bnd
            ...,
        ]

    @xm.setter
    def xm(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            0 : self.bnd,  # bnd
            ...,
        ] = value

    @property
    def inxm(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            self.bnd : 2 * self.bnd,  # bnd
            ...,
        ]

    @property
    def xp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            -self.bnd :,  # bnd
            ...,
        ]

    @xp.setter
    def xp(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            -self.bnd :,  # bnd
            ...,
        ] = value

    @property
    def inxp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            -2 * self.bnd : -self.bnd,  # bnd
            ...,
        ]

    @property
    def ym(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            0 : self.bnd,  # bnd
            self.bnd : -self.bnd,
            ...,
        ]

    @ym.setter
    def ym(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd,
            0 : self.bnd,  # bnd
            self.bnd : -self.bnd,
            ...,
        ] = value

    @property
    def inym(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            self.bnd : 2 * self.bnd,  # bnd
            self.bnd : -self.bnd,
            ...,
        ]

    @property
    def yp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            -self.bnd :,  # bnd
            self.bnd : -self.bnd,
            ...,
        ]

    @yp.setter
    def yp(self, value: torch.Tensor) -> None:
        self.raw[
            self.bnd : -self.bnd,
            -self.bnd :,  # bnd
            self.bnd : -self.bnd,
            ...,
        ] = value

    @property
    def inyp(self) -> torch.Tensor:
        return self.raw[
            self.bnd : -self.bnd,
            -2 * self.bnd : -self.bnd,  # bnd
            self.bnd : -self.bnd,
            ...,
        ]

    @property
    def zm(self) -> torch.Tensor:
        return self.raw[
            0 : self.bnd,  # bnd
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ]

    @zm.setter
    def zm(self, value: torch.Tensor) -> None:
        self.raw[
            0 : self.bnd,  # bnd
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ] = value

    @property
    def inzm(self) -> torch.Tensor:
        return self.raw[
            self.bnd : 2 * self.bnd,  # bnd
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ]

    @property
    def zp(self) -> torch.Tensor:
        return self.raw[
            -self.bnd :,  # bnd
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ]

    @zp.setter
    def zp(self, value: torch.Tensor) -> None:
        self.raw[
            -self.bnd :,  # bnd
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ] = value

    @property
    def inzp(self) -> torch.Tensor:
        return self.raw[
            -2 * self.bnd : -self.bnd,  # bnd
            self.bnd : -self.bnd,
            self.bnd : -self.bnd,
            ...,
        ]
