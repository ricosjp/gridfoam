from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.post.forces.coord import OrthonormalCoord


@dataclass(frozen=True)
class ForceCoeffs:
    """
    Aerodynamic force and moment coefficients at one time.

    Attributes
    ----------
    time : float
        Simulation time.
    Cd, Cs, Cl : torch.Tensor
        Drag, side-force and lift coefficients.
    CmRoll, CmPitch, CmYaw : torch.Tensor
        Moment coefficients about the local roll, pitch and yaw axes.
    Cd_f, Cd_r, Cs_f, Cs_r, Cl_f, Cl_r : torch.Tensor
        OpenFOAM-style front and rear axle constituents.
    """

    time: float
    Cd: Float[torch.Tensor, " 1"]
    Cd_f: Float[torch.Tensor, " 1"]
    Cd_r: Float[torch.Tensor, " 1"]
    Cl: Float[torch.Tensor, " 1"]
    Cl_f: Float[torch.Tensor, " 1"]
    Cl_r: Float[torch.Tensor, " 1"]
    CmPitch: Float[torch.Tensor, " 1"]
    CmRoll: Float[torch.Tensor, " 1"]
    CmYaw: Float[torch.Tensor, " 1"]
    Cs: Float[torch.Tensor, " 1"]
    Cs_f: Float[torch.Tensor, " 1"]
    Cs_r: Float[torch.Tensor, " 1"]

    @classmethod
    def from_force_moment(
        cls,
        *,
        time: float,
        force: Float[torch.Tensor, " 3"],
        moment: Float[torch.Tensor, " 3"],
        coord: OrthonormalCoord,
        force_scale: float,
        moment_scale: float,
    ) -> ForceCoeffs:
        Cd = torch.sum(force * coord.e1, dim=0, keepdim=True) / force_scale
        Cs = torch.sum(force * coord.e2, dim=0, keepdim=True) / force_scale
        Cl = torch.sum(force * coord.e3, dim=0, keepdim=True) / force_scale
        CmRoll = torch.sum(moment * coord.e1, dim=0, keepdim=True) / moment_scale
        CmPitch = torch.sum(moment * coord.e2, dim=0, keepdim=True) / moment_scale
        CmYaw = torch.sum(moment * coord.e3, dim=0, keepdim=True) / moment_scale

        return cls(
            time=time,
            Cd=Cd,
            Cd_f=0.5 * Cd + CmRoll,
            Cd_r=0.5 * Cd - CmRoll,
            Cl=Cl,
            Cl_f=0.5 * Cl + CmPitch,
            Cl_r=0.5 * Cl - CmPitch,
            CmPitch=CmPitch,
            CmRoll=CmRoll,
            CmYaw=CmYaw,
            Cs=Cs,
            Cs_f=0.5 * Cs + CmYaw,
            Cs_r=0.5 * Cs - CmYaw,
        )
