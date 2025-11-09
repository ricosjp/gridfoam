import torch
from jaxtyping import Float

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._grid import PyOctreeNode as Cube
from gridfoam._base._field._registry import FieldRegistry
from gridfoam.utils.enums import FieldLayout, FieldRole, Namespace


class RhieChowScheme:
    def __init__(
        self,
        velocity_fd: FieldDescriptor,
        pressure_fd: FieldDescriptor,
        field_registry: FieldRegistry,
    ):
        """
        Initialize the Rhie-Chow scheme.

        Parameters
        ----------
        velocity_fd : FieldDescriptor
            Field descriptor of the velocity field.
        pressure_fd : FieldDescriptor
            Field descriptor of the pressure field.
        """
        self._U_i_name = velocity_fd.canonical_name
        self._p_i_name = pressure_fd.canonical_name
        self._U_f_fd = field_registry.declare(
            name=velocity_fd.name,
            channels=velocity_fd.channels,
            dtype=velocity_fd.dtype,
            device=velocity_fd.device,
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.FACE,
            namespace=Namespace.RHIE_CHOW,
        )
        self._p_grad_fd = field_registry.declare(
            name=pressure_fd.name,
            channels=pressure_fd.channels,
            dtype=pressure_fd.dtype,
            device=pressure_fd.device,
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.FACE,
            namespace=Namespace.RHIE_CHOW,
        )

        self._U_f_name = self._U_f_fd.canonical_name
        self._p_grad_name = self._p_grad_fd.canonical_name

    def interpolate(
        self,
        cube: Cube,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> None:
        U_i = cube.old.cells[self._U_i_name]
        U_f = U_i.face_average()
        # TODO: Add Rhie-Chow correction
        cube.old.faces[self._U_f_name] = U_f
