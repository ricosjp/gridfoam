from collections.abc import ItemsView
from dataclasses import dataclass, field

import torch
from jaxtyping import Int32

from gridfoam._base.field_tensor import FieldTensor
from gridfoam.settings import FieldDataAttribute
from gridfoam.utils.enums import CubeType


@dataclass
class Cube:
    width: int
    bnd_width: int
    cube_type: CubeType
    depth: int
    global_index: Int32[torch.Tensor, " 3"]
    face_ids: Int32[torch.Tensor, " n_faces"]
    field_tensors: dict[str, FieldTensor] = field(default_factory=dict)
    device: torch.device = torch.device("cpu")

    def allocate_field_tensors(
        self, field_data_dict: dict[str, FieldDataAttribute]
    ) -> None:
        data_width = self.width + 2 * self.bnd_width
        for name, attr in field_data_dict.items():
            shape = (data_width, data_width, data_width, *attr.shape)
            data = torch.zeros(shape, dtype=attr.dtype, device=self.device)
            self.field_tensors[name] = FieldTensor(
                width=self.width,
                bnd=self.bnd_width,
                raw=data,
            )

    def __getitem__(self, name: str) -> FieldTensor:
        """
        Get a field tensor by name.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        torch.Tensor
            The corresponding tensor.
        """
        if name in self.field_tensors:
            return self.field_tensors[name]
        raise AttributeError(f"'FieldData' object has no attribute '{name}'")

    def __getattr__(self, name: str) -> FieldTensor:
        """
        Allow attribute-style access to fields.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        torch.Tensor
            The corresponding tensor.

        Raises
        ------
        AttributeError
            If the field does not exist.
        """
        if name in self.field_tensors:
            return self.field_tensors[name]
        raise AttributeError(f"'FieldData' object has no attribute '{name}'")

    def items(self) -> ItemsView[str, FieldTensor]:
        """
        Return an iterator over (field name, tensor) pairs.

        Returns
        -------
        Iterator[tuple[str, torch.Tensor]]
            Iterator over field data items.
        """
        return self.field_tensors.items()
