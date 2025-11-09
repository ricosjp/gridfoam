from dataclasses import dataclass

import torch

from gridfoam.utils.enums import FieldLayout, FieldRole, Namespace


@dataclass(frozen=True)
class FieldDescriptor:
    """
    Descriptor for a field.

    Parameters
    ----------
    name : str
        Name of the field.
    channels : int
        Number of channels of the field.
    dtype : torch.dtype
        Data type of the field.
    device : torch.device
        Device that the field is allocated on.
    role : FieldRole
        Role of the field.
    layout: FieldLayout
        Layout of the field.
    alias : str | None, default: Same as name
        Alias for the field.
        This is particularly useful for notation mapping.
        For example, a field named 'velocity' in code
        might have an alias 'u' to match the standard notation
        in fluid dynamics equations.
    units : str | None, default: None
        Units of the field.
    namespace : Namespace
        Namespace for the field. default: Namespace.USER
    requires_grad: bool = False
        Whether the field requires gradient.
    """
    name: str
    channels: int
    dtype: torch.dtype
    device: torch.device
    role: FieldRole
    layout: FieldLayout
    alias: str | None = None
    units: str | None = None
    namespace: Namespace = Namespace.USER
    requires_grad: bool = False

    def __post_init__(self) -> None:
        if self.alias is None:
            object.__setattr__(self, 'alias', self.name)

    @property
    def canonical_name(self) -> str:
        """
        Canonical name of the field.
        """
        return self.namespace.value + "." + self.name

    def __repr__(self) -> str:
        return f"{self.alias}"
