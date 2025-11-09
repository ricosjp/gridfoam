from collections.abc import Iterator

import torch

from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam.utils.enums import FieldLayout, FieldRole, Namespace


class FieldRegistry:
    """
    Registry for fields.
    """

    def __init__(self) -> None:
        self._registered_fd: dict[str, FieldDescriptor] = {}
        self._alias_to_canonical_name: dict[str, str] = {}

    def declare(
        self,
        name: str,
        channels: int,
        dtype: torch.dtype,
        device: torch.device,
        role: FieldRole,
        layout: FieldLayout,
        alias: str | None = None,
        units: str | None = None,
        namespace: Namespace = Namespace.USER,
        requires_grad: bool = False,
    ) -> FieldDescriptor:
        """
        Declare a field.
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
        fd = FieldDescriptor(
            name=name,
            channels=channels,
            dtype=dtype,
            device=device,
            role=role,
            layout=layout,
            alias=alias,
            units=units,
            namespace=namespace,
            requires_grad=requires_grad,
        )
        canonical_name = fd.canonical_name
        if canonical_name in self._registered_fd:
            raise ValueError(
                f"Field {fd.name} already declared \
                in namespace {fd.namespace.value}"
            )
        if fd.alias in self._alias_to_canonical_name:
            raise ValueError(f"Alias {fd.alias} already declared")
        self._alias_to_canonical_name[fd.alias] = canonical_name
        self._registered_fd[canonical_name] = fd
        return fd

    def get_field_desc_by_name(
        self, name: str, namespace: Namespace = Namespace.USER
    ) -> FieldDescriptor:
        """
        Get a field descriptor by name.
        Parameters
        ----------
        name : str
            Name of the field.
        namespace : Namespace, default: Namespace.USER
            Namespace of the field.
        Returns
        -------
        FieldDescriptor
            Field descriptor.
        """
        canonical_name = namespace.value + "." + name
        if canonical_name not in self._registered_fd:
            raise ValueError(
                f"Field {name} not declared \
                in namespace {namespace.value}"
            )
        return self._registered_fd[canonical_name]

    def get_field_desc_by_alias(self, alias: str) -> FieldDescriptor:
        """
        Get a field descriptor by alias.
        Parameters
        ----------
        alias : str
            Alias of the field.
        Returns
        -------
        FieldDescriptor
            Field descriptor.
        """
        if alias not in self._alias_to_canonical_name:
            raise ValueError(f"Alias {alias} not declared")
        return self._registered_fd[self._alias_to_canonical_name[alias]]

    def iter_all(self) -> Iterator[FieldDescriptor]:
        """
        Iterate over all field descriptors.
        Returns
        -------
        Iterator[FieldDescriptor]
            Iterator over all field descriptors.
        """
        return iter(self._registered_fd.values())
