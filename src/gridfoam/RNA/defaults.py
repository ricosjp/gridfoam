from .builtins import builtin_p, builtin_phi, builtin_U
from .registry import SimulationMetaRegistry

# _default_registry: SimulationMetaRegistry | None = None

# def default_registry() -> SimulationMetaRegistry:
#     global _default_registry
#     if _default_registry is None:
#         _default_registry = SimulationMetaRegistry()
#         _autoregister_builtin_fields(_default_registry)
#     return _default_registry


def default_registry() -> SimulationMetaRegistry:
    reg = SimulationMetaRegistry()
    _autoregister_builtin_fields(reg)
    return reg


def _autoregister_builtin_fields(reg: SimulationMetaRegistry) -> None:
    """
    Autoregister builtin fields.
    """
    reg.register_field(builtin_p())
    reg.register_field(builtin_U())
    reg.register_field(builtin_phi())
