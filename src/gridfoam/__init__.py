from gridfoam.runtime_config import runtime_type_checks_enabled

if runtime_type_checks_enabled():
    from beartype.claw import beartype_this_package

    beartype_this_package()

# Register runtime type checks before importing the public API.
from gridfoam import core, fv, meta  # noqa: E402

__all__ = ["core", "fv", "meta", "runtime_type_checks_enabled"]
