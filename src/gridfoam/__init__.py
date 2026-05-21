from gridfoam.runtime_config import runtime_type_checks_enabled

if runtime_type_checks_enabled():
    from beartype.claw import beartype_this_package

    beartype_this_package()
