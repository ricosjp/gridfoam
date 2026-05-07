from __future__ import annotations


def make_builtin_key(
    name: str,
    *,
    scope: str = "global",
    region: str | None = None,
    phase: str | None = None,
) -> str:
    """
    Build a hierarchical builtin-field key.

    The key format is designed to remain stable when multiphase or
    multi-region support is introduced.
    """
    parts = [f"scope={scope}"]
    if region is not None:
        parts.append(f"region={region}")
    if phase is not None:
        parts.append(f"phase={phase}")
    parts.append(name)
    return "/".join(parts)
