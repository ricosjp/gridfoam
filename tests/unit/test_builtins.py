import pytest

import gridfoam.algorithms.simple  # noqa: F401 — establish import order
from gridfoam.core.name import (
    FieldNameParts,
    make_field_name,
    parse_field_name,
)


@pytest.mark.parametrize(
    ("name", "phase", "expected"),
    [
        ("phi", None, "phi"),
        ("U", None, "U"),
        ("phi", "water", "phi.water"),
        ("HbyA", "air", "HbyA.air"),
    ],
)
def test_make_field_name(name: str, phase: str | None, expected: str) -> None:
    assert make_field_name(name, phase=phase) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("phi", FieldNameParts(base="phi", phase=None)),
        ("phi.water", FieldNameParts(base="phi", phase="water")),
        ("HbyA.air", FieldNameParts(base="HbyA", phase="air")),
    ],
)
def test_parse_field_name(key: str, expected: FieldNameParts) -> None:
    assert parse_field_name(key) == expected


def test_make_field_name_rejects_dots_in_name() -> None:
    with pytest.raises(ValueError, match="name"):
        make_field_name("phi.water")


def test_make_field_name_rejects_dots_in_phase() -> None:
    with pytest.raises(ValueError, match="phase"):
        make_field_name("phi", phase="a.b")


def test_parse_field_name_rejects_invalid_key() -> None:
    with pytest.raises(ValueError, match="invalid field name"):
        parse_field_name(".water")
