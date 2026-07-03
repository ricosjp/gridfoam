"""Validate loading example YAML configs into ``GridfoamConfig``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from gridfoam.meta.config import GridfoamConfig, RefinementRegionConfig


def test_example_cavity_config_loads():
    # The shipped cavity example must parse without validation errors.
    repo = Path(__file__).resolve().parents[3]
    config_path = (
        repo / "examples" / "cavity" / "gridfoam" / "data" / "config.yaml"
    )
    assert config_path.is_file(), f"missing example config: {config_path}"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = GridfoamConfig.model_validate(raw)
    assert cfg.simulator.device.value in ("cpu", "cuda")
    assert cfg.fluxel.ibm_type.value == "axis_projected"


def test_refinement_region_config_validates_box():
    # A well-formed refinement box must retain name and level.
    region = RefinementRegionConfig(
        name="wake",
        min=[0.0, -0.1, -0.1],
        max=[1.0, 0.1, 0.1],
        level=2,
    )

    assert region.name == "wake"
    assert region.level == 2


def test_refinement_region_config_rejects_invalid_box():
    # Degenerate boxes (min == max on an axis) must fail validation.
    try:
        RefinementRegionConfig(
            min=[0.0, 0.0, 0.0],
            max=[0.0, 1.0, 1.0],
            level=1,
        )
    except ValidationError:
        return

    raise AssertionError("invalid refinement region should fail validation")
