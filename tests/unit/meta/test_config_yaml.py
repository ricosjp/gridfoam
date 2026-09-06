"""Validate loading example YAML configs into ``GridfoamConfig``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from gridfoam.meta.config import GridfoamConfig, RefinementRegionConfig
from gridfoam.meta.enums import BoundaryConditionType


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
    assert cfg.fluxel.motion.value == "static"


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


def test_fluxel_motion_dynamic_from_yaml():
    repo = Path(__file__).resolve().parents[3]
    config_path = (
        repo / "examples" / "cavity" / "gridfoam" / "data" / "config.yaml"
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["fluxel"]["motion"] = "dynamic"
    cfg = GridfoamConfig.model_validate(raw)
    assert cfg.fluxel.motion.value == "dynamic"


def test_dynamic_motion_example_configs_load():
    repo = Path(__file__).resolve().parents[3]
    for rel in (
        "examples/dynamic_motions/update_ib/data/config.yaml",
        "examples/dynamic_motions/remesh/data/config.yaml",
    ):
        config_path = repo / rel
        assert config_path.is_file(), f"missing example config: {config_path}"
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        cfg = GridfoamConfig.model_validate(raw)
        assert cfg.fluxel.motion.value == "dynamic"
        u_wall = cfg.simulator.conditions["U"].boundary["wall"]
        assert u_wall.type == BoundaryConditionType.DIRICHLET
        assert u_wall.value == [1.5, 0.0, 0.0]
        p_wall = cfg.simulator.conditions["p"].boundary["wall"]
        assert p_wall.type == BoundaryConditionType.FIXED_FLUX_PRESSURE
        if "update_ib" in rel:
            assert len(cfg.fluxel.refinement_regions) >= 1
        else:
            assert cfg.fluxel.refinement_regions == []


def test_fluxel_motion_rejects_unknown_value():
    repo = Path(__file__).resolve().parents[3]
    config_path = (
        repo / "examples" / "cavity" / "gridfoam" / "data" / "config.yaml"
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["fluxel"]["motion"] = "moving"
    try:
        GridfoamConfig.model_validate(raw)
    except ValidationError:
        return
    raise AssertionError("unknown motion value should fail validation")
