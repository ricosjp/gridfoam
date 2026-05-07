"""Load example YAML and validate ``GridfoamConfig``."""

from __future__ import annotations

from pathlib import Path

import yaml

from gridfoam.meta.config import GridfoamConfig


def test_example_cavity_config_loads():
    repo = Path(__file__).resolve().parents[2]
    config_path = (
        repo / "examples" / "cavity" / "gridfoam" / "data" / "config.yml"
    )
    assert config_path.is_file(), f"missing example config: {config_path}"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = GridfoamConfig.model_validate(raw)
    assert cfg.simulator.device.value in ("cpu", "cuda")
    assert cfg.fluxel.ibm_type.value == "axis_projected"
