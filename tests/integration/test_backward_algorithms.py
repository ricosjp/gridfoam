"""Backward history advances once per time step through pressure correctors."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tests.helpers import channel_flow_config

from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
from gridfoam.fv.schemes.ddt import ddt_coefficients
from gridfoam.meta.config import PIMPLEAlgorithm, PISOAlgorithm, fvSchemesConfig
from gridfoam.meta.enums import AlgorithmType


@pytest.mark.parametrize("pimple", [False, True])
def test_backward_pressure_coupling_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pimple: bool
):
    algorithm = (
        PIMPLEAlgorithm(
            type=AlgorithmType.PIMPLE, nCorrectors=2, nOuterCorrectors=3
        )
        if pimple
        else PISOAlgorithm(type=AlgorithmType.PISO, nCorrectors=2)
    )
    config = channel_flow_config(tmp_path, algorithm)
    config = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={
                    "fvSchemes": fvSchemesConfig.model_validate(
                        {"ddtSchemes": {"default": "backward"}}
                    )
                }
            )
        }
    )
    grid = create_grid(config)
    algo = PIMPLE(grid) if pimple else PISO(grid)
    assert ddt_coefficients(algo.U) == (1.0, 1.0, 0.0)
    u_history = Mock(wraps=algo.U.update_history)
    phi_history = Mock(wraps=algo.phi.update_history)
    monkeypatch.setattr(algo.U, "update_history", u_history)
    monkeypatch.setattr(algo.phi, "update_history", phi_history)

    algo.step()
    first_u = algo.U.data.clone()
    first_phi = algo.phi.single_data.clone()
    assert ddt_coefficients(algo.U) == (1.5, 2.0, 0.5)
    algo.step()
    assert u_history.call_count == 2
    assert phi_history.call_count == 2
    torch.testing.assert_close(algo.U.older_data, first_u)
    torch.testing.assert_close(algo.phi.older_single_data, first_phi)
    assert float(fvc.div(algo.phi).data.abs().max()) < 1e-8

    grid.to("cpu")
    torch.testing.assert_close(algo.U.older_data, first_u)
    torch.testing.assert_close(algo.phi.older_single_data, first_phi)
    # The same fixed-topology synchronization is used after IBM updates.
    algo.U.sync_to_grid_topology()
    algo.phi.sync_to_grid_topology()
    assert ddt_coefficients(algo.U) == (1.0, 1.0, 0.0)
    assert algo.phi.older_single_data is None
