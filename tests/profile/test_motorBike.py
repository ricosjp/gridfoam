"""Optional pytest entry point for the shared CPU/CUDA profiling workload."""

import pytest
import torch
from tests.profile.case import (
    DEFAULT_CONFIG,
    configure_device,
    load_case,
    run_case,
)


@pytest.mark.profile
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_motorBike_profile(device: str) -> None:
    """
    The shared workload completes nonempty mesh and time-step work on each
    device.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    configure_device(device, cpu_threads=None)
    result = run_case(load_case(DEFAULT_CONFIG, device))
    assert result.n_cells > 0
    assert result.n_steps > 0
