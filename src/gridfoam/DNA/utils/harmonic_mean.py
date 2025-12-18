import torch


def harmonic_mean(
    forward: torch.Tensor,
    backward: torch.Tensor,
) -> torch.Tensor:
    denom = forward + backward
    face = torch.zeros_like(denom)
    nonzero = denom != 0
    face[nonzero] = (
        2.0 * forward[nonzero] * backward[nonzero] / denom[nonzero]
    )
    return face
