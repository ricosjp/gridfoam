import torch
from jaxtyping import Float


def non_orth_delta_coeffs(
    d_vec: Float[torch.Tensor, " F 3"],
    mag_Sf: Float[torch.Tensor, " F 1"],
    Sf: Float[torch.Tensor, " F 3"],
) -> Float[torch.Tensor, " F 1"]:
    """
    Compute OpenFOAM-style non-orthogonal delta coefficients.

    nonOrthDeltaCoeffs = 1 / max(n & d, 0.05 |d|).

    Parameters
    ----------
    d_vec : torch.Tensor
        Cell-centre distance vector for each face.
    mag_Sf : torch.Tensor
        Face area magnitude.
    Sf : torch.Tensor
        Face area vector.

    Returns
    -------
    torch.Tensor
        Non-orthogonal delta coefficients with shape ``[F, 1]``.
    """
    n_hat = Sf / mag_Sf
    mag_d = torch.linalg.vector_norm(d_vec, dim=1, keepdim=True)
    n_dot_d = torch.sum(n_hat * d_vec, dim=1, keepdim=True)
    return 1.0 / torch.clamp(n_dot_d, min=0.05 * mag_d)


def non_orth_correction_vectors(
    d_vec: Float[torch.Tensor, " F 3"],
    delta_coeffs: Float[torch.Tensor, " F 1"],
    mag_Sf: Float[torch.Tensor, " F 1"],
    Sf: Float[torch.Tensor, " F 3"],
) -> Float[torch.Tensor, " F 3"]:
    """
    Compute OpenFOAM-style non-orthogonal correction vectors.

    k = n - d * nonOrthDeltaCoeffs.

    Parameters
    ----------
    d_vec : torch.Tensor
        Cell-centre distance vector for each face.
    delta_coeffs : torch.Tensor
        Non-orthogonal delta coefficients.
    mag_Sf : torch.Tensor
        Face area magnitude.
    Sf : torch.Tensor
        Face area vector.

    Returns
    -------
    torch.Tensor
        Correction vectors with shape ``[F, 3]``.
    """
    n_hat = Sf / mag_Sf
    return n_hat - d_vec * delta_coeffs
