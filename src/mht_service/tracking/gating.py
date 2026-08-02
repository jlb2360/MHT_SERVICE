"""
Mahalanobis Distance Gating Implementation (Spec 02).
Pure, stateless mathematical engine for measurement gating.
"""

import numpy as np
import numpy.typing as npt
import scipy.linalg


def measurement_gate(
    x: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    chi: float,
) -> bool:
    """
    Determine whether a measurement is statistically close enough to the predicted state
    using Mahalanobis distance gating.

    Args:
        x: State mean array of shape (4,) representing [x, y, vx, vy].
        P: State covariance matrix of shape (4, 4).
        z: Measurement array of shape (2,) representing [x_measured, y_measured].
        R: Measurement noise covariance matrix of shape (2, 2).
        chi: Statistical threshold (chi-squared threshold for Mahalanobis distance squared).

    Returns:
        bool: True if measurement is within the gating threshold, False otherwise.
    """
    # Measurement matrix H (observes position [x, y])
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Innovation (residual) v = z - H @ x
    hx = H @ x
    v = z - hx

    # Innovation covariance S = H @ P @ H^T + R
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)

    # Cholesky decomposition of S for stable linear solving
    try:
        c, low = scipy.linalg.cho_factor(S, lower=True)
    except scipy.linalg.LinAlgError:
        # Fallback jitter if S is ill-conditioned or singular
        S_jittered = S + np.eye(2) * 1e-9
        c, low = scipy.linalg.cho_factor(S_jittered, lower=True)

    # Solve S @ inv_S_v = v  =>  inv_S_v = cho_solve((c, low), v)
    inv_S_v = scipy.linalg.cho_solve((c, low), v)

    # Mahalanobis distance squared d^2 = v^T * S^{-1} * v
    mahalanobis_sq = float(np.dot(v, inv_S_v))

    # Compare squared Mahalanobis distance against chi (chi-squared threshold)
    return mahalanobis_sq <= chi, mahalanobis_sq
