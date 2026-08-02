"""
2D Constant Velocity Kalman Filter Implementation (Spec 01).
Pure, stateless mathematical engine for state prediction and measurement update.
"""

import numpy as np
import numpy.typing as npt
import scipy.linalg


def predict(
    x: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    dt: float,
    q_variance: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Predict state mean and covariance for the next time step using a 2D constant velocity model.

    Args:
        x: State mean array of shape (4,) representing [x, y, vx, vy] in meters and meters/sec.
        P: State covariance matrix of shape (4, 4).
        dt: Time step in seconds (float >= 0).
        q_variance: Process noise intensity (float >= 0).

    Returns:
        tuple containing:
            - x_pred: Predicted state mean array of shape (4,).
            - P_pred: Predicted state covariance matrix of shape (4, 4), symmetrized.
    """
    # State transition matrix F for 2D constant velocity model
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Process noise covariance matrix Q (discrete white noise model)
    dt2 = dt**2 / 2.0
    dt3 = dt**3 / 3.0
    q = q_variance

    Q = np.zeros((4, 4), dtype=np.float64)
    Q[0, 0] = dt3 * q
    Q[1, 1] = dt3 * q
    Q[0, 2] = Q[2, 0] = dt2 * q
    Q[1, 3] = Q[3, 1] = dt2 * q
    Q[2, 2] = dt * q
    Q[3, 3] = dt * q

    # Predict state mean
    x_pred = F @ x

    # Predict covariance
    P_pred = F @ P @ F.T + Q

    # Enforce symmetry
    P_pred = 0.5 * (P_pred + P_pred.T)

    return x_pred, P_pred


def update(
    x: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """
    Perform measurement update on predicted state mean and covariance.

    Args:
        x: Predicted state mean array of shape (4,) representing [x, y, vx, vy].
        P: Predicted state covariance matrix of shape (4, 4).
        z: Measurement array of shape (2,) representing [x_measured, y_measured] in meters.
        R: Measurement noise covariance matrix of shape (2, 2).

    Returns:
        tuple containing:
            - x_upd: Updated state mean array of shape (4,).
            - P_upd: Updated state covariance matrix of shape (4, 4), symmetrized (Joseph form).
            - log_likelihood: Log-likelihood of the measurement innovation (float).
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

    # Cholesky decomposition of S for stable linear solving and log-determinant
    try:
        c, low = scipy.linalg.cho_factor(S, lower=True)
    except scipy.linalg.LinAlgError:
        # Fallback or jitter if S is ill-conditioned
        S_jittered = S + np.eye(2) * 1e-9
        c, low = scipy.linalg.cho_factor(S_jittered, lower=True)
        S = S_jittered

    # Kalman Gain K = P @ H^T @ inv(S)
    # Solve S @ K^T = H @ P  =>  K^T = cho_solve((c, low), H @ P)
    HP = H @ P
    K_transpose = scipy.linalg.cho_solve((c, low), HP)
    K = K_transpose.T

    # Update state mean
    x_upd = x + K @ v

    # Update state covariance using numerically stable Joseph form:
    # P_upd = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
    I_4 = np.eye(4, dtype=np.float64)
    I_KH = I_4 - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T

    # Enforce symmetry
    P_upd = 0.5 * (P_upd + P_upd.T)

    # Compute log-likelihood of innovation
    # ln L = -0.5 * (v^T S^{-1} v + ln|S| + k * ln(2*pi))
    inv_S_v = scipy.linalg.cho_solve((c, low), v)
    mahalanobis_sq = np.dot(v, inv_S_v)
    log_det_S = 2.0 * np.sum(np.log(np.diagonal(c)))
    k = 2.0  # dimensionality of measurement z
    log_likelihood = -0.5 * (mahalanobis_sq + log_det_S + k * np.log(2.0 * np.pi))

    return x_upd, P_upd, float(log_likelihood)
