# Spec 01: 2D Constant Velocity Kalman Filter

## Objective
Implement a discrete-time 2D Constant Velocity (CV) Kalman Filter to handle state prediction and measurement updates for individual tracks. This module must be a pure, stateless mathematical engine.

## Interfaces & Contracts

### 1. State Representation
*   **State Mean (`x`)**: NumPy array of shape `(4,)` representing `[x_position, y_position, x_velocity, y_velocity]`.
*   **State Covariance (`P`)**: NumPy array of shape `(4, 4)`.

### 2. Required Functions / Methods
*   **`predict(x, P, dt, q_variance)` -> `tuple[np.ndarray, np.ndarray]`**
    *   `dt`: Float (time step in seconds).
    *   `q_variance`: Float (process noise intensity).
    *   Returns the predicted state mean and covariance for the next time step.
*   **`update(x, P, z, R)` -> `tuple[np.ndarray, np.ndarray, float]`**
    *   `z`: NumPy array of shape `(2,)` representing `[x_measured, y_measured]`.
    *   `R`: NumPy array of shape `(2, 2)` representing measurement noise covariance.
    *   Returns the updated state mean, updated covariance, and the log-likelihood of the measurement innovation.

## Mathematical Constraints
*   **State Transition Matrix (F):** Must implement standard 2D kinematic equations for constant velocity based on `dt`.
*   **Process Noise Matrix (Q):** Must be constructed using the discrete white noise model (piecewise constant white noise) parameterized by `q_variance` and `dt`.
*   **Covariance Conditioning:** After both `predict` and `update` steps, the covariance matrix MUST be explicitly symmetrized: `P = 0.5 * (P + P.T)`.
*   **Joseph Form Update:** The measurement update step MUST use the numerically stable Joseph form for the covariance update to prevent loss of positive-definiteness:
    `P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T`
*   **Inversion:** Use `scipy.linalg.solve` or `scipy.linalg.cho_solve` for calculating the Kalman Gain. Do NOT use `numpy.linalg.inv`.

## Out of Scope
*   Handling multiple tracks or dictionaries of tracks.
*   Spatial gating (Mahalanobis distance validation).
*   Data association or track lifecycle management.

## Verification Command
Run the following command. The task is ONLY complete when this command exits with code 0. If the test file does not exist, you must write it first based on the constraints above.

`uv run pytest tests/unit/test_01_kalman.py -v --cov=src/mht_service/tracking --cov-fail-under=95`