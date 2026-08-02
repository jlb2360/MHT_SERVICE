# Spec 02: Mahalanobis Gating

## Objective
Implement a gating statistic based on the mahalanobis distance which will reject measurements from becoming another hypothesis branch. This module must be a pure, stateless mathematical engine.

## Interfaces & Contracts

### 1. State Representation
*   **Prediction State (`x`)**: NumPy array of shape `(4,)` representing `[x_position, y_position, x_velocity, y_velocity]`.
*   **Prediction Covariance (`P`)**: NumPy array of shape `(4, 4)`.
*   **Measurement State (`z`)**: NumPy array of shape `(2,)` representing `[x_position, y_position]`.
*   **Measurement Covariance (`R`)**: NumPy array of shape `(2, 2)`.

### 2. Required Functions / Methods
*   **`measurement_gate(x, P, z, R, chi)` -> Tuple[`bool`, `float`]**
    *   `chi`: Float (statistical threshold).
    *   Returns whether the measurement is close enough that it could statistically be the target as a boolean. It also returns the mahalanobis value.

## Mathematical Constraints
*   **Covariance Conditioning:** After both `predict` and `update` steps, the covariance matrix MUST be explicitly symmetrized: `P = 0.5 * (P + P.T)`.
*   **Inversion:** Use `scipy.linalg.solve` or `scipy.linalg.cho_solve` for calculating the mahalanobis distance by solving for the intermediate vector v from Lv=$\bar{y}$. Where $\bar{y}$ is the innovation. Do NOT use `numpy.linalg.inv`.

## Out of Scope
*   Handling multiple tracks or dictionaries of tracks.
*   peroming the prediction step or update step of the kalman filter
*   Data association or track lifecycle management.

## Verification Command
Run the following command. The task is ONLY complete when this command exits with code 0. If the test file does not exist, you must write it first based on the constraints above.

`uv run pytest tests/unit/test_02_mahalanobis.py -v --cov=src/mht_service/tracking --cov-fail-under=95`