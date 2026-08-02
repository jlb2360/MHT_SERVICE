"""
Unit tests for the 2D Constant Velocity Kalman Filter (Spec 01).
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst

from mht_service.tracking.kalman import predict, update


def test_predict_basic() -> None:
    """Test basic prediction step with known values."""
    x = np.array([10.0, 20.0, 5.0, -2.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64) * 2.0
    dt = 1.0
    q_variance = 0.1

    x_pred, P_pred = predict(x, P, dt, q_variance)

    # Check shapes
    assert x_pred.shape == (4,)
    assert P_pred.shape == (4, 4)

    # Expected mean: x + vx*dt, y + vy*dt, vx, vy
    # [10 + 5*1, 20 + (-2)*1, 5, -2] = [15.0, 18.0, 5.0, -2.0]
    expected_x = np.array([15.0, 18.0, 5.0, -2.0], dtype=np.float64)
    np.testing.assert_allclose(x_pred, expected_x, rtol=1e-5, atol=1e-8)

    # Check symmetry of P_pred
    np.testing.assert_allclose(P_pred, P_pred.T, rtol=1e-5, atol=1e-8)

    # Check positive semi-definiteness (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(P_pred)
    assert np.all(eigenvalues >= -1e-12)


def test_update_basic() -> None:
    """Test basic measurement update step with known values."""
    x = np.array([15.0, 18.0, 5.0, -2.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64) * 2.0
    z = np.array([15.2, 17.9], dtype=np.float64)
    R = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)

    x_upd, P_upd, log_likelihood = update(x, P, z, R)

    # Check shapes
    assert x_upd.shape == (4,)
    assert P_upd.shape == (4, 4)
    assert isinstance(log_likelihood, float)
    assert np.isfinite(log_likelihood)

    # Check symmetry of P_upd
    np.testing.assert_allclose(P_upd, P_upd.T, rtol=1e-5, atol=1e-8)

    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(P_upd)
    assert np.all(eigenvalues >= -1e-12)


def test_covariance_symmetrization() -> None:
    """Test that output covariances are strictly symmetric even with asymmetric inputs."""
    x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    # Construct an asymmetric positive-definite-like matrix
    P = np.array([
        [2.0, 0.1, 0.0, 0.0],
        [0.2, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.1],
        [0.0, 0.0, 0.2, 2.0],
    ], dtype=np.float64)

    dt = 0.5
    q_variance = 0.01

    x_pred, P_pred = predict(x, P, dt, q_variance)
    np.testing.assert_allclose(P_pred, P_pred.T, rtol=1e-5, atol=1e-8)

    z = np.array([0.5, 0.5], dtype=np.float64)
    R = np.eye(2, dtype=np.float64)
    x_upd, P_upd, ll = update(x_pred, P_pred, z, R)
    np.testing.assert_allclose(P_upd, P_upd.T, rtol=1e-5, atol=1e-8)


@given(
    dt=st.floats(min_value=0.1, max_value=5.0),
    q_variance=st.floats(min_value=1e-3, max_value=10.0),
)
def test_predict_properties(dt: float, q_variance: float) -> None:
    """Property-based tests for predict function using Hypothesis."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=4)
    # Generate random PSD covariance matrix via A * A^T + I
    A = rng.normal(size=(4, 4))
    P = A @ A.T + np.eye(4)

    x_pred, P_pred = predict(x, P, dt, q_variance)

    assert x_pred.shape == (4,)
    assert P_pred.shape == (4, 4)
    np.testing.assert_allclose(P_pred, P_pred.T, rtol=1e-5, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(P_pred) >= -1e-10)


@given(
    z_noise=npst.arrays(np.float64, (2,), elements=st.floats(min_value=-1.0, max_value=1.0)),
)
def test_update_properties(z_noise: np.ndarray) -> None:
    """Property-based tests for update function using Hypothesis."""
    rng = np.random.default_rng(123)
    x = np.array([10.0, 20.0, 2.0, 3.0], dtype=np.float64)
    A = rng.normal(size=(4, 4))
    P = A @ A.T + np.eye(4)

    # Measurement z near predicted position with some noise
    z = x[:2] + z_noise

    # Valid measurement noise covariance R
    B = rng.normal(size=(2, 2))
    R = B @ B.T + np.eye(2) * 0.1

    x_upd, P_upd, ll = update(x, P, z, R)

    assert x_upd.shape == (4,)
    assert P_upd.shape == (4, 4)
    np.testing.assert_allclose(P_upd, P_upd.T, rtol=1e-5, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(P_upd) >= -1e-10)
    assert isinstance(ll, float)
    assert np.isfinite(ll)


def test_update_singular_S_fallback() -> None:
    """Test update when innovation covariance S is singular/ill-conditioned to trigger fallback."""
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    P = np.zeros((4, 4), dtype=np.float64)
    z = np.array([0.0, 0.0], dtype=np.float64)
    R = np.zeros((2, 2), dtype=np.float64)  # S will be all zeros -> LinAlgError -> jitter fallback

    x_upd, P_upd, ll = update(x, P, z, R)
    assert x_upd.shape == (4,)
    assert P_upd.shape == (4, 4)
    assert np.isfinite(ll)

