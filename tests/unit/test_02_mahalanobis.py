"""
Unit tests for the Mahalanobis Distance Gating (Spec 02).
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst

from mht_service.tracking.gating import measurement_gate


def test_measurement_gate_basic() -> None:
    """Test basic measurement gating with known values."""
    x = np.array([10.0, 20.0, 5.0, -2.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64) * 2.0
    z = np.array([10.1, 19.9], dtype=np.float64)  # Very close to predicted position [10, 20]
    R = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    chi = 9.21  # 99% confidence threshold for 2 DOF chi-squared distribution

    # Should pass gate (true)
    assert measurement_gate(x, P, z, R, chi)[0] is True

    # Test measurement far away
    z_far = np.array([100.0, 200.0], dtype=np.float64)
    assert measurement_gate(x, P, z_far, R, chi)[0] is False


def test_measurement_gate_threshold_edge_case() -> None:
    """Test gating behavior with extreme thresholds (negative, zero, infinite)."""
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    z = np.array([0.1, 0.1], dtype=np.float64)
    R = np.eye(2, dtype=np.float64)

    # Negative chi: squared Mahalanobis distance is always >= 0, so gate should fail
    assert measurement_gate(x, P, z, R, -1.0)[0] is False

    # Inf chi: should always pass
    assert measurement_gate(x, P, z, R, float("inf"))[0] is True


def test_measurement_gate_singular_S_fallback() -> None:
    """Test measurement gating when innovation covariance S is singular/ill-conditioned to trigger fallback."""
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    P = np.zeros((4, 4), dtype=np.float64)
    z = np.array([0.0, 0.0], dtype=np.float64)
    R = np.zeros((2, 2), dtype=np.float64)  # S will be all zeros -> LinAlgError -> jitter fallback
    chi = 5.99

    # Should not raise exception and return bool
    result, _ = measurement_gate(x, P, z, R, chi)
    assert isinstance(result, bool)


@given(
    z_noise=npst.arrays(np.float64, (2,), elements=st.floats(min_value=-2.0, max_value=2.0)),
    chi=st.floats(min_value=1.0, max_value=20.0),
)
def test_measurement_gate_properties(z_noise: np.ndarray, chi: float) -> None:
    """Property-based tests for measurement_gate function using Hypothesis."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=4)
    # Generate random PSD covariance matrix via A * A^T + I
    A = rng.normal(size=(4, 4))
    P = A @ A.T + np.eye(4)

    # Measurement near predicted position + noise
    z = x[:2] + z_noise

    # Valid measurement noise covariance R
    B = rng.normal(size=(2, 2))
    R = B @ B.T + np.eye(2) * 0.1

    result = measurement_gate(x, P, z, R, chi)[0]
    assert isinstance(result, bool)

    # If chi is extremely large, should always be True
    assert measurement_gate(x, P, z, R, 1e12)[0] is True

    # If chi is negative, should always be False
    assert measurement_gate(x, P, z, R, -0.1)[0] is False
