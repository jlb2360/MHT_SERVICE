"""
Unit tests for TrackNode class (Spec 03).
Verifies state representation, prediction, gating, update child creation, missed child creation, child registration, and conflict detection.
"""

import numpy as np
import pytest

from mht_service.tracking.track_node import TrackNode


def test_track_node_initialization():
    """Test TrackNode initialization and attribute assignments."""
    x = np.array([10.0, 20.0, 1.0, 2.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    node = TrackNode(
        track_id=1,
        x=x,
        P=P,
        tree_id=100,
        parent_id=None,
        timestamp=0.0,
        score=0.0,
        history=()
    )

    assert node.track_id == 1
    assert node.parent_id is None
    assert node.tree_id == 100
    assert np.allclose(node.x, x)
    assert np.allclose(node.P, P)
    assert node.score == 0.0
    assert node.score_id == 0.0
    assert node.timestamp == 0.0
    assert node.children_ids == []
    assert node.history == ()


def test_score_setters():
    """Test score and score_id property setters."""
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    node = TrackNode(track_id=1, x=x, P=P, tree_id=1)

    node.score = 5.5
    assert node.score == 5.5
    assert node.score_id == 5.5

    node.score_id = 12.3
    assert node.score == 12.3
    assert node.score_id == 12.3


def test_track_node_predict():
    """Test predict method updates state mean, covariance, and timestamp."""
    x = np.array([0.0, 0.0, 10.0, 5.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    node = TrackNode(track_id=1, x=x, P=P, tree_id=1, timestamp=0.0)

    node.predict(dt=1.0, timestamp=1.0, q_variance=0.1)

    assert node.timestamp == 1.0
    assert np.allclose(node.x[:2], [10.0, 5.0])
    assert node.P.shape == (4, 4)
    # Check symmetry
    assert np.allclose(node.P, node.P.T)


def test_evaluate_gate():
    """Test evaluate_gate returns (bool, float) for Mahalanobis distance gating."""
    x = np.array([10.0, 10.0, 1.0, 1.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    R = np.eye(2, dtype=np.float64)
    node = TrackNode(track_id=1, x=x, P=P, tree_id=1)

    # Measurement close to prediction
    meas_close = np.array([10.5, 10.5], dtype=np.float64)
    inside, d2 = node.evaluate_gate(meas_close, R=R, gate_threshold=9.21)  # chi2(2) for p=0.01 is 9.21
    assert isinstance(inside, bool)
    assert isinstance(d2, float)
    assert inside is True
    assert d2 >= 0.0

    # Measurement far from prediction
    meas_far = np.array([100.0, 100.0], dtype=np.float64)
    inside_far, d2_far = node.evaluate_gate(meas_far, R=R, gate_threshold=9.21)
    assert inside_far is False
    assert d2_far > 9.21


def test_evaluate_gate_singular_matrix():
    """Test evaluate_gate handles singular covariance using jitter fallback."""
    x = np.zeros(4, dtype=np.float64)
    P = np.zeros((4, 4), dtype=np.float64)
    R = np.zeros((2, 2), dtype=np.float64)
    node = TrackNode(track_id=1, x=x, P=P, tree_id=1)

    inside, d2 = node.evaluate_gate(np.array([1.0, 1.0], dtype=np.float64), R=R, gate_threshold=9.21)
    assert isinstance(inside, bool)
    assert isinstance(d2, float)


def test_create_update_child():
    """Test creating an update child updates state, score, history, and registers child."""
    x = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    R = np.eye(2, dtype=np.float64)
    parent = TrackNode(track_id=1, x=x, P=P, tree_id=10, score=0.0, history=(101,))

    measurement = np.array([5.1, 4.9], dtype=np.float64)
    child = parent.create_update_child(
        child_id=2,
        meas_id=102,
        measurement=measurement,
        R=R,
        timestamp=1.0,
        p_d=0.9,
        lambda_fa=1e-4,
    )

    assert child.track_id == 2
    assert child.parent_id == 1
    assert child.tree_id == 10
    assert child.timestamp == 1.0
    assert child.history == (101, 102)
    assert child.score != 0.0
    assert 2 in parent.children_ids


def test_history_truncation():
    """Test that history is truncated to at most 3 items."""
    x = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    R = np.eye(2, dtype=np.float64)
    parent = TrackNode(track_id=1, x=x, P=P, tree_id=10, history=(1, 2, 3))

    child = parent.create_update_child(
        child_id=2,
        meas_id=4,
        measurement=np.array([0.0, 0.0], dtype=np.float64),
        R=R,
        timestamp=1.0,
        p_d=0.9,
        lambda_fa=1e-4,
    )

    assert child.history == (2, 3, 4)


def test_create_missed_child():
    """Test creating a missed detection child updates score and copies history."""
    x = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    parent = TrackNode(track_id=1, x=x, P=P, tree_id=10, score=0.0, history=(101,))

    child = parent.create_missed_child(
        child_id=3,
        timestamp=1.0,
        p_d=0.9,
    )

    assert child.track_id == 3
    assert child.parent_id == 1
    assert child.tree_id == 10
    assert child.timestamp == 1.0
    assert child.history == (101,)
    # Missed detection score penalty: ln(1 - p_d) = ln(0.1) < 0
    assert child.score < parent.score
    assert 3 in parent.children_ids


def test_register_child():
    """Test register_child adds child ID to list without duplicates."""
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    node = TrackNode(track_id=1, x=x, P=P, tree_id=1)

    node.register_child(5)
    node.register_child(6)
    node.register_child(5)  # duplicate

    assert node.children_ids == [5, 6]


def test_conflicts_with():
    """Test conflicts_with detects shared measurement history."""
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    node1 = TrackNode(track_id=1, x=x, P=P, tree_id=1, history=(10, 20))
    node2 = TrackNode(track_id=2, x=x, P=P, tree_id=1, history=(20, 30))
    node3 = TrackNode(track_id=3, x=x, P=P, tree_id=1, history=(30, 40))

    assert node1.conflicts_with(node2) is True  # shared measurement 20
    assert node1.conflicts_with(node3) is False # disjoint histories
