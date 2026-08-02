"""
Unit tests for TrackManager class (Spec 04).
Verifies initialization, root track creation, node addition, scan processing, threshold pruning, n-scan pruning, global hypothesis extraction via Lagrangian relaxation, and branch deletion.
"""

import numpy as np
import pytest

from mht_service.tracking.track_manager import TrackManager
from mht_service.tracking.track_node import TrackNode


def test_track_manager_initialization():
    """Test TrackManager initialization with default and custom parameters."""
    manager = TrackManager(
        vel_assumption=np.array([1.0, 2.0], dtype=np.float64),
        init_cov=np.eye(2, dtype=np.float64) * 5.0,
        q_variance=0.1,
        min_score=-10.0,
        n_steps=3,
        alpha=0.5
    )

    assert manager.next_node_id == 1
    assert manager.next_tree_id == 1
    assert len(manager.registry) == 0
    assert len(manager.active_leaves) == 0
    assert len(manager.trees) == 0
    assert np.allclose(manager.vel_assumption, [1.0, 2.0])
    assert np.allclose(manager.init_cov, np.eye(2) * 5.0)
    assert manager.q_variance == 0.1
    assert manager.min_score == -10.0
    assert manager.n_steps == 3
    assert manager.alpha == 0.5


def test_create_new_root_track():
    """Test _create_new_root_track correctly initializes root node and registry entries."""
    manager = TrackManager(
        vel_assumption=np.array([2.0, 3.0], dtype=np.float64),
        init_cov=np.eye(2, dtype=np.float64) * 4.0
    )

    measurement = np.array([10.0, 20.0], dtype=np.float64)
    R = np.eye(2, dtype=np.float64) * 1.0

    root_id = manager._create_new_root_track(
        meas_id=101,
        measurement=measurement,
        R=R,
        timestamp=0.0
    )

    assert root_id == 1
    assert manager.next_node_id == 2
    assert manager.next_tree_id == 2
    assert root_id in manager.registry
    assert root_id in manager.active_leaves
    assert 1 in manager.trees
    assert root_id in manager.trees[1]

    root_node = manager.registry[root_id]
    assert root_node.track_id == root_id
    assert root_node.tree_id == 1
    assert root_node.parent_id is None
    assert np.allclose(root_node.x[:2], [10.0, 20.0])
    assert np.allclose(root_node.x[2:], [2.0, 3.0])
    assert root_node.history == (101,)
    assert root_node.P.shape == (4, 4)


def test_add_and_delete_branch():
    """Test adding nodes and recursively deleting branches."""
    manager = TrackManager()
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    node1 = TrackNode(track_id=1, x=x, P=P, tree_id=10)
    manager._add_node(node1, manager.active_leaves)

    assert 1 in manager.registry
    assert 1 in manager.active_leaves
    assert 10 in manager.trees
    assert 1 in manager.trees[10]

    node2 = TrackNode(track_id=2, x=x, P=P, tree_id=10, parent_id=1)
    node1.register_child(2)
    manager._add_node(node2, manager.active_leaves)
    manager.active_leaves.discard(1)  # 1 is no longer a leaf

    assert 2 in manager.registry
    assert 2 in manager.active_leaves
    assert 1 not in manager.active_leaves

    # Delete branch rooted at node 1 (should delete 1 and 2)
    manager._delete_branch(1)

    assert 1 not in manager.registry
    assert 2 not in manager.registry
    assert 2 not in manager.active_leaves
    assert 10 not in manager.trees or len(manager.trees[10]) == 0


def test_process_scan():
    """Test process_scan handles prediction, missed detections, gating, update, and new roots."""
    manager = TrackManager(min_score=-50.0)

    # Initial scan creates a root track
    measurements_t0 = {1: np.array([0.0, 0.0], dtype=np.float64)}
    Rs_t0 = {1: np.eye(2, dtype=np.float64)}

    manager.process_scan(
        measurements=measurements_t0,
        Rs=Rs_t0,
        dt=1.0,
        timestamp=0.0,
        gate_threshold=9.21,
        p_d=0.9,
        lambda_fa=1e-4
    )

    assert len(manager.active_leaves) == 1
    root_id = list(manager.active_leaves)[0]

    # Second scan with an associated measurement and a new measurement (unassociated -> new root)
    measurements_t1 = {
        10: np.array([1.1, 0.9], dtype=np.float64),  # updates existing track
        11: np.array([50.0, 50.0], dtype=np.float64) # new root track
    }
    Rs_t1 = {
        10: np.eye(2, dtype=np.float64),
        11: np.eye(2, dtype=np.float64)
    }

    manager.process_scan(
        measurements=measurements_t1,
        Rs=Rs_t1,
        dt=1.0,
        timestamp=1.0,
        gate_threshold=9.21,
        p_d=0.9,
        lambda_fa=1e-4
    )

    # Should have update child, missed child (from original), and new root
    assert len(manager.active_leaves) >= 2


def test_prune_threshold():
    """Test prune_threshold removes leaves whose score is below min_score."""
    manager = TrackManager(min_score=-5.0)
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    node_good = TrackNode(track_id=1, x=x, P=P, tree_id=1, score=0.0)
    node_bad = TrackNode(track_id=2, x=x, P=P, tree_id=2, score=-10.0)

    manager._add_node(node_good, manager.active_leaves)
    manager._add_node(node_bad, manager.active_leaves)

    manager.prune_threshold()

    assert 1 in manager.registry
    assert 1 in manager.active_leaves
    assert 2 not in manager.registry
    assert 2 not in manager.active_leaves


def test_prune_n_scan():
    """Test prune_n_scan preserves ancestry up to n_steps and prunes older branches."""
    manager = TrackManager(n_steps=1)
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    # Build chain: root (1) -> child (2) -> grandchild (3)
    node1 = TrackNode(track_id=1, x=x, P=P, tree_id=1, timestamp=0.0)
    manager._add_node(node1, manager.active_leaves)
    manager.active_leaves.discard(1)

    node2 = TrackNode(track_id=2, x=x, P=P, tree_id=1, parent_id=1, timestamp=1.0)
    node1.register_child(2)
    manager._add_node(node2, manager.active_leaves)
    manager.active_leaves.discard(2)

    node3 = TrackNode(track_id=3, x=x, P=P, tree_id=1, parent_id=2, timestamp=2.0)
    node2.register_child(3)
    manager._add_node(node3, manager.active_leaves)

    # Prune with n_steps=1 should keep node 2 and node 3, but prune node 1 if no other children
    manager.prune_n_scan()
    assert 3 in manager.active_leaves


def test_get_best_global_hypothesis():
    """Test get_best_global_hypothesis resolves non-conflicting track leaves with highest score."""
    manager = TrackManager()
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    # Leaf 1 claims measurement 10
    leaf1 = TrackNode(track_id=1, x=x, P=P, tree_id=1, score=10.0, history=(10,))
    # Leaf 2 conflicts with Leaf 1 (also claims measurement 10)
    leaf2 = TrackNode(track_id=2, x=x, P=P, tree_id=1, score=15.0, history=(10,))
    # Leaf 3 is non-conflicting (claims measurement 20)
    leaf3 = TrackNode(track_id=3, x=x, P=P, tree_id=2, score=8.0, history=(20,))

    manager._add_node(leaf1, manager.active_leaves)
    manager._add_node(leaf2, manager.active_leaves)
    manager._add_node(leaf3, manager.active_leaves)

    # Among leaf1 and leaf2, leaf2 has higher score (15 > 10). Leaf3 is non-conflicting.
    # So best global hypothesis should include leaf2 and leaf3 (or leaf2 + leaf3).
    best_hyp = manager.get_best_global_hypothesis()
    best_ids = {node.track_id for node in best_hyp}

    assert 2 in best_ids
    assert 3 in best_ids
    assert 1 not in best_ids


def test_get_best_global_hypothesis_empty():
    """Test get_best_global_hypothesis returns empty list when no active leaves."""
    manager = TrackManager()
    assert manager.get_best_global_hypothesis() == []


def test_get_best_global_hypothesis_no_measurements():
    """Test get_best_global_hypothesis when active leaves have no measurements."""
    manager = TrackManager()
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    leaf1 = TrackNode(track_id=1, x=x, P=P, tree_id=1, score=10.0, history=())
    leaf2 = TrackNode(track_id=2, x=x, P=P, tree_id=2, score=25.0, history=())
    manager._add_node(leaf1, manager.active_leaves)
    manager._add_node(leaf2, manager.active_leaves)

    best = manager.get_best_global_hypothesis()
    assert len(best) == 1
    assert best[0].track_id == 2


def test_prune_n_scan_sibling_deletion():
    """Test prune_n_scan prunes sibling branches of the N-steps-back ancestor."""
    manager = TrackManager(n_steps=1)
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    # 0 -> 1 -> 2 (best leaf)
    #   -> 3 (sibling to 1 under 0)
    node0 = TrackNode(track_id=0, x=x, P=P, tree_id=1)
    manager._add_node(node0, manager.active_leaves)
    manager.active_leaves.discard(0)

    node1 = TrackNode(track_id=1, x=x, P=P, tree_id=1, parent_id=0)
    node0.register_child(1)
    manager._add_node(node1, manager.active_leaves)
    manager.active_leaves.discard(1)

    node3 = TrackNode(track_id=3, x=x, P=P, tree_id=1, parent_id=0, score=-10.0)
    node0.register_child(3)
    manager._add_node(node3, manager.active_leaves)

    node2 = TrackNode(track_id=2, x=x, P=P, tree_id=1, parent_id=1, score=10.0)
    node1.register_child(2)
    manager._add_node(node2, manager.active_leaves)

    # With n_steps=1, best_leaf is node 2. Walk back 1 step -> node 1.
    # ancestor = node 1. ancestor.parent_id = 0 (not None).
    # parent_of_ancestor = node 0. children = {1, 3}.
    # Sibling 3 should be deleted.
    manager.prune_n_scan()

    assert 3 not in manager.registry
    assert 2 in manager.active_leaves


def test_get_best_global_hypothesis_subgradient_and_fallback():
    """Test subgradient optimization coverage branches (<0 adjusted score, fallback with conflicts)."""
    manager = TrackManager(alpha=10.0)
    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)

    # Leaf 1 and Leaf 2 claim the same measurement (10), causing conflict in fallback
    leaf1 = TrackNode(track_id=1, x=x, P=P, tree_id=1, score=-10.0, history=(10,))
    leaf2 = TrackNode(track_id=2, x=x, P=P, tree_id=2, score=-5.0, history=(10,))

    manager._add_node(leaf1, manager.active_leaves)
    manager._add_node(leaf2, manager.active_leaves)

    best = manager.get_best_global_hypothesis()
    assert len(best) == 1
    assert best[0].track_id == 2



