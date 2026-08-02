"""
Track Manager implementation for Multiple Hypothesis Tracking (Spec 04).
Manages track hypothesis trees, registration, scan processing, gating, score pruning, n-scan pruning, and global hypothesis extraction via Lagrangian relaxation.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import numpy.typing as npt

from mht_service.tracking.track_node import TrackNode


class TrackManager:
    """
    Manages multiple hypothesis tracking trees, including root creation,
    leaf updates, missed detection branching, gating, pruning, and global association.
    """

    def __init__(
        self,
        vel_assumption: Optional[npt.NDArray[np.float64]] = None,
        init_cov: Optional[npt.NDArray[np.float64]] = None,
        q_variance: float = 0.1,
        min_score: float = -20.0,
        n_steps: int = 3,
        alpha: float = 0.1,
    ) -> None:
        """
        Initialize the TrackManager.

        Args:
            vel_assumption: 2D array [vx, vy] default velocity assumption for new roots.
            init_cov: 2D array (2, 2) initial velocity uncertainty covariance.
            q_variance: Process noise variance for Kalman prediction.
            min_score: Minimum log-likelihood score threshold for pruning.
            n_steps: Number of steps looked back for N-scan pruning.
            alpha: Step size for Lagrangian relaxation dual variable updates.
        """
        self.vel_assumption: npt.NDArray[np.float64] = (
            np.array(vel_assumption, dtype=np.float64)
            if vel_assumption is not None
            else np.zeros(2, dtype=np.float64)
        )
        self.init_cov: npt.NDArray[np.float64] = (
            np.array(init_cov, dtype=np.float64)
            if init_cov is not None
            else np.eye(2, dtype=np.float64) * 10.0
        )
        self.q_variance: float = float(q_variance)
        self.min_score: float = float(min_score)
        self.n_steps: int = int(n_steps)
        self.alpha: float = float(alpha)

        self.registry: Dict[int, TrackNode] = {}
        self.active_leaves: Set[int] = set()
        self.trees: Dict[int, Set[int]] = {}
        self.next_node_id: int = 1
        self.next_tree_id: int = 1

    def _add_node(self, node: TrackNode, new_leaves: Set[int]) -> None:
        """
        Add a node to the registry, active leaves collection, and tree index.

        Args:
            node: The TrackNode to add.
            new_leaves: Set tracking newly added leaf IDs during scan processing.
        """
        self.registry[node.track_id] = node
        new_leaves.add(node.track_id)
        self.trees.setdefault(node.tree_id, set()).add(node.track_id)
        if node.parent_id is not None:
            self.registry[node.parent_id].register_child(node.track_id)

    def _create_new_root_track(
        self,
        meas_id: int,
        measurement: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        timestamp: float,
        target_leaves: Optional[Set[int]] = None,
    ) -> int:
        """
        Create a new root track from an unassociated measurement.

        Args:
            meas_id: Identifier of the unassociated measurement.
            measurement: 2D position array [x, y].
            R: Measurement noise covariance (2, 2).
            timestamp: Current timestamp.
            target_leaves: Optional set to add the new root node ID into (defaults to active_leaves).

        Returns:
            int: The created root node track ID.
        """
        track_id = self.next_node_id
        tree_id = self.next_tree_id
        self.next_node_id += 1
        self.next_tree_id += 1

        # Construct 4D state vector [x, y, vx, vy]
        x_init = np.array(
            [
                measurement[0],
                measurement[1],
                self.vel_assumption[0],
                self.vel_assumption[1],
            ],
            dtype=np.float64,
        )

        # Construct 4x4 initial covariance matrix P0:
        # Top-left 2x2 is R (position uncertainty), bottom-right 2x2 is init_cov (velocity uncertainty)
        P_init = np.zeros((4, 4), dtype=np.float64)
        P_init[:2, :2] = R
        P_init[2:, 2:] = self.init_cov

        root_node = TrackNode(
            track_id=track_id,
            x=x_init,
            P=P_init,
            tree_id=tree_id,
            parent_id=None,
            timestamp=timestamp,
            score=0.0,
            history=(meas_id,),
        )

        target_set = target_leaves if target_leaves is not None else self.active_leaves
        self._add_node(root_node, target_set)
        return track_id

    def _delete_branch(self, node_id: int) -> None:
        """
        Recursively delete a node and all its descendants from registry, active leaves, and trees.

        Args:
            node_id: Root of the branch to delete.
        """

        node = self.registry[node_id]

        # Recursively delete children
        for child_id in list(node.children_ids):
            self._delete_branch(child_id)

        # Remove from active leaves
        self.active_leaves.discard(node_id)

        # Remove from tree group
        if node.tree_id in self.trees:
            self.trees[node.tree_id].discard(node_id)
            if not self.trees[node.tree_id]:
                del self.trees[node.tree_id]

        # Remove from parent's children list if parent exists
        if node.parent_id is not None and node.parent_id in self.registry:
            parent = self.registry[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)

        # Remove from registry
        del self.registry[node_id]

    def process_scan(
        self,
        measurements: Dict[int, npt.NDArray[np.float64]],
        Rs: Dict[int, npt.NDArray[np.float64]],
        dt: float,
        timestamp: float,
        gate_threshold: float,
        p_d: float,
        lambda_fa: float,
    ) -> None:
        """
        Process the current set of measurements through prediction, gating, update branching,
        missed detection branching, and new root initiation.

        Args:
            measurements: Dictionary mapping measurement ID to 2D measurement array.
            Rs: Dictionary mapping measurement ID to 2x2 measurement noise covariance R.
            dt: Time step size in seconds.
            timestamp: Current timestamp.
            gate_threshold: Mahalanobis distance gating threshold.
            p_d: Probability of detection.
            lambda_fa: False alarm density.
        """
        current_leaves = list(self.active_leaves)
        new_leaves: Set[int] = set()
        associated_measurements: Set[int] = set()

        # Step 1: Predict all active leaves
        for leaf_id in current_leaves:
            leaf = self.registry[leaf_id]
            leaf.predict(dt, timestamp, self.q_variance)

        # Step 2 & 3: Evaluate gating and generate missed/update child hypotheses
        for leaf_id in current_leaves:
            leaf = self.registry[leaf_id]

            # Remove parent leaf from active leaves as it now branches
            self.active_leaves.discard(leaf_id)

            # Spawn missed detection child hypothesis
            missed_child_id = self.next_node_id
            self.next_node_id += 1
            missed_child = leaf.create_missed_child(
                child_id=missed_child_id,
                timestamp=timestamp,
                p_d=p_d,
            )
            self._add_node(missed_child, new_leaves)

            # Evaluate gating against all measurements
            for meas_id, meas in measurements.items():
                R = Rs.get(meas_id, np.eye(2, dtype=np.float64))
                inside, _ = leaf.evaluate_gate(meas, R, gate_threshold)
                if inside:
                    associated_measurements.add(meas_id)
                    update_child_id = self.next_node_id
                    self.next_node_id += 1
                    update_child = leaf.create_update_child(
                        child_id=update_child_id,
                        meas_id=meas_id,
                        measurement=meas,
                        R=R,
                        timestamp=timestamp,
                        p_d=p_d,
                        lambda_fa=lambda_fa,
                    )
                    self._add_node(update_child, new_leaves)

        # Step 4: Spawn new roots for measurements that were not attached to any current hypotheses
        for meas_id, meas in measurements.items():
            if meas_id not in associated_measurements:
                R = Rs.get(meas_id, np.eye(2, dtype=np.float64))
                self._create_new_root_track(
                    meas_id=meas_id,
                    measurement=meas,
                    R=R,
                    timestamp=timestamp,
                    target_leaves=new_leaves,
                )

        self.active_leaves = new_leaves

        # Apply post-scan pruning
        self.prune_threshold()
        self.prune_n_scan()

    def prune_threshold(self) -> None:
        """
        Prune all leaves whose likelihood score has dropped below the threshold of min_score.
        """
        for leaf_id in list(self.active_leaves):
            if self.registry[leaf_id].score < self.min_score:
                self._delete_branch(leaf_id)

    def prune_n_scan(self) -> None:
        """
        Perform N-scan pruning by locking in the best history/root N steps ago.
        Branches older than n_steps that do not belong to surviving consensus trees are pruned.
        """
        # Iterate over a list to avoid runtime errors if trees are deleted during iteration
        for tree_id, leaf_ids in list(self.trees.items()):
            if not leaf_ids:
                continue
                
            # Find the highest scoring leaf in this family
            best_leaf_id = max(leaf_ids, key=lambda lid: self.registry[lid].score)
            best_leaf = self.registry[best_leaf_id]
            
            # Walk back N steps to find the "True" ancestor
            ancestor = best_leaf
            steps_back = 0
            while ancestor.parent_id is not None and steps_back < self.n_steps:
                ancestor = self.registry[ancestor.parent_id]
                steps_back += 1
                
            # If we reached an ancestor that isn't the root, re-root the tree
            if ancestor.parent_id is not None:
                parent_of_ancestor = self.registry[ancestor.parent_id]
                
                # 1. Sever the chosen ancestor from the rest of the tree
                parent_of_ancestor.children_ids.remove(ancestor.track_id)
                ancestor.parent_id = None
                
                # 2. Find the absolute root of this tree
                old_root = parent_of_ancestor
                while old_root.parent_id is not None:
                    old_root = self.registry[old_root.parent_id]
                    
                # 3. Delete the old root and all its children. 
                # Because 'ancestor' was severed, it (and its descendants) survive.
                self._delete_branch(old_root.track_id)

    def get_best_global_hypothesis(self) -> List[TrackNode]:
        """
        Find the best global hypothesis using Lagrangian relaxation and subgradient optimization,
        with a 50-iteration fallback to a greedy algorithm.

        Returns:
            List[TrackNode]: List of non-conflicting active leaf nodes forming the best global hypothesis.
        """
        leaves = [self.registry[lid] for lid in self.active_leaves if lid in self.registry]
        if not leaves:
            return []

        all_measurements = set()
        for leaf in leaves:
            for m in leaf.history:
                if m is not None and m != 0:
                    all_measurements.add(m)

        meas_list = sorted(list(all_measurements))
        num_leaves = len(leaves)
        num_meas = len(meas_list)

        if num_meas == 0 or num_leaves == 0:
            best_leaf = max(leaves, key=lambda n: n.score)
            return [best_leaf]

        meas_to_col = {m: idx for idx, m in enumerate(meas_list)}

        A = np.zeros((num_leaves, num_meas), dtype=np.float64)
        for i, leaf in enumerate(leaves):
            for m in leaf.history:
                if m in meas_to_col:
                    A[i, meas_to_col[m]] = 1.0

        scores = np.array([leaf.score for leaf in leaves], dtype=np.float64)
        lambdas = np.zeros(num_meas, dtype=np.float64)

        best_selected_indices: List[int] = []
        best_objective = -np.inf

        max_iter = 50
        for iteration in range(max_iter):
            adjusted_scores = scores - A @ lambdas

            selected = []
            used_measurements = set()

            sorted_indices = np.argsort(-adjusted_scores)
            for i in sorted_indices:
                if adjusted_scores[i] < 0:
                    continue
                leaf = leaves[i]
                conflict = False
                leaf_meas = set(m for m in leaf.history if m is not None and m != 0)
                if not leaf_meas.isdisjoint(used_measurements):
                    conflict = True
                for sel_idx in selected:
                    if leaves[sel_idx].tree_id == leaf.tree_id:
                        conflict = True
                        break

                if not conflict:
                    selected.append(i)
                    used_measurements.update(leaf_meas)

            x_sel = np.zeros(num_leaves, dtype=np.float64)
            for i in selected:
                x_sel[i] = 1.0

            usage_counts = A.T @ x_sel
            g_custom = np.zeros(num_meas, dtype=np.float64)
            for j in range(num_meas):
                cnt = usage_counts[j]
                if np.isclose(cnt, 1.0):
                    g_custom[j] = 0.0
                elif cnt < 1.0:
                    g_custom[j] = -1.0
                else:
                    g_custom[j] = cnt - 1.0

            current_obj = sum(scores[i] for i in selected) - np.sum(lambdas * (usage_counts - 1.0))
            if current_obj > best_objective and selected:
                best_objective = current_obj
                best_selected_indices = selected

            lambdas = np.maximum(0.0, lambdas + self.alpha * g_custom)

        if not best_selected_indices:
            selected = []
            used_measurements = set()
            sorted_indices = np.argsort(-scores)
            for i in sorted_indices:
                leaf = leaves[i]
                conflict = False
                leaf_meas = set(m for m in leaf.history if m is not None and m != 0)
                if not leaf_meas.isdisjoint(used_measurements):
                    conflict = True
                for sel_idx in selected:
                    if leaves[sel_idx].tree_id == leaf.tree_id:
                        conflict = True
                        break
                if not conflict:
                    selected.append(i)
                    used_measurements.update(leaf_meas)
            best_selected_indices = selected

        return [leaves[i] for i in best_selected_indices]
