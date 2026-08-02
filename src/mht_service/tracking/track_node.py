"""
Track Node implementation for Multiple Hypothesis Tracking (Spec 03).
Represents a node in the hypothesis tree containing state estimate, covariance, score, and ancestry.
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
import scipy.linalg

from mht_service.tracking import kalman
from mht_service.tracking import gating


class TrackNode:
    """
    Represents a single track hypothesis node in the MHT tree.
    """

    def __init__(
        self,
        track_id: int,
        x: npt.NDArray[np.float64],
        P: npt.NDArray[np.float64],
        tree_id: int,
        parent_id: Optional[int] = None,
        timestamp: float = 0.0,
        score: float = 0.0,
        history: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Initialize a TrackNode.

        Args:
            track_id: Integer identifier of the track.
            x: State mean array of shape (4,) [x, y, vx, vy].
            P: State covariance matrix of shape (4, 4).
            tree_id: Integer identifier of the track family/tree.
            parent_id: Optional integer identifier of the parent track.
            timestamp: Time that the node was created.
            score: Log-likelihood ratio score for the node (also accessible via score_id).
            history: Tuple of measurement identifiers that created the track node (max 3).
        """
        self.track_id: int = track_id
        self.x: npt.NDArray[np.float64] = np.array(x, dtype=np.float64)
        self.P: npt.NDArray[np.float64] = np.array(P, dtype=np.float64)
        self.tree_id: int = tree_id
        self.parent_id: Optional[int] = parent_id
        self._score: float = float(score)
        self.timestamp: float = float(timestamp)
        self.children_ids: list[int] = []
        self.history: Tuple[int, ...] = tuple(history) if history is not None else ()

        # Enforce initial covariance symmetry
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def score(self) -> float:
        """Log-likelihood ratio score for the node."""
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        self._score = float(value)

    @property
    def score_id(self) -> float:
        """Alias for score (log-likelihood ratio score)."""
        return self._score

    @score_id.setter
    def score_id(self, value: float) -> None:
        self._score = float(value)

    def predict(self, dt: float, timestamp: float, q_variance: float) -> None:
        """
        Update the current node's state prediction using the Kalman filter predict step.

        Args:
            dt: Time step in seconds.
            timestamp: Current system time.
            q_variance: Process noise intensity.
        """
        self.x, self.P = kalman.predict(self.x, self.P, dt, q_variance)
        self.timestamp = float(timestamp)

    def evaluate_gate(
        self,
        measurement: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        gate_threshold: float,
    ) -> Tuple[bool, float]:
        """
        Evaluate the Mahalanobis gate for a measurement compared to the prediction.

        Args:
            measurement: Measurement array of shape (2,) [x, y].
            R: Measurement noise covariance matrix of shape (2, 2).
            gate_threshold: Statistical threshold required to return True.

        Returns:
            tuple containing:
                - inside: bool indicating whether measurement is within gate.
                - d2: float squared Mahalanobis distance.
        """
        
        return gating.measurement_gate(self.x, self.P, measurement, R, gate_threshold)
        

    def create_update_child(
        self,
        child_id: int,
        meas_id: int,
        measurement: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        timestamp: float,
        p_d: float,
        lambda_fa: float,
    ) -> "TrackNode":
        """
        Create a child node based on a new measurement update.

        Args:
            child_id: Identifier for the new child node.
            meas_id: Identifier of the measurement being associated.
            measurement: Measurement array of shape (2,).
            R: Measurement noise covariance matrix of shape (2, 2).
            timestamp: Time of the child node.
            p_d: Probability of detection.
            lambda_fa: False alarm density.

        Returns:
            TrackNode: The newly created update child node.
        """
        x_upd, P_upd, log_likelihood = kalman.update(self.x, self.P, measurement, R)

        # MHT LLR score increment for an update:
        # ln(p_d) - ln(lambda_fa) + log_likelihood
        safe_p_d = max(1e-12, p_d)
        safe_lambda_fa = max(1e-12, lambda_fa)
        score_increment = np.log(safe_p_d) - np.log(safe_lambda_fa) + log_likelihood
        new_score = self._score + float(score_increment)

        # History: append meas_id, keep at most 3 items
        history_list = list(self.history)
        history_list.append(int(meas_id))
        if len(history_list) > 3:
            history_list = history_list[-3:]
        new_history = tuple(history_list)

        child = TrackNode(
            track_id=child_id,
            x=x_upd,
            P=P_upd,
            tree_id=self.tree_id,
            parent_id=self.track_id,
            timestamp=timestamp,
            score=new_score,
            history=new_history,
        )

        self.register_child(child_id)
        return child

    def create_missed_child(
        self,
        child_id: int,
        timestamp: float,
        p_d: float,
    ) -> "TrackNode":
        """
        Create a child node representing a missed detection (no measurement association).

        Args:
            child_id: Identifier for the new child node.
            timestamp: Time of the child node.
            p_d: Probability of detection.

        Returns:
            TrackNode: The newly created missed child node.
        """
        # MHT score increment for missed detection: ln(1 - p_d)
        safe_p_d = min(max(0.0, p_d), 1.0 - 1e-12)
        score_increment = np.log(1.0 - safe_p_d)
        new_score = self._score + float(score_increment)

        # Copy down history
        new_history = self.history

        child = TrackNode(
            track_id=child_id,
            x=self.x.copy(),
            P=self.P.copy(),
            tree_id=self.tree_id,
            parent_id=self.track_id,
            timestamp=timestamp,
            score=new_score,
            history=new_history,
        )

        self.register_child(child_id)
        return child

    def register_child(self, child_id: int) -> None:
        """
        Add a child identifier to the track node's list of children.

        Args:
            child_id: Integer identifier of the child track.
        """
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def conflicts_with(self, other: "TrackNode") -> bool:
        """
        Check if another track node conflicts with the current node
        (i.e. share any measurement IDs in their history).

        Args:
            other: Another TrackNode to check against.

        Returns:
            bool: True if there is a conflict (shared measurement IDs), False otherwise.
        """
        self_meas = set(m for m in self.history if m is not None and m != 0)
        other_meas = set(m for m in other.history if m is not None and m != 0)
        return not self_meas.isdisjoint(other_meas)
