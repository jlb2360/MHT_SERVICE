# Spec 03: Track Node

## Objective
Implement a track node data structure which should contain all information needed for a given node in the hypothesis tree. This should be a python class.

## Interfaces & Contracts

### 1. State Representation
*   **Prediction State (`x`)**: NumPy array of shape `(4,)` representing `[x_position, y_position, x_velocity, y_velocity]`.
*   **Prediction Covariance (`P`)**: NumPy array of shape `(4, 4)`.
*   **Track ID (`track_id`)**: integer identifier of the track
*   **Parent ID (`parent_id`)**: optional integer identifier of the parent track
*   **Tree ID (`tree_id`)**: integer identifier of the family of tracks that this node belongs to.
*   **Score (`score_id`)**: log-likelihood ratio score for the node
*   **Timestamp (`timestamp`)**: Time that the node was created
*   **Children IDs (`children_ids`)**: List of IDs of all of the children that this parent track spawned.
*   **History (`history`)**: The set of measurement Identifiers that created the track node. This is at most three.


### 2. Required Functions / Methods
*   **`predict(self, dt: float, timestamp: float, q_variance: float) -> None`**
    *   `dt:` current step in time
    *   `timestamp:` the current time of the system
    *   `q_variance`: Float (process noise intensity).
    *   updates the current node to predict what the next measurement should be
*   **`evaluate_gate(self, measurement: np.ndarray, R: np.ndarray, gate_threshold: float) -> Tuple[bool, float]`**
    *   `measurement:` the measurement we want to compare to the prediction
    *   `gate_threshold:` the statistical threshold needed to cross to return true
    *   evaluates the mahalanobis gate for the measurement compared to prediction
*   **`create_update_child(self, child_id: int, meas_id: int, measurement: np.ndarray, R: np.ndarray, timestamp: float, p_d: float, lambda_fa: float) -> "TrackNode"`**
    * `p_d` is the probability of detection
    * `lambda_fa` the false alarm density
    *   create a child that is based on a new measurement, record the likelihood score, add a measurement and child to the history
*   **`create_missed_child(self, child_id: int, timestamp: float, p_d: float) -> "TrackNode"`**
    *   create a child that is based on the prediction, record the likelihood score, copy down the history
*   **`register_child(child_id: int) -> None`**
    *   add a child to the list of ids
*   **`conflicts_with(self, other: "TrackNode") -> bool:`
    * check if a different node conflicts with the current node

## Out of Scope
*   track tree management
*   track pruning
*   resolving the global hypothesis

## Verification Command
Run the following command. The task is ONLY complete when this command exits with code 0. If the test file does not exist, you must write it first based on the constraints above.

`uv run pytest tests/unit/test_03_track_node.py -v --cov=src/mht_service/tracking/track_node.py --cov-fail-under=95`