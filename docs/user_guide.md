# MHT Radar Tracker Microservice: User Guide & API Reference

Welcome to the **Multiple Hypothesis Tracking (MHT) Radar Tracker Microservice**. This service provides a robust, state-of-the-art Track-Oriented Multiple Hypothesis Tracking (TOMHT) engine exposed via a clean, high-performance FastAPI interface. It is designed for multi-target tracking in cluttered environments with measurement noise, false alarms, and missed detections.

---

## 1. System Architecture & Components

The microservice integrates several rigorous mathematical and algorithmic components located in `src/mht_service/tracking/`:

1. **Kalman Predictor (`kalman.py`)**: Implements constant-velocity state estimation using a 4-dimensional state vector $\mathbf{x} = [x, y, v_x, v_y]^T$ and covariance matrix $\mathbf{P}$. Supports covariance prediction, innovation covariance calculation, Kalman gain computation, and state updates using the numerically stable Joseph form.
2. **Mahalanobis Gating (`gating.py`)**: Computes squared Mahalanobis distances $d^2 = \mathbf{v}^T \mathbf{S}^{-1} \mathbf{v}$ between predicted track states and incoming measurements, evaluating against a Chi-squared threshold (e.g., $\chi^2_2(0.99) \approx 9.21$) to determine valid measurement associations.
3. **Track Node (`track_node.py`)**: Represents individual hypotheses within tree structures. Each node maintains its state vector, covariance, log-likelihood score, ancestry history, child references, and mutual conflict sets (`conflicts_with`).
4. **Track Manager (`track_manager.py`)**: Orchestrates hypothesis generation across radar scans, handling prediction, gating, missed detection branching, measurement updates, new root track birth, score threshold pruning, N-scan sliding window pruning, and global hypothesis extraction via **Lagrangian Relaxation**.

---

## 2. Pydantic Schemas (`src/mht_service/schemas/schema.py`)

### `TrackerConfig`
Configuration parameters for initializing a tracker session:
* `vel_assumption`: Tuple `[vx, vy]` default velocity assumption for new roots (default: `(0.0, 0.0)`).
* `init_cov`: $2 \times 2$ velocity uncertainty matrix (default: `[[10.0, 0.0], [0.0, 10.0]]`).
* `q_variance`: Process noise variance $Q$ for Kalman prediction (default: `0.1`).
* `min_score`: Minimum log-likelihood score threshold for pruning unpromising tracks (default: `-20.0`).
* `n_steps`: Lookback window size for N-scan pruning (default: `3`).
* `alpha`: Step size for Lagrangian relaxation dual variable updates (default: `0.1`).

### `MeasurementModel`
* `meas_id`: Unique integer identifier for the measurement.
* `z`: 2D position tuple `[x, y]`.
* `R`: $2 \times 2$ measurement noise covariance matrix `[[Rxx, Rxy], [Ryx, Ryy]]`.

### `ScanPayload`
* `measurements`: List of `MeasurementModel` objects detected in the current scan.
* `dt`: Time step size in seconds.
* `timestamp`: Current system/sensor timestamp.
* `gate_threshold`: Chi-squared gating threshold (default: `9.21`).
* `p_d`: Probability of target detection (default: `0.9`).
* `lambda_fa`: False alarm spatial density (default: `1e-4`).

### `TrackState`
* `track_id`: Unique track node identifier.
* `tree_id`: Root tree identifier.
* `x`: 4D state vector `[x, y, vx, vy]`.
* `P`: $4 \times 4$ covariance matrix.
* `score`: Cumulative log-likelihood score.
* `history`: Tuple of ancestor node IDs.
* `timestamp`: Timestamp of the track state.

---

## 3. REST API Endpoints (`src/mht_service/services/api.py`)

### 1. Initialize Tracker Session
* **URL:** `POST /api/v1/trackers/{session_id}`
* **Description:** Spawns a stateful `TrackManager` instance for the specified session ID using custom or default `TrackerConfig`.
* **Request Body (Optional):** `TrackerConfig` JSON.
* **Response:** `200 OK` with initialization success message.

### 2. Process Radar Scan
* **URL:** `POST /api/v1/trackers/{session_id}/scan`
* **Description:** Ingests a new scan of measurements, predicts active track states, performs Mahalanobis gating, spawns update and missed-detection child hypotheses, births new root tracks for unassociated measurements, and applies threshold and N-scan pruning.
* **Request Body:** `ScanPayload` JSON.
* **Response:** `200 OK` with processing summary and active hypothesis count.

### 3. Extract Best Global Hypothesis
* **URL:** `GET /api/v1/trackers/{session_id}/global-hypothesis`
* **Description:** Solves the global association problem across all active tracking trees using Lagrangian relaxation and subgradient optimization, returning the optimal set of mutually compatible track states.
* **Response:** `200 OK` with a JSON list of `TrackState` objects.

### 4. Delete Tracker Session
* **URL:** `DELETE /api/v1/trackers/{session_id}`
* **Description:** Destroys the session and frees associated memory.
* **Response:** `200 OK` with deletion confirmation.

---

## 4. Usage Example (Python `requests`)

Here is how a client interacts with the MHT microservice:

```python
import requests

BASE_URL = "http://localhost:8000"
session_id = "radar_sector_alpha"

# 1. Initialize Tracker Session
config = {
    "q_variance": 0.2,
    "min_score": -15.0,
    "n_steps": 3
}
response = requests.post(f"{BASE_URL}/api/v1/trackers/{session_id}", json=config)
print(response.json())

# 2. Send Scan 1 (New target at x=100, y=200)
scan_1 = {
    "measurements": [
        {
            "meas_id": 1,
            "z": [100.0, 200.0],
            "R": [[1.0, 0.0], [0.0, 1.0]]
        }
    ],
    "dt": 1.0,
    "timestamp": 1000.0,
    "gate_threshold": 9.21,
    "p_d": 0.9,
    "lambda_fa": 1e-4
}
res1 = requests.post(f"{BASE_URL}/api/v1/trackers/{session_id}/scan", json=scan_1)
print("Scan 1:", res1.json())

# 3. Send Scan 2 (Target moved to x=105, y=205)
scan_2 = {
    "measurements": [
        {
            "meas_id": 2,
            "z": [105.0, 205.0],
            "R": [[1.0, 0.0], [0.0, 1.0]]
        }
    ],
    "dt": 1.0,
    "timestamp": 1001.0,
    "gate_threshold": 9.21,
    "p_d": 0.9,
    "lambda_fa": 1e-4
}
res2 = requests.post(f"{BASE_URL}/api/v1/trackers/{session_id}/scan", json=scan_2)
print("Scan 2:", res2.json())

# 4. Retrieve Best Global Hypothesis
hyp_res = requests.get(f"{BASE_URL}/api/v1/trackers/{session_id}/global-hypothesis")
tracks = hyp_res.json()
print(f"Optimal Active Tracks: {len(tracks)}")
for track in tracks:
    print(f"Track ID: {track['track_id']}, Position: {track['x'][:2]}, Score: {track['score']:.2f}")

# 5. Cleanup Session
requests.delete(f"{BASE_URL}/api/v1/trackers/{session_id}")
```

---

## 5. Running the Service Locally

To run the FastAPI microservice using `uv` and `uvicorn`:

```bash
uv run uvicorn mht_service.services.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation (Swagger UI / OpenAPI) is available at:
* **`http://localhost:8000/docs`**
