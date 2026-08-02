# MHT Radar Tracker Microservice

A high-performance, rigorously tested **Track-Oriented Multiple Hypothesis Tracking (TOMHT)** microservice built with Python, NumPy, Pydantic, and FastAPI.

## Key Features
* **Kalman Predictor**: Constant-velocity state estimation with Joseph-form covariance updates (`src/mht_service/tracking/kalman.py`).
* **Mahalanobis Gating**: Ellipsoidal spatial gating using Chi-squared distribution tests (`src/mht_service/tracking/gating.py`).
* **Track Nodes & Trees**: Hypothesis tree management with scoring, ancestry history, and conflict resolution (`src/mht_service/tracking/track_node.py`).
* **Track Manager**: Full scan ingestion, missed-detection branching, measurement association, threshold & N-scan pruning, and global association via **Lagrangian Relaxation** (`src/mht_service/tracking/track_manager.py`).
* **FastAPI REST API**: Stateful session management and endpoints for ingestion and hypothesis extraction (`src/mht_service/services/api.py`).

## Getting Started
See [User Guide & API Reference](docs/user_guide.md) for full documentation, schema definitions, and usage examples.

### Running the API
```bash
uv run python main.py
```

### Testing the API
```bash
uv run bokeh serve --show sim.py
```
