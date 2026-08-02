from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple

# --- Configuration & Initialization Schemas ---

class TrackerConfig(BaseModel):
    vel_assumption: Tuple[float, float] = Field((0.0, 0.0), description="Default [vx, vy] for new roots.")
    init_cov: Tuple[Tuple[float, float], Tuple[float, float]] = Field(
        ((10.0, 0.0), (0.0, 10.0)), description="Initial 2x2 velocity uncertainty."
    )
    q_variance: float = Field(0.1, description="Process noise variance for Kalman prediction.")
    min_score: float = Field(-20.0, description="Minimum log-likelihood score threshold for pruning.")
    n_steps: int = Field(3, description="Number of steps looked back for N-scan pruning.")
    alpha: float = Field(0.1, description="Step size for Lagrangian relaxation.")

# --- Measurement & Scan Processing Schemas ---

class MeasurementModel(BaseModel):
    meas_id: int = Field(..., description="Unique identifier for the measurement.")
    z: Tuple[float, float] = Field(..., description="2D position array [x, y].")
    R: Tuple[Tuple[float, float], Tuple[float, float]] = Field(
        ..., description="2x2 measurement noise covariance matrix."
    )

class ScanPayload(BaseModel):
    measurements: List[MeasurementModel]
    dt: float = Field(..., description="Time step size in seconds.")
    timestamp: float = Field(..., description="Current system timestamp.")
    gate_threshold: float = Field(9.21, description="Chi-squared threshold for Mahalanobis gating.")
    p_d: float = Field(0.9, description="Probability of detection.")
    lambda_fa: float = Field(1e-4, description="False alarm density.")

# --- Output Schemas ---

class TrackState(BaseModel):
    track_id: int
    tree_id: int
    x: Tuple[float, float, float, float]
    P: List[List[float]] # 4x4 Covariance
    score: float
    history: Tuple[int, ...]
    timestamp: float