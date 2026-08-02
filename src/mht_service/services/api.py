from fastapi import FastAPI, HTTPException, Path
import numpy as np
from typing import Dict, List

# Assuming your provided spec files are in the following structure:
from mht_service.tracking.track_manager import TrackManager #
from mht_service.tracking.track_node import TrackNode 
from mht_service.schemas.schema import *

app = FastAPI(
    title="MHT Radar Tracker Microservice",
    description="API for multi-hypothesis target tracking and state estimation.",
    version="1.0.0"
)

# In-memory session store for stateful TrackManagers. 
# Maps session_id -> TrackManager instance.
active_trackers: Dict[str, TrackManager] = {}


@app.post("/api/v1/trackers/{session_id}")
def create_tracker(session_id: str = Path(...), config: TrackerConfig = TrackerConfig()) -> dict:
    """Initialize a new TrackManager instance for a given session."""
    if session_id in active_trackers:
        raise HTTPException(status_code=400, detail="Tracker session already exists.")
    
    vel_assumption = np.array(config.vel_assumption, dtype=np.float64)
    init_cov = np.array(config.init_cov, dtype=np.float64)
    
    active_trackers[session_id] = TrackManager(
        vel_assumption=vel_assumption,
        init_cov=init_cov,
        q_variance=config.q_variance,
        min_score=config.min_score,
        n_steps=config.n_steps,
        alpha=config.alpha
    ) #[cite: 3]
    
    return {"message": f"Tracker {session_id} initialized successfully."}


@app.post("/api/v1/trackers/{session_id}/scan")
def process_scan(session_id: str, payload: ScanPayload) -> dict:
    """
    Process a scan of measurements through prediction, gating, and pruning.
    """
    tracker = active_trackers.get(session_id)
    if not tracker:
        raise HTTPException(status_code=404, detail="Tracker session not found.")

    # Convert Pydantic payloads to NumPy arrays for the mathematical engine
    meas_dict: Dict[int, np.ndarray] = {}
    r_dict: Dict[int, np.ndarray] = {}
    
    for m in payload.measurements:
        meas_dict[m.meas_id] = np.array(m.z, dtype=np.float64)
        r_dict[m.meas_id] = np.array(m.R, dtype=np.float64)

    tracker.process_scan(
        measurements=meas_dict,
        Rs=r_dict,
        dt=payload.dt,
        timestamp=payload.timestamp,
        gate_threshold=payload.gate_threshold,
        p_d=payload.p_d,
        lambda_fa=payload.lambda_fa
    ) #[cite: 3]

    return {
        "message": "Scan processed successfully.",
        "active_hypotheses_count": len(tracker.active_leaves) 
    }


@app.get("/api/v1/trackers/{session_id}/global-hypothesis", response_model=List[TrackState])
def get_global_hypothesis(session_id: str):
    """
    Extract the best non-conflicting global hypothesis via Lagrangian relaxation.
    """
    tracker = active_trackers.get(session_id)
    if not tracker:
        raise HTTPException(status_code=404, detail="Tracker session not found.")

    best_leaves: List[TrackNode] = tracker.get_best_global_hypothesis() 
    
    # Serialize the output
    response = []
    for leaf in best_leaves:
        response.append(TrackState(
            track_id=leaf.track_id,    
            tree_id=leaf.tree_id,      
            x=tuple(leaf.x.tolist()),  
            P=leaf.P.tolist(),         
            score=leaf.score,            
            history=leaf.history,       
            timestamp=leaf.timestamp    
        ))
        
    return response


@app.delete("/api/v1/trackers/{session_id}")
def delete_tracker(session_id: str) -> dict:
    """Clean up and remove a tracker session."""
    if session_id in active_trackers:
        del active_trackers[session_id]
        return {"message": f"Tracker {session_id} deleted."}
    raise HTTPException(status_code=404, detail="Tracker session not found.")