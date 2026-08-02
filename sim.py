import numpy as np
import requests
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.palettes import Dark2_8 as palette
import itertools

# --- Configuration ---
API_BASE_URL = "http://localhost:8000/api/v1/trackers"
SESSION_ID = "drone_swarm_sim_attack"

DT = 1.0                # Time step in seconds
NUM_STEPS = 40          # Total number of simulation steps
MEASUREMENT_NOISE = 3.0 # Standard deviation of position noise (meters)
P_D = 0.90              # Probability of detection (lowered to induce more missed detections)
GATE_THRESHOLD = 11.83  # 99.7% chi-squared threshold for 2 DOF to handle maneuvers

class MHTSimulationApp:
    def __init__(self):
        self.step_count = 0
        self.meas_id_counter = 1
        
        # Drone Swarm configuration for a converging attack
        # angle: approach azimuth, radius: starting distance, speed: radial velocity towards center
        # weave_amp: cross-range oscillation amplitude, weave_freq: maneuver frequency
        self.swarm_config = [
            {"angle": 0.0,          "radius": 450, "speed": 8,  "weave_amp": 30, "weave_freq": 0.4},
            {"angle": np.pi/3.5,    "radius": 400, "speed": 7,  "weave_amp": 45, "weave_freq": 0.3},
            {"angle": 2*np.pi/3,    "radius": 480, "speed": 10, "weave_amp": 20, "weave_freq": 0.5},
            {"angle": 4*np.pi/3,    "radius": 420, "speed": 9,  "weave_amp": 35, "weave_freq": 0.45},
            {"angle": 5*np.pi/2.8,  "radius": 460, "speed": 8,  "weave_amp": 25, "weave_freq": 0.6},
            {"angle": 1.5*np.pi,    "radius": 500, "speed": 11, "weave_amp": 40, "weave_freq": 0.35},
        ]
        
        self.true_states = [np.zeros(4) for _ in range(len(self.swarm_config))]
        self.R_matrix = [[MEASUREMENT_NOISE**2, 0.0], [0.0, MEASUREMENT_NOISE**2]]
        self.track_history = {}
        
        self._initialize_api()
        self._setup_bokeh_figure()

    def _initialize_api(self):
        """Reset and initialize the MHT tracker session."""
        url = f"{API_BASE_URL}/{SESSION_ID}"
        try:
            requests.delete(url)
            payload = {
                "vel_assumption": [0.0, 0.0],
                "init_cov": [[100.0, 0.0], [0.0, 100.0]],
                "q_variance": 15.0, # High process noise to allow the CV model to track maneuvers
                "min_score": -15.0,
                "n_steps": 4,
                "alpha": 0.1
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("API Connected. Tracker Initialized for Swarm Defense.")
        except requests.exceptions.ConnectionError:
            print("CRITICAL: Could not connect to API. Is the FastAPI service running?")
            raise

    def _setup_bokeh_figure(self):
        """Configure the Bokeh figure with fixed axes centered on the tracker."""
        self.plot = figure(
            title="MHT Swarm Defense (Weaving Attackers)",
            x_axis_label="X Position (meters)",
            y_axis_label="Y Position (meters)",
            x_range=(-500, 500), # Fixed Display Boundary
            y_range=(-500, 500), # Fixed Display Boundary
            width=800,
            height=800,
            match_aspect=True
        )
        
        # Center Target (Us)
        self.plot.scatter([0], [0], size=15, color='red', marker='star', legend_label="Tracker Center")

        # Data source for Ground Truth (Dashed Lines)
        self.source_truth = ColumnDataSource(data=dict(xs=[[] for _ in self.swarm_config], ys=[[] for _ in self.swarm_config]))
        self.plot.multi_line(
            xs='xs', ys='ys', source=self.source_truth,
            line_width=1.5, line_dash='dashed', line_color='black', 
            alpha=0.3, legend_label="True Kinematic Path"
        )
        
        # Data source for MHT Tracks (Solid Lines with Colors)
        self.source_tracks = ColumnDataSource(data=dict(xs=[], ys=[], line_color=[]))
        self.plot.multi_line(
            xs='xs', ys='ys', line_color='line_color', source=self.source_tracks,
            line_width=2.5, alpha=0.9, legend_label="MHT Global Hypothesis"
        )
        
        # Data source for Current Measurements (Scatter)
        self.source_measurements = ColumnDataSource(data=dict(x=[], y=[]))
        self.plot.scatter(
            x='x', y='y', source=self.source_measurements,
            size=5, color='gray', marker='cross', alpha=0.7, legend_label="Sensor Measurements"
        )

        self.plot.legend.location = "top_left"
        self.plot.legend.click_policy = "hide"
        self.colors = itertools.cycle(palette)

    def iterate(self):
        """Main loop executed by the Bokeh periodic callback."""
        if self.step_count >= NUM_STEPS:
            return 
            
        t = self.step_count * DT
        current_measurements = []
        meas_x, meas_y = [], []
        
        # 1. Update true positions using a weaving kinematic model
        for i, config in enumerate(self.swarm_config):
            theta = config["angle"]
            
            # Radial distance closing in on the origin
            current_radius = max(0.0, config["radius"] - config["speed"] * t)
            
            if current_radius > 0:
                # Base position on the straight-line vector
                base_x = current_radius * np.cos(theta)
                base_y = current_radius * np.sin(theta)
                
                # Cross-range weave (perpendicular to the attack vector)
                weave_offset = config["weave_amp"] * np.sin(config["weave_freq"] * t)
                
                pos_x = base_x - weave_offset * np.sin(theta)
                pos_y = base_y + weave_offset * np.cos(theta)
                
                # Apparent velocities for the state vector
                v_radial = -config["speed"]
                v_weave = config["weave_amp"] * config["weave_freq"] * np.cos(config["weave_freq"] * t)
                
                vx = v_radial * np.cos(theta) - v_weave * np.sin(theta)
                vy = v_radial * np.sin(theta) + v_weave * np.cos(theta)
                
                self.true_states[i] = np.array([pos_x, pos_y, vx, vy])
                
                self.source_truth.data['xs'][i].append(pos_x)
                self.source_truth.data['ys'][i].append(pos_y)
                
                # Generate noisy measurement
                if np.random.rand() <= P_D:
                    z_x = pos_x + np.random.normal(0, MEASUREMENT_NOISE)
                    z_y = pos_y + np.random.normal(0, MEASUREMENT_NOISE)
                    
                    meas_x.append(z_x)
                    meas_y.append(z_y)
                    
                    current_measurements.append({
                        "meas_id": self.meas_id_counter,
                        "z": [z_x, z_y],
                        "R": self.R_matrix
                    })
                    self.meas_id_counter += 1

        self.source_measurements.data = dict(x=meas_x, y=meas_y)
        self.source_truth.data = dict(self.source_truth.data)

        # 2. Send scan to API
        scan_payload = {
            "measurements": current_measurements,
            "dt": DT,
            "timestamp": t,
            "gate_threshold": GATE_THRESHOLD,
            "p_d": P_D,
            "lambda_fa": 1e-4
        }
        requests.post(f"{API_BASE_URL}/{SESSION_ID}/scan", json=scan_payload).raise_for_status()
        
        # 3. Retrieve global hypothesis
        hyp_resp = requests.get(f"{API_BASE_URL}/{SESSION_ID}/global-hypothesis")
        hyp_resp.raise_for_status()
        active_tracks = hyp_resp.json()

        # 4. Update track histories
        for track in active_tracks:
            tid = track["tree_id"]
            if tid not in self.track_history:
                self.track_history[tid] = {'x': [], 'y': [], 'color': next(self.colors)}
            
            self.track_history[tid]['x'].append(track["x"][0])
            self.track_history[tid]['y'].append(track["x"][1])

        # 5. Push track updates to Bokeh
        track_xs, track_ys, track_colors = [], [], []
        for tid, data in self.track_history.items():
            if len(data['x']) > 2:
                track_xs.append(data['x'])
                track_ys.append(data['y'])
                track_colors.append(data['color'])
                
        self.source_tracks.data = dict(xs=track_xs, ys=track_ys, line_color=track_colors)
        self.step_count += 1

app = MHTSimulationApp()
doc = curdoc()
doc.add_root(column(app.plot))
doc.title = "MHT Swarm Attack"
doc.add_periodic_callback(app.iterate, 250)