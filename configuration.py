"""Shared, public experiment settings. Hidden scene contents live in scene/build.py."""
from pathlib import Path
from dataclasses import dataclass, asdict
import json

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    timestep: float = .002
    flight_speed: float = .65
    clearance: float = .34
    grid_resolution: float = .05
    sobol_power: int = 10
    inertia_min: float = .05
    inertia_max: float = .80
    friction_min: float = .01
    friction_max: float = 8.
    hinge_damping: float = .05
    probe_duration: float = 1.2
    probe_force_max: float = 2.2
    push_force_max: float = 2.2
    push_timeout: float = 10.
    force_scale: float = .30
    angle_scale: float = .035
    alpha: float = 2.
    geometry_threshold: float = 3.
    disagreement_threshold: float = .36
    sensor_period: float = .1
    camera_width: int = 480
    camera_height: int = 320
    camera_fovy: float = 68.
    scan_range: float = 6.
    scan_rays: int = 161
    confirmation_frames: int = 3
    workers: int = 4
    seed: int = 22

def architecture():
    return json.loads((ROOT/'config'/'architecture.json').read_text(encoding='utf-8'))

def load_config():
    p = ROOT/'config'/'settings.json'
    return Config(**json.loads(p.read_text(encoding='utf-8'))) if p.exists() else Config()
