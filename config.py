"""
config.py - Central configuration for the aimbot system.
All tunable parameters live here. Edit this file to customize behavior.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "lib" / "best.onnx"   # exported ONNX model
CONFIG_DIR = ROOT_DIR / "lib" / "config"
SENS_FILE  = CONFIG_DIR / "config.json"

# ---------------------------------------------------------------------------
# Runtime / inference
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    model_path:        str   = str(MODEL_PATH)
    confidence_thresh: float = 0.50
    iou_thresh:        float = 0.15
    input_size:        int   = 640       # YOLO input resolution (must match export)
    use_fp16:          bool  = True      # enable FP16 when provider supports it
    # Provider priority order – first available wins
    provider_priority: list  = field(default_factory=lambda: [
        "DmlExecutionProvider",          # AMD / DirectML
        "CUDAExecutionProvider",         # NVIDIA CUDA
        "CPUExecutionProvider",          # fallback
    ])

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
@dataclass
class CaptureConfig:
    detection_box_size: int   = 400      # square crop around screen centre (px)
    target_fps:         int   = 144      # max capture FPS cap
    output_color:       str   = "BGR"

# ---------------------------------------------------------------------------
# Aimbot / mouse movement
# ---------------------------------------------------------------------------
@dataclass
class AimbotConfig:
    keybind:          int   = 0x02      # 0x02 = right mouse button
    use_controller:   bool  = False
    pixel_increment:  float = 0.5       # pixels per movement step (smoothness)
    mouse_delay:      float = 0.0003    # seconds between movement steps
    head_offset_ratio: float = 0.25     # vertical offset above bbox centre for head

# ---------------------------------------------------------------------------
# Overlay / UI
# ---------------------------------------------------------------------------
@dataclass
class OverlayConfig:
    enabled:           bool  = True
    box_opacity:       float = 0.35
    crosshair_size:    int   = 6
    box_color_rgb:     tuple = (0, 255, 0)
    crosshair_color_rgb: tuple = (0, 255, 0)
    line_thickness:    int   = 1
    show_fps:          bool  = True

# ---------------------------------------------------------------------------
# Sensitivity (loaded from / saved to SENS_FILE)
# ---------------------------------------------------------------------------
@dataclass
class SensitivityConfig:
    xy_sens:         float = 10.0
    targeting_sens:  float = 10.0
    xy_scale:        float = 1.0
    targeting_scale: float = 1.0

    @classmethod
    def load(cls) -> "SensitivityConfig":
        if SENS_FILE.exists():
            with open(SENS_FILE) as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SENS_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def recalculate(self) -> None:
        self.xy_scale       = 10.0 / self.xy_sens if self.xy_sens else 1.0
        self.targeting_scale = 1000.0 / (self.targeting_sens * self.xy_sens) if (self.targeting_sens and self.xy_sens) else 1.0

# ---------------------------------------------------------------------------
# Master config bundle
# ---------------------------------------------------------------------------
@dataclass
class Config:
    inference:   InferenceConfig   = field(default_factory=InferenceConfig)
    capture:     CaptureConfig     = field(default_factory=CaptureConfig)
    aimbot:      AimbotConfig      = field(default_factory=AimbotConfig)
    overlay:     OverlayConfig     = field(default_factory=OverlayConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    @classmethod
    def default(cls) -> "Config":
        cfg = cls()
        cfg.sensitivity = SensitivityConfig.load()
        return cfg


# Singleton accessible as `from config import CFG`
CFG = Config.default()
