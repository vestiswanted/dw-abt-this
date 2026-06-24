"""
main.py - Entry point.

Pipeline:
  CaptureThread  →  queue  →  InferenceThread  →  mouse movement
                                     ↓
                              OverlayThread (FPS display)

Keybinds (global, via pynput):
  F1 – toggle aimbot on/off
  F2 – quit
"""

from __future__ import annotations
import json
import logging
import math
import os
import sys
import time
import threading
from queue import Queue, Empty
from typing import Optional, List

import cv2
import win32api
from pynput import keyboard

# Local modules
from config import CFG, SensitivityConfig, SENS_FILE, CONFIG_DIR
from utils import (
    setup_logging, precise_sleep, move_mouse_relative,
    interpolate_to_target, get_screen_resolution, BANNER
)
from detector import Detector, Detection, ModelLoader
from capture import ScreenCapture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_aimbot_enabled = True
_running        = True


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def _on_key_release(key) -> None:
    global _aimbot_enabled, _running
    if key == keyboard.Key.f1:
        _aimbot_enabled = not _aimbot_enabled
        status = "ENABLED" if _aimbot_enabled else "DISABLED"
        print(f"\r[!] AIMBOT {status}          ", end="", flush=True)
    elif key == keyboard.Key.f2:
        print("\n[INFO] F2 pressed – shutting down...")
        _running = False


# ---------------------------------------------------------------------------
# Sensitivity setup wizard
# ---------------------------------------------------------------------------

def _run_setup() -> SensitivityConfig:
    print("\n[SETUP] X and Y axis sensitivity MUST be the same in-game.\n")

    def prompt(msg: str) -> float:
        while True:
            try:
                return float(input(msg))
            except ValueError:
                print("  Enter a number only (e.g. 6.5)")

    xy_sens        = prompt("X/Y Axis Sensitivity (from in-game settings): ")
    targeting_sens = prompt("Targeting / Scoped Sensitivity:               ")

    cfg = SensitivityConfig(
        xy_sens=xy_sens,
        targeting_sens=targeting_sens,
    )
    cfg.recalculate()
    cfg.save()
    print("[SETUP] Configuration saved.\n")
    return cfg


# ---------------------------------------------------------------------------
# Input check
# ---------------------------------------------------------------------------

def _is_aim_active() -> bool:
    """Return True when the aim keybind is held.

    GetAsyncKeyState reads hardware state directly — no message-pump needed.
    The high-order bit (0x8000) is set while the key/button is physically held.
    GetKeyState only reflects state as of the last Win32 message processed by
    the calling thread; in a non-GUI thread it almost always reads 0.
    """
    try:
        if win32api.GetAsyncKeyState(CFG.aimbot.keybind) & 0x8000:
            return True
    except Exception:
        pass

    if CFG.aimbot.use_controller:
        from utils import is_controller_triggered
        return is_controller_triggered()

    return False


# ---------------------------------------------------------------------------
# Mouse mover
# ---------------------------------------------------------------------------

def _move_towards(
    target_abs_x: float,
    target_abs_y: float,
    screen_cx: int,
    screen_cy: int,
) -> None:
    scale = CFG.sensitivity.targeting_scale
    incr  = CFG.aimbot.pixel_increment
    delay = CFG.aimbot.mouse_delay

    for rel_x, rel_y in interpolate_to_target(
        (target_abs_x, target_abs_y),
        (screen_cx, screen_cy),
        scale,
        incr,
    ):
        move_mouse_relative(rel_x, rel_y)
        precise_sleep(delay)


# ---------------------------------------------------------------------------
# Detection → target selection
# ---------------------------------------------------------------------------

def _pick_target(
    detections: List[Detection],
    box_size: int,
    margin_x: int,
    margin_bottom: int,
) -> Optional[Detection]:
    """
    Return the detection closest to the crosshair, ignoring own-player detections
    (very close to left edge, or bottom-left corner of the detection box).
    """
    half = box_size / 2
    best_dist: Optional[float] = None
    best_det:  Optional[Detection] = None

    for det in detections:
        # Own-player heuristic: huge bounding box anchored at left/bottom
        own_player = (
            det.x1 < margin_x
            or (det.x1 < box_size // 5 and det.y2 > box_size - margin_bottom)
        )
        if own_player:
            continue

        hx, hy = det.head
        dist = math.dist((hx, hy), (half, half))

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_det  = det

    return best_det


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    global _running

    setup_logging(logging.INFO)
    os.system("cls" if os.name == "nt" else "clear")
    print(BANNER)

    # ------------------------------------------------------------------
    # Sensitivity config
    # ------------------------------------------------------------------
    if not SENS_FILE.exists() or "--setup" in sys.argv:
        if not SENS_FILE.exists():
            print("[!] No sensitivity config found.")
        CFG.sensitivity = _run_setup()
    else:
        CFG.sensitivity = SensitivityConfig.load()
        print(f"[INFO] Loaded sensitivity from {SENS_FILE}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    loader = ModelLoader(
        model_path=CFG.inference.model_path,
        provider_priority=CFG.inference.provider_priority,
        use_fp16=CFG.inference.use_fp16,
    )
    loader.load()
    print(f"[INFO] Provider : {loader.selected_provider}")
    print(f"[INFO] Model    : {CFG.inference.model_path}")

    detector = Detector(
        loader=loader,
        input_size=CFG.inference.input_size,
        conf_thresh=CFG.inference.confidence_thresh,
        iou_thresh=CFG.inference.iou_thresh,
        use_fp16=CFG.inference.use_fp16,
    )

    # ------------------------------------------------------------------
    # Screen / capture
    # ------------------------------------------------------------------
    sw, sh = get_screen_resolution()
    screen_cx = sw // 2
    screen_cy = sh // 2
    box       = CFG.capture.detection_box_size

    left   = (sw - box) // 2
    top    = (sh - box) // 2
    right  = left + box
    bottom = top  + box

    capture = ScreenCapture(
        region=(left, top, right, bottom),
        target_fps=CFG.capture.target_fps,
        output_color=CFG.capture.output_color,
    )

    # ------------------------------------------------------------------
    # Overlay  (QApplication MUST be created in the main thread)
    # ------------------------------------------------------------------
    if CFG.overlay.enabled:
        from overlay import init_overlay, tick as overlay_tick, update_fps
        init_overlay()
    else:
        def update_fps(_fps: float) -> None:  # type: ignore[misc]
            pass
        def overlay_tick() -> None:           # type: ignore[misc]
            pass

    # ------------------------------------------------------------------
    # Keyboard listener
    # ------------------------------------------------------------------
    listener = keyboard.Listener(on_release=_on_key_release)
    listener.start()

    print("\n[INFO] PRESS F1 TO TOGGLE AIMBOT | F2 TO QUIT\n")

    # ------------------------------------------------------------------
    # Start capture thread
    # ------------------------------------------------------------------
    capture.start()

    # ------------------------------------------------------------------
    # Inference loop (runs in main thread)
    # ------------------------------------------------------------------
    fps_history: List[float] = []

    try:
        while _running:
            # Pump Qt events every iteration (keeps overlay alive in main thread)
            overlay_tick()

            frame = capture.latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            t0          = time.perf_counter()
            detections  = detector.infer(frame)
            elapsed     = time.perf_counter() - t0
            fps         = 1.0 / elapsed if elapsed > 0 else 0.0

            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            update_fps(avg_fps)

            aim_active = _is_aim_active()
            if _aimbot_enabled and aim_active:
                target = _pick_target(
                    detections,
                    box_size=box,
                    margin_x=15,
                    margin_bottom=int(box * 0.17),
                )
                if target is not None:
                    abs_hx = target.head[0] + left
                    abs_hy = target.head[1] + top
                    _move_towards(abs_hx, abs_hy, screen_cx, screen_cy)
            elif detections and not aim_active:
                # Target seen but keybind not held — helpful reminder on console
                sys.stdout.write(
                    f"\r[~] Target detected ({len(detections)}) — hold keybind to aim | FPS {avg_fps:.0f}   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        _running = False
        capture.stop()
        listener.stop()
        print("[INFO] Shutdown complete.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
