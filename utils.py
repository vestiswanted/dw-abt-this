"""
utils.py - Shared utility helpers.
"""

from __future__ import annotations
import math
import time
import ctypes
import logging
import subprocess
import sys
from typing import Generator, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


# ---------------------------------------------------------------------------
# Precision sleep (more accurate than time.sleep on Windows)
# ---------------------------------------------------------------------------

def precise_sleep(duration: float) -> None:
    if duration <= 0:
        return
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        pass


# ---------------------------------------------------------------------------
# Mouse movement (Win32 SendInput)
# ---------------------------------------------------------------------------

PUL = ctypes.POINTER(ctypes.c_ulong)


class _KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]


class _Input_I(ctypes.Union):
    _fields_ = [("ki", _KeyBdInput), ("mi", _MouseInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _Input_I)]


_extra = ctypes.c_ulong(0)
_ii    = _Input_I()


def move_mouse_relative(dx: int, dy: int) -> None:
    """Send a relative mouse movement via Win32 SendInput."""
    _ii.mi = _MouseInput(dx, dy, 0, 0x0001, 0, ctypes.pointer(_extra))
    inp = _Input(ctypes.c_ulong(0), _ii)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# ---------------------------------------------------------------------------
# Coordinate interpolation
# ---------------------------------------------------------------------------

def interpolate_to_target(
    target_abs: Tuple[float, float],
    screen_center: Tuple[int, int],
    scale: float,
    pixel_increment: float,
) -> Generator[Tuple[int, int], None, None]:
    """
    Yields (rel_x, rel_y) pixel steps to smoothly move from the screen centre
    toward *target_abs* using the provided sensitivity scale.
    """
    diff_x = (target_abs[0] - screen_center[0]) * scale / pixel_increment
    diff_y = (target_abs[1] - screen_center[1]) * scale / pixel_increment
    length = int(math.dist((0, 0), (diff_x, diff_y)))

    if length == 0:
        return

    unit_x = (diff_x / length) * pixel_increment
    unit_y = (diff_y / length) * pixel_increment
    x = y = sum_x = sum_y = 0.0

    for k in range(length):
        sum_x += x
        sum_y += y
        x = unit_x * k - sum_x
        y = unit_y * k - sum_y
        yield round(x), round(y)


# ---------------------------------------------------------------------------
# Screen resolution
# ---------------------------------------------------------------------------

def get_screen_resolution() -> Tuple[int, int]:
    """Return (width, height) of the primary monitor."""
    try:
        w = ctypes.windll.user32.GetSystemMetrics(0)
        h = ctypes.windll.user32.GetSystemMetrics(1)
        return w, h
    except Exception:
        return 1920, 1080


# ---------------------------------------------------------------------------
# Controller trigger check (Xbox / XInput style)
# ---------------------------------------------------------------------------

def is_controller_triggered() -> bool:
    """Return True when the right trigger on a connected controller is pressed."""
    try:
        import pygame
        pygame.event.pump()
        for i in range(pygame.joystick.get_count()):
            joy = pygame.joystick.Joystick(i)
            # Axis 4 is the right trigger on most XInput pads
            if joy.get_axis(4) > 0.5:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# ONNX export helper (run once offline to produce best.onnx)
# ---------------------------------------------------------------------------

def export_yolov5_to_onnx(pt_path: str, out_path: str, img_size: int = 640) -> bool:
    """
    Export a YOLOv5 .pt model to ONNX.
    Requires torch + ultralytics yolov5 installed (done separately).
    Returns True on success.
    """
    try:
        import torch
        logger.info(f"Exporting {pt_path} → {out_path} (img_size={img_size})")
        model = torch.hub.load("ultralytics/yolov5", "custom", path=pt_path)
        model.eval()
        dummy = torch.zeros(1, 3, img_size, img_size)
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        )
        logger.info("Export complete.")
        return True
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Pretty banner
# ---------------------------------------------------------------------------

BANNER = r"""
██████╗░░█████╗░███████╗██╗░░░██╗░██████╗  ██╗░░░██╗░░███╗░░░░░███████╗
██╔══██╗██╔══██╗██╔════╝██║░░░██║██╔════╝  ██║░░░██║░████║░░░░░╚════██║
██████╔╝██║░░██║█████╗░░██║░░░██║╚█████╗░  ╚██╗░██╔╝██╔██║░░░░░░░░░██╔╝
██╔══██╗██║░░██║██╔══╝░░██║░░░██║░╚═══██╗  ░╚████╔╝░╚═╝██║░░░░░░░░██╔╝░
██║░░██║╚█████╔╝██║░░░░░╚██████╔╝██████╔╝  ░░╚██╔╝░░███████╗██╗░░██╔╝░░
╚═╝░░╚═╝░╚════╝░╚═╝░░░░░░╚═════╝░╚═════╝░  ░░░╚═╝░░░╚══════╝╚═╝░░╚═╝░░░
          AMD DirectML Edition  |  ONNX Runtime Inference
"""
