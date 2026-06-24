"""
capture.py - Non-blocking screen capture using BetterCam.

Runs in a dedicated thread and pushes frames into a queue so the inference
thread never blocks waiting for a new frame.
"""

from __future__ import annotations
import logging
import threading
import time
from queue import Queue, Full
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Wraps BetterCam in a background thread.

    Usage:
        cap = ScreenCapture(region=(left, top, right, bottom), target_fps=144)
        cap.start()
        frame = cap.latest_frame()   # non-blocking; returns None if no frame yet
        cap.stop()
    """

    def __init__(
        self,
        region: Tuple[int, int, int, int],
        target_fps: int  = 144,
        queue_size: int  = 2,
        output_color: str = "BGR",
    ) -> None:
        self.region       = region
        self.target_fps   = target_fps
        self.output_color = output_color

        self._queue: Queue[np.ndarray] = Queue(maxsize=queue_size)
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._camera      = None

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._thread.start()
        logger.info("Screen capture thread started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
        logger.info("Screen capture stopped.")

    # ------------------------------------------------------------------

    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame without blocking, or None."""
        frame = None
        while not self._queue.empty():
            try:
                frame = self._queue.get_nowait()
            except Exception:
                break
        return frame

    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        try:
            import bettercam
        except ImportError:
            raise ImportError(
                "bettercam not installed.\n"
                "Install with: pip install git+https://github.com/RootKit-Org/BetterCam.git"
            )

        self._camera = bettercam.create(
            output_color=self.output_color,
            region=self.region,
        )

        frame_interval = 1.0 / self.target_fps
        logger.info(f"Capture region: {self.region} @ target {self.target_fps} FPS")

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            frame = self._camera.grab()
            if frame is not None:
                # Drop old frame rather than block the capture thread
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                try:
                    self._queue.put_nowait(frame)
                except Full:
                    pass

            elapsed = time.perf_counter() - t0
            wait    = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)
