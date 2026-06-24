"""
overlay.py - Transparent PyQt5 overlay window.

QApplication MUST live in the main thread.
Call init_overlay() once from main() before the inference loop,
then call tick() every loop iteration so Qt can process its events.
update_fps() may be called from any thread.
"""

from __future__ import annotations
import sys
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, Q_ARG
from PyQt5.QtGui import QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import QApplication, QWidget

from config import CFG
from utils import get_screen_resolution


# ---------------------------------------------------------------------------
# Module-level singletons (populated by init_overlay)
# ---------------------------------------------------------------------------

_app:    Optional[QApplication]  = None
_widget: Optional["OverlayWidget"] = None


# ---------------------------------------------------------------------------
# Signal bridge — lets non-Qt threads safely trigger a repaint
# ---------------------------------------------------------------------------

class _Bridge(QObject):
    fps_signal = pyqtSignal(float)

_bridge = _Bridge()


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------

class OverlayWidget(QWidget):

    def __init__(self) -> None:
        super().__init__()
        cfg = CFG.overlay
        self._box_size    = CFG.capture.detection_box_size
        self._opacity     = cfg.box_opacity
        self._xhair_size  = cfg.crosshair_size
        self._box_color   = QColor(*cfg.box_color_rgb)
        self._xhair_color = QColor(*cfg.crosshair_color_rgb)
        self._thickness   = cfg.line_thickness
        self._show_fps    = cfg.show_fps
        self._fps: float  = 0.0

        sw, sh = get_screen_resolution()

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool                    # hide from taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)

        ox = (sw - self._box_size) // 2
        oy = (sh - self._box_size) // 2
        self.setGeometry(ox, oy, self._box_size, self._box_size)

        # Connect signal in the widget's thread (main thread = Qt thread)
        _bridge.fps_signal.connect(self._on_fps)

    def _on_fps(self, fps: float) -> None:
        self._fps = fps
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Detection-box border
        p.setOpacity(self._opacity)
        pen = QPen(self._box_color, self._thickness, Qt.SolidLine)
        p.setPen(pen)
        p.drawRect(0, 0, w - 1, h - 1)

        # Crosshair
        p.setOpacity(0.9)
        cx, cy = w // 2, h // 2
        s = self._xhair_size
        pen.setColor(self._xhair_color)
        p.setPen(pen)
        p.drawLine(cx, cy - s, cx, cy + s)
        p.drawLine(cx - s, cy, cx + s, cy)

        # FPS counter
        if self._show_fps and self._fps > 0:
            p.setPen(QPen(QColor(255, 220, 0), 1))
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            p.drawText(6, 18, f"FPS: {self._fps:.0f}")

        p.end()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_overlay() -> None:
    """
    Create QApplication and the overlay widget in the calling (main) thread.
    Call ONCE before the inference loop.
    """
    global _app, _widget
    _app = QApplication.instance() or QApplication(sys.argv)
    _widget = OverlayWidget()
    _widget.show()


def tick() -> None:
    """
    Pump the Qt event queue.  Call once per inference loop iteration.
    This replaces app.exec_() so Qt stays in the main thread without blocking.
    """
    if _app is not None:
        _app.processEvents()


def update_fps(fps: float) -> None:
    """Thread-safe FPS update — emit via queued signal so Qt handles the repaint."""
    _bridge.fps_signal.emit(fps)
