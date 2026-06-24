"""
detector.py - ONNX-based YOLOv5 inference with automatic provider selection.

Provider priority (first available wins):
  1. DmlExecutionProvider  – AMD RX 6600 XT via DirectML
  2. CUDAExecutionProvider – NVIDIA via CUDA
  3. CPUExecutionProvider  – CPU fallback
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def head(self) -> Tuple[int, int]:
        """Approximate head position – upper quarter of the bounding box."""
        cx = (self.x1 + self.x2) // 2
        cy = self.y1 + (self.y2 - self.y1) // 4
        return cx, cy

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Loads a YOLOv5 ONNX model and selects the best available
    ONNX Runtime execution provider.
    """

    def __init__(
        self,
        model_path: str,
        provider_priority: List[str],
        use_fp16: bool = True,
    ) -> None:
        self.model_path      = Path(model_path)
        self.provider_priority = provider_priority
        self.use_fp16        = use_fp16
        self.session         = None
        self.selected_provider: Optional[str] = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}\n"
                "Run the export script first:  python export_model.py"
            )

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed.  "
                "Run:  pip install onnxruntime-directml"
            )

        available = ort.get_available_providers()
        logger.debug(f"Available ORT providers: {available}")

        chosen = None
        for prov in self.provider_priority:
            if prov in available:
                chosen = prov
                break

        if chosen is None:
            chosen = "CPUExecutionProvider"
            logger.warning("No preferred provider found – falling back to CPU.")

        self.selected_provider = chosen
        logger.info(f"Inference provider: {chosen}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.log_severity_level = 3  # suppress verbose ORT logs

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=[chosen],
            sess_options=sess_opts,
        )

        self._input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input shape: {input_shape}")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Wraps model inference + NMS post-processing for YOLOv5 ONNX output.
    """

    def __init__(
        self,
        loader: ModelLoader,
        input_size: int      = 640,
        conf_thresh: float   = 0.50,
        iou_thresh: float    = 0.15,
        use_fp16: bool       = True,   # reserved; ONNX input is always float32
    ) -> None:
        self.loader      = loader
        self.input_size  = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        # ONNX Runtime (DirectML / CUDA / CPU) all expect float32 *input* tensors.
        # FP16 optimisation happens inside the graph — we must never cast the blob.

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Letterbox resize → BGR→RGB → HWC→CHW → normalise → add batch dim.
        Returns (blob, scale, pad_x, pad_y).
        """
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # Always float32 — ONNX Runtime requires it regardless of provider
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]           # (1, 3, H, W)

        return blob, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # Post-processing (YOLOv5 raw output → Detection list)
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[Detection]:
        """
        raw shape: (1, num_anchors, 85)  – [cx, cy, w, h, obj_conf, *cls_conf]
        """
        preds = raw[0]  # (N, 85)

        # Filter by objectness
        obj_conf = preds[:, 4]
        mask     = obj_conf >= self.conf_thresh
        preds    = preds[mask]

        if preds.shape[0] == 0:
            return []

        # Class scores
        cls_scores = preds[:, 5:]
        class_ids  = cls_scores.argmax(axis=1)
        scores     = obj_conf[mask] * cls_scores[np.arange(len(preds)), class_ids]

        # Filter by combined score
        keep   = scores >= self.conf_thresh
        preds  = preds[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if preds.shape[0] == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2 in original image coords
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = ((cx - bw / 2 - pad_x) / scale).astype(np.int32)
        y1 = ((cy - bh / 2 - pad_y) / scale).astype(np.int32)
        x2 = ((cx + bw / 2 - pad_x) / scale).astype(np.int32)
        y2 = ((cy + bh / 2 - pad_y) / scale).astype(np.int32)

        # OpenCV NMS
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores.tolist(),
            self.conf_thresh,
            self.iou_thresh,
        )

        detections: List[Detection] = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                detections.append(Detection(
                    x1=int(x1[i]), y1=int(y1[i]),
                    x2=int(x2[i]), y2=int(y2[i]),
                    confidence=float(scores[i]),
                    class_id=int(class_ids[i]),
                ))

        return detections

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run full pipeline on a BGR frame; returns list of Detections."""
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        raw = self.loader.session.run(
            None,
            {self.loader._input_name: blob},
        )[0]
        return self._postprocess(raw, scale, pad_x, pad_y)
