from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import CONFIG


class HazardDetector:
    """Optional YOLO segmentation wrapper. Works in heuristic fallback mode when no model is installed."""
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or CONFIG.yolo_model
        self.model = None
        self.names = {}
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.names = self.model.names
        except Exception:
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, image_paths: Iterable[str], conf: float = 0.35) -> list[dict]:
        results: list[dict] = []
        if self.model is None:
            return results
        for path in image_paths:
            try:
                outputs = self.model.predict(source=path, conf=conf, verbose=False)
                for out in outputs:
                    boxes = getattr(out, "boxes", None)
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        score = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
                        name = self.names.get(cls_id, str(cls_id))
                        results.append({"image": Path(path).name, "class": str(name).lower(), "confidence": score, "bbox": [round(x,1) for x in xyxy]})
            except Exception as exc:
                results.append({"image": Path(path).name, "class": "inference_error", "confidence": 0.0, "error": str(exc)})
        return results

    @staticmethod
    def heuristic_hazards(image_paths: Iterable[str]) -> list[dict]:
        """Conservative demo-only visual indicators; not a substitute for a trained detector."""
        hazards = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Red/orange dominance can indicate possible fire-like pixels.
            red1 = cv2.inRange(hsv, (0,80,100), (15,255,255))
            red2 = cv2.inRange(hsv, (160,80,100), (179,255,255))
            red_ratio = float((red1|red2).mean()/255.0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dark_ratio = float((gray < 55).mean())
            if red_ratio > 0.03:
                hazards.append({"image":Path(path).name,"class":"fire","confidence":min(0.9,red_ratio*5)})
            if dark_ratio > 0.30:
                hazards.append({"image":Path(path).name,"class":"smoke","confidence":min(0.75,dark_ratio)})
        return hazards
