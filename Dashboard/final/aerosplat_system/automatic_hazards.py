"""Automatic visual hazard-indicator detection for AeroSplat.

This module intentionally returns *indicators* with confidence scores rather
than declaring a structure safe/unsafe. A custom YOLO model can optionally be
used by setting AEROSPLAT_HAZARD_MODEL to a compatible weights file and
installing ultralytics.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _norm_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> List[float]:
    return [
        max(0.0, min(1.0, x1 / max(w, 1))),
        max(0.0, min(1.0, y1 / max(h, 1))),
        max(0.0, min(1.0, x2 / max(w, 1))),
        max(0.0, min(1.0, y2 / max(h, 1))),
    ]


def _component_boxes(mask: np.ndarray, min_area: int, max_items: int = 8) -> List[Tuple[int, int, int, int, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area:
            continue
        boxes.append((x, y, x + w, y + h, area))
    boxes.sort(key=lambda item: item[-1], reverse=True)
    return boxes[:max_items]


def _heuristic_detect(image: np.ndarray) -> List[Dict[str, Any]]:
    h, w = image.shape[:2]
    small_scale = min(1.0, 1100.0 / max(w, h))
    small = cv2.resize(image, (max(1, int(w * small_scale)), max(1, int(h * small_scale))), interpolation=cv2.INTER_AREA) if small_scale < 1 else image.copy()
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    results: List[Dict[str, Any]] = []

    # Fire-like / hot-colored regions.
    lower1 = np.array([0, 115, 145], dtype=np.uint8)
    upper1 = np.array([38, 255, 255], dtype=np.uint8)
    fire = cv2.inRange(hsv, lower1, upper1)
    fire = cv2.morphologyEx(fire, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fire = cv2.morphologyEx(fire, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fire_ratio = float(np.mean(fire > 0))
    if fire_ratio > 0.006:
        for x1, y1, x2, y2, area in _component_boxes(fire, max(30, int(sh * sw * 0.00015)), 3):
            area_ratio = area / max(sh * sw, 1)
            conf = min(0.96, 0.58 + area_ratio * 12.0)
            results.append({
                "kind": "fire_like", "label": "Fire / heat visual indicator",
                "confidence": round(conf, 3), "bbox": _norm_box(x1, y1, x2, y2, sw, sh),
                "weight": 10.0, "hard": True, "source": "heuristic_hsv",
            })

    # Water-like regions.
    water = ((hsv[:, :, 0] >= 82) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 50)).astype(np.uint8) * 255
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    water_ratio = float(np.mean(water > 0))
    if water_ratio > 0.03:
        for x1, y1, x2, y2, area in _component_boxes(water, max(50, int(sh * sw * 0.0008)), 3):
            area_ratio = area / max(sh * sw, 1)
            conf = min(0.92, 0.52 + area_ratio * 4.5)
            results.append({
                "kind": "water_like", "label": "Water / standing-flow visual indicator",
                "confidence": round(conf, 3), "bbox": _norm_box(x1, y1, x2, y2, sw, sh),
                "weight": 4.0, "hard": False, "source": "heuristic_hsv",
            })

    # Crack-like / narrow dark line indicators.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
    _, crack = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    crack = cv2.morphologyEx(crack, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    crack = cv2.dilate(crack, np.ones((2, 2), np.uint8), iterations=1)
    crack_ratio = float(np.mean(crack > 0))
    if crack_ratio > 0.004:
        for x1, y1, x2, y2, area in _component_boxes(crack, max(20, int(sh * sw * 0.00008)), 6):
            bw, bh = x2 - x1, y2 - y1
            elongation = max(bw, bh) / max(min(bw, bh), 1)
            if elongation < 1.8 and area < sh * sw * 0.015:
                continue
            conf = min(0.88, 0.47 + 0.08 * min(elongation, 5.0) + min(area / max(sh * sw, 1) * 8, 0.2))
            results.append({
                "kind": "crack_like", "label": "Crack-like visual indicator",
                "confidence": round(conf, 3), "bbox": _norm_box(x1, y1, x2, y2, sw, sh),
                "weight": 6.0, "hard": False, "source": "heuristic_blackhat",
            })

    # Dense irregular texture as rubble/debris indicator.
    edges = cv2.Canny(gray, 60, 150)
    edge_density = cv2.GaussianBlur((edges > 0).astype(np.float32), (19, 19), 0)
    tile = max(32, min(sh, sw) // 8)
    for y in range(0, sh, tile):
        for x in range(0, sw, tile):
            patch = edge_density[y:min(y + tile, sh), x:min(x + tile, sw)]
            if patch.size == 0:
                continue
            density = float(np.mean(patch))
            if density < 0.26:
                continue
            region = gray[y:min(y + tile, sh), x:min(x + tile, sw)]
            texture = float(np.std(region)) if region.size else 0.0
            if texture < 22:
                continue
            x2, y2 = min(x + tile, sw), min(y + tile, sh)
            conf = min(0.82, 0.45 + density * 0.9 + min(texture / 255.0, 0.22))
            results.append({
                "kind": "debris_like", "label": "Dense debris / rubble visual indicator",
                "confidence": round(conf, 3), "bbox": _norm_box(x, y, x2, y2, sw, sh),
                "weight": 5.0, "hard": False, "source": "heuristic_texture",
            })

    # Poor-visibility zones are useful route penalties but not physical hazards.
    dark = (gray < 32).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    dark_ratio = float(np.mean(dark > 0))
    if dark_ratio > 0.12:
        results.append({
            "kind": "obscured_area", "label": "Poor-visibility / obscured area",
            "confidence": round(min(0.9, 0.55 + dark_ratio), 3),
            "bbox": [0.0, 0.0, 1.0, 1.0], "weight": 3.0, "hard": False,
            "source": "heuristic_visibility",
        })

    return results


def _try_yolo(image: np.ndarray) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    model_path = os.getenv("AEROSPLAT_HAZARD_MODEL", "").strip()
    if not model_path or not Path(model_path).exists():
        return [], None
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return [], "ultralytics_not_installed"
    try:
        model = YOLO(model_path)
        prediction = model.predict(source=image, verbose=False, conf=0.25, imgsz=960)[0]
        names = prediction.names
        results: List[Dict[str, Any]] = []
        h, w = image.shape[:2]
        if prediction.boxes is None:
            return results, "yolo"
        for box in prediction.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            raw_name = str(names.get(cls, cls)).lower().strip()
            mapping = {
                "crack": ("crack", "Crack", 6.0, False),
                "rubble": ("rubble", "Rubble", 5.0, False),
                "debris": ("debris", "Debris", 5.0, False),
                "fire": ("fire", "Fire", 10.0, True),
                "smoke": ("smoke", "Smoke", 7.0, True),
                "water": ("water", "Water", 4.0, False),
                "collapse": ("collapse_indicator", "Collapse indicator", 10.0, True),
            }
            selected = next((value for key, value in mapping.items() if key in raw_name), None)
            if not selected:
                continue
            kind, label, weight, hard = selected
            results.append({
                "kind": kind, "label": label, "confidence": round(conf, 3),
                "bbox": _norm_box(x1, y1, x2, y2, w, h), "weight": weight,
                "hard": hard, "source": "yolo",
            })
        return results, "yolo"
    except Exception:
        return [], "yolo_error"


def analyze_frame(image_path: Path) -> Dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    yolo, yolo_status = _try_yolo(image)
    heuristic = _heuristic_detect(image)
    hazards = yolo if yolo else heuristic
    if yolo and heuristic:
        hazards.extend([h for h in heuristic if h["kind"] not in {x["kind"] for x in yolo}][:6])

    # Keep strongest findings and give every detection a stable id.
    hazards = sorted(hazards, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)[:30]
    for idx, hazard in enumerate(hazards, start=1):
        hazard["id"] = f"HZ-{idx:03d}"
        bbox = hazard.get("bbox") or [0, 0, 1, 1]
        hazard["point"] = [
            round((float(bbox[0]) + float(bbox[2])) / 2.0, 5),
            round((float(bbox[1]) + float(bbox[3])) / 2.0, 5),
        ]
        hazard["radius"] = round(max(0.025, min(0.22, max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) * 0.55)), 4)

    counts: Dict[str, int] = {}
    for h in hazards:
        counts[h["kind"]] = counts.get(h["kind"], 0) + 1
    return {
        "frame": image_path.name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "detector": "yolo+heuristic" if yolo else "heuristic",
        "yolo_status": yolo_status,
        "hazards": hazards,
        "counts": counts,
        "note": "Automatic detections are visual indicators and require field verification; they are not structural certification.",
    }
