from __future__ import annotations

import math
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


class StructuralDamageQuantifier:
    def __init__(self, metric_scale: float = 1.0):
        self.metric_scale = float(metric_scale)

    def _empty(self, note: str):
        return {"status": "insufficient_data", "note": note, "confidence": 0.0}

    def analyze_wall_tilt(self, points: np.ndarray) -> dict:
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 8:
            return self._empty("Not enough wall points")
        pts = points * self.metric_scale
        pca = PCA(n_components=3).fit(pts)
        # For a roughly vertical wall, the smallest-variance PCA component approximates wall normal.
        normal = pca.components_[-1]
        vertical = np.array([0.0, 0.0, 1.0])
        wall_vertical_dir = pca.components_[np.argmax(np.abs(pca.components_ @ vertical))]
        if wall_vertical_dir[2] < 0:
            wall_vertical_dir = -wall_vertical_dir
        angle = math.degrees(math.acos(float(np.clip(np.dot(wall_vertical_dir, vertical), -1, 1))))
        spread = np.std(pts, axis=0)
        confidence = float(np.clip((len(points)/500.0) * (spread.max()/(spread.min()+1e-6)) / 10.0, 0.2, 0.98))
        return {"tilt_degrees": round(angle, 3), "normal": normal.tolist(), "confidence": round(confidence, 3)}

    def analyze_crack(self, points: np.ndarray) -> dict:
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 8:
            return self._empty("Not enough crack candidate points")
        pts = points * self.metric_scale
        pca = PCA(n_components=3).fit(pts)
        centered = pts - pca.mean_
        proj = centered @ pca.components_.T
        length = float(np.ptp(proj[:, 0]))
        width = float(2.0 * np.std(proj[:, 1])) if proj.shape[1] > 1 else 0.0
        return {"length_m": round(length, 3), "width_mm": round(width * 1000.0, 3), "confidence": round(float(np.clip(len(points)/1000,0.2,0.95)),3)}

    def analyze_debris_volume(self, points: np.ndarray) -> dict:
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 8:
            return self._empty("Not enough debris points")
        pts = points * self.metric_scale
        min_xyz = pts.min(axis=0); max_xyz = pts.max(axis=0)
        extent = np.maximum(max_xyz-min_xyz, 1e-6)
        # Conservative prototype estimate: OBB volume times occupancy factor estimated from point count.
        occupancy = float(np.clip(len(points)/2000.0, 0.15, 0.75))
        volume = float(np.prod(extent) * occupancy)
        return {"volume_m3": round(volume,3), "bbox_extent_m": extent.round(3).tolist(), "confidence": round(float(np.clip(len(points)/1500,0.2,0.9)),3)}

    def analyze_crack_mask(self, mask: np.ndarray, meters_per_pixel: Optional[float] = None) -> dict:
        if mask is None or mask.size == 0:
            return self._empty("Empty crack mask")
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return self._empty("Not enough segmented crack pixels")
        scale = float(meters_per_pixel or self.metric_scale)
        p = np.c_[xs, ys].astype(np.float64) * scale
        pca = PCA(n_components=2).fit(p)
        proj = (p-pca.mean_) @ pca.components_.T
        return {"length_m": round(float(np.ptp(proj[:,0])),3), "width_mm": round(float(2*np.std(proj[:,1])*1000),3), "confidence": round(float(np.clip(len(xs)/10000,0.25,0.98)),3)}
