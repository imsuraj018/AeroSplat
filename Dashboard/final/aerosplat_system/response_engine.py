"""3D evidence mapping and route visualization helpers."""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def read_cameras_bin(path: Path) -> Dict[int, Dict[str, Any]]:
    # COLMAP camera model id -> (model name, parameter count)
    models = {
        0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
        4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    }
    result: Dict[int, Dict[str, Any]] = {}
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            return result
        count = struct.unpack("<Q", raw)[0]
        for _ in range(count):
            head = stream.read(24)
            if len(head) != 24:
                break
            cam_id, model_id = struct.unpack_from("<II", head, 0)
            width, height = struct.unpack_from("<QQ", head, 8)
            model_name, n_params = models.get(model_id, (f"MODEL_{model_id}", 4))
            params_raw = stream.read(n_params * 8)
            if len(params_raw) != n_params * 8:
                break
            params = struct.unpack("<" + "d" * n_params, params_raw)
            if model_name == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx = fy = f
            elif model_name in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
                fx, fy, cx, cy = params[:4]
            else:
                f, cx, cy = params[:3]
                fx = fy = f
            result[int(cam_id)] = {
                "model": model_name, "width": int(width), "height": int(height),
                "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            }
    return result


def read_images_bin(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            return result
        count = struct.unpack("<Q", raw)[0]
        for _ in range(count):
            raw = stream.read(52)
            if len(raw) != 52:
                break
            image_id = struct.unpack_from("<Q", raw, 0)[0]
            qw, qx, qy, qz = struct.unpack_from("<dddd", raw, 8)
            tx, ty, tz = struct.unpack_from("<ddd", raw, 40)
            camera_id = struct.unpack_from("<I", raw, 48)[0]
            name_bytes = bytearray()
            while True:
                ch = stream.read(1)
                if not ch:
                    break
                if ch == b"\x00":
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8", errors="replace")
            raw = stream.read(8)
            if len(raw) != 8:
                break
            point_count = struct.unpack("<Q", raw)[0]
            stream.seek(point_count * 24, 1)
            R = quaternion_to_rotation(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            position = -(R.T @ t)
            forward = R.T @ np.array([0.0, 0.0, 1.0])
            forward /= max(np.linalg.norm(forward), 1e-12)
            result[name] = {
                "image_id": int(image_id), "camera_id": int(camera_id),
                "position": position, "rotation": R, "forward": forward,
            }
    return result


def read_points3d_bin(path: Path, max_points: int = 90_000) -> np.ndarray:
    data: List[Tuple[float, float, float]] = []
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            return np.empty((0, 3), dtype=np.float64)
        total = int(struct.unpack("<Q", raw)[0])
        keep = min(total, int(max_points))
        for idx in range(total):
            record = stream.read(51)
            if len(record) != 51:
                break
            _, x, y, z, _, _, _, track_len = struct.unpack("<QdddBBBdQ", record)
            stream.seek(int(track_len) * 16, 1)
            if idx < keep and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                data.append((x, y, z))
    return np.asarray(data, dtype=np.float64)


def find_colmap_model(dense_sparse: Path, legacy: Optional[Path] = None) -> Optional[Path]:
    candidates = [dense_sparse, dense_sparse / "0"]
    if legacy:
        candidates.extend([legacy, legacy / "0"])
    for candidate in candidates:
        if (candidate / "images.bin").exists() and (candidate / "cameras.bin").exists() and (candidate / "points3D.bin").exists():
            return candidate
    return None


def project_points(points: np.ndarray, camera: Dict[str, Any], pose: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.empty((0, 2)), np.empty((0,))
    R = pose["rotation"]
    t = -R @ pose["position"]
    cam = (R @ points.T).T + t
    z = cam[:, 2]
    valid = z > 1e-4
    x = camera["fx"] * cam[:, 0] / np.maximum(z, 1e-4) + camera["cx"]
    y = camera["fy"] * cam[:, 1] / np.maximum(z, 1e-4) + camera["cy"]
    pix = np.column_stack([x, y])
    valid &= pix[:, 0] >= 0
    valid &= pix[:, 0] < camera["width"]
    valid &= pix[:, 1] >= 0
    valid &= pix[:, 1] < camera["height"]
    return pix[valid], np.flatnonzero(valid)


def bbox_to_3d(bbox: List[float], pixel: np.ndarray, world_points: np.ndarray, valid_indices: np.ndarray, camera: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if pixel.size == 0 or world_points.size == 0 or valid_indices.size == 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    px1, py1 = x1 * camera["width"], y1 * camera["height"]
    px2, py2 = x2 * camera["width"], y2 * camera["height"]
    inside = (
        (pixel[:, 0] >= px1) & (pixel[:, 0] <= px2) &
        (pixel[:, 1] >= py1) & (pixel[:, 1] <= py2)
    )
    candidates = world_points[valid_indices[inside]]
    if len(candidates) < 3:
        return None
    center = np.median(candidates, axis=0)
    spread = np.percentile(np.linalg.norm(candidates - center, axis=1), 75) if len(candidates) > 2 else 0.0
    return {"position": [round(float(v), 5) for v in center], "radius": round(float(max(spread, 0.05)), 5), "support_points": int(len(candidates))}


def map_hazards_to_3d(hazards: List[Dict[str, Any]], points: np.ndarray, camera: Dict[str, Any], pose: Dict[str, Any]) -> List[Dict[str, Any]]:
    pixels, valid_idx = project_points(points, camera, pose)
    mapped: List[Dict[str, Any]] = []
    for hazard in hazards:
        item = dict(hazard)
        mapped_3d = bbox_to_3d(item.get("bbox") or [0, 0, 1, 1], pixels, points, valid_idx, camera)
        if mapped_3d:
            item["position_3d"] = mapped_3d["position"]
            item["support_points"] = mapped_3d["support_points"]
            item["spatial_radius"] = mapped_3d["radius"]
            item["spatial_status"] = "PROJECTED_FROM_COLMAP"
        else:
            item["position_3d"] = None
            item["spatial_status"] = "NO_SUPPORTING_3D_POINTS"
        mapped.append(item)
    return mapped


def route_2d_to_3d(path: List[List[float]], points: np.ndarray, camera: Dict[str, Any], pose: Dict[str, Any], max_vertices: int = 120) -> List[List[float]]:
    if not path or points.size == 0:
        return []
    pixels, valid_idx = project_points(points, camera, pose)
    if len(pixels) == 0:
        return []
    pts_norm = np.column_stack([pixels[:, 0] / camera["width"], pixels[:, 1] / camera["height"]])
    stride = max(1, len(path) // max_vertices)
    samples = path[::stride]
    if samples[-1] != path[-1]:
        samples.append(path[-1])
    out: List[List[float]] = []
    used = set()
    for x, y in samples:
        d2 = np.sum((pts_norm - np.array([x, y])) ** 2, axis=1)
        order = np.argsort(d2)[:20]
        pick = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(pick)
        world = points[int(valid_idx[pick])]
        out.append([round(float(v), 5) for v in world])
    return out


def recommend_approach_point(hazards: List[Dict[str, Any]], margin: float = 0.04) -> Dict[str, Any]:
    # Boundary candidates represent observation/approach points visible in the selected frame.
    candidates = []
    for i in range(8):
        t = i / 7.0
        candidates.extend([[t, margin], [t, 1.0 - margin], [margin, t], [1.0 - margin, t]])
    unique = {(round(x, 4), round(y, 4)) for x, y in candidates}
    candidates = [[x, y] for x, y in sorted(unique)]

    def score(pt: List[float]) -> float:
        total = 0.0
        for h in hazards:
            hx, hy = h.get("point", [0.5, 0.5])[:2]
            dist = math.hypot(pt[0] - hx, pt[1] - hy)
            weight = float(h.get("weight", 3.0)) * float(h.get("confidence", 0.5))
            total += weight / max(dist, 0.05)
            if h.get("hard") and dist < float(h.get("radius", 0.08)) * 1.5:
                total += 100.0
        return total

    best = min(candidates, key=score)
    side = "left" if best[0] < 0.2 else "right" if best[0] > 0.8 else "top" if best[1] < 0.2 else "bottom"
    return {
        "point": [round(best[0], 4), round(best[1], 4)],
        "side": side,
        "score": round(score(best), 2),
        "type": "approach_candidate",
        "reason": "Lowest heuristic hazard-exposure among visible frame-boundary candidates; not a verified doorway or structural clearance.",
    }
