"""COLMAP sparse reconstruction loader with optional synthetic fallback."""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import CONFIG

try:
    import cv2
except ImportError:
    cv2 = None


CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2 * (q2*q2 + q3*q3), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
        [2 * (q1*q2 + q0*q3), 1 - 2 * (q1*q1 + q3*q3), 2 * (q2*q3 - q0*q1)],
        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1*q1 + q2*q2)],
    ], dtype=np.float64)


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


@dataclass
class ColmapImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def rotation(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    @property
    def camera_center(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec


class ColmapDataset:
    def __init__(self, colmap_dir: str | None = None, images_dir: str | None = None):
        self.colmap_dir = colmap_dir or CONFIG.colmap_dir
        self.images_dir = images_dir or CONFIG.images_dir
        self.points3d = np.empty((0, 3), dtype=np.float32)
        self.point_colors = np.empty((0, 3), dtype=np.float32)
        self.point_errors = np.empty((0,), dtype=np.float32)
        self.camera_poses: Dict[str, ColmapImagePose] = {}
        self.cameras: Dict[int, dict] = {}
        self.point_ids = np.empty((0,), dtype=np.int64)
        self.crack_indices = np.empty((0,), dtype=np.int64)
        self.wall_indices = np.empty((0,), dtype=np.int64)
        self.debris_indices = np.empty((0,), dtype=np.int64)
        self._load()

    def _load(self) -> None:
        points_path = os.path.join(self.colmap_dir, "points3D.bin")
        images_path = os.path.join(self.colmap_dir, "images.bin")
        cameras_path = os.path.join(self.colmap_dir, "cameras.bin")
        if os.path.exists(points_path):
            self._load_points3d_bin(points_path)
        elif os.path.exists(os.path.join(self.colmap_dir, "points3D.txt")):
            self._load_points3d_txt(os.path.join(self.colmap_dir, "points3D.txt"))
        if os.path.exists(images_path):
            self._load_images_bin(images_path)
        if os.path.exists(cameras_path):
            self._load_cameras_bin(cameras_path)
        if len(self.points3d):
            self._derive_regions()

    def _load_points3d_bin(self, path: str) -> None:
        xyz, rgb, err, ids = [], [], [], []
        with open(path, "rb") as f:
            num_points = read_next_bytes(f, 8, "Q")[0]
            for _ in range(num_points):
                point_id = read_next_bytes(f, 8, "Q")[0]
                x, y, z = read_next_bytes(f, 24, "ddd")
                r, g, b = read_next_bytes(f, 3, "BBB")
                error = read_next_bytes(f, 8, "d")[0]
                track_len = read_next_bytes(f, 8, "Q")[0]
                f.seek(8 * track_len * 2, 1)
                ids.append(point_id); xyz.append((x, y, z)); rgb.append((r/255, g/255, b/255)); err.append(error)
        self.point_ids = np.asarray(ids, dtype=np.int64)
        self.points3d = np.asarray(xyz, dtype=np.float32)
        self.point_colors = np.asarray(rgb, dtype=np.float32)
        self.point_errors = np.asarray(err, dtype=np.float32)

    def _load_points3d_txt(self, path: str) -> None:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.split()
                rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), int(p[4]), int(p[5]), int(p[6]), float(p[7])))
        if rows:
            a = np.asarray(rows, dtype=np.float64)
            self.point_ids = a[:, 0].astype(np.int64)
            self.points3d = a[:, 1:4].astype(np.float32)
            self.point_colors = (a[:, 4:7] / 255.0).astype(np.float32)
            self.point_errors = a[:, 7].astype(np.float32)

    def _load_images_bin(self, path: str) -> None:
        with open(path, "rb") as f:
            num = read_next_bytes(f, 8, "Q")[0]
            for _ in range(num):
                image_id = read_next_bytes(f, 4, "i")[0]
                qvec = np.array(read_next_bytes(f, 32, "dddd"), dtype=np.float64)
                tvec = np.array(read_next_bytes(f, 24, "ddd"), dtype=np.float64)
                camera_id = read_next_bytes(f, 4, "i")[0]
                name = bytearray()
                while True:
                    c = f.read(1)
                    if c == b"\\x00":
                        break
                    name.extend(c)
                name = name.decode("utf-8", errors="replace")
                num_points2d = read_next_bytes(f, 8, "Q")[0]
                f.seek(num_points2d * (8 + 8 + 8), 1)
                self.camera_poses[name] = ColmapImagePose(image_id, qvec, tvec, camera_id, name)

    def _load_cameras_bin(self, path: str) -> None:
        with open(path, "rb") as f:
            num = read_next_bytes(f, 8, "Q")[0]
            for _ in range(num):
                camera_id = read_next_bytes(f, 4, "i")[0]
                model_id = read_next_bytes(f, 4, "i")[0]
                width = read_next_bytes(f, 8, "Q")[0]
                height = read_next_bytes(f, 8, "Q")[0]
                model_name, nparams = CAMERA_MODEL_IDS.get(model_id, ("UNKNOWN", 4))
                params = np.array(read_next_bytes(f, 8*nparams, "d"*nparams), dtype=np.float64)
                self.cameras[camera_id] = {"model": model_name, "width": width, "height": height, "params": params}

    def _derive_regions(self) -> None:
        if len(self.points3d) < 20:
            idx = np.arange(len(self.points3d))
            self.crack_indices = idx
            self.wall_indices = idx
            self.debris_indices = idx
            return
        p = self.points3d
        c = self.point_colors
        z = p[:, 2]
        med = np.median(z)
        spread = np.std(z) + 1e-6
        dark = np.mean(c, axis=1) < np.quantile(np.mean(c, axis=1), 0.12)
        low = z < med - 0.25 * spread
        high = z > med + 0.20 * spread
        self.crack_indices = np.where(dark)[0]
        self.wall_indices = np.where(high)[0]
        self.debris_indices = np.where(low)[0]
        # Ensure non-empty masks.
        if len(self.crack_indices) < 8:
            self.crack_indices = np.argsort(np.mean(c, axis=1))[:max(8, len(p)//20)]
        if len(self.wall_indices) < 8:
            self.wall_indices = np.argsort(p[:,2])[-max(8, len(p)//10):]
        if len(self.debris_indices) < 8:
            self.debris_indices = np.argsort(p[:,2])[:max(8, len(p)//10)]

    def synthesize_disaster_geometry(self, seed: int = 17, n: int = 6500) -> None:
        """Synthetic fallback for demos; never used to claim real-world measurements."""
        rng = np.random.default_rng(seed)
        walls = []
        colors = []
        # Building facade with controlled lean and windows.
        for x0 in np.linspace(0, 18, 7):
            y = rng.uniform(-0.15, 0.15, n//12)
            z = rng.uniform(0, 9, n//12)
            x = x0 + rng.normal(0, 0.04, len(z)) + 0.08*z
            walls.append(np.c_[x, y, z]); colors.append(np.tile([0.65,0.65,0.65], (len(z),1)))
        # Side wall.
        x = rng.uniform(-0.1, 0.1, n//12); y = rng.uniform(0, 8, n//12); z = rng.uniform(0, 9, n//12)
        walls.append(np.c_[x,y,z]); colors.append(np.tile([0.58,0.60,0.62], (len(z),1)))
        # Rubble mound.
        x = rng.uniform(0, 18, n//5); y = rng.uniform(-2, 3, n//5); z = np.maximum(0, rng.normal(0.7,0.5,n//5))
        walls.append(np.c_[x,y,z]); colors.append(rng.uniform(0.25,0.50,(len(z),3)))
        # Dark cracks as a narrow ribbon.
        x = rng.uniform(4, 13, n//20); z = rng.uniform(2, 7, n//20); y = rng.normal(-0.18,0.03,n//20)
        walls.append(np.c_[x,y,z]); colors.append(rng.uniform(0.03,0.10,(len(z),3)))
        self.points3d = np.vstack(walls).astype(np.float32)
        self.point_colors = np.vstack(colors).astype(np.float32)
        self.point_errors = np.zeros(len(self.points3d), dtype=np.float32)
        self.point_ids = np.arange(len(self.points3d), dtype=np.int64)
        self._derive_regions()
