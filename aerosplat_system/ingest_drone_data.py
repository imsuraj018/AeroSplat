from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG


class DroneDataIngestor:
    def __init__(self):
        CONFIG.ensure_dirs()
        self.images_dir = CONFIG.images_dir
        self.db_path = CONFIG.colmap_database
        self.sparse_dir = CONFIG.sparse_dir

    @staticmethod
    def clean_path(raw_path: str) -> str:
        if not raw_path:
            return ""
        p = raw_path.strip().strip("'\"")
        if p.startswith("& "):
            p = p[2:].strip().strip("'\"")
        return p

    def clear_reconstruction(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.sparse_dir):
            shutil.rmtree(self.sparse_dir, ignore_errors=True)
        os.makedirs(self.sparse_dir, exist_ok=True)

    def copy_images(self, source_dir: str) -> int:
        source = Path(self.clean_path(source_dir))
        if not source.is_dir():
            raise FileNotFoundError(f"Image directory not found: {source}")
        valid = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        copied = 0
        for p in sorted(source.iterdir()):
            if p.suffix.lower() in valid:
                shutil.copy2(p, Path(self.images_dir)/p.name)
                copied += 1
        return copied

    def extract_frames_from_video(self, video_path: str, fps_sample: float | None = None, blur_threshold: float | None = None) -> int:
        video_path = self.clean_path(video_path)
        sample = fps_sample or CONFIG.frame_sample_fps
        blur = blur_threshold or CONFIG.blur_threshold
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(round(video_fps / max(sample, 0.1))))
        frame_idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness >= blur or saved < 10:
                    out = Path(self.images_dir) / f"frame_{saved:05d}.jpg"
                    cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    saved += 1
            frame_idx += 1
        cap.release()
        return saved

    def _run(self, args: list[str]) -> None:
        subprocess.run(args, check=True)

    def run_colmap(self, is_video: bool = True) -> str:
        images = [p for p in Path(self.images_dir).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        if len(images) < 3:
            raise RuntimeError("COLMAP requires at least 3 overlapping images for a useful reconstruction.")
        self.clear_reconstruction()
        colmap = CONFIG.colmap_exe
        feat = [colmap, "feature_extractor", "--database_path", self.db_path, "--image_path", self.images_dir,
                "--ImageReader.camera_model", CONFIG.camera_model, "--ImageReader.single_camera", "1" if CONFIG.single_camera else "0",
                "--FeatureExtraction.use_gpu", "1" if CONFIG.use_gpu_features else "0"]
        self._run(feat)
        if is_video:
            match = [colmap, "sequential_matcher", "--database_path", self.db_path,
                     "--SequentialMatching.overlap", str(CONFIG.sequential_overlap),
                     "--FeatureMatching.use_gpu", "1" if CONFIG.use_gpu_matching else "0"]
        else:
            matcher = "exhaustive_matcher" if len(images) <= CONFIG.max_images_for_exhaustive_matching else "sequential_matcher"
            match = [colmap, matcher, "--database_path", self.db_path,
                     *( ["--SequentialMatching.overlap", str(CONFIG.sequential_overlap)] if matcher == "sequential_matcher" else []),
                     "--FeatureMatching.use_gpu", "1" if CONFIG.use_gpu_matching else "0"]
        self._run(match)
        self._run([colmap, "mapper", "--database_path", self.db_path, "--image_path", self.images_dir, "--output_path", self.sparse_dir])
        return os.path.join(self.sparse_dir, "0")

    def calibrate_scale_from_apriltags(self, known_tag_size_m: float | None = None) -> float:
        known = known_tag_size_m or CONFIG.known_apriltag_size_meters
        try:
            import apriltag
            detector = apriltag.Detector()
        except Exception:
            return CONFIG.default_metric_scale_m_per_sfm_unit
        for path in sorted(Path(self.images_dir).iterdir())[:30]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            results = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if not results:
                continue
            widths = [float(np.linalg.norm(r.corners[0]-r.corners[1])) for r in results]
            if max(widths) > 1:
                # This is a pixel-to-meter image scale, not necessarily an SfM metric scale.
                # Treat it as a weak fallback only.
                return float(known / np.median(widths))
        return CONFIG.default_metric_scale_m_per_sfm_unit
