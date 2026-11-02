import os
from dataclasses import dataclass, field


def _default_workspace() -> str:
    return os.path.abspath(os.environ.get("AEROSPLAT_WORKSPACE", os.path.join(os.getcwd(), "aerosplat_workspace")))


@dataclass
class SystemConfig:
    workspace_dir: str = field(default_factory=_default_workspace)
    colmap_exe: str = field(default_factory=lambda: os.environ.get("COLMAP_EXE", "colmap"))
    gs_repo_dir: str = field(default_factory=lambda: os.environ.get("AEROSPLAT_3DGS_REPO", os.path.abspath(os.path.join(os.getcwd(), "third_party", "gaussian-splatting"))))
    yolo_model: str = field(default_factory=lambda: os.environ.get("AEROSPLAT_YOLO_MODEL", os.path.abspath(os.path.join(os.getcwd(), "models", "disaster_seg.pt"))))

    # Processing
    frame_sample_fps: float = 2.0
    blur_threshold: float = 40.0
    max_images_for_exhaustive_matching: int = 120
    max_points_for_browser: int = 50000

    # COLMAP
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    use_gpu_features: bool = False
    use_gpu_matching: bool = False
    sequential_overlap: int = 10

    # 3DGS adapter
    gs_iterations: int = 7000
    gs_resolution: int = -1
    gs_sh_degree: int = 3

    # Metric calibration
    known_apriltag_size_meters: float = 0.20
    default_metric_scale_m_per_sfm_unit: float = 0.05

    # Preliminary screening only; NOT official ATC-20 thresholds.
    warning_tilt_deg: float = 2.5
    critical_tilt_deg: float = 5.0
    warning_crack_width_mm: float = 5.0
    critical_crack_width_mm: float = 10.0
    warning_debris_vol_m3: float = 1.5
    critical_debris_vol_m3: float = 5.0

    # Risk weights
    route_weight_distance: float = 1.0
    route_weight_risk: float = 8.0
    route_weight_obstruction: float = 10.0
    route_weight_flood: float = 9.0
    route_weight_fire: float = 12.0

    @property
    def images_dir(self) -> str:
        return os.path.join(self.workspace_dir, "images")

    @property
    def frames_dir(self) -> str:
        return os.path.join(self.workspace_dir, "frames")

    @property
    def colmap_database(self) -> str:
        return os.path.join(self.workspace_dir, "database.db")

    @property
    def sparse_dir(self) -> str:
        return os.path.join(self.workspace_dir, "sparse")

    @property
    def colmap_dir(self) -> str:
        return os.path.join(self.sparse_dir, "0")

    @property
    def dense_dir(self) -> str:
        return os.path.join(self.workspace_dir, "dense")

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.workspace_dir, "outputs")

    @property
    def scene_ply(self) -> str:
        return os.path.join(self.outputs_dir, "scene_model.ply")

    @property
    def detections_json(self) -> str:
        return os.path.join(self.outputs_dir, "detections.json")

    @property
    def route_json(self) -> str:
        return os.path.join(self.outputs_dir, "route.json")

    def ensure_dirs(self) -> None:
        for path in [self.workspace_dir, self.images_dir, self.frames_dir, self.sparse_dir, self.dense_dir, self.outputs_dir]:
            os.makedirs(path, exist_ok=True)


CONFIG = SystemConfig()
CONFIG.ensure_dirs()
