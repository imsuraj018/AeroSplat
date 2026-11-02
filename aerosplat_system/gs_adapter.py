from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import CONFIG
from .gaussian_model import save_initial_splat_ply


class GaussianSplatAdapter:
    """Adapter to the official GraphDeco-Inria Gaussian Splatting implementation."""
    def __init__(self, repo_dir: str | None = None):
        self.repo_dir = Path(repo_dir or CONFIG.gs_repo_dir)

    @property
    def train_script(self) -> Path:
        return self.repo_dir / "train.py"

    @property
    def repo_available(self) -> bool:
        return self.train_script.exists()

    def status(self) -> dict:
        return {"available": self.repo_available, "repo_dir": str(self.repo_dir), "train_script": str(self.train_script)}

    def run(self, source_dir: str | None = None, model_dir: str | None = None, iterations: int | None = None) -> dict:
        source = Path(source_dir or CONFIG.workspace_dir).resolve()
        model = Path(model_dir or os.path.join(CONFIG.outputs_dir, "3dgs_model")).resolve()
        model.mkdir(parents=True, exist_ok=True)
        if not self.repo_available:
            return {"status":"not_installed","message":"Run scripts/setup_3dgs.py or set AEROSPLAT_3DGS_REPO to the official 3DGS repo."}
        cmd = ["python", str(self.train_script), "-s", str(source), "-m", str(model), "--iterations", str(iterations or CONFIG.gs_iterations), "--sh_degree", str(CONFIG.gs_sh_degree), "--eval"]
        try:
            result = subprocess.run(cmd, cwd=str(self.repo_dir), text=True, capture_output=True, check=True)
            ply = model / "point_cloud" / f"iteration_{iterations or CONFIG.gs_iterations}" / "point_cloud.ply"
            if not ply.exists():
                candidates = list((model/"point_cloud").glob("iteration_*/point_cloud.ply"))
                ply = sorted(candidates)[-1] if candidates else None
            if ply and ply.exists():
                shutil.copy2(ply, CONFIG.scene_ply)
            return {"status":"trained","model_dir":str(model),"scene_ply":str(ply) if ply else None,"stdout_tail":result.stdout[-2000:]}
        except subprocess.CalledProcessError as exc:
            return {"status":"error","returncode":exc.returncode,"stdout_tail":exc.stdout[-1500:] if exc.stdout else "","stderr_tail":exc.stderr[-2500:] if exc.stderr else ""}
