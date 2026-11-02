from __future__ import annotations

import argparse
import json
import os
import shutil

from .colmap_loader import ColmapDataset
from .config import CONFIG
from .damage_quantifier import StructuralDamageQuantifier
from .gs_adapter import GaussianSplatAdapter
from .hazard_detector import HazardDetector
from .ingest_drone_data import DroneDataIngestor
from .risk_engine import RiskEngine
from .route_planner import demo_route
from .triage_engine import ATCTriageEngine
from .gaussian_model import save_initial_splat_ply


def run_pipeline(source: str | None = None, mode: str = "auto", train_3dgs: bool = False) -> dict:
    CONFIG.ensure_dirs()
    ingestor = DroneDataIngestor()
    if source:
        p = ingestor.clean_path(source)
        if os.path.isdir(p):
            ingestor.copy_images(p)
            if mode in {"auto", "images"}:
                ingestor.run_colmap(is_video=False)
        elif os.path.isfile(p):
            ingestor.extract_frames_from_video(p)
            ingestor.run_colmap(is_video=True)
        else:
            raise FileNotFoundError(p)

    dataset = ColmapDataset(CONFIG.colmap_dir, CONFIG.images_dir)
    if len(dataset.points3d) == 0:
        dataset.synthesize_disaster_geometry()
    save_initial_splat_ply(CONFIG.scene_ply, dataset.points3d, dataset.point_colors, radius=max(CONFIG.default_metric_scale_m_per_sfm_unit*0.5, 0.002))

    gs_result = {"status": "initialization_only"}
    if train_3dgs:
        gs_result = GaussianSplatAdapter().run()

    quant = StructuralDamageQuantifier(CONFIG.default_metric_scale_m_per_sfm_unit)
    metrics = {
        "crack": quant.analyze_crack(dataset.points3d[dataset.crack_indices]),
        "wall_tilt": quant.analyze_wall_tilt(dataset.points3d[dataset.wall_indices]),
        "debris": quant.analyze_debris_volume(dataset.points3d[dataset.debris_indices]),
    }
    images = [os.path.join(CONFIG.images_dir, f) for f in sorted(os.listdir(CONFIG.images_dir)) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    detector = HazardDetector()
    hazards = detector.predict(images[:20]) if detector.available else detector.heuristic_hazards(images[:20])
    risk = RiskEngine().evaluate(metrics, hazards).as_dict()
    triage = ATCTriageEngine.evaluate(metrics, risk)
    route = demo_route()
    report = {"metrics":metrics,"hazards":hazards,"risk":risk,"triage":triage,"route":route,"gs":gs_result}
    with open(os.path.join(CONFIG.outputs_dir,"report.json"),"w",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroSplat post-disaster analysis pipeline")
    parser.add_argument("source", nargs="?", help="Drone video or folder of images")
    parser.add_argument("--train-3dgs", action="store_true", help="Run the official 3DGS optimizer through the adapter")
    args = parser.parse_args()
    report = run_pipeline(args.source, train_3dgs=args.train_3dgs)
    print(json.dumps({"triage":report["triage"],"risk":report["risk"],"route":report["route"]}, indent=2))


if __name__ == "__main__":
    main()
