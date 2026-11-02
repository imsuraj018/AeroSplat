"""Risk-aware candidate route generation for AeroSplat.

The module generates multiple image-plane route candidates from a grid.  It is
intended to be a deterministic path-planning layer.  A separate Laya decision
layer can compare the resulting candidates; Laya never supplies path
coordinates and never overrides hard safety constraints.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def normalize_point(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return [clamp01(raw[0]), clamp01(raw[1])]


def build_cost_map(
    width: int,
    height: int,
    hazards: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return blocked mask and continuous hazard risk map."""
    blocked = np.zeros((height, width), dtype=bool)
    risk = np.zeros((height, width), dtype=np.float64)

    cell_x = np.arange(width, dtype=np.float64)[None, :] / max(width - 1, 1)
    cell_y = np.arange(height, dtype=np.float64)[:, None] / max(height - 1, 1)

    for hazard in hazards:
        hx, hy = hazard["point"]
        # Slightly normalize x to account for non-square planning grids.
        dx = (cell_x - hx) * max(width / max(height, 1), 1.0)
        dy = cell_y - hy
        distance = np.sqrt(dx * dx + dy * dy)
        radius = max(float(hazard.get("radius", 0.06)), 1e-4)
        influence = np.clip(1.0 - distance / radius, 0.0, 1.0)
        risk += float(hazard.get("weight", 3.0)) * influence
        if bool(hazard.get("hard", False)):
            blocked |= distance <= radius

    return blocked, risk


def _cell(point: List[float], width: int, height: int) -> Tuple[int, int]:
    return (
        min(width - 1, max(0, int(round(point[0] * (width - 1))))),
        min(height - 1, max(0, int(round(point[1] * (height - 1))))),
    )


def _path_to_norm(path_cells: List[Tuple[int, int]], width: int, height: int) -> List[List[float]]:
    return [
        [
            round(x / max(width - 1, 1), 5),
            round(y / max(height - 1, 1), 5),
        ]
        for x, y in path_cells
    ]


def _simplify(path_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(path_cells) <= 2:
        return path_cells
    out = [path_cells[0]]
    for idx in range(1, len(path_cells) - 1):
        ax, ay = path_cells[idx - 1]
        bx, by = path_cells[idx]
        cx, cy = path_cells[idx + 1]
        if (bx - ax, by - ay) != (cx - bx, cy - by):
            out.append(path_cells[idx])
    out.append(path_cells[-1])
    return out


def _astar(
    start: Tuple[int, int],
    target: Tuple[int, int],
    blocked: np.ndarray,
    risk: np.ndarray,
    diversity_penalty: Optional[np.ndarray] = None,
) -> Optional[List[Tuple[int, int]]]:
    height, width = blocked.shape
    sx, sy = start
    tx, ty = target
    if blocked[sy, sx] or blocked[ty, tx]:
        return None

    moves = [
        (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
    ]

    open_heap: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    g_cost: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while open_heap:
        f_score, cx, cy = heapq.heappop(open_heap)
        current_g = g_cost.get((cx, cy))
        if current_g is None:
            continue
        if (cx, cy) == (tx, ty):
            path = [(tx, ty)]
            current = (tx, ty)
            while current != (sx, sy):
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy, step_distance in moves:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or blocked[ny, nx]:
                continue
            # Prevent diagonal corner-cutting through hard hazards.
            if dx != 0 and dy != 0:
                if blocked[cy, nx] or blocked[ny, cx]:
                    continue
            exposure = float(risk[ny, nx]) / 7.0
            diversity = float(diversity_penalty[ny, nx]) if diversity_penalty is not None else 0.0
            tentative = current_g + step_distance * (1.0 + exposure) + diversity
            if tentative >= g_cost.get((nx, ny), float("inf")):
                continue
            g_cost[(nx, ny)] = tentative
            came_from[(nx, ny)] = (cx, cy)
            heuristic = math.hypot(tx - nx, ty - ny)
            heapq.heappush(open_heap, (tentative + heuristic, nx, ny))

    return None


def _route_metrics(
    path_cells: List[Tuple[int, int]],
    risk: np.ndarray,
    hazards: List[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, Any]:
    distance = 0.0
    exposure_cost = 0.0
    risk_values = []
    min_hard_distance = 1.0

    hard_hazards = [h for h in hazards if bool(h.get("hard", False))]

    for (ax, ay), (bx, by) in zip(path_cells, path_cells[1:]):
        step = math.hypot(bx - ax, by - ay)
        distance += step
        exposure_cost += step * (1.0 + float(risk[by, bx]) / 7.0)
        risk_values.append(float(risk[by, bx]))

    for cx, cy in path_cells:
        px = cx / max(width - 1, 1)
        py = cy / max(height - 1, 1)
        for hazard in hard_hazards:
            hx, hy = hazard["point"]
            min_hard_distance = min(min_hard_distance, math.hypot(px - hx, py - hy))

    normalized_exposure = min(1.0, exposure_cost / max(distance * 4.5, 1.0))
    clearance = min(1.0, min_hard_distance / 0.20) if hard_hazards else 1.0
    blocked = False

    return {
        "distance": round(distance, 2),
        "exposure_cost": round(exposure_cost, 2),
        "hazard_exposure": round(normalized_exposure, 4),
        "clearance": round(clearance, 4),
        "blocked": blocked,
        "mean_cell_risk": round(float(np.mean(risk_values)) if risk_values else 0.0, 4),
    }


def generate_candidate_routes(
    start: List[float],
    target: List[float],
    hazards: List[Dict[str, Any]],
    k: int = 5,
    width: int = 100,
    height: int = 70,
) -> Dict[str, Any]:
    start = normalize_point(start) or [0.1, 0.5]
    target = normalize_point(target) or [0.9, 0.5]
    width = max(40, min(180, int(width)))
    height = max(30, min(120, int(height)))
    k = max(1, min(8, int(k)))

    blocked, risk = build_cost_map(width, height, hazards)
    start_cell = _cell(start, width, height)
    target_cell = _cell(target, width, height)

    if blocked[target_cell[1], target_cell[0]]:
        return {
            "status": "TARGET_IN_BLOCKED_ZONE",
            "candidates": [],
            "graph": {"width": width, "height": height, "nodes": int(width * height), "edges": 0},
        }

    candidates: List[Dict[str, Any]] = []
    seen_paths = set()
    diversity_penalty = np.zeros_like(risk)

    for idx in range(k):
        path_cells = _astar(
            start_cell,
            target_cell,
            blocked,
            risk,
            diversity_penalty if idx else None,
        )
        if not path_cells:
            break

        path_key = tuple(path_cells)
        if path_key in seen_paths:
            break
        seen_paths.add(path_key)

        simplified_cells = _simplify(path_cells)
        metrics = _route_metrics(simplified_cells, risk, hazards, width, height)
        route = {
            "route_id": f"R{idx + 1:02d}",
            "path": _path_to_norm(simplified_cells, width, height),
            **metrics,
            "hazards": hazards,
            "planner": "weighted_astar_candidate_generator",
        }
        candidates.append(route)

        # Penalize a narrow band around the selected path to encourage a
        # materially different next candidate while preserving feasibility.
        if idx < k - 1:
            for x, y in path_cells:
                y0, y1 = max(0, y - 3), min(height, y + 4)
                x0, x1 = max(0, x - 3), min(width, x + 4)
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
                diversity_penalty[y0:y1, x0:x1] += np.clip(5.0 - dist, 0.0, 5.0)

    return {
        "status": "CANDIDATES_READY" if candidates else "NO_PATH",
        "candidates": candidates,
        "graph": {
            "width": width,
            "height": height,
            "nodes": int(width * height),
            "edges": int(max(0, width * height * 4)),
        },
    }
