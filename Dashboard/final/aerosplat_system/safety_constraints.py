"""Hard response-planning constraints for AeroSplat."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

HARD_HAZARDS = {
    "fire", "fire_like", "gas", "utility", "void", "restricted",
    "unstable_wall", "falling_debris", "collapse_indicator", "smoke",
}


def route_intersects_hard_hazard(route: Dict[str, Any]) -> bool:
    if bool(route.get("blocked", False)):
        return True
    hazards = route.get("hazards") or []
    path = route.get("path") or []
    if not path:
        return True

    for hazard in hazards:
        if not bool(hazard.get("hard", False)) and str(hazard.get("kind")) not in HARD_HAZARDS:
            continue
        point = hazard.get("point") or [0.5, 0.5]
        radius = float(hazard.get("radius", 0.06))
        for px, py in path:
            dx = px - point[0]
            dy = py - point[1]
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                return True
    return False


def filter_eligible_routes(routes: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for route in routes:
        item = dict(route)
        if route_intersects_hard_hazard(item):
            item["blocked"] = True
            item["eligibility"] = "REJECTED_HARD_CONSTRAINT"
            rejected.append(item)
        else:
            item["eligibility"] = "ELIGIBLE"
            eligible.append(item)
    return eligible, rejected
