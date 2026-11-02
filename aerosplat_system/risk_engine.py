from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CONFIG


@dataclass
class RiskResult:
    score: float
    level: str
    factors: list[str]
    route_penalty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "factors": self.factors,
            "route_penalty": round(self.route_penalty, 3),
        }


class RiskEngine:
    def evaluate(self, metrics: dict, hazards: list[dict] | None = None) -> RiskResult:
        hazards = hazards or []
        score = 0.0
        factors: list[str] = []
        tilt = float(metrics.get("wall_tilt", {}).get("tilt_degrees", 0.0) or 0.0)
        crack = float(metrics.get("crack", {}).get("width_mm", 0.0) or 0.0)
        debris = float(metrics.get("debris", {}).get("volume_m3", 0.0) or 0.0)
        score += min(1.0, tilt / max(CONFIG.critical_tilt_deg, 0.1)) * 0.40
        score += min(1.0, crack / max(CONFIG.critical_crack_width_mm, 0.1)) * 0.30
        score += min(1.0, debris / max(CONFIG.critical_debris_vol_m3, 0.1)) * 0.20
        hazard_weight = {"fire": 0.20, "smoke": 0.12, "flood": 0.16, "debris": 0.12, "collapse": 0.22, "blocked_road": 0.18}
        route_penalty = 0.0
        for h in hazards:
            name = str(h.get("class", "unknown")).lower()
            conf = float(h.get("confidence", 0.0))
            contribution = hazard_weight.get(name, 0.05) * conf
            score += contribution
            if name in {"fire", "collapse", "blocked_road", "flood"}:
                route_penalty += contribution
                factors.append(f"{name} ({conf:.0%})")
        if tilt >= CONFIG.warning_tilt_deg:
            factors.append(f"wall tilt {tilt:.2f}°")
        if crack >= CONFIG.warning_crack_width_mm:
            factors.append(f"crack width {crack:.1f} mm")
        if debris >= CONFIG.warning_debris_vol_m3:
            factors.append(f"debris {debris:.2f} m³")
        score = min(score, 1.0)
        level = "HIGH" if score >= 0.66 else "MEDIUM" if score >= 0.33 else "LOW"
        return RiskResult(score, level, factors, route_penalty)
