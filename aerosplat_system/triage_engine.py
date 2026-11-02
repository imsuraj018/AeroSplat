from __future__ import annotations

from .config import CONFIG


class ATCTriageEngine:
    """ATC-20-informed preliminary screening. Not a structural engineering certification."""
    @staticmethod
    def evaluate(metrics: dict, risk: dict | None = None) -> dict:
        tilt = float(metrics.get("wall_tilt", {}).get("tilt_degrees", 0.0) or 0.0)
        crack = float(metrics.get("crack", {}).get("width_mm", 0.0) or 0.0)
        debris = float(metrics.get("debris", {}).get("volume_m3", 0.0) or 0.0)
        factors = []
        red = []
        yellow = []
        if tilt >= CONFIG.critical_tilt_deg: red.append(f"wall tilt {tilt:.2f}°")
        elif tilt >= CONFIG.warning_tilt_deg: yellow.append(f"wall tilt {tilt:.2f}°")
        if crack >= CONFIG.critical_crack_width_mm: red.append(f"crack width {crack:.1f} mm")
        elif crack >= CONFIG.warning_crack_width_mm: yellow.append(f"crack width {crack:.1f} mm")
        if debris >= CONFIG.critical_debris_vol_m3: red.append(f"estimated debris {debris:.2f} m³")
        elif debris >= CONFIG.warning_debris_vol_m3: yellow.append(f"estimated debris {debris:.2f} m³")
        if risk and risk.get("level") == "HIGH": red.append("high composite visual/geometric risk")
        factors.extend(red or yellow)
        if red:
            tag, severity = "RED", "CRITICAL"
            directives = ["Do not authorize routine entry based on this automated screen.", "Establish a controlled exclusion zone and request qualified structural inspection."]
        elif yellow:
            tag, severity = "YELLOW", "MODERATE"
            directives = ["Restrict access to authorized responders.", "Perform qualified on-site structural verification before occupancy."]
        else:
            tag, severity = "GREEN", "LOW"
            directives = ["No critical indicator exceeded this preliminary screening model.", "Continue monitoring and obtain professional verification."]
        return {
            "safety_tag": tag,
            "label": {"GREEN":"INSPECTED / PRELIMINARY LOW RISK","YELLOW":"RESTRICTED USE / PRELIMINARY","RED":"UNSAFE / PRELIMINARY"}[tag],
            "severity_level": severity,
            "factors": factors,
            "directives": directives,
            "disclaimer": "AI-assisted preliminary screening informed by ATC-20 terminology; not an engineering certification.",
        }
