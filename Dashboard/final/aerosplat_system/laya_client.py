"""Small HTTP client for the official Laya self-hosted server.

AeroSplat uses Laya only as a decision/evaluation layer over already-generated
route candidates. It does not allow Laya to create geometry or bypass hard
constraints.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LAYA_URL = os.getenv("LAYA_URL", "http://127.0.0.1:8100/v1/systemone")
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "").strip()
LAYA_TIMEOUT = float(os.getenv("LAYA_TIMEOUT", "90"))


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if LAYA_API_KEY:
        headers["Authorization"] = f"Bearer {LAYA_API_KEY}"
    return headers


def health() -> Dict[str, Any]:
    url = LAYA_URL.rsplit("/v1/systemone", 1)[0].rstrip("/") + "/health"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        return {"available": True, **data}
    except (URLError, HTTPError, TimeoutError, ValueError) as exc:
        return {"available": False, "error": str(exc), "url": url}


def evaluate_routes(
    *,
    site_state: Dict[str, Any],
    routes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not routes:
        return {
            "available": False,
            "status": "NO_ELIGIBLE_ROUTES",
            "selected_route": None,
            "raw": None,
        }

    route_descriptions: Dict[str, str] = {}
    for route in routes:
        rid = str(route["route_id"])
        route_descriptions[rid] = (
            f"Candidate {rid}: distance {float(route.get('distance', 0)):.2f} grid units; "
            f"hazard exposure {float(route.get('hazard_exposure', 0)):.3f}; "
            f"exposure cost {float(route.get('exposure_cost', 0)):.2f}; "
            f"clearance indicator {float(route.get('clearance', 0)):.3f}; "
            f"mean cell risk {float(route.get('mean_cell_risk', 0)):.3f}; "
            f"eligible {route.get('eligibility', 'UNKNOWN')}."
        )

    state = {
        "mission": "Evaluate responder approach route candidates for a collapsed/disaster site.",
        "site": site_state,
        "candidates": routes,
        "safety_rule": "Never choose a route marked REJECTED_HARD_CONSTRAINT.",
    }

    questions = {
        "recommended_route": {
            "type": "choice",
            "instructions": (
                "Choose the eligible route candidate that best balances lower observed hazard exposure, "
                "greater clearance, and reasonable route length. Do not choose a rejected hard-constraint route."
            ),
            "criteria": route_descriptions,
        },
        "review_priority": {
            "type": "score",
            "instructions": (
                "Rate how strongly the proposed route should require qualified human field review before operational use."
            ),
            "criteria": [
                "routine review",
                "moderate review",
                "high review priority",
                "critical review priority",
            ],
        },
        "needs_reassessment": {
            "type": "noul",
            "instructions": (
                "Should the route be reassessed before operational use because of hazard uncertainty, "
                "incomplete evidence, or changing disaster conditions?"
            ),
        },
    }

    payload: Dict[str, Any] = {
        "state": state,
        "questions": questions,
        "model": "typed-decisions",
    }
    request = Request(
        LAYA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )

    try:
        with urlopen(request, timeout=LAYA_TIMEOUT) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ValueError) as exc:
        return {
            "available": False,
            "status": "LAYA_UNAVAILABLE",
            "error": str(exc),
            "selected_route": None,
            "raw": None,
        }

    answers = raw.get("answers") or {}
    recommended_answer = answers.get("recommended_route") or {}
    selected = recommended_answer.get("choice")
    valid_ids = {str(route["route_id"]) for route in routes}
    if selected not in valid_ids:
        selected = None

    return {
        "available": True,
        "status": "LAYA_EVALUATED",
        "selected_route": selected,
        "selected_route_confidence": (answers.get("recommended_route") or {}).get("confidence"),
        "review_priority": (answers.get("review_priority") or {}).get("score"),
        "reassessment_probability": (answers.get("needs_reassessment") or {}).get("noul"),
        "routing": raw.get("routing") or {},
        "raw": raw,
    }
