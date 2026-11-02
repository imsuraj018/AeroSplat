"""AeroSplat HTTP client for the Laya decision server.

Laya is advisory only. AeroSplat's deterministic safety-first route optimizer
is the authoritative selector of a feasible route. Laya evaluates the
candidate set and returns an advisory comparison/review signal.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LAYA_URL = os.getenv("LAYA_URL", "http://127.0.0.1:8100/v1/systemone").strip()
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "").strip()
LAYA_TIMEOUT = float(os.getenv("LAYA_TIMEOUT", "60"))


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
        return {"available": True, "url": url, **data}
    except Exception as exc:
        return {"available": False, "url": url, "error": str(exc)}


def evaluate_routes(*, site_state: Dict[str, Any], routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not routes:
        return {
            "available": False,
            "status": "NO_ELIGIBLE_ROUTES",
            "selected_route": None,
            "selected_route_confidence": None,
            "review_priority": None,
            "reassessment_probability": None,
            "routing": {},
            "raw": None,
        }

    route_descriptions: Dict[str, str] = {}
    for route in routes:
        rid = str(route.get("route_id", "UNKNOWN"))
        route_descriptions[rid] = (
            f"{rid}: distance {float(route.get('distance', 0.0)):.2f}; "
            f"hazard exposure {float(route.get('hazard_exposure', 0.0)):.3f}; "
            f"mean risk {float(route.get('mean_cell_risk', 0.0)):.3f}; "
            f"clearance {float(route.get('clearance', 0.0)):.3f}; "
            f"exposure cost {float(route.get('exposure_cost', 0.0)):.2f}; "
            f"eligibility {route.get('eligibility', 'ELIGIBLE')}"
        )

    state = {
        "mission": "Evaluate eligible responder approach route candidates for a disaster site.",
        "site": site_state,
        "candidates": routes,
        "safety_rule": "Only eligible candidates are supplied. Hard constraints have already been applied.",
    }

    questions = {
        "route_advisory": {
            "type": "choice",
            "instructions": (
                "Which eligible route candidate appears preferable based on lower observed hazard exposure, "
                "better clearance and reasonable distance? This is advisory only."
            ),
            "criteria": route_descriptions,
        },
        "review_priority": {
            "type": "score",
            "instructions": "How strongly should this route plan require qualified human field review before operational use?",
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
                "Does the current route plan need reassessment before operational use because of uncertainty, "
                "incomplete evidence, or changing disaster conditions?"
            ),
            "criteria": {
                "false": "the current evidence does not indicate an additional reassessment need",
                "true": "uncertainty or changing conditions justify reassessment before operational use",
            },
        },
    }

    payload = {
        "state": state,
        "questions": questions,
        # Valid Laya checkpoint for typed decisions. See the official router names.
        "model": "typed-decisions",
    }
    request = Request(
        LAYA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
            "selected_route_confidence": None,
            "review_priority": None,
            "reassessment_probability": None,
            "routing": {},
            "raw": None,
        }

    answers = raw.get("answers") or {}
    advisory = answers.get("route_advisory") or {}
    choice = advisory.get("choice")
    valid_ids = {str(route.get("route_id")) for route in routes}
    if choice not in valid_ids:
        choice = None

    review = answers.get("review_priority") or {}
    reassess = answers.get("needs_reassessment") or {}

    return {
        "available": True,
        "status": "LAYA_EVALUATED",
        "selected_route": choice,
        "selected_route_confidence": advisory.get("confidence"),
        "review_priority": review.get("score"),
        "reassessment_probability": reassess.get("noul"),
        "routing": raw.get("routing") or {},
        "raw": raw,
    }
