from candidate_routes import generate_candidate_routes
from safety_constraints import filter_eligible_routes


def test_generates_multiple_candidates():
    hazards = [
        {"point": [0.5, 0.5], "kind": "debris", "weight": 4, "hard": False, "radius": 0.10},
        {"point": [0.55, 0.5], "kind": "restricted", "weight": 10, "hard": True, "radius": 0.08},
    ]
    result = generate_candidate_routes([0.08, 0.50], [0.92, 0.50], hazards, k=5)
    assert result["status"] == "CANDIDATES_READY"
    assert len(result["candidates"]) >= 2
    eligible, rejected = filter_eligible_routes(result["candidates"])
    assert len(eligible) >= 1
    assert all(not r["blocked"] for r in eligible)


def test_target_hazard_is_rejected():
    hazards = [
        {"point": [0.9, 0.5], "kind": "fire", "weight": 10, "hard": True, "radius": 0.12},
    ]
    result = generate_candidate_routes([0.1, 0.5], [0.9, 0.5], hazards, k=3)
    assert result["status"] == "TARGET_IN_BLOCKED_ZONE"
    assert result["candidates"] == []


if __name__ == "__main__":
    test_generates_multiple_candidates()
    test_target_hazard_is_rejected()
    print("candidate route tests: 2/2 passed")
