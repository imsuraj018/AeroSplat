import tempfile
from pathlib import Path
import numpy as np

from automatic_hazards import analyze_frame
from response_engine import recommend_approach_point


def test_approach_candidate_prefers_clear_side():
    hazards = [
        {"point": [0.5, 0.5], "weight": 10, "confidence": 1.0, "hard": True, "radius": 0.3}
    ]
    result = recommend_approach_point(hazards)
    assert result["type"] == "approach_candidate"
    assert 0 <= result["point"][0] <= 1
    assert 0 <= result["point"][1] <= 1


def test_heuristic_detector_on_simple_image():
    import cv2
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fire.png"
        image = np.zeros((240, 320, 3), np.uint8)
        image[:] = (35, 35, 35)
        cv2.circle(image, (160, 120), 60, (0, 120, 255), -1)
        cv2.imwrite(str(path), image)
        result = analyze_frame(path)
        assert result["hazards"]


if __name__ == "__main__":
    test_approach_candidate_prefers_clear_side()
    test_heuristic_detector_on_simple_image()
    print("2/2 response-engine tests passed")
