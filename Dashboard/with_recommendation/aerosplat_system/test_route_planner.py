import os
import sys
import unittest

os.environ.setdefault("AEROSPLAT_WORKSPACE", os.path.join(os.path.dirname(__file__), "_test_workspace"))
sys.path.insert(0, os.path.dirname(__file__))

import server


class RoutePlannerTests(unittest.TestCase):
    def setUp(self):
        self.analysis = server.default_analysis()
        self.triage = server.make_triage(self.analysis)

    def test_approach_detours_hard_hazard(self):
        route = server.plan_route(
            {
                "mode": "approach",
                "entry_authorized": False,
                "start": [0.05, 0.50],
                "target": [0.95, 0.50],
                "frame_name": "frame.jpg",
                "hazards": [{"point": [0.50, 0.50], "kind": "unstable_wall"}],
            },
            self.analysis,
            self.triage,
            "frame.jpg",
        )
        self.assertEqual(route["status"], "APPROACH_ROUTE")
        self.assertGreater(len(route["path"]), 2)

    def test_controlled_entry_requires_authorization(self):
        route = server.plan_route(
            {
                "mode": "controlled_entry",
                "entry_authorized": False,
                "start": [0.05, 0.50],
                "target": [0.95, 0.50],
                "frame_name": "frame.jpg",
            },
            self.analysis,
            self.triage,
            "frame.jpg",
        )
        self.assertEqual(route["status"], "ENTRY_AUTHORIZATION_REQUIRED")

    def test_red_screening_blocks_entry_route(self):
        red = dict(self.triage)
        red["label"] = "RED"
        route = server.plan_route(
            {
                "mode": "controlled_entry",
                "entry_authorized": True,
                "start": [0.05, 0.50],
                "target": [0.95, 0.50],
                "frame_name": "frame.jpg",
            },
            self.analysis,
            red,
            "frame.jpg",
        )
        self.assertEqual(route["status"], "ENTRY_ROUTE_DISABLED")


if __name__ == "__main__":
    unittest.main()
