from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from core.perception.canvas import (
    annotate_canvas, annotate_node_layout, layout_issues,
    observe_canvas, observe_canvas_detailed,
)


class CanvasPerceptionTest(unittest.TestCase):
    def test_empty_canvas_returns_no_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.png")
            _write_grid_canvas(path)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(nodes, [])

    def test_detailed_empty_canvas_has_debug_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.png")
            _write_grid_canvas(path)

            detail = observe_canvas_detailed(path, region=(0, 0, 500, 300))

        self.assertEqual(detail["nodes"], [])
        self.assertEqual(detail["theme"], "light")
        self.assertIn("accepted_candidates", detail)
        self.assertIn("rejected_candidates", detail)

    def test_rejected_candidate_includes_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "small_shape.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (30, 30), (60, 50), (40, 40, 40), 2)
            cv2.imwrite(path, img)

            detail = observe_canvas_detailed(path, region=(0, 0, 500, 300))

        self.assertEqual(detail["nodes"], [])
        self.assertTrue(detail["rejected_candidates"])
        self.assertTrue(detail["rejected_candidates"][0]["reasons"])

    def test_rectangle_canvas_returns_one_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rectangle.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (120, 90), (300, 170), (40, 40, 40), 2)
            cv2.imwrite(path, img)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["id"], "Observed_Node_1")
        self.assertGreater(nodes[0]["confidence"], 0)

    def test_tall_narrow_rectangle_canvas_returns_one_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tall_narrow_rectangle.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (220, 55), (300, 245), (40, 40, 40), 2)
            cv2.imwrite(path, img)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(len(nodes), 1)
        self.assertLess(nodes[0]["w"], nodes[0]["h"])

    def test_dark_empty_canvas_returns_no_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dark_empty.png")
            _write_dark_grid_canvas(path)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(nodes, [])

    def test_dark_rectangle_canvas_returns_one_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dark_rectangle.png")
            img = _write_dark_grid_canvas(path)
            cv2.rectangle(img, (120, 90), (300, 170), (235, 235, 235), 2)
            cv2.imwrite(path, img)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["id"], "Observed_Node_1")
        self.assertGreater(nodes[0]["confidence"], 0)

    def test_selected_rectangle_handles_still_returns_one_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "selected.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (120, 90), (300, 170), (40, 40, 40), 2)
            for x, y in [(120, 90), (210, 90), (300, 90), (120, 130),
                         (300, 130), (120, 170), (210, 170), (300, 170)]:
                cv2.circle(img, (x, y), 5, (230, 160, 20), -1)
            cv2.imwrite(path, img)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(len(nodes), 1)

    def test_detailed_selected_rectangle_has_accepted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "selected_detail.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (120, 90), (300, 170), (40, 40, 40), 2)
            cv2.imwrite(path, img)

            detail = observe_canvas_detailed(path, region=(0, 0, 500, 300))

        self.assertEqual(len(detail["nodes"]), 1)
        self.assertEqual(len(detail["accepted_candidates"]), 1)
        self.assertEqual(detail["accepted_candidates"][0]["id"], "Raw_Node_1")

    def test_text_below_rectangle_does_not_create_extra_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "text_below.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (120, 90), (300, 170), (40, 40, 40), 2)
            cv2.putText(img, "Cache", (175, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
            cv2.imwrite(path, img)

            nodes = observe_canvas(path, region=(0, 0, 500, 300))

        self.assertEqual(len(nodes), 1)

    def test_nested_circles_inside_rectangle_remain_separate_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "traffic_light.png")
            img = np.full((600, 800, 3), 255, dtype=np.uint8)
            cv2.rectangle(img, (300, 140), (420, 370), (40, 40, 40), 2)
            for y in (190, 255, 320):
                cv2.ellipse(img, (360, y), (34, 24), 0, 0, 360, (40, 40, 40), 2)
            cv2.imwrite(path, img)

            nodes = observe_canvas(path, region=(0, 0, 800, 600))

        self.assertEqual(len(nodes), 4)

    def test_annotate_canvas_writes_debug_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rectangle.png")
            out = os.path.join(tmp, "annotated.png")
            img = _write_grid_canvas(path)
            cv2.rectangle(img, (120, 90), (300, 170), (40, 40, 40), 2)
            cv2.imwrite(path, img)
            nodes = observe_canvas(path, region=(0, 0, 500, 300))

            detail = observe_canvas_detailed(path, region=(0, 0, 500, 300))
            result = annotate_canvas(path, nodes, out, detection=detail)

            self.assertEqual(result, out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)

    def test_layout_issues_reports_vertical_misalignment(self) -> None:
        nodes = [
            {"id": "Observed_Node_1", "x": 300, "y": 100, "w": 100, "h": 60},
            {"id": "Observed_Node_2", "x": 180, "y": 220, "w": 100, "h": 60},
            {"id": "Observed_Node_3", "x": 300, "y": 340, "w": 100, "h": 60},
        ]

        issues = layout_issues(nodes)

        self.assertTrue(any(i["type"] == "not_vertically_aligned" for i in issues))

    def test_layout_issues_reports_overlap(self) -> None:
        nodes = [
            {"id": "Observed_Node_1", "x": 200, "y": 200, "w": 120, "h": 80},
            {"id": "Observed_Node_2", "x": 230, "y": 210, "w": 120, "h": 80},
        ]

        issues = layout_issues(nodes)

        self.assertEqual(issues[0]["type"], "overlap")

    def test_annotate_node_layout_adds_relative_position(self) -> None:
        nodes = [{"id": "Observed_Node_1", "x": 250, "y": 100, "w": 100, "h": 60}]

        with unittest.mock.patch(
            "core.perception.canvas.config.canvas_region",
            return_value=(0, 0, 1000, 600),
        ):
            with unittest.mock.patch("core.perception.canvas.config.screen_scale", return_value=2):
                annotated = annotate_node_layout(nodes)

        self.assertEqual(annotated[0]["position"], "top")


def _write_grid_canvas(path: str) -> np.ndarray:
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    for x in range(0, 500, 20):
        cv2.line(img, (x, 0), (x, 300), (232, 232, 232), 1)
    for y in range(0, 300, 20):
        cv2.line(img, (0, y), (500, y), (232, 232, 232), 1)
    cv2.imwrite(path, img)
    return img


def _write_dark_grid_canvas(path: str) -> np.ndarray:
    img = np.full((300, 500, 3), 24, dtype=np.uint8)
    for x in range(0, 500, 20):
        cv2.line(img, (x, 0), (x, 300), (48, 48, 48), 1)
    for y in range(0, 300, 20):
        cv2.line(img, (0, y), (500, y), (48, 48, 48), 1)
    cv2.imwrite(path, img)
    return img


if __name__ == "__main__":
    unittest.main()
