from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.verification import verify_action


class VerificationTest(unittest.TestCase):
    def test_place_passes_when_tracked_count_increases(self) -> None:
        result = _verify(
            "place_shape",
            {},
            _graph([]),
            _graph([_node("Observed_Node_1", 100, 100)], new_tracks=["Observed_Node_1"]),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "observed_canvas_node_count_increased")
        self.assertEqual(result["new_tracks"], ["Observed_Node_1"])

    def test_closed_shape_place_fails_when_count_does_not_increase(self) -> None:
        result = _verify(
            "place_shape_to_zone",
            {"tool_name": "Rectangle_Tool"},
            _graph([_node("Observed_Node_1", 100, 100)]),
            _graph([_node("Observed_Node_1", 100, 100)]),
            changed=True,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "closed_shape_place_did_not_create_new_node")

    def test_connector_place_allows_node_count_to_stay_same(self) -> None:
        result = _verify(
            "place_shape_to_zone",
            {"tool_name": "Arrow_Tool"},
            _graph([_node("Observed_Node_1", 100, 100)]),
            _graph([_node("Observed_Node_1", 100, 100)]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "connector_canvas_changed_node_count_unchanged")

    def test_connect_nodes_verifies_canvas_change_without_new_node(self) -> None:
        result = _verify(
            "connect_nodes",
            {"source_node": "Observed_Node_1", "target_node": "Observed_Node_2"},
            _graph([
                _node("Observed_Node_1", 100, 100),
                _node("Observed_Node_2", 100, 220),
            ]),
            _graph([
                _node("Observed_Node_1", 100, 100),
                _node("Observed_Node_2", 100, 220),
            ]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["confidence"], "weak")
        self.assertEqual(result["reason"], "connector_canvas_changed")

    def test_drag_right_requires_same_node_moved_right(self) -> None:
        result = _verify(
            "move_node_to_zone_and_deselect",
            {"node_ref": "Observed_Node_1", "zone": "right"},
            _graph([_node("Observed_Node_1", 100, 100)]),
            _graph([_node("Observed_Node_1", 180, 100)]),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "target_node_moved_expected_direction")
        self.assertEqual(result["movement_delta"], {"dx": 80, "dy": 0})

    def test_adjacent_relation_expects_matching_direction(self) -> None:
        result = _verify(
            "move_node_adjacent_and_deselect",
            {
                "node_ref": "Observed_Node_2",
                "reference_node": "Observed_Node_1",
                "relation": "attached_below",
            },
            _graph([
                _node("Observed_Node_1", 100, 100),
                _node("Observed_Node_2", 200, 100),
            ]),
            _graph([
                _node("Observed_Node_1", 100, 100),
                _node("Observed_Node_2", 100, 190),
            ]),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "target_node_moved_expected_direction")
        self.assertEqual(result["expected_direction"], "down")

    def test_adjacent_relation_allows_target_to_merge_with_reference(self) -> None:
        result = _verify(
            "move_node_adjacent_and_deselect",
            {
                "node_ref": "Observed_Node_2",
                "reference_node": "Observed_Node_1",
                "relation": "attached_below",
            },
            _graph([
                _node("Observed_Node_1", 100, 100, w=80, h=80),
                _node("Observed_Node_2", 180, 180, w=40, h=60),
            ]),
            _graph([
                _node("Observed_Node_1", 115, 130, w=90, h=140),
            ]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["reason"],
            "target_node_merged_with_reference_after_adjacent_move",
        )
        self.assertEqual(result["confidence"], "weak")

    def test_delete_passes_when_target_disappears(self) -> None:
        result = _verify(
            "delete_node",
            {"node_ref": "Observed_Node_1"},
            _graph([_node("Observed_Node_1", 100, 100)]),
            _graph([]),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "target_node_disappeared_after_delete")

    def test_drag_reidentifies_same_sized_node_in_expected_direction(self) -> None:
        result = _verify(
            "move_node_to_zone_and_deselect",
            {"node_ref": "Observed_Node_1", "zone": "lower_right"},
            _graph([_node("Observed_Node_1", 100, 100)]),
            _graph([_node("Observed_Node_2", 240, 220)]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["confidence"], "weak")
        self.assertEqual(result["reason"], "target_node_reidentified_after_drag")
        self.assertEqual(result["reidentified_node_id"], "Observed_Node_2")

    def test_text_action_remains_weak(self) -> None:
        result = _verify(
            "type_label",
            {"text": "Cache"},
            _graph([_node("Observed_Node_1", 100, 100)]),
            _graph([_node("Observed_Node_1", 100, 100)]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["confidence"], "weak")

    def test_rotation_passes_when_size_swaps(self) -> None:
        result = _verify(
            "rotate_node_90_and_deselect",
            {"node_ref": "Observed_Node_1", "direction": "clockwise"},
            _graph([_node("Observed_Node_1", 100, 100, w=80, h=40)]),
            _graph([_node("Observed_Node_1", 100, 100, w=40, h=80)]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "target_node_rotated_size_swapped")

    def test_rotation_does_not_pass_when_canvas_unchanged(self) -> None:
        result = _verify(
            "rotate_node_90_and_deselect",
            {"node_ref": "Observed_Node_1", "direction": "clockwise"},
            _graph([_node("Observed_Node_1", 100, 100, w=121, h=145)]),
            _graph([_node("Observed_Node_1", 100, 100, w=121, h=145)]),
            changed=False,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "target_node_did_not_rotate")

    def test_resize_requires_target_size_change(self) -> None:
        result = _verify(
            "resize_node",
            {"node_ref": "Observed_Node_1", "new_width": 60, "new_height": 150},
            _graph([_node("Observed_Node_1", 100, 100, w=120, h=60)]),
            _graph([_node("Observed_Node_1", 100, 100, w=60, h=150)]),
            changed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "target_node_resized")

    def test_resize_fails_when_canvas_unchanged(self) -> None:
        result = _verify(
            "resize_node",
            {"node_ref": "Observed_Node_1", "new_width": 60, "new_height": 150},
            _graph([_node("Observed_Node_1", 100, 100, w=120, h=60)]),
            _graph([_node("Observed_Node_1", 100, 100, w=120, h=60)]),
            changed=False,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "target_node_did_not_resize")


def _verify(tool, params, before_graph, after_graph, changed=False):
    with tempfile.TemporaryDirectory() as tmp:
        before = os.path.join(tmp, "before.png")
        after = os.path.join(tmp, "after.png")
        _write_canvas(before)
        _write_canvas(after, changed=changed)
        with patch("core.verification.config.canvas_region", return_value=(0, 0, 100, 100)):
            return verify_action(
                tool,
                params,
                before_graph,
                after_graph,
                before,
                after,
                {"status": "ok"},
            )


def _graph(nodes, new_tracks=None):
    return {
        "Canvas_Nodes": nodes,
        "_canvas_tracking": {"new_tracks": new_tracks or []},
    }


def _node(node_id, x, y, w=120, h=60):
    return {"id": node_id, "x": x, "y": y, "w": w, "h": h, "text": ""}


def _write_canvas(path, changed=False):
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    if changed:
        cv2.rectangle(img, (10, 10), (40, 40), (0, 0, 0), 2)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    unittest.main()
