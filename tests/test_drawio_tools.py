from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("pyautogui", types.SimpleNamespace(
    FAILSAFE=True,
    PAUSE=0,
    click=lambda *args, **kwargs: None,
    typewrite=lambda *args, **kwargs: None,
    hotkey=lambda *args, **kwargs: None,
    moveTo=lambda *args, **kwargs: None,
    mouseDown=lambda *args, **kwargs: None,
    mouseUp=lambda *args, **kwargs: None,
    screenshot=lambda *args, **kwargs: None,
))

import domains.drawio.tools as drawio_tools
import core.tools.primitives as primitives
from core.tools.registry import resolve_tool


class DrawioCompoundToolTest(unittest.TestCase):
    def test_resolve_tool_accepts_family_default_and_plain_name(self) -> None:
        graph = {
            "UI_Elements": {
                "Triangle_Tool": {"x": 188, "y": 338},
            },
            "Tool_Families": {
                "Triangle_Family": {
                    "default": "Triangle_Tool",
                    "candidates": ["Triangle_Tool"],
                }
            },
        }

        self.assertEqual(resolve_tool(graph, "Triangle_Family"), (188, 338))
        self.assertEqual(resolve_tool(graph, "Triangle"), (188, 338))
        self.assertEqual(resolve_tool(graph, "triangle"), (188, 338))

    def test_place_shape_then_edit_label_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        with patch.object(drawio_tools, "_fn_place_shape", fake("place_shape")):
            with patch.object(drawio_tools, "_fn_press_escape", fake("press_escape")):
                with patch.object(drawio_tools, "_fn_press_enter", fake("press_enter")):
                    with patch.object(drawio_tools, "_fn_select_all", fake("select_all")):
                        with patch.object(drawio_tools, "_fn_type_label", fake("type_label")):
                            with patch.object(drawio_tools, "_fn_click_empty_canvas",
                                              fake("click_empty_canvas")):
                                result = drawio_tools.place_shape_then_edit_label(
                                    {"UI_Elements": {}}, "Rectangle_Tool", "Cache",
                                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls, [
            "place_shape",
            "press_escape",
            "press_enter",
            "select_all",
            "type_label",
            "press_escape",
            "click_empty_canvas",
        ])

    def test_drag_node_to_zone_resolves_named_zone(self) -> None:
        captured = {}

        def fake_drag(ui_graph, node_ref, target_x, target_y):
            captured["args"] = (ui_graph, node_ref, target_x, target_y)
            return {"status": "ok", "tool": "drag_node"}

        with patch.object(primitives.config, "canvas_region",
                          return_value=(100, 100, 900, 700)):
            with patch.object(primitives.config, "screen_scale", return_value=2):
                with patch.object(primitives, "_fn_drag_node", fake_drag):
                    result = primitives.drag_node_to_zone(
                        {"Canvas_Nodes": [{"id": "Observed_Node_1", "x": 50, "y": 50}]},
                        "Observed_Node_1",
                        "right",
                    )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool"], "drag_node_to_zone")
        self.assertEqual(result["zone"], "right")
        self.assertEqual(captured["args"][1:], ("Observed_Node_1", 350, 200))

    def test_drag_selected_to_zone_uses_default_shape_point(self) -> None:
        calls = []
        fake_pyautogui = types.SimpleNamespace(
            moveTo=lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
            mouseDown=lambda *args, **kwargs: calls.append(("mouseDown", args, kwargs)),
            mouseUp=lambda *args, **kwargs: calls.append(("mouseUp", args, kwargs)),
        )

        with patch.object(primitives, "pyautogui", fake_pyautogui):
            with patch.object(primitives.config, "default_shape_point",
                              return_value=(100, 100)):
                with patch.object(primitives.config, "canvas_region",
                                  return_value=(100, 100, 900, 700)):
                    with patch.object(primitives.config, "screen_scale", return_value=2):
                        result = primitives.drag_selected_to_zone("right")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["from"], [100, 100])
        self.assertEqual(result["to"], [350, 200])
        self.assertEqual([c[0] for c in calls], ["moveTo", "mouseDown", "moveTo", "mouseUp"])

    def test_resize_selected_drags_default_bottom_right_handle(self) -> None:
        calls = []
        fake_pyautogui = types.SimpleNamespace(
            moveTo=lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
            mouseDown=lambda *args, **kwargs: calls.append(("mouseDown", args, kwargs)),
            mouseUp=lambda *args, **kwargs: calls.append(("mouseUp", args, kwargs)),
        )

        with patch.object(primitives, "pyautogui", fake_pyautogui):
            with patch.object(primitives.config, "default_shape_point",
                              return_value=(100, 100)):
                result = primitives.resize_selected(60, 30)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["handle_from"], [160, 140])
        self.assertEqual(result["handle_to"], [130, 115])
        self.assertEqual([c[0] for c in calls], ["moveTo", "mouseDown", "moveTo", "mouseUp"])

    def test_drag_selected_to_node_slot_uses_container_geometry(self) -> None:
        calls = []
        fake_pyautogui = types.SimpleNamespace(
            moveTo=lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
            mouseDown=lambda *args, **kwargs: calls.append(("mouseDown", args, kwargs)),
            mouseUp=lambda *args, **kwargs: calls.append(("mouseUp", args, kwargs)),
        )
        graph = {
            "Canvas_Nodes": [
                {"id": "Housing", "x": 300, "y": 300, "w": 100, "h": 200},
            ]
        }

        with patch.object(primitives, "pyautogui", fake_pyautogui):
            with patch.object(primitives.config, "default_shape_point",
                              return_value=(100, 100)):
                result = primitives.drag_selected_to_node_slot(
                    graph, "Housing", "bottom",
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["slot"], "bottom")
        self.assertEqual(result["from"], [100, 100])
        self.assertEqual(result["to"], [300, 360])
        self.assertEqual([c[0] for c in calls], ["moveTo", "mouseDown", "moveTo", "mouseUp"])

    def test_drag_node_adjacent_attaches_below_reference(self) -> None:
        captured = {}

        def fake_drag(ui_graph, node_ref, target_x, target_y):
            captured["args"] = (ui_graph, node_ref, target_x, target_y)
            return {"status": "ok", "tool": "drag_node"}

        graph = {
            "Canvas_Nodes": [
                {"id": "Trunk", "x": 300, "y": 300, "w": 40, "h": 80},
                {"id": "Foliage", "x": 200, "y": 200, "w": 100, "h": 100},
            ]
        }
        with patch.object(primitives, "_fn_drag_node", fake_drag):
            result = primitives.drag_node_adjacent(
                graph, "Trunk", "Foliage", "attached_below",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool"], "drag_node_adjacent")
        self.assertEqual(result["relation"], "attached_below")
        self.assertEqual(captured["args"][1:], ("Trunk", 200, 290))

    def test_connect_nodes_drags_from_source_edge_to_target_edge(self) -> None:
        calls = []
        fake_pyautogui = types.SimpleNamespace(
            click=lambda *args, **kwargs: calls.append(("click", args, kwargs)),
            moveTo=lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
            mouseDown=lambda *args, **kwargs: calls.append(("mouseDown", args, kwargs)),
            mouseUp=lambda *args, **kwargs: calls.append(("mouseUp", args, kwargs)),
        )
        graph = {
            "UI_Elements": {"Line_Tool": {"x": 12, "y": 34}},
            "Canvas_Nodes": [
                {"id": "Top", "x": 300, "y": 100, "w": 120, "h": 60},
                {"id": "Bottom", "x": 300, "y": 240, "w": 120, "h": 60},
            ],
        }

        with patch.object(primitives, "pyautogui", fake_pyautogui):
            result = primitives.connect_nodes(graph, "Top", "Bottom")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["connector_tool"], "Line_Tool")
        self.assertEqual(result["from"], [300, 130])
        self.assertEqual(result["to"], [300, 210])
        self.assertEqual([c[0] for c in calls], ["click", "moveTo", "mouseDown", "moveTo", "mouseUp"])

    def test_place_shape_to_zone_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        with patch.object(drawio_tools, "_fn_place_shape", fake("place_shape")):
            with patch.object(drawio_tools, "_fn_press_escape", fake("press_escape")):
                with patch.object(drawio_tools, "_fn_drag_selected_to_zone",
                                  fake("drag_selected_to_zone")):
                    with patch.object(drawio_tools, "_fn_click_empty_canvas",
                                      fake("click_empty_canvas")):
                        result = drawio_tools.place_shape_to_zone(
                            {"UI_Elements": {}}, "Triangle_Tool", "upper_right",
                        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls, [
            "place_shape", "drag_selected_to_zone", "press_escape",
            "click_empty_canvas",
        ])

    def test_place_shape_in_node_slot_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        graph = {
            "Canvas_Nodes": [
                {"id": "Observed_Node_1", "x": 300, "y": 300, "w": 120, "h": 180},
            ]
        }
        with patch.object(drawio_tools, "_fn_place_shape", fake("place_shape")):
            with patch.object(drawio_tools, "_fn_resize_selected",
                              fake("resize_selected")):
                with patch.object(drawio_tools, "_fn_drag_selected_to_node_slot",
                                  fake("drag_selected_to_node_slot")):
                    with patch.object(drawio_tools, "_fn_press_escape",
                                      fake("press_escape")):
                        with patch.object(drawio_tools, "_fn_click_empty_canvas",
                                          fake("click_empty_canvas")):
                            result = drawio_tools.place_shape_in_node_slot(
                                graph, "Ellipse_Tool", "Observed_Node_1", "top",
                            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["target_size"], [66, 32])
        self.assertEqual(calls, [
            "place_shape", "resize_selected", "drag_selected_to_node_slot",
            "press_escape", "click_empty_canvas",
        ])

    def test_move_node_to_zone_and_deselect_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        with patch.object(drawio_tools, "_fn_drag_node_to_zone",
                          fake("drag_node_to_zone")):
            with patch.object(drawio_tools, "_fn_click_empty_canvas",
                              fake("click_empty_canvas")):
                result = drawio_tools.move_node_to_zone_and_deselect(
                    {"Canvas_Nodes": []}, "Observed_Node_1", "right",
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls, ["drag_node_to_zone", "click_empty_canvas"])

    def test_move_node_adjacent_and_deselect_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        with patch.object(drawio_tools, "_fn_drag_node_adjacent",
                          fake("drag_node_adjacent")):
            with patch.object(drawio_tools, "_fn_click_empty_canvas",
                              fake("click_empty_canvas")):
                result = drawio_tools.move_node_adjacent_and_deselect(
                    {"Canvas_Nodes": []},
                    "Observed_Node_2",
                    "Observed_Node_1",
                    "overlap_below",
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls, ["drag_node_adjacent", "click_empty_canvas"])

    def test_rotate_node_90_calls_rotation_handle_drag(self) -> None:
        calls = []

        fake_pyautogui = types.SimpleNamespace(
            click=lambda *args, **kwargs: calls.append(("click", args, kwargs)),
            moveTo=lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
            mouseDown=lambda *args, **kwargs: calls.append(("mouseDown", args, kwargs)),
            mouseUp=lambda *args, **kwargs: calls.append(("mouseUp", args, kwargs)),
        )

        with patch.object(primitives, "pyautogui", fake_pyautogui):
            result = primitives.rotate_node_90(
                {"Canvas_Nodes": [{"id": "Observed_Node_1", "x": 100, "y": 100, "w": 80, "h": 40}]},
                "Observed_Node_1",
                "clockwise",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool"], "rotate_node_90")
        self.assertEqual(result["direction"], "clockwise")
        self.assertEqual([c[0] for c in calls], ["click", "moveTo", "mouseDown", "moveTo", "mouseUp"])

    def test_rotate_node_90_and_deselect_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        with patch.object(drawio_tools, "_fn_rotate_node_90",
                          fake("rotate_node_90")):
            with patch.object(drawio_tools, "_fn_click_empty_canvas",
                              fake("click_empty_canvas")):
                result = drawio_tools.rotate_node_90_and_deselect(
                    {"Canvas_Nodes": []}, "Observed_Node_1", "clockwise",
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls, ["rotate_node_90", "click_empty_canvas"])

    def test_reshape_node_drags_named_handle(self) -> None:
        calls = []
        fake_pyautogui = types.SimpleNamespace(
            click=lambda *args, **kwargs: calls.append(("click", args, kwargs)),
            moveTo=lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
            mouseDown=lambda *args, **kwargs: calls.append(("mouseDown", args, kwargs)),
            mouseUp=lambda *args, **kwargs: calls.append(("mouseUp", args, kwargs)),
        )

        with patch.object(primitives, "pyautogui", fake_pyautogui):
            result = primitives.reshape_node(
                {"Canvas_Nodes": [{"id": "Observed_Node_1", "x": 100, "y": 100, "w": 80, "h": 40}]},
                "Observed_Node_1",
                "bottom",
                0,
                60,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool"], "reshape_node")
        self.assertEqual(result["handle"], "bottom")
        self.assertEqual(result["handle_from"], [100, 120])
        self.assertEqual(result["handle_to"], [100, 180])
        self.assertEqual([c[0] for c in calls], ["click", "moveTo", "mouseDown", "moveTo", "mouseUp"])

    def test_reshape_node_and_deselect_sequence(self) -> None:
        calls = []

        def fake(name):
            def _inner(*args, **kwargs):
                calls.append(name)
                return {"status": "ok", "tool": name}
            return _inner

        with patch.object(drawio_tools, "_fn_reshape_node", fake("reshape_node")):
            with patch.object(drawio_tools, "_fn_click_empty_canvas",
                              fake("click_empty_canvas")):
                result = drawio_tools.reshape_node_and_deselect(
                    {"Canvas_Nodes": []}, "Observed_Node_1", "bottom", 0, 120,
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls, ["reshape_node", "click_empty_canvas"])


if __name__ == "__main__":
    unittest.main()
