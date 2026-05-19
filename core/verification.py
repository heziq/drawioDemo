"""
Post-action verification for closed-loop pipeline control.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from core import config


def verify_action(
    tool_name: str,
    params: Dict[str, Any],
    before_graph: Dict[str, Any],
    after_graph: Dict[str, Any],
    before_screenshot: str,
    after_screenshot: str,
    dispatch_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a lightweight verification record for one executed tool."""
    before_count = len(before_graph.get("Canvas_Nodes", []))
    after_count = len(after_graph.get("Canvas_Nodes", []))
    changed = _canvas_changed(before_screenshot, after_screenshot)

    if dispatch_result.get("status") == "error":
        return _result(False, "dispatch_error", before_count, after_count, changed)

    if tool_name in {
        "place_shape", "place_and_label", "place_shape_then_edit_label",
        "place_shape_to_zone", "place_shape_in_node_slot",
    }:
        new_tracks = after_graph.get("_canvas_tracking", {}).get("new_tracks", [])
        if after_count == before_count + 1 or new_tracks:
            return _result(True, "observed_canvas_node_count_increased",
                           before_count, after_count, changed,
                           new_tracks=new_tracks)
        if _is_connector_like_tool(params.get("tool_name")):
            if changed:
                return _result(True, "connector_canvas_changed_node_count_unchanged",
                               before_count, after_count, changed, confidence="weak")
            return _result(False, "no_canvas_change_after_connector_place",
                           before_count, after_count, changed)
        if changed:
            return _result(False, "closed_shape_place_did_not_create_new_node",
                           before_count, after_count, changed)
        return _result(False, "no_canvas_change_after_place",
                       before_count, after_count, changed)

    if tool_name == "connect_nodes":
        if changed:
            return _result(True, "connector_canvas_changed",
                           before_count, after_count, changed, confidence="weak")
        return _result(False, "no_canvas_change_after_connect_nodes",
                       before_count, after_count, changed)

    if tool_name in {"type_label", "edit_label"}:
        if changed:
            return _result(True, "canvas_changed_after_text_action",
                           before_count, after_count, changed, confidence="weak")
        return _result(False, "no_canvas_change_after_text_action",
                       before_count, after_count, changed)

    if tool_name in {"press_escape", "click_empty_canvas", "press_enter"}:
        return _result(True, "selection_state_not_strictly_verified",
                       before_count, after_count, changed, confidence="weak")

    if tool_name in {
        "drag_node", "drag_node_near", "drag_node_to_zone", "drag_node_adjacent",
        "drag_selected_to_zone", "move_and_deselect", "move_node_to_zone_and_deselect",
        "move_node_adjacent_and_deselect",
    }:
        return _verify_drag(params, before_graph, after_graph, before_count, after_count, changed)

    if tool_name in {"rotate_node_90", "rotate_node_90_and_deselect"}:
        return _verify_rotation(params, before_graph, after_graph, before_count, after_count, changed)

    if tool_name in {"resize_node", "reshape_node", "reshape_node_and_deselect"}:
        return _verify_resize(params, before_graph, after_graph, before_count, after_count, changed)

    if tool_name == "delete_node":
        target = params.get("node_ref")
        before_node = _find_node(before_graph, target)
        after_node = _find_node(after_graph, target)
        if before_node and after_node is None:
            return _result(True, "target_node_disappeared_after_delete",
                           before_count, after_count, changed,
                           target_node_id=target,
                           before_position=_position(before_node))
        if after_count < before_count:
            return _result(True, "observed_canvas_node_count_decreased",
                           before_count, after_count, changed,
                           target_node_id=target)
        return _result(False, "target_node_not_deleted",
                       before_count, after_count, changed)

    return _result(True, "no_specific_verifier_for_tool",
                   before_count, after_count, changed, confidence="weak")


def _result(
    passed: bool,
    reason: str,
    before_count: int,
    after_count: int,
    changed: bool,
    *,
    confidence: str = "strong",
    **extra: Any,
) -> Dict[str, Any]:
    result = {
        "passed": passed,
        "confidence": confidence,
        "reason": reason,
        "before_node_count": before_count,
        "after_node_count": after_count,
        "canvas_changed": changed,
        "text_placement": "unknown",
    }
    result.update(extra)
    return result


def _verify_drag(
    params: Dict[str, Any],
    before_graph: Dict[str, Any],
    after_graph: Dict[str, Any],
    before_count: int,
    after_count: int,
    changed: bool,
) -> Dict[str, Any]:
    target = params.get("node_ref")
    before_node = _find_node(before_graph, target)
    after_node = _find_node(after_graph, target)
    if not before_node or not after_node:
        merged = _adjacent_merge_candidate(params, before_node, before_graph, after_graph)
        if merged:
            return _result(
                True,
                "target_node_merged_with_reference_after_adjacent_move",
                before_count,
                after_count,
                changed,
                confidence="weak",
                **merged,
            )
        fallback = _drag_reidentified_candidate(params, before_node, after_graph)
        if fallback:
            return _result(
                True,
                "target_node_reidentified_after_drag",
                before_count,
                after_count,
                changed,
                confidence="weak",
                **fallback,
            )
        return _result(False, "target_node_not_tracked_across_drag",
                       before_count, after_count, changed,
                       target_node_id=target)

    dx = after_node["x"] - before_node["x"]
    dy = after_node["y"] - before_node["y"]
    zone = params.get("zone")
    expected = _expected_direction(zone) or _expected_relation_direction(params.get("relation"))
    direction_ok = _direction_matches(dx, dy, expected)
    movement_delta = {"dx": dx, "dy": dy}
    common = {
        "target_node_id": target,
        "before_position": _position(before_node),
        "after_position": _position(after_node),
        "expected_direction": expected,
        "movement_delta": movement_delta,
    }
    if expected and direction_ok:
        return _result(True, "target_node_moved_expected_direction",
                       before_count, after_count, changed, **common)
    if not expected and (abs(dx) >= 8 or abs(dy) >= 8):
        return _result(True, "target_node_moved",
                       before_count, after_count, changed, **common)
    if changed:
        return _result(True, "canvas_changed_but_target_motion_not_confirmed",
                       before_count, after_count, changed,
                       confidence="weak", **common)
    return _result(False, "target_node_did_not_move_after_drag",
                   before_count, after_count, changed, **common)


def _adjacent_merge_candidate(
    params: Dict[str, Any],
    before_node: Dict[str, Any] | None,
    before_graph: Dict[str, Any],
    after_graph: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not before_node or not params.get("relation"):
        return None
    reference = _find_node(before_graph, params.get("reference_node"))
    if not reference:
        return None
    if len(after_graph.get("Canvas_Nodes", [])) >= len(before_graph.get("Canvas_Nodes", [])):
        return None

    union = _union_position(before_node, reference)
    candidates = [
        node for node in after_graph.get("Canvas_Nodes", [])
        if _overlaps_position(node, union, min_iou=0.35)
    ]
    if not candidates:
        return None
    merged = candidates[0]
    return {
        "target_node_id": params.get("node_ref"),
        "reference_node_id": params.get("reference_node"),
        "merged_node_id": merged.get("id"),
        "relation": params.get("relation"),
        "before_position": _position(before_node),
        "reference_position": _position(reference),
        "after_position": _position(merged),
    }


def _verify_rotation(
    params: Dict[str, Any],
    before_graph: Dict[str, Any],
    after_graph: Dict[str, Any],
    before_count: int,
    after_count: int,
    changed: bool,
) -> Dict[str, Any]:
    target = params.get("node_ref")
    before_node = _find_node(before_graph, target)
    after_node = _find_node(after_graph, target)
    if not before_node or not after_node:
        return _result(False, "target_node_not_tracked_across_rotation",
                       before_count, after_count, changed,
                       target_node_id=target)

    before_position = _position(before_node)
    after_position = _position(after_node)
    before_ratio = _aspect_ratio(before_node)
    after_ratio = _aspect_ratio(after_node)
    swapped = _size_swapped(before_node, after_node)
    common = {
        "target_node_id": target,
        "before_position": before_position,
        "after_position": after_position,
        "before_aspect_ratio": round(before_ratio, 3),
        "after_aspect_ratio": round(after_ratio, 3),
    }
    if swapped and changed:
        return _result(True, "target_node_rotated_size_swapped",
                       before_count, after_count, changed, **common)
    if changed and abs(before_ratio - after_ratio) >= 0.25:
        return _result(True, "target_node_rotation_geometry_changed",
                       before_count, after_count, changed, **common)
    if changed:
        return _result(True, "canvas_changed_but_rotation_not_confirmed",
                       before_count, after_count, changed,
                       confidence="weak", **common)
    return _result(False, "target_node_did_not_rotate",
                   before_count, after_count, changed, **common)


def _verify_resize(
    params: Dict[str, Any],
    before_graph: Dict[str, Any],
    after_graph: Dict[str, Any],
    before_count: int,
    after_count: int,
    changed: bool,
) -> Dict[str, Any]:
    target = params.get("node_ref")
    before_node = _find_node(before_graph, target)
    after_node = _find_node(after_graph, target)
    if not before_node or not after_node:
        return _result(False, "target_node_not_tracked_across_resize",
                       before_count, after_count, changed,
                       target_node_id=target)

    old_size = {"w": before_node.get("w"), "h": before_node.get("h")}
    new_size = {"w": after_node.get("w"), "h": after_node.get("h")}
    requested = {
        "w": params.get("new_width"),
        "h": params.get("new_height"),
    }
    dw = abs((after_node.get("w") or 0) - (before_node.get("w") or 0))
    dh = abs((after_node.get("h") or 0) - (before_node.get("h") or 0))
    common = {
        "target_node_id": target,
        "before_size": old_size,
        "after_size": new_size,
        "requested_size": requested,
    }
    if changed and (dw >= 8 or dh >= 8):
        return _result(True, "target_node_resized",
                       before_count, after_count, changed, **common)
    if changed:
        return _result(True, "canvas_changed_but_resize_not_confirmed",
                       before_count, after_count, changed,
                       confidence="weak", **common)
    return _result(False, "target_node_did_not_resize",
                   before_count, after_count, changed, **common)


def _find_node(graph: Dict[str, Any], ref: str | None) -> Dict[str, Any] | None:
    if not ref:
        return None
    for node in graph.get("Canvas_Nodes", []):
        if node.get("id") == ref or node.get("text") == ref:
            return node
    return None


def _is_connector_like_tool(tool_name: str | None) -> bool:
    if not tool_name:
        return False
    lowered = tool_name.lower()
    return "arrow" in lowered or "line" in lowered or "connector" in lowered


def _drag_reidentified_candidate(
    params: Dict[str, Any],
    before_node: Dict[str, Any] | None,
    after_graph: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not before_node:
        return None
    candidates = [
        node for node in after_graph.get("Canvas_Nodes", [])
        if _same_size(before_node, node)
    ]
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    dx = candidate["x"] - before_node["x"]
    dy = candidate["y"] - before_node["y"]
    expected = _expected_direction(params.get("zone")) or _expected_relation_direction(
        params.get("relation")
    )
    if expected and not _direction_matches(dx, dy, expected):
        return None
    if not expected and abs(dx) < 8 and abs(dy) < 8:
        return None
    return {
        "target_node_id": params.get("node_ref"),
        "reidentified_node_id": candidate.get("id"),
        "before_position": _position(before_node),
        "after_position": _position(candidate),
        "expected_direction": expected,
        "movement_delta": {"dx": dx, "dy": dy},
    }


def _same_size(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    aw, ah = max(a.get("w", 1), 1), max(a.get("h", 1), 1)
    bw, bh = max(b.get("w", 1), 1), max(b.get("h", 1), 1)
    return abs(aw - bw) / max(aw, bw) <= 0.15 and abs(ah - bh) / max(ah, bh) <= 0.15


def _position(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "x": node.get("x"),
        "y": node.get("y"),
        "w": node.get("w"),
        "h": node.get("h"),
    }


def _union_position(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ax1, ay1, ax2, ay2 = _bounds(a)
    bx1, by1, bx2, by2 = _bounds(b)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    return {
        "x": int((x1 + x2) / 2),
        "y": int((y1 + y2) / 2),
        "w": int(x2 - x1),
        "h": int(y2 - y1),
    }


def _overlaps_position(node: Dict[str, Any], pos: Dict[str, Any], min_iou: float) -> bool:
    ax1, ay1, ax2, ay2 = _bounds(node)
    bx1, by1, bx2, by2 = _bounds(pos)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    area_b = max((bx2 - bx1) * (by2 - by1), 1)
    return inter / (area_a + area_b - inter) >= min_iou


def _bounds(node: Dict[str, Any]) -> tuple[float, float, float, float]:
    x, y = float(node.get("x", 0)), float(node.get("y", 0))
    w, h = float(node.get("w", 1)), float(node.get("h", 1))
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def _aspect_ratio(node: Dict[str, Any]) -> float:
    return node.get("w", 1) / max(node.get("h", 1), 1)


def _size_swapped(before_node: Dict[str, Any], after_node: Dict[str, Any]) -> bool:
    bw, bh = max(before_node.get("w", 1), 1), max(before_node.get("h", 1), 1)
    aw, ah = max(after_node.get("w", 1), 1), max(after_node.get("h", 1), 1)
    width_matches_old_height = abs(aw - bh) / max(aw, bh) <= 0.25
    height_matches_old_width = abs(ah - bw) / max(ah, bw) <= 0.25
    return width_matches_old_height and height_matches_old_width


def _expected_direction(zone: str | None) -> str | None:
    if not zone:
        return None
    zone_name = str(zone).lower().replace("-", "_").replace(" ", "_")
    if "right" in zone_name:
        return "right"
    if "left" in zone_name:
        return "left"
    if "top" in zone_name or "upper" in zone_name:
        return "up"
    if "bottom" in zone_name or "lower" in zone_name:
        return "down"
    return None


def _expected_relation_direction(relation: str | None) -> str | None:
    if not relation:
        return None
    relation_name = str(relation).lower().replace("-", "_").replace(" ", "_")
    if "below" in relation_name:
        return "down"
    if "above" in relation_name:
        return "up"
    if "right" in relation_name:
        return "right"
    if "left" in relation_name:
        return "left"
    return None


def _direction_matches(dx: int, dy: int, expected: str | None) -> bool:
    threshold = 8
    if expected == "right":
        return dx >= threshold
    if expected == "left":
        return dx <= -threshold
    if expected == "up":
        return dy <= -threshold
    if expected == "down":
        return dy >= threshold
    return False


def _canvas_changed(before_path: str, after_path: str) -> bool:
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    if before is None or after is None:
        return False

    before_crop = _canvas_crop(before)
    after_crop = _canvas_crop(after)
    if before_crop.shape != after_crop.shape or before_crop.size == 0:
        return False

    diff = cv2.absdiff(before_crop, after_crop)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    changed_pixels = int(np.count_nonzero(gray > 18))
    return changed_pixels > 50


def _canvas_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    region = config.canvas_region()
    if region is None:
        sidebar = config.sidebar_region()
        region = (sidebar[2], 0, w, h)

    x1, y1, x2, y2 = region
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return img[y1:y2, x1:x2]
