"""
Primitives — L0 atomic tools.

Domain-agnostic mouse/keyboard primitives. Each function wraps a single
``pyautogui`` call and returns a status dict. ToolNodes are self-registered
at import time.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import pyautogui

from core import config
from core.tools.registry import (
    ToolNode, register, resolve_tool, resolve_node,
)


# ===========================================================================
# Leaf tool functions (level 0 — single atomic operations)
# ===========================================================================

def _fn_place_shape(ui_graph: Dict[str, Any], tool_name: str) -> dict:
    x, y = resolve_tool(ui_graph, tool_name)
    print(f"  [L0] place_shape('{tool_name}') → click ({x}, {y})")
    pyautogui.click(x, y)
    return {"status": "ok", "tool": "place_shape", "tool_name": tool_name,
            "x": x, "y": y}


def _fn_type_label(text: str) -> dict:
    print(f"  [L0] type_label('{text}')")
    pyautogui.typewrite(text, interval=config.type_interval())
    return {"status": "ok", "tool": "type_label", "text": text}


def _fn_press_escape() -> dict:
    print("  [L0] press_escape")
    pyautogui.hotkey("Escape")
    return {"status": "ok", "tool": "press_escape"}


def _fn_press_enter() -> dict:
    print("  [L0] press_enter")
    pyautogui.hotkey("Return")
    return {"status": "ok", "tool": "press_enter"}


def _fn_press_delete() -> dict:
    print("  [L0] press_delete")
    pyautogui.hotkey("BackSpace")
    return {"status": "ok", "tool": "press_delete"}


def _fn_select_all() -> dict:
    print("  [L0] select_all (Cmd+A)")
    pyautogui.hotkey("command", "a")
    return {"status": "ok", "tool": "select_all"}


def _fn_click_empty_canvas() -> dict:
    x, y = config.empty_canvas_point()
    print(f"  [L0] click_empty_canvas → ({x}, {y})")
    pyautogui.click(x, y)
    return {"status": "ok", "tool": "click_empty_canvas", "x": x, "y": y}


def _fn_click_node(ui_graph: Dict[str, Any], node_ref: str, clicks: int = 1) -> dict:
    node = resolve_node(ui_graph, node_ref)
    x, y = node["x"], node["y"]
    print(f"  [L0] click_node('{node_ref}', clicks={clicks}) → ({x}, {y})")
    pyautogui.click(x, y, clicks=clicks)
    return {"status": "ok", "tool": "click_node", "node_ref": node_ref,
            "x": x, "y": y}


def _fn_double_click_node(ui_graph: Dict[str, Any], node_ref: str) -> dict:
    return _fn_click_node(ui_graph, node_ref, clicks=2)


def _fn_drag_node(
    ui_graph: Dict[str, Any], node_ref: str, target_x: int, target_y: int,
) -> dict:
    node = resolve_node(ui_graph, node_ref)
    sx, sy = node["x"], node["y"]
    dur = config.drag_duration()
    print(f"  [L0] drag_node('{node_ref}') → ({sx},{sy}) → ({target_x},{target_y})")
    pyautogui.moveTo(sx, sy)
    pyautogui.mouseDown()
    pyautogui.moveTo(target_x, target_y, duration=dur)
    pyautogui.mouseUp()
    return {"status": "ok", "tool": "drag_node", "node_ref": node_ref,
            "from": [sx, sy], "to": [target_x, target_y]}


def _fn_drag_node_near(
    ui_graph: Dict[str, Any], node_ref: str, reference_node: str,
    offset_x: int = 200, offset_y: int = 0,
) -> dict:
    ref = resolve_node(ui_graph, reference_node)
    return _fn_drag_node(ui_graph, node_ref, ref["x"] + offset_x, ref["y"] + offset_y)


def _fn_drag_node_to_zone(ui_graph: Dict[str, Any], node_ref: str, zone: str) -> dict:
    target_x, target_y = _canvas_zone_point(zone)
    print(f"  [L0] drag_node_to_zone('{node_ref}', zone='{zone}')")
    result = _fn_drag_node(ui_graph, node_ref, target_x, target_y)
    result["tool"] = "drag_node_to_zone"
    result["zone"] = _normalize_zone(zone)
    return result


def _fn_drag_node_adjacent(
    ui_graph: Dict[str, Any],
    node_ref: str,
    reference_node: str,
    relation: str,
) -> dict:
    node = resolve_node(ui_graph, node_ref)
    ref = resolve_node(ui_graph, reference_node)
    target_x, target_y = _adjacent_target(node, ref, relation)
    relation_name = _normalize_relation(relation)
    print(
        f"  [L0] drag_node_adjacent('{node_ref}', reference='{reference_node}', "
        f"relation='{relation_name}')"
    )
    result = _fn_drag_node(ui_graph, node_ref, target_x, target_y)
    result["tool"] = "drag_node_adjacent"
    result["reference_node"] = reference_node
    result["relation"] = relation_name
    return result


def _fn_connect_nodes(
    ui_graph: Dict[str, Any],
    source_node: str,
    target_node: str,
) -> dict:
    source = resolve_node(ui_graph, source_node)
    target = resolve_node(ui_graph, target_node)
    connector_name = _connector_tool_name(ui_graph)
    tool_x, tool_y = resolve_tool(ui_graph, connector_name)
    start_x, start_y = _node_edge_point(source, target)
    end_x, end_y = _node_edge_point(target, source)
    print(
        f"  [L0] connect_nodes('{source_node}' → '{target_node}') "
        f"with '{connector_name}'"
    )
    pyautogui.click(tool_x, tool_y)
    time.sleep(0.15)
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=config.drag_duration())
    pyautogui.mouseUp()
    return {
        "status": "ok",
        "tool": "connect_nodes",
        "connector_tool": connector_name,
        "source_node": source_node,
        "target_node": target_node,
        "from": [start_x, start_y],
        "to": [end_x, end_y],
    }


def _fn_drag_selected_to_zone(zone: str) -> dict:
    sx, sy = config.default_shape_point()
    target_x, target_y = _canvas_zone_point(zone)
    print(f"  [L0] drag_selected_to_zone(zone='{zone}') → ({sx},{sy}) → ({target_x},{target_y})")
    pyautogui.moveTo(sx, sy)
    pyautogui.mouseDown()
    pyautogui.moveTo(target_x, target_y, duration=config.drag_duration())
    pyautogui.mouseUp()
    return {
        "status": "ok",
        "tool": "drag_selected_to_zone",
        "zone": _normalize_zone(zone),
        "from": [sx, sy],
        "to": [target_x, target_y],
    }


def _fn_resize_selected(
    new_width: int,
    new_height: int,
    default_width: int = 120,
    default_height: int = 80,
) -> dict:
    sx, sy = config.default_shape_point()
    start_x, start_y = sx + int(default_width) // 2, sy + int(default_height) // 2
    target_x, target_y = sx + int(new_width) // 2, sy + int(new_height) // 2
    print(f"  [L0] resize_selected({new_width}×{new_height})")
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(target_x, target_y, duration=config.drag_duration())
    pyautogui.mouseUp()
    return {
        "status": "ok",
        "tool": "resize_selected",
        "from_size": [int(default_width), int(default_height)],
        "new_size": [int(new_width), int(new_height)],
        "handle_from": [start_x, start_y],
        "handle_to": [target_x, target_y],
    }


def _fn_drag_selected_to_node_slot(
    ui_graph: Dict[str, Any],
    container_node: str,
    slot: str,
) -> dict:
    node = resolve_node(ui_graph, container_node)
    sx, sy = config.default_shape_point()
    target_x, target_y = _node_slot_point(node, slot)
    slot_name = _normalize_slot(slot)
    print(
        f"  [L0] drag_selected_to_node_slot(container='{container_node}', "
        f"slot='{slot_name}') → ({sx},{sy}) → ({target_x},{target_y})"
    )
    pyautogui.moveTo(sx, sy)
    pyautogui.mouseDown()
    pyautogui.moveTo(target_x, target_y, duration=config.drag_duration())
    pyautogui.mouseUp()
    return {
        "status": "ok",
        "tool": "drag_selected_to_node_slot",
        "container_node": container_node,
        "slot": slot_name,
        "from": [sx, sy],
        "to": [target_x, target_y],
    }


def _fn_rotate_node_90(
    ui_graph: Dict[str, Any], node_ref: str, direction: str = "clockwise",
) -> dict:
    node = resolve_node(ui_graph, node_ref)
    x, y = node["x"], node["y"]
    w, h = node.get("w", 120), node.get("h", 60)
    rotate_dir = _normalize_direction(direction)
    radius = max(w, h) // 2 + 28
    start_x = x + w // 2 + 16
    start_y = y - h // 2 - 16
    if rotate_dir == "counterclockwise":
        target_x, target_y = x - radius, y - radius
    else:
        target_x, target_y = x + radius, y + radius

    print(f"  [L0] rotate_node_90('{node_ref}', direction='{rotate_dir}')")
    pyautogui.click(x, y)
    time.sleep(0.2)
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(target_x, target_y, duration=config.drag_duration())
    pyautogui.mouseUp()
    return {
        "status": "ok",
        "tool": "rotate_node_90",
        "node_ref": node_ref,
        "direction": rotate_dir,
        "handle_from": [start_x, start_y],
        "handle_to": [target_x, target_y],
    }


def _fn_resize_node(
    ui_graph: Dict[str, Any], node_ref: str, new_width: int, new_height: int,
) -> dict:
    node = resolve_node(ui_graph, node_ref)
    x, y = node["x"], node["y"]
    w, h = node.get("w", 120), node.get("h", 60)
    handle_x, handle_y = x + w // 2, y + h // 2
    new_hx, new_hy = x + new_width // 2, y + new_height // 2
    print(f"  [L0] resize_node('{node_ref}', {new_width}×{new_height})")
    pyautogui.click(x, y)
    time.sleep(0.2)
    pyautogui.moveTo(handle_x, handle_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(new_hx, new_hy, duration=0.3)
    pyautogui.mouseUp()
    return {"status": "ok", "tool": "resize_node", "node_ref": node_ref,
            "new_size": [new_width, new_height]}


def _fn_reshape_node(
    ui_graph: Dict[str, Any],
    node_ref: str,
    handle: str,
    delta_x: int,
    delta_y: int,
) -> dict:
    node = resolve_node(ui_graph, node_ref)
    x, y = node["x"], node["y"]
    handle_name = _normalize_handle(handle)
    start_x, start_y = _node_handle_point(node, handle_name)
    target_x, target_y = start_x + int(delta_x), start_y + int(delta_y)
    print(
        f"  [L0] reshape_node('{node_ref}', handle='{handle_name}', "
        f"delta=({delta_x},{delta_y}))"
    )
    pyautogui.click(x, y)
    time.sleep(0.2)
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(target_x, target_y, duration=config.drag_duration())
    pyautogui.mouseUp()
    return {
        "status": "ok",
        "tool": "reshape_node",
        "node_ref": node_ref,
        "handle": handle_name,
        "handle_from": [start_x, start_y],
        "handle_to": [target_x, target_y],
        "delta": [int(delta_x), int(delta_y)],
    }


def _fn_hotkey(keys: Any, *extra: str) -> dict:
    if extra:
        parsed = [keys, *extra]
    elif isinstance(keys, str):
        parsed = [k.strip() for k in keys.replace("+", ",").split(",") if k.strip()]
    else:
        parsed = list(keys)
    combo = " + ".join(parsed)
    print(f"  [L0] hotkey({combo})")
    pyautogui.hotkey(*parsed)
    return {"status": "ok", "tool": "hotkey", "keys": parsed}


def _fn_undo() -> dict:
    print("  [L0] undo (Cmd+Z)")
    pyautogui.hotkey("command", "z")
    return {"status": "ok", "tool": "undo"}


def _canvas_zone_point(zone: str) -> Tuple[int, int]:
    region = config.canvas_region()
    if region is None:
        raise ValueError("explorer.canvas_region is required for drag_node_to_zone")

    zone_name = _normalize_zone(zone)
    zone_fractions = {
        "center": (0.50, 0.50),
        "left": (0.25, 0.50),
        "right": (0.75, 0.50),
        "top": (0.50, 0.25),
        "bottom": (0.50, 0.75),
        "upper_left": (0.25, 0.25),
        "upper_right": (0.75, 0.25),
        "lower_left": (0.25, 0.75),
        "lower_right": (0.75, 0.75),
    }
    if zone_name not in zone_fractions:
        valid = ", ".join(sorted(zone_fractions))
        raise ValueError(f"Unknown canvas zone '{zone}'. Valid zones: {valid}")

    x1, y1, x2, y2 = region
    fx, fy = zone_fractions[zone_name]
    scale = max(config.screen_scale(), 1)
    return (
        int((x1 + (x2 - x1) * fx) / scale),
        int((y1 + (y2 - y1) * fy) / scale),
    )


def _normalize_zone(zone: str) -> str:
    return str(zone).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_direction(direction: str) -> str:
    value = str(direction).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"ccw", "counter_clockwise", "counterclockwise", "left"}:
        return "counterclockwise"
    return "clockwise"


def _normalize_handle(handle: str) -> str:
    value = str(handle).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "n": "top",
        "s": "bottom",
        "w": "left",
        "e": "right",
        "nw": "top_left",
        "ne": "top_right",
        "sw": "bottom_left",
        "se": "bottom_right",
        "upper_left": "top_left",
        "upper_right": "top_right",
        "lower_left": "bottom_left",
        "lower_right": "bottom_right",
    }
    value = aliases.get(value, value)
    valid = {
        "top", "bottom", "left", "right",
        "top_left", "top_right", "bottom_left", "bottom_right",
    }
    if value not in valid:
        raise ValueError(f"Unknown resize handle '{handle}'. Valid handles: {sorted(valid)}")
    return value


def _node_handle_point(node: Dict[str, Any], handle: str) -> Tuple[int, int]:
    x, y = int(node["x"]), int(node["y"])
    w, h = int(node.get("w", 120)), int(node.get("h", 60))
    left, right = x - w // 2, x + w // 2
    top, bottom = y - h // 2, y + h // 2
    points = {
        "top": (x, top),
        "bottom": (x, bottom),
        "left": (left, y),
        "right": (right, y),
        "top_left": (left, top),
        "top_right": (right, top),
        "bottom_left": (left, bottom),
        "bottom_right": (right, bottom),
    }
    return points[handle]


def _normalize_slot(slot: str) -> str:
    value = str(slot).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "upper": "top",
        "upper_light": "top",
        "top_light": "top",
        "center": "middle",
        "centre": "middle",
        "mid": "middle",
        "middle_light": "middle",
        "lower": "bottom",
        "lower_light": "bottom",
        "bottom_light": "bottom",
    }
    value = aliases.get(value, value)
    valid = {"top", "middle", "bottom"}
    if value not in valid:
        raise ValueError(f"Unknown node slot '{slot}'. Valid slots: {sorted(valid)}")
    return value


def _node_slot_point(node: Dict[str, Any], slot: str) -> Tuple[int, int]:
    x, y = int(node["x"]), int(node["y"])
    h = int(node.get("h", 120))
    slot_name = _normalize_slot(slot)
    offsets = {
        "top": -0.30,
        "middle": 0.0,
        "bottom": 0.30,
    }
    return x, int(y + h * offsets[slot_name])


def _node_edge_point(node: Dict[str, Any], other: Dict[str, Any]) -> Tuple[int, int]:
    x, y = int(node["x"]), int(node["y"])
    w, h = int(node.get("w", 120)), int(node.get("h", 60))
    dx = int(other["x"]) - x
    dy = int(other["y"]) - y
    if abs(dy) >= abs(dx):
        return x, y + (h // 2 if dy >= 0 else -h // 2)
    return x + (w // 2 if dx >= 0 else -w // 2), y


def _connector_tool_name(ui_graph: Dict[str, Any]) -> str:
    elements = ui_graph.get("UI_Elements", {})
    families = ui_graph.get("Tool_Families", {})
    for family in ("Connector_Family", "Line_Family", "Arrow_Family"):
        default = families.get(family, {}).get("default")
        if default in elements:
            return default
    for name in ("Line_Tool", "Arrow_Tool", "Arrow_Tool_1"):
        if name in elements:
            return name
    candidates = [name for name in elements if "Line" in name or "Arrow" in name]
    if candidates:
        return sorted(candidates)[0]
    raise KeyError("No connector-like tool found. Need Line_Tool or Arrow_Tool in UI_Elements.")


def inner_slot_shape_size(node: Dict[str, Any]) -> Tuple[int, int]:
    """Return a conservative size for a small shape placed inside a container."""
    w = int(node.get("w", 120))
    h = int(node.get("h", 120))
    return (
        max(24, min(80, int(w * 0.55))),
        max(18, min(45, int(h * 0.18))),
    )


def _normalize_relation(relation: str) -> str:
    value = str(relation).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "on_top_of": "overlap_above",
        "stack_above": "overlap_above",
        "stacked_above": "overlap_above",
        "stack_below": "overlap_below",
        "stacked_below": "overlap_below",
        "under": "attached_below",
        "underneath": "attached_below",
        "trunk_below": "attached_below",
        "touching_below": "attached_below",
        "touching_above": "attached_above",
        "touching_left": "attached_left",
        "touching_right": "attached_right",
    }
    return aliases.get(value, value)


def _adjacent_target(
    node: Dict[str, Any],
    ref: Dict[str, Any],
    relation: str,
) -> Tuple[int, int]:
    relation_name = _normalize_relation(relation)
    nx, ny = int(node["x"]), int(node["y"])
    nw, nh = int(node.get("w", 120)), int(node.get("h", 60))
    rx, ry = int(ref["x"]), int(ref["y"])
    rw, rh = int(ref.get("w", 120)), int(ref.get("h", 60))
    gap = 0

    if relation_name == "attached_above":
        return rx, int(ry - rh / 2 - nh / 2 - gap)
    if relation_name == "attached_below":
        return rx, int(ry + rh / 2 + nh / 2 + gap)
    if relation_name == "attached_left":
        return int(rx - rw / 2 - nw / 2 - gap), ry
    if relation_name == "attached_right":
        return int(rx + rw / 2 + nw / 2 + gap), ry
    if relation_name == "overlap_above":
        return rx, int(ry - max(rh, nh) * 0.35)
    if relation_name == "overlap_below":
        return rx, int(ry + max(rh, nh) * 0.35)
    if relation_name == "centered_on":
        return rx, ry

    # Safe fallback: preserve current position rather than inventing a relation.
    return nx, ny


# ===========================================================================
# Leaf ToolNodes (level 0)
# ===========================================================================

N_PLACE_SHAPE = ToolNode(
    name="place_shape", fn=_fn_place_shape,
    params=["tool_name"], needs_ui_graph=True,
    description="Click a sidebar shape to place it on the canvas.",
)

N_TYPE_LABEL = ToolNode(
    name="type_label", fn=_fn_type_label,
    params=["text"], needs_ui_graph=False,
    description="Type a text label into the active shape.",
)

N_PRESS_ESCAPE = ToolNode(
    name="press_escape", fn=_fn_press_escape,
    params=[], needs_ui_graph=False,
    description="Press Escape to exit text editing or deselect.",
)

N_PRESS_ENTER = ToolNode(
    name="press_enter", fn=_fn_press_enter,
    params=[], needs_ui_graph=False,
    description="Press Enter to confirm input.",
)

N_PRESS_DELETE = ToolNode(
    name="press_delete", fn=_fn_press_delete,
    params=[], needs_ui_graph=False,
    description="Press Delete to remove the selected element.",
)

N_SELECT_ALL = ToolNode(
    name="select_all", fn=_fn_select_all,
    params=[], needs_ui_graph=False,
    description="Select all text in active field (Cmd+A).",
)

N_CLICK_EMPTY = ToolNode(
    name="click_empty_canvas", fn=_fn_click_empty_canvas,
    params=[], needs_ui_graph=False,
    description="Click empty canvas area to deselect.",
)

N_CLICK_NODE = ToolNode(
    name="click_node", fn=_fn_click_node,
    params=["node_ref", "clicks"], needs_ui_graph=True,
    description="Click on an existing canvas node.",
)

N_DOUBLE_CLICK_NODE = ToolNode(
    name="double_click_node", fn=_fn_double_click_node,
    params=["node_ref"], needs_ui_graph=True,
    description="Double-click a node to enter text-edit mode.",
)

N_DRAG_NODE = ToolNode(
    name="drag_node", fn=_fn_drag_node,
    params=["node_ref", "target_x", "target_y"], needs_ui_graph=True,
    description="Drag a node to a new position.",
)

N_DRAG_NODE_NEAR = ToolNode(
    name="drag_node_near", fn=_fn_drag_node_near,
    params=["node_ref", "reference_node", "offset_x", "offset_y"],
    needs_ui_graph=True,
    description="Move a node to a position relative to another node.",
)

N_DRAG_NODE_TO_ZONE = ToolNode(
    name="drag_node_to_zone", fn=_fn_drag_node_to_zone,
    params=["node_ref", "zone"], needs_ui_graph=True,
    description=(
        "Drag a node to a named canvas zone: center, left, right, top, "
        "bottom, upper_left, upper_right, lower_left, lower_right."
    ),
)

N_DRAG_NODE_ADJACENT = ToolNode(
    name="drag_node_adjacent", fn=_fn_drag_node_adjacent,
    params=["node_ref", "reference_node", "relation"], needs_ui_graph=True,
    description=(
        "Drag a node into a named relation with another node: attached_above, "
        "attached_below, attached_left, attached_right, overlap_above, "
        "overlap_below, centered_on."
    ),
)

N_CONNECT_NODES = ToolNode(
    name="connect_nodes", fn=_fn_connect_nodes,
    params=["source_node", "target_node"], needs_ui_graph=True,
    description=(
        "Draw a connector/arrow from one observed node to another by dragging "
        "from the source edge to the target edge. Use for flowcharts."
    ),
)

N_DRAG_SELECTED_TO_ZONE = ToolNode(
    name="drag_selected_to_zone", fn=_fn_drag_selected_to_zone,
    params=["zone"], needs_ui_graph=False,
    description=(
        "Drag the currently selected/newly inserted shape from Draw.io's "
        "default insertion point to a named canvas zone."
    ),
)

N_RESIZE_SELECTED = ToolNode(
    name="resize_selected", fn=_fn_resize_selected,
    params=["new_width", "new_height"], needs_ui_graph=False,
    description=(
        "Resize the currently selected/newly inserted shape from its default "
        "size by dragging its bottom-right handle."
    ),
)

N_DRAG_SELECTED_TO_NODE_SLOT = ToolNode(
    name="drag_selected_to_node_slot", fn=_fn_drag_selected_to_node_slot,
    params=["container_node", "slot"], needs_ui_graph=True,
    description=(
        "Drag the currently selected/newly inserted shape into a slot inside "
        "an existing container node. Valid slots: top, middle, bottom."
    ),
)

N_ROTATE_NODE_90 = ToolNode(
    name="rotate_node_90", fn=_fn_rotate_node_90,
    params=["node_ref", "direction"], needs_ui_graph=True,
    description="Rotate a canvas node by about 90 degrees using its rotation handle.",
)

N_RESIZE_NODE = ToolNode(
    name="resize_node", fn=_fn_resize_node,
    params=["node_ref", "new_width", "new_height"], needs_ui_graph=True,
    description="Resize a node by dragging its handle.",
)

N_RESHAPE_NODE = ToolNode(
    name="reshape_node", fn=_fn_reshape_node,
    params=["node_ref", "handle", "delta_x", "delta_y"], needs_ui_graph=True,
    description=(
        "Reshape a selected-style node by dragging one of its 8 blue handles: "
        "top, bottom, left, right, top_left, top_right, bottom_left, bottom_right."
    ),
)

N_HOTKEY = ToolNode(
    name="hotkey", fn=_fn_hotkey,
    params=["keys"], needs_ui_graph=False,
    description="Press a keyboard shortcut.",
)

N_UNDO = ToolNode(
    name="undo", fn=_fn_undo,
    params=[], needs_ui_graph=False,
    description="Undo last action (Cmd+Z).",
)


# ===========================================================================
# Self-register all primitives
# ===========================================================================

for _n in (
    N_PLACE_SHAPE, N_TYPE_LABEL, N_PRESS_ESCAPE, N_PRESS_ENTER,
    N_PRESS_DELETE, N_SELECT_ALL, N_CLICK_EMPTY, N_CLICK_NODE,
    N_DOUBLE_CLICK_NODE, N_DRAG_NODE, N_DRAG_NODE_NEAR,
    N_DRAG_NODE_TO_ZONE, N_DRAG_NODE_ADJACENT, N_CONNECT_NODES,
    N_DRAG_SELECTED_TO_ZONE,
    N_RESIZE_SELECTED, N_DRAG_SELECTED_TO_NODE_SLOT, N_ROTATE_NODE_90,
    N_RESIZE_NODE, N_RESHAPE_NODE, N_HOTKEY, N_UNDO,
):
    register(_n)


# ===========================================================================
# Public function aliases (for direct script use)
# ===========================================================================

place_shape = _fn_place_shape
type_label = _fn_type_label
press_escape = _fn_press_escape
press_enter = _fn_press_enter
press_delete = _fn_press_delete
select_all_text = _fn_select_all
click_empty_canvas = _fn_click_empty_canvas
click_node = _fn_click_node
double_click_node = _fn_double_click_node
drag_node = _fn_drag_node
drag_node_near = _fn_drag_node_near
drag_node_to_zone = _fn_drag_node_to_zone
drag_node_adjacent = _fn_drag_node_adjacent
connect_nodes = _fn_connect_nodes
drag_selected_to_zone = _fn_drag_selected_to_zone
resize_selected = _fn_resize_selected
drag_selected_to_node_slot = _fn_drag_selected_to_node_slot
rotate_node_90 = _fn_rotate_node_90
resize_node = _fn_resize_node
reshape_node = _fn_reshape_node
hotkey = _fn_hotkey
undo = _fn_undo
