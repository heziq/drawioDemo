"""
drawio compound tools (level 1+).

Domain-specific tool compositions for draw.io. Composes generic
primitives from ``core.tools.primitives`` into multi-step actions
matching draw.io's interaction model.

## draw.io interaction model

    Click sidebar shape → shape placed at default position (text cursor active)
    Type label → text goes into the shape
    Escape → exit text editing (shape still selected)
    Drag shape → move to desired position
    Drag handle → resize
    Click empty → deselect
"""

from __future__ import annotations

import time
from typing import Any, Dict

from core.tools.registry import ToolNode, register
from core.tools.primitives import (
    _fn_place_shape, _fn_type_label, _fn_press_escape, _fn_click_empty_canvas,
    _fn_double_click_node, _fn_select_all, _fn_click_node, _fn_press_delete,
    _fn_drag_node, _fn_drag_node_to_zone, _fn_drag_node_adjacent,
    _fn_drag_selected_to_zone, _fn_resize_selected,
    _fn_drag_selected_to_node_slot, inner_slot_shape_size,
    _fn_rotate_node_90, _fn_reshape_node, _fn_press_enter,
    N_PLACE_SHAPE, N_TYPE_LABEL, N_PRESS_ESCAPE, N_CLICK_EMPTY,
    N_DOUBLE_CLICK_NODE, N_SELECT_ALL, N_CLICK_NODE, N_PRESS_DELETE,
    N_DRAG_NODE, N_DRAG_NODE_TO_ZONE, N_DRAG_NODE_ADJACENT,
    N_DRAG_SELECTED_TO_ZONE, N_RESIZE_SELECTED, N_DRAG_SELECTED_TO_NODE_SLOT,
    N_ROTATE_NODE_90, N_RESHAPE_NODE, N_PRESS_ENTER,
)
from core.tools.registry import resolve_node


# Configurable pause between compound sub-steps
_STEP_PAUSE = 0.3


# ===========================================================================
# Compound tool functions (auto-level from children)
# ===========================================================================

def _fn_place_and_label(
    ui_graph: Dict[str, Any], tool_name: str, label: str,
) -> dict:
    """Place a shape, label it, then deselect."""
    steps = []
    print(f"\n  [L{N_PLACE_AND_LABEL.level}] place_and_label('{tool_name}', '{label}')")
    steps.append(_fn_place_shape(ui_graph, tool_name))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_type_label(label))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_escape())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {"status": "ok" if ok else "partial", "tool": "place_and_label",
            "steps": steps}


def _fn_place_shape_then_edit_label(
    ui_graph: Dict[str, Any], tool_name: str, label: str,
) -> dict:
    """Place a shape, enter edit mode explicitly, label it, then deselect."""
    steps = []
    print(f"\n  [L{N_PLACE_SHAPE_THEN_EDIT_LABEL.level}] "
          f"place_shape_then_edit_label('{tool_name}', '{label}')")
    steps.append(_fn_place_shape(ui_graph, tool_name))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_escape())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_enter())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_select_all())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_type_label(label))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_escape())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {"status": "ok" if ok else "partial",
            "tool": "place_shape_then_edit_label", "steps": steps}


def _fn_place_shape_to_zone(
    ui_graph: Dict[str, Any], tool_name: str, zone: str,
) -> dict:
    """Place a shape and immediately move it away from the default insertion point."""
    steps = []
    print(f"\n  [L{N_PLACE_SHAPE_TO_ZONE.level}] "
          f"place_shape_to_zone('{tool_name}', '{zone}')")
    steps.append(_fn_place_shape(ui_graph, tool_name))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_drag_selected_to_zone(zone))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_escape())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {"status": "ok" if ok else "partial",
            "tool": "place_shape_to_zone", "steps": steps}


def _fn_place_shape_in_node_slot(
    ui_graph: Dict[str, Any],
    tool_name: str,
    container_node: str,
    slot: str,
) -> dict:
    """Place a small shape inside a named slot of an existing container node."""
    steps = []
    container = resolve_node(ui_graph, container_node)
    target_width, target_height = inner_slot_shape_size(container)
    print(f"\n  [L{N_PLACE_SHAPE_IN_NODE_SLOT.level}] "
          f"place_shape_in_node_slot('{tool_name}', '{container_node}', '{slot}')")
    steps.append(_fn_place_shape(ui_graph, tool_name))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_resize_selected(target_width, target_height))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_drag_selected_to_node_slot(ui_graph, container_node, slot))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_escape())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {
        "status": "ok" if ok else "partial",
        "tool": "place_shape_in_node_slot",
        "container_node": container_node,
        "slot": slot,
        "target_size": [target_width, target_height],
        "steps": steps,
    }


def _fn_edit_label(
    ui_graph: Dict[str, Any], node_ref: str, new_label: str,
) -> dict:
    """Re-label an existing canvas node."""
    steps = []
    print(f"\n  [L{N_EDIT_LABEL.level}] edit_label('{node_ref}', '{new_label}')")
    steps.append(_fn_double_click_node(ui_graph, node_ref))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_select_all())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_type_label(new_label))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_escape())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {"status": "ok" if ok else "partial", "tool": "edit_label",
            "steps": steps}


def _fn_delete_node(ui_graph: Dict[str, Any], node_ref: str) -> dict:
    """Select and delete a canvas node."""
    steps = []
    print(f"\n  [L{N_DELETE_NODE.level}] delete_node('{node_ref}')")
    steps.append(_fn_click_node(ui_graph, node_ref))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_press_delete())
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {"status": "ok" if ok else "partial", "tool": "delete_node",
            "steps": steps}


def _fn_move_and_deselect(
    ui_graph: Dict[str, Any], node_ref: str, target_x: int, target_y: int,
) -> dict:
    """Drag a node to a position and deselect."""
    steps = []
    print(f"\n  [L{N_MOVE_AND_DESELECT.level}] move_and_deselect('{node_ref}')")
    steps.append(_fn_drag_node(ui_graph, node_ref, target_x, target_y))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {"status": "ok" if ok else "partial", "tool": "move_and_deselect",
            "steps": steps}


def _fn_move_node_to_zone_and_deselect(
    ui_graph: Dict[str, Any], node_ref: str, zone: str,
) -> dict:
    """Drag a node to a named canvas zone and deselect."""
    steps = []
    print(f"\n  [L{N_MOVE_NODE_TO_ZONE_AND_DESELECT.level}] "
          f"move_node_to_zone_and_deselect('{node_ref}', '{zone}')")
    steps.append(_fn_drag_node_to_zone(ui_graph, node_ref, zone))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {
        "status": "ok" if ok else "partial",
        "tool": "move_node_to_zone_and_deselect",
        "steps": steps,
    }


def _fn_move_node_adjacent_and_deselect(
    ui_graph: Dict[str, Any],
    node_ref: str,
    reference_node: str,
    relation: str,
) -> dict:
    """Drag a node into a named relation with another node and deselect."""
    steps = []
    print(f"\n  [L{N_MOVE_NODE_ADJACENT_AND_DESELECT.level}] "
          f"move_node_adjacent_and_deselect('{node_ref}', '{reference_node}', '{relation}')")
    steps.append(_fn_drag_node_adjacent(ui_graph, node_ref, reference_node, relation))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {
        "status": "ok" if ok else "partial",
        "tool": "move_node_adjacent_and_deselect",
        "steps": steps,
    }


def _fn_rotate_node_90_and_deselect(
    ui_graph: Dict[str, Any], node_ref: str, direction: str = "clockwise",
) -> dict:
    """Rotate a node 90 degrees and deselect."""
    steps = []
    print(f"\n  [L{N_ROTATE_NODE_90_AND_DESELECT.level}] "
          f"rotate_node_90_and_deselect('{node_ref}', '{direction}')")
    steps.append(_fn_rotate_node_90(ui_graph, node_ref, direction))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {
        "status": "ok" if ok else "partial",
        "tool": "rotate_node_90_and_deselect",
        "steps": steps,
    }


def _fn_reshape_node_and_deselect(
    ui_graph: Dict[str, Any],
    node_ref: str,
    handle: str,
    delta_x: int,
    delta_y: int,
) -> dict:
    """Reshape a node by dragging a named blue handle and deselect."""
    steps = []
    print(f"\n  [L{N_RESHAPE_NODE_AND_DESELECT.level}] "
          f"reshape_node_and_deselect('{node_ref}', '{handle}', {delta_x}, {delta_y})")
    steps.append(_fn_reshape_node(ui_graph, node_ref, handle, delta_x, delta_y))
    time.sleep(_STEP_PAUSE)
    steps.append(_fn_click_empty_canvas())
    ok = all(s.get("status") == "ok" for s in steps)
    return {
        "status": "ok" if ok else "partial",
        "tool": "reshape_node_and_deselect",
        "steps": steps,
    }


# ===========================================================================
# Compound ToolNodes (level auto-derived from children)
# ===========================================================================

N_PLACE_AND_LABEL = ToolNode(
    name="place_and_label", fn=_fn_place_and_label,
    params=["tool_name", "label"], needs_ui_graph=True,
    description="Place a shape, label it, then deselect.",
    children=[N_PLACE_SHAPE, N_TYPE_LABEL, N_PRESS_ESCAPE, N_CLICK_EMPTY],
)

N_PLACE_SHAPE_THEN_EDIT_LABEL = ToolNode(
    name="place_shape_then_edit_label", fn=_fn_place_shape_then_edit_label,
    params=["tool_name", "label"], needs_ui_graph=True,
    description=(
        "Place a shape, explicitly enter label edit mode, type the label, "
        "then deselect."
    ),
    children=[
        N_PLACE_SHAPE, N_PRESS_ESCAPE, N_PRESS_ENTER, N_SELECT_ALL,
        N_TYPE_LABEL, N_PRESS_ESCAPE, N_CLICK_EMPTY,
    ],
)

N_PLACE_SHAPE_TO_ZONE = ToolNode(
    name="place_shape_to_zone", fn=_fn_place_shape_to_zone,
    params=["tool_name", "zone"], needs_ui_graph=True,
    description=(
        "Place a shape, drag the newly selected shape to a named canvas zone, "
        "then deselect. Use for multi-shape diagrams to avoid overlap."
    ),
    children=[N_PLACE_SHAPE, N_DRAG_SELECTED_TO_ZONE, N_PRESS_ESCAPE, N_CLICK_EMPTY],
)

N_PLACE_SHAPE_IN_NODE_SLOT = ToolNode(
    name="place_shape_in_node_slot",
    fn=_fn_place_shape_in_node_slot,
    params=["tool_name", "container_node", "slot"], needs_ui_graph=True,
    description=(
        "Place a small shape inside an existing container node at slot top, "
        "middle, or bottom. Use for nested diagrams such as traffic lights."
    ),
    children=[
        N_PLACE_SHAPE, N_RESIZE_SELECTED, N_DRAG_SELECTED_TO_NODE_SLOT,
        N_PRESS_ESCAPE, N_CLICK_EMPTY,
    ],
)

N_EDIT_LABEL = ToolNode(
    name="edit_label", fn=_fn_edit_label,
    params=["node_ref", "new_label"], needs_ui_graph=True,
    description="Re-label an existing canvas node.",
    children=[N_DOUBLE_CLICK_NODE, N_SELECT_ALL, N_TYPE_LABEL,
              N_PRESS_ESCAPE, N_CLICK_EMPTY],
)

N_DELETE_NODE = ToolNode(
    name="delete_node", fn=_fn_delete_node,
    params=["node_ref"], needs_ui_graph=True,
    description="Select and delete a canvas node.",
    children=[N_CLICK_NODE, N_PRESS_DELETE, N_CLICK_EMPTY],
)

N_MOVE_AND_DESELECT = ToolNode(
    name="move_and_deselect", fn=_fn_move_and_deselect,
    params=["node_ref", "target_x", "target_y"], needs_ui_graph=True,
    description="Drag a node and deselect.",
    children=[N_DRAG_NODE, N_CLICK_EMPTY],
)

N_MOVE_NODE_TO_ZONE_AND_DESELECT = ToolNode(
    name="move_node_to_zone_and_deselect",
    fn=_fn_move_node_to_zone_and_deselect,
    params=["node_ref", "zone"], needs_ui_graph=True,
    description="Drag a node to a named canvas zone and deselect.",
    children=[N_DRAG_NODE_TO_ZONE, N_CLICK_EMPTY],
)

N_MOVE_NODE_ADJACENT_AND_DESELECT = ToolNode(
    name="move_node_adjacent_and_deselect",
    fn=_fn_move_node_adjacent_and_deselect,
    params=["node_ref", "reference_node", "relation"], needs_ui_graph=True,
    description=(
        "Drag a node into a named relation with another node and deselect. "
        "Use for connected assemblies like stacked tree foliage or a trunk "
        "attached below foliage."
    ),
    children=[N_DRAG_NODE_ADJACENT, N_CLICK_EMPTY],
)

N_ROTATE_NODE_90_AND_DESELECT = ToolNode(
    name="rotate_node_90_and_deselect",
    fn=_fn_rotate_node_90_and_deselect,
    params=["node_ref", "direction"], needs_ui_graph=True,
    description="Rotate a node about 90 degrees using its rotation handle and deselect.",
    children=[N_ROTATE_NODE_90, N_CLICK_EMPTY],
)

N_RESHAPE_NODE_AND_DESELECT = ToolNode(
    name="reshape_node_and_deselect",
    fn=_fn_reshape_node_and_deselect,
    params=["node_ref", "handle", "delta_x", "delta_y"], needs_ui_graph=True,
    description=(
        "Reshape a node by dragging one of its named blue selection handles "
        "and deselect. Use for making shapes taller, wider, narrower, or shorter."
    ),
    children=[N_RESHAPE_NODE, N_CLICK_EMPTY],
)


# ===========================================================================
# Self-register compound tools with the framework registry
# ===========================================================================

for _n in (
    N_PLACE_AND_LABEL, N_PLACE_SHAPE_THEN_EDIT_LABEL, N_PLACE_SHAPE_TO_ZONE,
    N_PLACE_SHAPE_IN_NODE_SLOT, N_EDIT_LABEL, N_DELETE_NODE, N_MOVE_AND_DESELECT,
    N_MOVE_NODE_TO_ZONE_AND_DESELECT, N_MOVE_NODE_ADJACENT_AND_DESELECT,
    N_ROTATE_NODE_90_AND_DESELECT, N_RESHAPE_NODE_AND_DESELECT,
):
    register(_n)


# ===========================================================================
# Public function aliases (for direct script use)
# ===========================================================================

place_and_label = _fn_place_and_label
place_shape_then_edit_label = _fn_place_shape_then_edit_label
place_shape_to_zone = _fn_place_shape_to_zone
place_shape_in_node_slot = _fn_place_shape_in_node_slot
edit_label = _fn_edit_label
delete_node = _fn_delete_node
move_and_deselect = _fn_move_and_deselect
move_node_to_zone_and_deselect = _fn_move_node_to_zone_and_deselect
move_node_adjacent_and_deselect = _fn_move_node_adjacent_and_deselect
rotate_node_90_and_deselect = _fn_rotate_node_90_and_deselect
reshape_node_and_deselect = _fn_reshape_node_and_deselect
