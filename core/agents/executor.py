"""
Executor agent — picks the next tool given a task and current UI graph.

Constructs a coordinate-free prompt from the tool catalog and detected
element names, sends it to a local LLM via Ollama, and parses the JSON
response. The executor never sees pixel coordinates — it only picks
named tools and references elements by name/id.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import ollama

from core import config
from core.perception.canvas import tool_families
from core.tools import TOOL_CATALOG


# ---------------------------------------------------------------------------
# Prompt construction — coordinate-free
# ---------------------------------------------------------------------------

def _tool_table() -> str:
    """Render the tool catalog as a markdown table for the prompt."""
    lines = []
    for name, node in TOOL_CATALOG.items():
        params = ", ".join(node.params) if node.params else "(none)"
        lines.append(f"| {name} | L{node.level} | {params} | {node.description} |")
    header = "| tool | level | params | description |\n|------|-------|--------|-------------|"
    return header + "\n" + "\n".join(lines)


def _element_summary(ui_graph: Dict[str, Any]) -> str:
    """
    Coordinate-free summary of detected elements.
    Shows names only — no (x, y) values.
    """
    parts = []

    tools = list(ui_graph.get("UI_Elements", {}).keys())
    if tools:
        parts.append("### Sidebar Shapes (use with `place_shape`)")
        for t in tools:
            parts.append(f"- `{t}`")

        families = ui_graph.get("Tool_Families") or tool_families(ui_graph.get("UI_Elements", {}))
        if families:
            parts.append("\n### Ambiguous Sidebar Families")
            parts.append(
                "Use family defaults for common shapes, but dispatch exact tool names."
            )
            for family, spec in families.items():
                candidates = spec.get("candidates", [])
                default = spec.get("default")
                joined = ", ".join(f"`{c}`" for c in candidates)
                parts.append(f"- `{family}` default: `{default}`; candidates: {joined}")

    nodes = ui_graph.get("Canvas_Nodes", [])
    parts.append("\n### Observed Canvas")
    if nodes:
        parts.append(f"Visible node count: {len(nodes)}")
        for n in nodes:
            text = n.get("text", "")
            confidence = n.get("confidence")
            source = n.get("source", "unknown")
            parts.append(
                f"- id=`{n['id']}`, text=`{text}`, "
                f"position=`{n.get('position', 'unknown')}`, "
                f"confidence=`{confidence}`, source=`{source}`"
            )
    else:
        parts.append("Visible node count: 0")

    issues = ui_graph.get("Layout_Issues", [])
    if issues:
        parts.append("\n### Layout Issues")
        for issue in issues:
            node_list = ", ".join(f"`{n}`" for n in issue.get("nodes", []))
            parts.append(
                f"- `{issue.get('type')}` severity=`{issue.get('severity')}` "
                f"nodes={node_list}"
            )

    edges = ui_graph.get("Canvas_Edges", [])
    if edges:
        parts.append("\n### Canvas Edges")
        for e in edges:
            parts.append(f"- `{e['source']}` → `{e['target']}`")

    return "\n".join(parts)


_SYSTEM_TEMPLATE = """\
You are the **Planner** agent for draw.io.

## RULES
1. You **CANNOT** specify or produce any pixel coordinates.
2. Choose ONLY from the AVAILABLE TOOLS below.
3. Reference elements by **name** or **id** only.
4. If the required element is not listed, use `"request_rescan"`.
5. Output **exactly ONE tool call** per response.

## draw.io WORKFLOW (important!)
- You are responsible for decomposing abstract drawing requests into concrete
  component shapes and operations. The user should be able to say "draw a tree"
  or "draw a tree with 2 triangles and 1 rectangle"; you must infer placement,
  rotation, resizing, and arrangement steps.
- For tasks like "add/place a rectangle labelled X", prefer
  `place_shape_then_edit_label` with `tool_name` and `label`.
- `place_shape_then_edit_label` is more reliable because it explicitly enters
  label edit mode before typing.
- `tool_name` parameters should normally be exact sidebar tool names from
  "Sidebar Shapes" such as `Rectangle_Tool` or `Triangle_Tool`. Do not invent
  names like `Triangle` or `General_shapes_panel_triangle`. Families are only
  summaries; if a family is listed, use its exact `default` candidate.
- Use raw `place_shape` followed by `type_label` only when the task explicitly
  asks for step-by-step primitive actions.
- For multi-shape diagrams, prefer `place_shape_to_zone` over raw `place_shape`
  so the newly inserted shape is moved away from Draw.io's default insertion
  point before the next shape is added.
- For vertical flowcharts, place boxes with zones `top`, `center`, and
  `bottom` so they share the same vertical line. Do not use `upper_right` or
  `lower_right` unless the task asks for right-side placement.
- Use `connect_nodes` to draw arrows/connectors between two observed boxes.
  Do not place `Arrow_Tool` as a standalone shape for flowchart connectors.
  Connectors/arrow lines are not reliable `Canvas_Nodes`, so never invent a
  new id such as `Observed_Node_4` for an arrow unless it appears in Observed
  Canvas.
- If Layout Issues report `not_vertically_aligned`, move boxes to `top`,
  `center`, and `bottom` before adding connectors. If Layout Issues report
  `overlap`, move one of the overlapping nodes away before continuing.
- For nested/container diagrams, prefer `place_shape_in_node_slot` over
  `place_shape_to_zone` for shapes that should appear inside another shape.
  Valid slots are `top`, `middle`, and `bottom`.
- For a traffic light, make a rectangle housing first, reshape it taller if
  needed, then place three ellipse/circle lights with
  `place_shape_in_node_slot` in `top`, `middle`, and `bottom`. Do not place
  the lights in global canvas zones and do not flatten them manually.
- For rearrange/move/drag tasks, prefer `move_node_to_zone_and_deselect` with
  `node_ref` and a named `zone`.
- Valid zones: `center`, `left`, `right`, `top`, `bottom`, `upper_left`,
  `upper_right`, `lower_left`, `lower_right`.
- For connected multi-part objects, zones are only a rough first placement.
  After all required parts are visible, use `move_node_adjacent_and_deselect`
  to attach or overlap components using named relations. Valid relations:
  `attached_above`, `attached_below`, `attached_left`, `attached_right`,
  `overlap_above`, `overlap_below`, `centered_on`.
- Do not use `drag_node` or `move_and_deselect` unless the user/test explicitly
  provides target coordinates.
- For rotation/turning tasks, use `rotate_node_90_and_deselect` with
  `node_ref` and `direction`, usually `clockwise`. Do not invent hotkeys or
  mouse gestures such as `ctrl+drag`.
- For shape resizing/reshaping tasks, prefer `reshape_node_and_deselect`.
  Select a named blue handle and drag it: use `bottom` with positive `delta_y`
  to make a shape taller, `right` with positive `delta_x` to make it wider,
  `left` with positive `delta_x` or `right` with negative `delta_x` to make it
  narrower, and `bottom_right` to change width and height together.
- For container objects such as traffic lights, first make the housing taller
  with the `bottom` handle. Avoid aggressive narrowing unless the task requires
  it; the housing must stay wide enough for internal shapes and for perception
  to keep tracking the same node.
- Canvas node `text` may be empty because OCR is not implemented. If the prior
  action just placed a shape and exactly one visible canvas node exists, use
  that node id directly instead of requesting rescan for the label text.
- Draw.io often places every new sidebar shape at the same default canvas
  location. For multi-shape diagrams, do not leave a newly placed shape at the
  default point. Use `place_shape_to_zone` for each component, otherwise shapes
  may overlap and perception may merge them into one node.
- For tree-like diagrams, use this recipe unless the user says otherwise:
  place the first exact default candidate from `Triangle_Family` with
  `place_shape_to_zone` in `upper_right`, place the second triangle in `right`,
  place the `Rectangle_Family` default in `lower_right`, resize the rectangle
  to be tall/narrow if needed while it is still a separate detected node, then
  assemble them into one connected object: move the second triangle
  `overlap_below` the first triangle and move the rectangle `attached_below`
  the lower triangle. Only rotate triangles if the user asks for
  rotated/sideways foliage. A tree is not complete while its parts are merely
  near each other; foliage and trunk should touch or overlap.
- After `press_escape`, the shape is still selected. Use `click_empty_canvas` to deselect.
- `double_click_node` is ONLY needed to re-edit an existing node's label.

## AVAILABLE TOOLS
{tool_table}

**Special signals** (no params):
| tool | description |
|------|-------------|
| request_rescan | Re-perceive the screen |
| task_complete  | Signal task is finished |

## DETECTED ELEMENTS
{element_summary}

## OUTPUT FORMAT
Respond with a single JSON object — no markdown, no commentary:
{{
  "reasoning": "<your step-by-step logic>",
  "tool": "<tool_name>",
  "params": {{}}
}}
"""


def build_prompt(ui_graph: Dict[str, Any]) -> str:
    """Build the system prompt for the LLM."""
    return _SYSTEM_TEMPLATE.format(
        tool_table=_tool_table(),
        element_summary=_element_summary(ui_graph),
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_response(raw: str) -> Dict[str, Any]:
    """Extract a JSON dict from the LLM's raw text output."""
    text = raw.strip()

    # Try raw JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fenced JSON block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return json.loads(match.group(1))

    # First { … last }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Could not parse JSON from LLM response:\n{text}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(
    task: str,
    ui_graph: Dict[str, Any],
    screenshot_path: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Ask the LLM to choose the next tool.

    Args:
        task:            Natural-language task description.
        ui_graph:        Current UI graph (element names shown, no coords).
        screenshot_path: Path to screenshot (sent as image).
        history:         Prior conversation turns for multi-step reasoning.

    Returns:
        Dict with keys: ``reasoning``, ``tool``, ``params``.
    """
    model = config.llm_model()
    prompt = build_prompt(ui_graph)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": prompt}]
    if history:
        messages.extend(history)

    with open(screenshot_path, "rb") as f:
        image_bytes = f.read()

    messages.append({
        "role": "user",
        "content": f"Task: {task}",
        "images": [image_bytes],
    })

    print(f"[EXECUTOR] Querying {model} …")
    response = ollama.chat(model=model, messages=messages)
    raw = response["message"]["content"]
    print(f"[EXECUTOR] Raw response:\n{raw}")

    result = parse_response(raw)

    # Normalize: accept "action" key as alias for "tool"
    if "tool" not in result and "action" in result:
        result["tool"] = result.pop("action")
    if "tool" not in result:
        raise ValueError(f"Executor response missing 'tool' key: {result}")

    print(f"[EXECUTOR] Decided: {result['tool']}  {result.get('params', {})}")
    return result
