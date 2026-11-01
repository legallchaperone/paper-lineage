from __future__ import annotations

import json

from ..graph import LineageGraph


def render_json(g: LineageGraph) -> str:
    return json.dumps(g.to_dict(), indent=2, ensure_ascii=False)
