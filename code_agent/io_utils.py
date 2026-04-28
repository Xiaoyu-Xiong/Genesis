from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_object(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} input must be a JSON object: {path}")
    return payload


def dump_json(data: Any, path: Path | None) -> None:
    content = json.dumps(data, indent=2, ensure_ascii=False)
    if path is None:
        print(content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
