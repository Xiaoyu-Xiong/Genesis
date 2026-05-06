from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(data: Any, path: Path) -> None:
    content = json.dumps(data, indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
