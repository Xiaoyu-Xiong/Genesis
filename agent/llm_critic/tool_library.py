from __future__ import annotations

import json
from typing import Any


class CriticToolLibrary:
    def __init__(
        self,
        *,
        ir: dict[str, Any],
        event_pack: dict[str, Any],
        xml_texts_by_body: dict[str, str],
    ) -> None:
        self.ir = ir
        self.event_pack = event_pack
        self.xml_texts_by_body = xml_texts_by_body
        self._tool_funcs = {
            "get_critic_bootstrap": self._get_critic_bootstrap,
            "get_ir_scene": self._get_ir_scene,
            "get_ir_body": self._get_ir_body,
            "get_ir_actions": self._get_ir_actions,
            "get_event_execution": self._get_event_execution,
            "get_event_crash": self._get_event_crash,
            "get_event_observations": self._get_event_observations,
            "get_xml_body": self._get_xml_body,
        }

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_critic_bootstrap",
                    "description": "Return compact bootstrap metadata about available IR bodies, event timeline, crash status, and XML coverage.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ir_scene",
                    "description": "Return the IR scene object only.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ir_body",
                    "description": "Return one body from the IR by body name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"body_name": {"type": "string"}},
                        "required": ["body_name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ir_actions",
                    "description": "Return a slice of the IR action list.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer"},
                            "count": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_event_execution",
                    "description": "Return execution metadata from the event pack.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_event_crash",
                    "description": "Return crash information from the event pack, if any.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_event_observations",
                    "description": "Return filtered observation timeline entries from the event pack.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "tag": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_xml_body",
                    "description": "Return the XML text for one articulated body, if available.",
                    "parameters": {
                        "type": "object",
                        "properties": {"body_name": {"type": "string"}},
                        "required": ["body_name"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute_tool_calls_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in calls:
            name = call.get("name")
            arguments_json = call.get("arguments_json")
            if not isinstance(name, str) or name not in self._tool_funcs:
                results.append({"ok": False, "error": f"Unknown tool `{name}`."})
                continue
            try:
                args = self._parse_arguments(arguments_json)
                results.append(self._tool_funcs[name](args))
            except Exception as exc:  # noqa: BLE001
                results.append({"ok": False, "error": str(exc)})
        return results

    @staticmethod
    def _parse_arguments(arguments_json: str | None) -> dict[str, Any]:
        if arguments_json is None or not arguments_json.strip():
            return {}
        parsed = json.loads(arguments_json)
        if not isinstance(parsed, dict):
            raise ValueError("critic tool arguments root must be an object")
        return parsed

    def _get_critic_bootstrap(self, _: dict[str, Any]) -> dict[str, Any]:
        bodies = self.ir.get("bodies")
        observations = self.event_pack.get("observations")
        timeline = observations.get("timeline") if isinstance(observations, dict) else None
        tags = sorted(
            {
                item.get("tag")
                for item in timeline
                if isinstance(item, dict) and isinstance(item.get("tag"), str) and item.get("tag")
            }
        ) if isinstance(timeline, list) else []
        return {
            "ok": True,
            "body_names": [body.get("name") for body in bodies if isinstance(body, dict) and isinstance(body.get("name"), str)]
            if isinstance(bodies, list)
            else [],
            "observation_count": len(timeline) if isinstance(timeline, list) else 0,
            "observation_tags": tags,
            "has_crash": isinstance(self.event_pack.get("crash"), dict),
            "xml_bodies": sorted(self.xml_texts_by_body.keys()),
        }

    def _get_ir_scene(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "scene": self.ir.get("scene")}

    def _get_ir_body(self, args: dict[str, Any]) -> dict[str, Any]:
        body_name = args.get("body_name")
        if not isinstance(body_name, str) or not body_name.strip():
            return {"ok": False, "error": "`body_name` must be a non-empty string."}
        for body in self.ir.get("bodies", []):
            if isinstance(body, dict) and body.get("name") == body_name:
                return {"ok": True, "body": body}
        return {"ok": False, "error": f"Body `{body_name}` not found."}

    def _get_ir_actions(self, args: dict[str, Any]) -> dict[str, Any]:
        actions = self.ir.get("actions", [])
        if not isinstance(actions, list):
            return {"ok": True, "actions": []}
        start = args.get("start", 0)
        count = args.get("count", 20)
        if not isinstance(start, int):
            start = 0
        if not isinstance(count, int):
            count = 20
        start = max(start, 0)
        count = max(count, 1)
        return {"ok": True, "start": start, "count": count, "actions": actions[start : start + count]}

    def _get_event_execution(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "execution": self.event_pack.get("execution")}

    def _get_event_crash(self, _: dict[str, Any]) -> dict[str, Any]:
        crash = self.event_pack.get("crash")
        return {"ok": True, "crash": crash}

    def _get_event_observations(self, args: dict[str, Any]) -> dict[str, Any]:
        observations = self.event_pack.get("observations")
        timeline = observations.get("timeline") if isinstance(observations, dict) else None
        if not isinstance(timeline, list):
            return {"ok": True, "timeline": []}
        entity = args.get("entity")
        tag = args.get("tag")
        limit = args.get("limit", 20)
        if not isinstance(limit, int):
            limit = 20
        limit = max(limit, 1)
        filtered: list[dict[str, Any]] = []
        for item in timeline:
            if not isinstance(item, dict):
                continue
            if isinstance(entity, str) and item.get("entity") != entity:
                continue
            if isinstance(tag, str) and item.get("tag") != tag:
                continue
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return {"ok": True, "timeline": filtered}

    def _get_xml_body(self, args: dict[str, Any]) -> dict[str, Any]:
        body_name = args.get("body_name")
        if not isinstance(body_name, str) or not body_name.strip():
            return {"ok": False, "error": "`body_name` must be a non-empty string."}
        xml_text = self.xml_texts_by_body.get(body_name)
        if xml_text is None:
            return {"ok": False, "error": f"No XML text available for `{body_name}`."}
        return {"ok": True, "body_name": body_name, "xml_text": xml_text}
