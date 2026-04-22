from __future__ import annotations

import json
from typing import Any


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted_tool: dict[str, Any] = {"type": "function", "name": name}
        description = function.get("description")
        parameters = function.get("parameters")
        if isinstance(description, str):
            converted_tool["description"] = description
        if isinstance(parameters, dict):
            converted_tool["parameters"] = parameters
        converted.append(converted_tool)
    return converted


def convert_tool_choice(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return "auto"
    if tool_choice.get("type") != "function":
        return "auto"
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return "auto"
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return "auto"
    return {"type": "function", "name": name}


def convert_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        if role == "tool":
            converted.extend(_convert_tool_message(message))
            continue

        content = _convert_message_content(message.get("content"), role=role)
        if role == "assistant":
            if content:
                converted.append({"role": role, "content": content})
            converted.extend(_convert_assistant_tool_calls(message.get("tool_calls")))
            continue

        converted.append({"role": role, "content": content})
    return converted


def assistant_message_from_responses(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                content_parts.extend(_extract_response_message_text(item))
                continue
            if item_type == "function_call":
                tool_call = _convert_response_function_call(item, len(tool_calls))
                if tool_call is not None:
                    tool_calls.append(tool_call)

    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        content_parts.append(output_text)

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(part for part in content_parts if part).strip()}
    response_id = response.get("id")
    if isinstance(response_id, str) and response_id:
        message["_response_id"] = response_id
    usage = response.get("usage")
    if isinstance(usage, dict):
        message["_usage"] = usage
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        if isinstance(item.get("content"), str):
            parts.append(item["content"])
    return "\n".join(part for part in parts if part).strip()


def _convert_message_content(content: Any, *, role: str) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        return [{"type": text_type, "text": ""}]

    converted: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            converted.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str):
                converted.append({"type": text_type, "text": text})
            continue
        if role != "assistant" and part_type in {"image_url", "input_image"}:
            image = part.get("image_url")
            if isinstance(image, dict):
                image = image.get("url")
            if isinstance(image, str):
                converted.append({"type": "input_image", "image_url": image})
    if converted:
        return converted
    return [{"type": text_type, "text": ""}]


def _convert_tool_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_name = message.get("name")
    call_id = message.get("tool_call_id")
    tool_content = message.get("content", "")
    if not isinstance(tool_content, str):
        tool_content = json.dumps(tool_content, ensure_ascii=False)

    if isinstance(call_id, str) and call_id:
        return [{"type": "function_call_output", "call_id": call_id, "output": tool_content}]

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Tool result from `{tool_name}` (no call_id): {tool_content}",
                }
            ],
        }
    ]


def _convert_assistant_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    converted: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        call_id = tool_call.get("id")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        if not isinstance(call_id, str) or not call_id:
            call_id = f"tool_call_{index}"
        converted.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }
        )
    return converted


def _extract_response_message_text(item: dict[str, Any]) -> list[str]:
    message_content = item.get("content")
    if not isinstance(message_content, list):
        return []

    parts: list[str] = []
    for part in message_content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {"output_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return parts


def _convert_response_function_call(item: dict[str, Any], index: int) -> dict[str, Any] | None:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None

    arguments = item.get("arguments")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    if not isinstance(arguments, str):
        arguments = "{}"

    tool_call_id = item.get("call_id")
    if not isinstance(tool_call_id, str):
        item_id = item.get("id")
        tool_call_id = item_id if isinstance(item_id, str) else f"tool_call_{index}"

    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
