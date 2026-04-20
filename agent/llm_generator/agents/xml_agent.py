from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..client import OpenAIResponsesClient
from .prompt_utils import truncate_prompt_text


class XMLGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class XMLGenerationAttemptLog:
    attempt: int
    model_response: str
    validation_error: str | None


@dataclass(slots=True)
class XMLGenerationResult:
    model: str
    attempts: int
    xml_path: str
    xml_content: str
    logs: list[XMLGenerationAttemptLog]


XML_SYSTEM_PROMPT = (
    "You are an MJCF XML generator for Genesis articulated rigid bodies. "
    "Return one JSON object with keys: `filename`, `xml_content`. "
    "`xml_content` must be valid MJCF with root `<mujoco>` and at least one joint. "
    "The XML must describe only the robot body tree. "
    "Do not add ground planes, tables, lights, cameras, or any other background/environment elements. "
    "Under `<worldbody>`, include only the robot root `<body>` tree. "
    "Use only simple primitive geoms inside the articulated body tree; do not define mesh assets or geom type='mesh'. "
    "Every body/link in the articulated tree should have at least one collision-enabled primitive geom; avoid empty "
    "grouping bodies or non-colliding support-only links. "
    "Do not include `<actuator>` blocks or actuator tags under `<default>`; actuator definitions belong to IR. "
    "Do not include `<contact><exclude .../></contact>` blocks. "
    "Do not wrap XML in markdown fences."
)


ACTUATOR_DEFAULT_TAGS = {
    "general",
    "motor",
    "position",
    "velocity",
    "intvelocity",
    "damper",
    "cylinder",
    "muscle",
    "adhesion",
    "plugin",
}


def _slugify_filename(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = "articulated_model"
    return sanitized


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise XMLGenerationError("XML agent output does not contain a JSON object.")
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise XMLGenerationError("XML agent output root must be a JSON object.")
    return parsed


def _extract_xml_payload(payload: dict[str, Any]) -> tuple[str, str]:
    filename = payload.get("filename")
    if not isinstance(filename, str) or filename.strip() == "":
        filename = "articulated_model.xml"

    if not filename.endswith(".xml"):
        filename = f"{filename}.xml"

    xml_content = payload.get("xml_content")
    if not isinstance(xml_content, str):
        xml_content = payload.get("mjcf")
    if not isinstance(xml_content, str):
        xml_content = payload.get("xml")

    if not isinstance(xml_content, str) or xml_content.strip() == "":
        raise XMLGenerationError("XML agent output must include non-empty `xml_content` string.")

    xml_content = xml_content.strip()

    # Remove markdown fences if model wrapped it.
    if xml_content.startswith("```"):
        xml_content = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", xml_content)
        xml_content = re.sub(r"\n```$", "", xml_content)

    return filename, xml_content.strip()


def _validate_mjcf(xml_content: str) -> None:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise XMLGenerationError(f"Generated XML is not well-formed: {exc}") from exc

    if root.tag != "mujoco":
        raise XMLGenerationError(f"Expected root tag `<mujoco>`, got `<{root.tag}>`.")

    if not any(elem.tag == "joint" for elem in root.iter()):
        raise XMLGenerationError("Generated MJCF must contain at least one `<joint>` element.")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise XMLGenerationError("Generated MJCF must contain exactly one `<worldbody>` for the robot root body.")

    direct_bodies = [child for child in list(worldbody) if child.tag == "body"]
    direct_non_bodies = [child.tag for child in list(worldbody) if child.tag != "body"]
    if direct_non_bodies:
        raise XMLGenerationError(
            "Generated MJCF must not include background elements under `<worldbody>`; "
            f"found direct children {direct_non_bodies}."
        )
    if len(direct_bodies) != 1:
        raise XMLGenerationError(
            "Generated MJCF must contain exactly one direct robot root `<body>` under `<worldbody>`."
        )

    if any(elem.tag == "mesh" for elem in root.iter()):
        raise XMLGenerationError("Generated MJCF must not define `<mesh>` assets; use simple primitive geoms only.")
    if any(elem.tag == "geom" and elem.attrib.get("type") == "mesh" for elem in root.iter()):
        raise XMLGenerationError("Generated MJCF must not use `geom type=\"mesh\"`; use simple primitive geoms only.")

    try:
        import mujoco
    except Exception:
        # Fallback to XML well-formedness + structural checks if mujoco is unavailable.
        return

    try:
        mujoco.MjModel.from_xml_string(xml_content)
    except Exception as exc:  # noqa: BLE001
        raise XMLGenerationError(f"Generated MJCF failed MuJoCo schema validation: {exc}") from exc


def _strip_actuator_tags(root: ET.Element) -> bool:
    changed = False

    # Remove `<actuator>` blocks globally.
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in list(parent):
            if child.tag == "actuator":
                parent.remove(child)
                changed = True
                continue
            stack.append(child)

    # Remove actuator-default tags under every `<default>`.
    for default_elem in root.iter("default"):
        for child in list(default_elem):
            if child.tag in ACTUATOR_DEFAULT_TAGS:
                default_elem.remove(child)
                changed = True

    return changed


def _strip_worldbody_background(root: ET.Element) -> bool:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return False

    changed = False
    for child in list(worldbody):
        if child.tag != "body":
            worldbody.remove(child)
            changed = True
    return changed


def _strip_contact_excludes(root: ET.Element) -> bool:
    changed = False
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in list(parent):
            if child.tag == "contact":
                for grandchild in list(child):
                    if grandchild.tag == "exclude":
                        child.remove(grandchild)
                        changed = True
                if len(child) == 0:
                    parent.remove(child)
                    changed = True
                    continue
            stack.append(child)
    return changed


def _sanitize_mjcf(xml_content: str) -> str:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return xml_content

    changed = _strip_actuator_tags(root)
    changed = _strip_worldbody_background(root) or changed
    changed = _strip_contact_excludes(root) or changed
    if not changed:
        return xml_content

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _build_user_prompt(task: str, *, previous_error: str | None = None, file_stem: str) -> str:
    lines = [
        "Constraints:",
        "- Output JSON only.",
        "- xml_content must be valid MJCF.",
        "- The XML must include only the robot itself. No ground, no background, no environment objects.",
        "- Under `<worldbody>`, include only the robot root `<body>` tree.",
        "- Use only simple primitive geoms inside the articulated body. Do not define mesh assets or `geom type=\"mesh\"`.",
        "- Every body/link in the articulated tree should have at least one collision-enabled primitive geom. Avoid empty grouping bodies and avoid links whose geoms are all non-colliding.",
        "- Include at least one movable joint.",
        "- Do not include `<actuator>` blocks or actuator-default tags under `<default>`.",
        "- Do not include `<contact><exclude .../></contact>` blocks.",
        "- Use reasonable inertial/geom defaults for stable simulation.",
        "",
        "Task:",
        task.strip(),
        f"Preferred filename stem: `{file_stem}`",
    ]
    if previous_error:
        lines.extend(["", "Previous attempt failed validation:", previous_error.strip()])
    return "\n".join(lines)


def _build_revision_prompt(
    *,
    task: str,
    previous_error: str | None,
    file_stem: str,
    previous_xml_text: str | None,
) -> str:
    base = _build_user_prompt(task, previous_error=previous_error, file_stem=file_stem)
    if previous_xml_text is None or not previous_xml_text.strip():
        return base
    return "\n".join(
        [
            base,
            "",
            "Revision mode:",
            "- Revise the previous XML instead of generating a completely different robot.",
            "- Make targeted changes to the XML file to address the existing problem.",
            "",
            "Previous XML to revise:",
            truncate_prompt_text(previous_xml_text.strip()),
        ]
    )


def _build_prompt_cache_key() -> str:
    digest = hashlib.sha1(XML_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    return f"rigid_xml_agent:{digest}"


def generate_articulated_xml_with_openai(
    *,
    task: str,
    model: str,
    client: OpenAIResponsesClient,
    output_dir: str | Path,
    file_stem: str,
    previous_xml_text: str | None = None,
    max_attempts: int = 4,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> XMLGenerationResult:
    if max_attempts < 1:
        raise ValueError("`max_attempts` must be >= 1.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    safe_stem = _slugify_filename(file_stem)
    previous_error: str | None = None
    logs: list[XMLGenerationAttemptLog] = []
    prompt_cache_key = _build_prompt_cache_key()

    for attempt in range(1, max_attempts + 1):
        text = client.responses_json(
            model=model,
            system_prompt=XML_SYSTEM_PROMPT,
            user_prompt=_build_revision_prompt(
                task=task,
                previous_error=previous_error,
                file_stem=safe_stem,
                previous_xml_text=previous_xml_text,
            ),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
        )

        try:
            payload = _extract_json(text)
            filename, xml_content = _extract_xml_payload(payload)

            # Normalize filename and keep within output dir.
            filename = _slugify_filename(Path(filename).stem) + ".xml"

            xml_content = _sanitize_mjcf(xml_content)
            _validate_mjcf(xml_content)

            path = output_dir_path / filename
            path.write_text(xml_content + "\n", encoding="utf-8")

            logs.append(
                XMLGenerationAttemptLog(
                    attempt=attempt,
                    model_response=text,
                    validation_error=None,
                )
            )
            return XMLGenerationResult(
                model=model,
                attempts=attempt,
                xml_path=str(path.as_posix()),
                xml_content=xml_content,
                logs=logs,
            )
        except Exception as exc:  # noqa: BLE001
            previous_error = str(exc)
            logs.append(
                XMLGenerationAttemptLog(
                    attempt=attempt,
                    model_response=text,
                    validation_error=previous_error,
                )
            )

    raise XMLGenerationError(
        "Failed to generate valid MJCF XML after "
        f"{max_attempts} attempts. Last error: {previous_error}"
    )


def list_named_joint_names(xml_file: str | Path) -> tuple[str, ...]:
    path = Path(xml_file)
    root = ET.fromstring(path.read_text(encoding="utf-8"))

    names: list[str] = []
    seen: set[str] = set()
    for elem in root.iter("joint"):
        name = elem.attrib.get("name")
        if name is None:
            continue
        stripped = name.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        names.append(stripped)

    return tuple(names)
