from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET


ACTUATOR_TAGS = {
    "adhesion",
    "cylinder",
    "general",
    "intvelocity",
    "motor",
    "muscle",
    "position",
    "velocity",
}
FORBIDDEN_TAGS = {"camera", "include", "light"}
FORBIDDEN_GEOM_TYPES = {"hfield", "plane"}
MESH_ASSET_TAGS = {"hfield", "mesh"}
GLOBAL_SIMULATION_TAGS = {"option"}


def tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1] if "}" in element.tag else element.tag


def single_child(root: ET.Element, child_tag: str, errors: list[str]) -> ET.Element | None:
    matches = [child for child in list(root) if tag(child) == child_tag]
    if len(matches) != 1:
        errors.append(f"Exactly one <{child_tag}> element is required; found {len(matches)}.")
        return matches[0] if matches else None
    return matches[0]


def forbidden_elements(root: ET.Element) -> list[str]:
    found: list[str] = []
    for element in root.iter():
        element_tag = tag(element)
        if element_tag in FORBIDDEN_TAGS:
            name = element.attrib.get("name")
            found.append(f"<{element_tag}{' name=' + name if name else ''}>")
    return found


def global_simulation_elements(root: ET.Element) -> list[str]:
    return [f"<{tag(element)}>" for element in list(root) if tag(element) in GLOBAL_SIMULATION_TAGS]


def has_mesh_asset(root: ET.Element) -> bool:
    return any(tag(element) in MESH_ASSET_TAGS for element in root.iter())


def collect_body_tree(
    body: ET.Element,
    *,
    parent_path: str,
    body_infos: list[dict[str, Any]],
    joint_infos: list[dict[str, Any]],
    geom_infos: list[dict[str, Any]],
    site_infos: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    body_name = body.attrib.get("name") or f"unnamed_body_{len(body_infos)}"
    body_path = f"{parent_path}/{body_name}" if parent_path else body_name
    child_bodies = [child for child in list(body) if tag(child) == "body"]
    direct_geoms = [child for child in list(body) if tag(child) == "geom"]
    direct_joints = [child for child in list(body) if tag(child) in {"joint", "freejoint"}]
    body_infos.append(
        {
            "name": body_name,
            "path": body_path,
            "pos": body.attrib.get("pos"),
            "quat": body.attrib.get("quat"),
            "child_body_count": len(child_bodies),
            "direct_geom_count": len(direct_geoms),
            "direct_joint_count": len(direct_joints),
        }
    )

    if not direct_geoms and direct_joints:
        warnings.append(f"Body `{body_path}` has joints but no direct geom.")

    for joint in direct_joints:
        joint_tag = tag(joint)
        joint_name = joint.attrib.get("name") or f"unnamed_{joint_tag}_{len(joint_infos)}"
        joint_infos.append(
            {
                "name": joint_name,
                "tag": joint_tag,
                "type": "free" if joint_tag == "freejoint" else joint.attrib.get("type", "hinge"),
                "body": body_path,
                "axis": joint.attrib.get("axis"),
                "range": float_list(joint.attrib.get("range")),
                "limited": joint.attrib.get("limited"),
                "damping": float_or_none(joint.attrib.get("damping")),
            }
        )
        if not joint.attrib.get("name"):
            errors.append(f"Every joint must be named; unnamed {joint_tag} found on body `{body_path}`.")

    for geom in direct_geoms:
        geom_type = geom.attrib.get("type")
        geom_name = geom.attrib.get("name") or f"unnamed_geom_{len(geom_infos)}"
        geom_infos.append(
            {
                "name": geom_name,
                "type": geom_type,
                "body": body_path,
                "size": float_list(geom.attrib.get("size")),
                "fromto": float_list(geom.attrib.get("fromto")),
                "pos": float_list(geom.attrib.get("pos")),
                "rgba": float_list(geom.attrib.get("rgba")),
                "contype": geom.attrib.get("contype"),
                "conaffinity": geom.attrib.get("conaffinity"),
            }
        )
        if geom_type in FORBIDDEN_GEOM_TYPES:
            errors.append(
                f"Forbidden geom type `{geom_type}` found on `{geom_name}`. "
                "Generated XML assets must use non-plane body geoms; mesh geoms are allowed only with generated "
                "case-workspace mesh files declared in the XML asset section."
            )

    for site in [child for child in list(body) if tag(child) == "site"]:
        site_infos.append(
            {
                "name": site.attrib.get("name") or f"unnamed_site_{len(site_infos)}",
                "body": body_path,
                "pos": float_list(site.attrib.get("pos")),
                "type": site.attrib.get("type"),
                "size": float_list(site.attrib.get("size")),
            }
        )
        if not site.attrib.get("name"):
            warnings.append(f"Unnamed site found on body `{body_path}`.")

    for child_body in child_bodies:
        collect_body_tree(
            child_body,
            parent_path=body_path,
            body_infos=body_infos,
            joint_infos=joint_infos,
            geom_infos=geom_infos,
            site_infos=site_infos,
            errors=errors,
            warnings=warnings,
        )


def collect_tendons(root: ET.Element) -> list[dict[str, Any]]:
    tendons: list[dict[str, Any]] = []
    tendon_root = root.find("tendon")
    if tendon_root is None:
        return tendons
    for tendon in list(tendon_root):
        tendon_tag = tag(tendon)
        name = tendon.attrib.get("name") or f"unnamed_{tendon_tag}_{len(tendons)}"
        tendons.append(
            {
                "name": name,
                "type": tendon_tag,
                "range": float_list(tendon.attrib.get("range")),
                "limited": tendon.attrib.get("limited"),
            }
        )
    return tendons


def collect_actuators(root: ET.Element) -> list[dict[str, Any]]:
    actuator_root = root.find("actuator")
    if actuator_root is None:
        return []
    actuators: list[dict[str, Any]] = []
    for actuator in list(actuator_root):
        actuator_tag = tag(actuator)
        if actuator_tag not in ACTUATOR_TAGS:
            continue
        name = actuator.attrib.get("name") or f"unnamed_{actuator_tag}_{len(actuators)}"
        ctrlrange = float_list(actuator.attrib.get("ctrlrange"))
        actuators.append(
            {
                "name": name,
                "type": actuator_tag,
                "joint": actuator.attrib.get("joint"),
                "tendon": actuator.attrib.get("tendon"),
                "site": actuator.attrib.get("site"),
                "body": actuator.attrib.get("body"),
                "ctrlrange": ctrlrange,
                "neutral_command": neutral_command(ctrlrange),
                "gear": float_list(actuator.attrib.get("gear")),
                "kp": float_or_none(actuator.attrib.get("kp")),
                "kv": float_or_none(actuator.attrib.get("kv")),
            }
        )
    return actuators


def collect_equalities(root: ET.Element) -> list[dict[str, Any]]:
    equality_root = root.find("equality")
    if equality_root is None:
        return []
    equalities: list[dict[str, Any]] = []
    for equality in list(equality_root):
        equality_tag = tag(equality)
        name = equality.attrib.get("name") or f"unnamed_{equality_tag}_equality_{len(equalities)}"
        equalities.append(
            {
                "name": name,
                "type": equality_tag,
                "joint1": equality.attrib.get("joint1"),
                "joint2": equality.attrib.get("joint2"),
                "body1": equality.attrib.get("body1"),
                "body2": equality.attrib.get("body2"),
                "geom1": equality.attrib.get("geom1"),
                "geom2": equality.attrib.get("geom2"),
                "site1": equality.attrib.get("site1"),
                "site2": equality.attrib.get("site2"),
                "polycoef": float_list(equality.attrib.get("polycoef")),
            }
        )
    return equalities


def float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    try:
        return [float(part) for part in raw.split()]
    except ValueError:
        return None


def float_or_none(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def neutral_command(ctrlrange: list[float] | None) -> float:
    if ctrlrange is None or len(ctrlrange) != 2:
        return 0.0
    return 0.5 * (float(ctrlrange[0]) + float(ctrlrange[1]))
