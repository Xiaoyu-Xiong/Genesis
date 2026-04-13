from __future__ import annotations

import hashlib
import json
from typing import Any

from ..defaults import DEFAULTS
from .video_sampler import SampledFrame


if DEFAULTS.deformable.simulation_backend == "pbd":
    _DEFORMABLE_TUNING_GUIDANCE = (
        "For deformable bodies, prefer fixes that adjust `deformable_material.stretch_compliance`, "
        "`deformable_material.volume_compliance`, or `deformable_material.rho` before suggesting hidden solver "
        "hyperparameters, because the deformable v1 pipeline intentionally fixes those internal settings."
    )
else:
    _DEFORMABLE_TUNING_GUIDANCE = (
        "For deformable bodies, prefer fixes that adjust `deformable_material.E`, `deformable_material.nu`, "
        "or `deformable_material.rho` before suggesting hidden IPC/FEM solver hyperparameters, because the "
        "deformable v1 pipeline intentionally fixes those internal settings."
    )


CRITIC_SYSTEM_PROMPT = """You are a simulation critic for robotics/physics outputs.
You will receive:
1) a task prompt,
2) the generated IR JSON,
3) optional articulated asset texts by body,
4) the raw event-pack JSON,
5) sampled video frames in chronological order.

Your job:
- evaluate whether the output satisfies the task,
- cross-check IR, XML, event-pack, and video evidence,
- Do not be too strict on the result, pass some borderline cases if the overall behavior seems mostly correct.
- identify contradictions and uncertainty,
- propose concrete fixes.
- The IR may contain multiple bodies. Structure your critique by IR layers: global `scene`, global `actions`, and per-body analysis in `by_body`.
- Prioritize overall task fulfillment, visible behavior, physical plausibility, and whether the robot does the right thing.
- Do not let minor numeric discrepancies dominate the critique unless they clearly indicate a major behavioral problem, instability, or contradiction.
- Read and use the provided generator tool-library descriptions, especially generation_guide constraints, parameter_notes, parameter_relationship_notes, schema field descriptions, and any provided mesh bounding-box metadata.
- When a major issue involves parameter tuning, use those descriptions to identify the likely root cause instead of giving vague advice. Distinguish between insufficient stiffness, insufficient damping, insufficient output limit, unstable restitution, camera-lag settings, and similar cases when the evidence supports it.
- Keep `priority_fixes` focused on the few biggest issues blocking success, not on small cleanups.
- Prefer a small number of major issues with detailed modifications over a long list of shallow comments.
- In addition to correctness, consider IR conciseness (but not at the expense of clarity). If multiple actions can be merged into one equivalent multi-entity action without changing behavior, prefer that as the cleaner formulation. This is only a suggestion, but should not be used to determine success vs failure.
- When the task leaves scene composition open-ended, provide suggestions leading to visible motion, meaningful contact, and noticeable evolution over time. Do not treat this as hard requirement and use it to judge success.
- For deformable or soft-body tasks, evaluate both whole-body motion and visible deformation. Use deformable observation fields such as bounding-box changes and vertex-displacement summaries when present, and do not judge soft bodies by rigid-body pose stability alone.
- Base all suggested fixes on the provided generator tool-library capability only.
- Do not suggest unavailable controllers, target-tracking systems, sensors, or new runtime abilities that the current tool library cannot express.
- Every item in `priority_fixes` must be implementable through the provided tool library and current IR/XML path.
- If the active deformable backend is FEM+IPC, treat any initial penetration or interpenetration between bodies as a serious setup error and explicitly call it out. When this is the issue, prefer fixes that change only `bodies[*].initial_pose.pos` to create small positive clearance, and do not recommend changing shape, size, scale, material, density, stiffness, or actions unless there is separate evidence for those changes.
- Do not over-focus on duration alone; prioritize content correctness, physical plausibility, and control logic.
- For each major issue, make the `fix` field concrete: name the IR field(s) or actuator setting(s) to adjust, the direction of change, and the intended effect on behavior.
- For mesh objects, calibrate their orientation and overall size from the video evidence and any provided mesh bounding-box metadata. If the mesh is globally too large or too small, explicitly suggest adjusting `bodies[*].shape.scale` to resize the whole mesh uniformly. If orientation is wrong, provide specific `quat` adjustments with the intended effect on behavior.
- """ + _DEFORMABLE_TUNING_GUIDANCE + """
- When performing numerical parameter tuning, prefer exponential and more aggressive changes if the evidence suggests a major problem (e.g. objects supposed to move are almost static), and prefer smaller, more precise adjustments if the issue seems more borderline.
- Do not recommend verbose IR rewrites when a shorter equivalent IR is possible.

Return ONLY a JSON object with this schema:
{
  "verdict": "pass" | "partial" | "fail",
  "overall_score": 0-100,
  "summary": string,
  "by_section": {
    "scene": {
      "score": 0-100,
      "summary": string,
      "strengths": [string],
      "issues": [
        {
          "severity": "high" | "medium" | "low",
          "title": string,
          "evidence": [string],
          "fix": string
        }
      ]
    },
    "actions": {
      "score": 0-100,
      "summary": string,
      "strengths": [string],
      "issues": [
        {
          "severity": "high" | "medium" | "low",
          "title": string,
          "evidence": [string],
          "fix": string
        }
      ]
    }
  },
  "by_body": {
    "<body_name>": {
      "score": 0-100,
      "summary": string,
      "strengths": [string],
      "issues": [
        {
          "severity": "high" | "medium" | "low",
          "title": string,
          "evidence": [string],
          "fix": string
        }
      ]
    }
  },
  "cross_checks": {
    "ir_vs_event": string,
    "event_vs_video": string,
    "ir_vs_video": string
  },
  "priority_fixes": [string]
}
Use only provided evidence. Do not invent unseen details.
"""


def build_critic_prompt_cache_key() -> str:
    digest = hashlib.sha1(CRITIC_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    return f"rigid_critic:{digest}"


def build_compact_critic_prompt_cache_key() -> str:
    digest = hashlib.sha1((CRITIC_SYSTEM_PROMPT + "\ncompact").encode("utf-8")).hexdigest()[:16]
    return f"rigid_critic_compact:{digest}"


def build_critic_hosted_prompt_ref(
    *,
    hosted_prompt_id: str | None,
    hosted_prompt_version: str | None,
) -> dict[str, Any] | None:
    if hosted_prompt_id is None:
        return None
    prompt: dict[str, Any] = {"id": hosted_prompt_id}
    if hosted_prompt_version is not None:
        prompt["version"] = hosted_prompt_version
    return prompt


def build_critic_user_content(
    *,
    task: str,
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_texts_by_body: dict[str, str],
    input_digest: dict[str, Any],
    sampled_frames: list[SampledFrame],
) -> list[dict[str, Any]]:
    if xml_texts_by_body:
        rendered_xml_text = json.dumps(xml_texts_by_body, ensure_ascii=False, indent=2)
    else:
        rendered_xml_text = "<none provided>"
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Evaluate whether the simulation output satisfies the task.\n\n"
                f"Task prompt:\n{task}\n\n"
                "Raw IR JSON:\n"
                f"{json.dumps(ir, ensure_ascii=False, indent=2)}\n\n"
                "Raw articulated asset texts by body (optional):\n"
                f"{rendered_xml_text}\n\n"
                "Raw event-pack JSON:\n"
                f"{json.dumps(event_pack, ensure_ascii=False, indent=2)}\n\n"
                "Input digest (supporting summary and metadata, not the primary grading target):\n"
                f"{json.dumps(input_digest, ensure_ascii=False, indent=2)}\n\n"
                "The digest includes the generator tool-library capability summary. "
                "You must constrain your fixes to that capability set. "
                "Use the provided parameter notes, parameter relationship notes, and schema descriptions when deciding "
                "which field is actually responsible for a major problem. "
                "When a broad task could support a more dynamic scene, do not give full credit to outputs that are technically valid but largely static or visually uneventful. "
                "If deformable observation fields are present, use them to judge whether the soft body actually deforms in a meaningful way. "
                "When there are multiple bodies, assign body-specific issues to `by_body` using the actual body names from the IR. "
                "Also consider whether repeated same-payload actions could be merged using multi-entity `entity` lists "
                "to keep the IR shorter without changing behavior. "
                "Do not let minor numeric mismatches in the digest outweigh clear overall behavioral evidence from the task, "
                "video, and event trends.\n\n"
                "The following images are sampled frames in chronological order."
            ),
        }
    ]
    for frame in sampled_frames:
        content.append({"type": "input_text", "text": f"Frame {frame.index + 1} / {len(sampled_frames)}"})
        content.append({"type": "input_image", "image_url": frame.data_url})
    content.append(
        {
            "type": "input_text",
            "text": (
                "Now return the required JSON object using evidence from task, event-pack, and video frames. "
                "Focus on the main blockers, but for each main blocker provide a detailed, field-level modification."
            ),
        }
    )
    return content


def build_compact_critic_user_content(
    *,
    task: str,
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_texts_by_body: dict[str, str],
    input_digest: dict[str, Any],
    sampled_frames: list[SampledFrame],
) -> list[dict[str, Any]]:
    if xml_texts_by_body:
        rendered_xml_text = json.dumps(xml_texts_by_body, ensure_ascii=False, indent=2)
    else:
        rendered_xml_text = "<none provided>"
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Evaluate whether the simulation output satisfies the task.\n\n"
                f"Task prompt:\n{task}\n\n"
                "Raw IR JSON:\n"
                f"{json.dumps(ir, ensure_ascii=False, indent=2)}\n\n"
                "Raw articulated asset texts by body (optional):\n"
                f"{rendered_xml_text}\n\n"
                "Raw event-pack JSON:\n"
                f"{json.dumps(event_pack, ensure_ascii=False, indent=2)}\n\n"
                "Compact input digest (compact capability summary plus compact metadata):\n"
                f"{json.dumps(input_digest, ensure_ascii=False, indent=2)}\n\n"
                "The compact digest intentionally summarizes only the tool-library capability boundary, key parameter notes, "
                "and critical multi-body / multi-XML rules. Stay within that capability boundary when proposing fixes. "
                "When the task is broad and the scene could be more dynamic, treat mostly static outputs as weaker even if they are superficially valid. "
                "If deformable observation fields are present, use them to assess whether the soft body meaningfully deforms. "
                "When there are multiple bodies, assign body-specific issues to `by_body` using actual IR body names. "
                "Also consider whether repeated same-payload actions could be merged using multi-entity `entity` lists "
                "to keep the IR shorter without changing behavior.\n\n"
                "The following images are sampled frames in chronological order."
            ),
        }
    ]
    for frame in sampled_frames:
        content.append({"type": "input_text", "text": f"Frame {frame.index + 1} / {len(sampled_frames)}"})
        content.append({"type": "input_image", "image_url": frame.data_url})
    content.append(
        {
            "type": "input_text",
            "text": (
                "Now return the required JSON object using evidence from task, event-pack, and video frames. "
                "Focus on the main blockers, but for each main blocker provide a detailed, field-level modification."
            ),
        }
    )
    return content
