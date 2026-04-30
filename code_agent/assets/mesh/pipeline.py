from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ...io_utils import dump_json
from .models import (
    MeshRepairConfig,
    MeshyApiConfig,
    MeshyGenerationConfig,
    MeshyGenerationResult,
    MeshyTextureConfig,
    MeshyTextureResult,
    TextToMeshBundle,
)
from .repair.sanity import run_mesh_manifold_check
from .texture.transfer import transfer_texture_to_repaired_mesh
from .workflow.meshy import MeshyClient
from .workflow.steps import (
    augment_meshy_geometry_prompt,
    extract_preview_task_id,
    run_repair_pipeline,
    run_texture_pipeline,
    select_pipeline_source_mesh,
    slugify_prompt,
    time_stage,
)


@dataclass(slots=True)
class DownloadedMeshyAsset:
    generation: MeshyGenerationResult
    generation_config: MeshyGenerationConfig
    texture_config: MeshyTextureConfig | None
    texture: MeshyTextureResult | None
    pipeline_source_mesh_path: Path
    pipeline_source_kind: str
    profile_sec: dict[str, float]
    started_at_monotonic: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DownloadedMeshyAsset":
        generation_data = data["generation"]
        generation_config_data = data["generation_config"]
        texture_config_data = data.get("texture_config")
        texture_data = data.get("texture")

        generation = MeshyGenerationResult(
            provider=str(generation_data["provider"]),
            prompt=str(generation_data["prompt"]),
            output_dir=Path(generation_data["output_dir"]),
            mesh_path=Path(generation_data["mesh_path"]),
            prompt_path=Path(generation_data["prompt_path"]),
            submit_response_path=Path(generation_data["submit_response_path"]),
            final_response_path=Path(generation_data["final_response_path"]),
            metadata_path=Path(generation_data["metadata_path"]),
            preview_task_id=str(generation_data["preview_task_id"]),
            final_status=str(generation_data["final_status"]),
            stage_durations_sec=dict(generation_data.get("stage_durations_sec") or {}),
            submit_response={},
            final_response={},
        )

        generation_config = MeshyGenerationConfig(
            prompt=str(generation_config_data["prompt"]),
            output_dir=Path(generation_config_data["output_dir"]),
            mesh_format=str(generation_config_data["mesh_format"]),
            ai_model=str(generation_config_data["ai_model"]),
            art_style=str(generation_config_data["art_style"]),
            should_remesh=bool(generation_config_data["should_remesh"]),
            topology=str(generation_config_data["topology"]),
            target_polycount=generation_config_data.get("target_polycount"),
            symmetry_mode=str(generation_config_data["symmetry_mode"]),
            moderation=bool(generation_config_data["moderation"]),
            negative_prompt=generation_config_data.get("negative_prompt"),
            auto_size=bool(generation_config_data["auto_size"]),
            origin_at=generation_config_data.get("origin_at"),
            poll_interval_sec=float(generation_config_data["poll_interval_sec"]),
            max_wait_sec=float(generation_config_data["max_wait_sec"]),
            extra_payload=dict(generation_config_data.get("extra_payload") or {}),
        )

        texture_config = None
        if texture_config_data is not None:
            texture_config = MeshyTextureConfig(
                enabled=bool(texture_config_data["enabled"]),
                texture_prompt=texture_config_data.get("texture_prompt"),
                ai_model=texture_config_data.get("ai_model"),
                enable_pbr=bool(texture_config_data["enable_pbr"]),
                remove_lighting=bool(texture_config_data["remove_lighting"]),
            )

        texture = None
        if texture_data is not None:
            texture = MeshyTextureResult(
                requested=bool(texture_data["requested"]),
                ok=bool(texture_data["ok"]),
                prompt=str(texture_data["prompt"]),
                output_dir=Path(texture_data["output_dir"]),
                preview_task_id=str(texture_data["preview_task_id"]),
                refine_task_id=texture_data.get("refine_task_id"),
                submit_response_path=_optional_path(texture_data.get("submit_response_path")),
                final_response_path=_optional_path(texture_data.get("final_response_path")),
                textured_mesh_path=_optional_path(texture_data.get("textured_mesh_path")),
                textured_mtl_path=_optional_path(texture_data.get("textured_mtl_path")),
                texture_paths={key: Path(path) for key, path in dict(texture_data.get("texture_paths") or {}).items()},
                ai_model=texture_data.get("ai_model"),
                enable_pbr=bool(texture_data["enable_pbr"]),
                remove_lighting=bool(texture_data["remove_lighting"]),
                final_status=texture_data.get("final_status"),
                stage_durations_sec=dict(texture_data.get("stage_durations_sec") or {}),
                submit_response={},
                final_response={},
                error=texture_data.get("error"),
            )

        return cls(
            generation=generation,
            generation_config=generation_config,
            texture_config=texture_config,
            texture=texture,
            pipeline_source_mesh_path=Path(data["pipeline_source_mesh_path"]),
            pipeline_source_kind=str(data["pipeline_source_kind"]),
            profile_sec=dict(data.get("profile_sec") or {}),
            started_at_monotonic=time.monotonic(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation.to_dict(),
            "generation_config": self.generation_config.to_dict(),
            "texture_config": None if self.texture_config is None else self.texture_config.to_dict(),
            "texture": None if self.texture is None else self.texture.to_dict(),
            "pipeline_source_mesh_path": str(self.pipeline_source_mesh_path),
            "pipeline_source_kind": self.pipeline_source_kind,
            "profile_sec": self.profile_sec,
        }


def default_mesh_output_dir(prompt: str, *, root: Path = Path("code_agent/generated_meshes")) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify_prompt(prompt)
    return root / timestamp / slug


def generate_meshy_mesh_from_text(
    *,
    prompt: str,
    api_config: MeshyApiConfig,
    generation_config: MeshyGenerationConfig,
    texture_config: MeshyTextureConfig | None = None,
    repair_config: MeshRepairConfig | None = None,
) -> TextToMeshBundle:
    downloaded = download_meshy_mesh_from_text(
        prompt=prompt,
        api_config=api_config,
        generation_config=generation_config,
        texture_config=texture_config,
    )
    return process_downloaded_meshy_mesh(
        downloaded=downloaded,
        repair_config=repair_config,
    )


def download_meshy_mesh_from_text(
    *,
    prompt: str,
    api_config: MeshyApiConfig,
    generation_config: MeshyGenerationConfig,
    texture_config: MeshyTextureConfig | None = None,
) -> DownloadedMeshyAsset:
    meshy_geometry_prompt = augment_meshy_geometry_prompt(prompt)
    generation_config.prompt = meshy_geometry_prompt

    output_dir = generation_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_sec: dict[str, float] = {}
    total_start = time.monotonic()

    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(meshy_geometry_prompt.strip() + "\n", encoding="utf-8")

    client = MeshyClient(api_config)
    submit_response = time_stage(profile_sec, "submit_request", lambda: client.submit_text_to_mesh(generation_config))
    submit_response_path = output_dir / "meshy_submit_response.json"
    dump_json(submit_response, submit_response_path)

    preview_task_id = extract_preview_task_id(submit_response)
    final_response = time_stage(
        profile_sec,
        "wait_preview",
        lambda: client.wait_for_preview_completion(
            preview_task_id=preview_task_id,
            poll_interval_sec=generation_config.poll_interval_sec,
            max_wait_sec=generation_config.max_wait_sec,
        ),
    )
    final_response_path = output_dir / "meshy_final_response.json"
    dump_json(final_response, final_response_path)

    mesh_path = time_stage(
        profile_sec,
        "download_mesh",
        lambda: client.download_mesh(
            task_response=final_response,
            output_dir=output_dir,
            mesh_format=generation_config.mesh_format,
        ),
    )

    generation_result = MeshyGenerationResult(
        provider="meshy",
        prompt=meshy_geometry_prompt,
        output_dir=output_dir,
        mesh_path=mesh_path,
        prompt_path=prompt_path,
        submit_response_path=submit_response_path,
        final_response_path=final_response_path,
        metadata_path=output_dir / "metadata.json",
        preview_task_id=preview_task_id,
        final_status=str(final_response.get("status", "")),
        stage_durations_sec={
            "submit_request": profile_sec["submit_request"],
            "wait_preview": profile_sec["wait_preview"],
            "download_mesh": profile_sec["download_mesh"],
        },
        submit_response=submit_response,
        final_response=final_response,
    )

    texture_result = run_texture_pipeline(
        client=client,
        output_dir=output_dir,
        preview_task_id=preview_task_id,
        prompt=prompt,
        generation_config=generation_config,
        texture_config=texture_config,
        profile_sec=profile_sec,
    )

    pipeline_source_mesh_path, pipeline_source_kind = select_pipeline_source_mesh(
        raw_mesh_path=mesh_path,
        texture_config=texture_config,
        texture_result=texture_result,
    )

    return DownloadedMeshyAsset(
        generation=generation_result,
        generation_config=generation_config,
        texture_config=texture_config,
        texture=texture_result,
        pipeline_source_mesh_path=pipeline_source_mesh_path,
        pipeline_source_kind=pipeline_source_kind,
        profile_sec=profile_sec,
        started_at_monotonic=total_start,
    )


def process_downloaded_meshy_mesh(
    *,
    downloaded: DownloadedMeshyAsset,
    repair_config: MeshRepairConfig | None = None,
) -> TextToMeshBundle:
    generation_result = downloaded.generation
    texture_result = downloaded.texture
    output_dir = generation_result.output_dir
    profile_sec = dict(downloaded.profile_sec)

    raw_manifold_result = time_stage(
        profile_sec,
        "raw_manifold_check",
        lambda: run_mesh_manifold_check(downloaded.pipeline_source_mesh_path),
    )
    dump_json(raw_manifold_result.to_dict(), output_dir / "raw_manifold_check.json")

    repair_result, repair_attempts, final_manifold_result = run_repair_pipeline(
        mesh_path=downloaded.pipeline_source_mesh_path,
        output_dir=output_dir,
        repair_config=repair_config,
        raw_manifold_result=raw_manifold_result,
        profile_sec=profile_sec,
    )

    texture_transfer_result = None
    if (
        texture_result is not None
        and texture_result.ok
        and texture_result.textured_mesh_path is not None
        and repair_result is not None
        and repair_result.ok
    ):
        base_color_path = texture_result.texture_paths.get("base_color")
        if base_color_path is not None:
            texture_transfer_result = time_stage(
                profile_sec,
                "texture_transfer_total",
                lambda: transfer_texture_to_repaired_mesh(
                    source_mesh_path=texture_result.textured_mesh_path,
                    source_base_color_path=base_color_path,
                    target_mesh_path=repair_result.output_mesh_path,
                    output_dir=output_dir,
                    alignment_translation=repair_result.centroid_before_translation,
                ),
            )
            if (
                texture_transfer_result.ok
                and texture_transfer_result.output_mesh_path == repair_result.output_mesh_path
            ):
                final_manifold_result = time_stage(
                    profile_sec,
                    "post_texture_manifold_check",
                    lambda: run_mesh_manifold_check(repair_result.output_mesh_path),
                )

    dump_json(final_manifold_result.to_dict(), output_dir / "manifold_check.json")
    profile_sec["total"] = time.monotonic() - downloaded.started_at_monotonic
    dump_json(profile_sec, output_dir / "profile.json")

    metadata = {
        "provider": "meshy",
        "generation_config": downloaded.generation_config.to_dict(),
        "generation": generation_result.to_dict(),
        "texture_config": None if downloaded.texture_config is None else downloaded.texture_config.to_dict(),
        "pipeline_source_mesh_path": str(downloaded.pipeline_source_mesh_path),
        "pipeline_source_kind": downloaded.pipeline_source_kind,
        "raw_manifold": raw_manifold_result.to_dict(),
        "manifold": final_manifold_result.to_dict(),
        "profile_sec": profile_sec,
    }
    if texture_result is not None:
        metadata["texture"] = texture_result.to_dict()
    if repair_result is not None:
        metadata["repair"] = repair_result.to_dict()
    if texture_transfer_result is not None:
        metadata["texture_transfer"] = texture_transfer_result.to_dict()
    if repair_attempts:
        metadata["repair_attempts"] = [attempt.to_dict() for attempt in repair_attempts]
    dump_json(metadata, generation_result.metadata_path)

    return TextToMeshBundle(
        generation=generation_result,
        texture=texture_result,
        repair=repair_result,
        texture_transfer=texture_transfer_result,
        repair_attempts=tuple(repair_attempts),
        raw_manifold=raw_manifold_result,
        manifold=final_manifold_result,
        profile_sec=profile_sec,
    )


def parse_extra_payload(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    candidate = value.strip()
    if not candidate:
        return {}
    if candidate.startswith("{"):
        parsed = json.loads(candidate)
    else:
        path = Path(candidate)
        if path.exists():
            parsed = json.loads(path.read_text(encoding="utf-8"))
        else:
            parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Extra payload must be a JSON object.")
    return parsed


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))
