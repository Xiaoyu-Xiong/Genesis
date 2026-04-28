from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MESH_FORMAT_VALUES = ("obj", "glb", "stl")
MESHY_AI_MODEL_VALUES = ("meshy-5", "meshy-6", "latest")
MESHY_ART_STYLE_VALUES = ("realistic", "sculpture")
MESHY_TOPOLOGY_VALUES = ("triangle", "quad")
MESHY_SYMMETRY_VALUES = ("off", "auto", "on")
MESHY_TEXTURE_MAP_KEYS = ("base_color", "metallic", "normal", "roughness")


class MeshyRequestError(RuntimeError):
    pass


@dataclass(slots=True)
class MeshyApiConfig:
    api_key: str
    base_url: str = "https://api.meshy.ai"
    text_to_3d_path: str = "/openapi/v2/text-to-3d"
    timeout_sec: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "MESHY_API_KEY",
        base_url_env: str = "MESHY_API_BASE_URL",
        timeout_sec: float = 120.0,
    ) -> "MeshyApiConfig":
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise MeshyRequestError(
                f"Missing Meshy API key env `{api_key_env}`. "
                "Generate an API key in Meshy and export it before running the mesh generator."
            )
        base_url = os.getenv(base_url_env, "https://api.meshy.ai").rstrip("/")
        return cls(api_key=api_key, base_url=base_url, timeout_sec=timeout_sec)

    def auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = "<redacted>"
        return data


@dataclass(slots=True)
class MeshyGenerationConfig:
    prompt: str
    output_dir: Path
    mesh_format: str = "obj"
    ai_model: str = "latest"
    art_style: str = "realistic"
    should_remesh: bool = False
    topology: str = "triangle"
    target_polycount: int | None = None
    symmetry_mode: str = "auto"
    moderation: bool = False
    negative_prompt: str | None = None
    auto_size: bool = False
    origin_at: str | None = None
    poll_interval_sec: float = 2.0
    max_wait_sec: float = 300.0
    extra_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mesh_format not in MESH_FORMAT_VALUES:
            allowed = ", ".join(MESH_FORMAT_VALUES)
            raise ValueError(f"Unsupported mesh_format `{self.mesh_format}`. Expected one of: {allowed}.")
        if self.ai_model not in MESHY_AI_MODEL_VALUES:
            allowed = ", ".join(MESHY_AI_MODEL_VALUES)
            raise ValueError(f"Unsupported ai_model `{self.ai_model}`. Expected one of: {allowed}.")
        if self.art_style not in MESHY_ART_STYLE_VALUES:
            allowed = ", ".join(MESHY_ART_STYLE_VALUES)
            raise ValueError(f"Unsupported art_style `{self.art_style}`. Expected one of: {allowed}.")
        if self.topology not in MESHY_TOPOLOGY_VALUES:
            allowed = ", ".join(MESHY_TOPOLOGY_VALUES)
            raise ValueError(f"Unsupported topology `{self.topology}`. Expected one of: {allowed}.")
        if self.symmetry_mode not in MESHY_SYMMETRY_VALUES:
            allowed = ", ".join(MESHY_SYMMETRY_VALUES)
            raise ValueError(f"Unsupported symmetry_mode `{self.symmetry_mode}`. Expected one of: {allowed}.")
        if self.target_polycount is not None and self.target_polycount <= 0:
            raise ValueError("`target_polycount` must be > 0 when provided.")
        if self.origin_at is not None and self.origin_at not in {"bottom", "center"}:
            raise ValueError("`origin_at` must be one of: bottom, center.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(slots=True)
class MeshyTextureConfig:
    enabled: bool = False
    texture_prompt: str | None = None
    ai_model: str | None = None
    enable_pbr: bool = False
    remove_lighting: bool = True

    def __post_init__(self) -> None:
        if self.ai_model is not None and self.ai_model not in MESHY_AI_MODEL_VALUES:
            allowed = ", ".join(MESHY_AI_MODEL_VALUES)
            raise ValueError(f"Unsupported texture ai_model `{self.ai_model}`. Expected one of: {allowed}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MeshRepairConfig:
    component_count_face_cap: int = 100000
    min_component_faces: int = 100
    max_repair_attempts: int = 4
    merge_vertices: bool = True
    merge_digits_vertex: int | None = 6
    fix_normals: bool = True
    process_validate: bool = True
    keep_largest_component: bool = True
    ftetwild_edge_length_fac: float = 0.05
    ftetwild_edge_length_abs: float | None = None
    ftetwild_optimize: bool = True
    ftetwild_simplify: bool = True
    ftetwild_epsilon: float = 1e-3
    ftetwild_stop_energy: float = 10.0
    ftetwild_coarsen: bool = False
    ftetwild_num_threads: int = 0
    ftetwild_num_opt_iter: int = 80
    ftetwild_quiet: bool = True
    ftetwild_disable_filtering: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MeshyGenerationResult:
    provider: str
    prompt: str
    output_dir: Path
    mesh_path: Path
    prompt_path: Path
    submit_response_path: Path
    final_response_path: Path
    metadata_path: Path
    preview_task_id: str
    final_status: str
    submit_response: dict[str, Any]
    final_response: dict[str, Any]
    stage_durations_sec: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "prompt": self.prompt,
            "output_dir": str(self.output_dir),
            "mesh_path": str(self.mesh_path),
            "prompt_path": str(self.prompt_path),
            "submit_response_path": str(self.submit_response_path),
            "final_response_path": str(self.final_response_path),
            "metadata_path": str(self.metadata_path),
            "preview_task_id": self.preview_task_id,
            "final_status": self.final_status,
            "stage_durations_sec": self.stage_durations_sec,
        }


@dataclass(slots=True)
class MeshyTextureResult:
    requested: bool
    ok: bool
    prompt: str
    output_dir: Path
    preview_task_id: str
    refine_task_id: str | None
    submit_response_path: Path | None
    final_response_path: Path | None
    textured_mesh_path: Path | None
    textured_mtl_path: Path | None
    texture_paths: dict[str, Path]
    ai_model: str | None
    enable_pbr: bool
    remove_lighting: bool
    final_status: str | None
    stage_durations_sec: dict[str, float] = field(default_factory=dict)
    submit_response: dict[str, Any] | None = None
    final_response: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "ok": self.ok,
            "prompt": self.prompt,
            "output_dir": str(self.output_dir),
            "preview_task_id": self.preview_task_id,
            "refine_task_id": self.refine_task_id,
            "submit_response_path": None if self.submit_response_path is None else str(self.submit_response_path),
            "final_response_path": None if self.final_response_path is None else str(self.final_response_path),
            "textured_mesh_path": None if self.textured_mesh_path is None else str(self.textured_mesh_path),
            "textured_mtl_path": None if self.textured_mtl_path is None else str(self.textured_mtl_path),
            "texture_paths": {key: str(path) for key, path in sorted(self.texture_paths.items())},
            "ai_model": self.ai_model,
            "enable_pbr": self.enable_pbr,
            "remove_lighting": self.remove_lighting,
            "final_status": self.final_status,
            "stage_durations_sec": self.stage_durations_sec,
            "error": self.error,
        }


@dataclass(slots=True)
class MeshRepairResult:
    ok: bool
    input_mesh_path: Path
    output_mesh_path: Path
    attempt_index: int
    strategy_name: str
    operations: tuple[str, ...]
    vertex_count_before: int
    face_count_before: int
    component_count_before: int
    vertex_count_after: int
    face_count_after: int
    component_count_after: int
    centroid_before_translation: tuple[float, float, float] | None = None
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None
    bbox_size: tuple[float, float, float] | None = None
    centroid_at_origin: bool = False
    config_snapshot: dict[str, Any] | None = None
    stage_durations_sec: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "input_mesh_path": str(self.input_mesh_path),
            "output_mesh_path": str(self.output_mesh_path),
            "attempt_index": self.attempt_index,
            "strategy_name": self.strategy_name,
            "operations": list(self.operations),
            "vertex_count_before": self.vertex_count_before,
            "face_count_before": self.face_count_before,
            "component_count_before": self.component_count_before,
            "vertex_count_after": self.vertex_count_after,
            "face_count_after": self.face_count_after,
            "component_count_after": self.component_count_after,
            "centroid_before_translation": (
                list(self.centroid_before_translation) if self.centroid_before_translation is not None else None
            ),
            "bbox_min": list(self.bbox_min) if self.bbox_min is not None else None,
            "bbox_max": list(self.bbox_max) if self.bbox_max is not None else None,
            "bbox_size": list(self.bbox_size) if self.bbox_size is not None else None,
            "centroid_at_origin": self.centroid_at_origin,
            "config_snapshot": self.config_snapshot,
            "stage_durations_sec": self.stage_durations_sec,
            "error": self.error,
        }


@dataclass(slots=True)
class MeshTextureTransferResult:
    ok: bool
    source_mesh_path: Path
    source_base_color_path: Path
    target_mesh_path: Path
    output_mesh_path: Path | None
    output_mtl_path: Path | None
    output_texture_path: Path | None
    alignment_translation: tuple[float, float, float] | None = None
    source_texture_size: tuple[int, int] | None = None
    parameterization_filter: str | None = None
    transfer_filter: str | None = None
    debug_dir: Path | None = None
    stage_durations_sec: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_mesh_path": str(self.source_mesh_path),
            "source_base_color_path": str(self.source_base_color_path),
            "target_mesh_path": str(self.target_mesh_path),
            "output_mesh_path": None if self.output_mesh_path is None else str(self.output_mesh_path),
            "output_mtl_path": None if self.output_mtl_path is None else str(self.output_mtl_path),
            "output_texture_path": None if self.output_texture_path is None else str(self.output_texture_path),
            "alignment_translation": list(self.alignment_translation) if self.alignment_translation is not None else None,
            "source_texture_size": list(self.source_texture_size) if self.source_texture_size is not None else None,
            "parameterization_filter": self.parameterization_filter,
            "transfer_filter": self.transfer_filter,
            "debug_dir": None if self.debug_dir is None else str(self.debug_dir),
            "stage_durations_sec": self.stage_durations_sec,
            "error": self.error,
        }


@dataclass(slots=True)
class MeshManifoldCheckResult:
    ok: bool
    mesh_path: Path
    vertex_count: int
    face_count: int
    component_count: int
    is_watertight: bool
    is_winding_consistent: bool
    volume: float | None
    tetgen_ready: bool | None = None
    tetgen_message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mesh_path": str(self.mesh_path),
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "component_count": self.component_count,
            "is_watertight": self.is_watertight,
            "is_winding_consistent": self.is_winding_consistent,
            "volume": self.volume,
            "tetgen_ready": self.tetgen_ready,
            "tetgen_message": self.tetgen_message,
            "error": self.error,
        }


@dataclass(slots=True)
class TextToMeshBundle:
    generation: MeshyGenerationResult
    texture: MeshyTextureResult | None = None
    repair: MeshRepairResult | None = None
    texture_transfer: MeshTextureTransferResult | None = None
    repair_attempts: tuple[MeshRepairResult, ...] = ()
    raw_manifold: MeshManifoldCheckResult | None = None
    manifold: MeshManifoldCheckResult | None = None
    profile_sec: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"generation": self.generation.to_dict()}
        if self.texture is not None:
            data["texture"] = self.texture.to_dict()
        if self.repair is not None:
            data["repair"] = self.repair.to_dict()
        if self.texture_transfer is not None:
            data["texture_transfer"] = self.texture_transfer.to_dict()
        if self.repair_attempts:
            data["repair_attempts"] = [item.to_dict() for item in self.repair_attempts]
        if self.raw_manifold is not None:
            data["raw_manifold"] = self.raw_manifold.to_dict()
        if self.manifold is not None:
            data["manifold"] = self.manifold.to_dict()
        if self.profile_sec is not None:
            data["profile_sec"] = self.profile_sec
        return data
