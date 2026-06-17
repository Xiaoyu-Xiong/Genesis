from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_agent.dataset.utils import hamming_distance_hex, now_iso, safe_relpath, short_hash, slugify
from code_agent.io_utils import dump_json, load_json_object

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
MANIFEST_VERSION = 1
CV_DUPLICATE_MIN_SCORE = 0.72
NEAR_DUPLICATE_FINGERPRINT_DISTANCE = 4
NEAR_DUPLICATE_FRAME_DISTANCE = 8
NEAR_DUPLICATE_FRAME_MATCH_FRACTION = 0.55
NEAR_DUPLICATE_HISTOGRAM_DISTANCE = 0.22
NEAR_DUPLICATE_COMPONENT_DISTANCE = 12
NEAR_DUPLICATE_COMPONENT_PHASH_DISTANCE = 16
NEAR_DUPLICATE_COMPONENT_MATCH_FRACTION = 0.50
NEAR_DUPLICATE_FOREGROUND_HISTOGRAM_DISTANCE = 0.36
CLIP_LONG_FIELD_ORDER = (
    "frame_fingerprints",
    "color_histogram",
    "foreground_component_fingerprints",
    "foreground_component_phashes",
    "foreground_color_histogram",
)
CLIP_SHORT_FIELD_ORDER = (
    "id",
    "source_video_id",
    "source_url",
    "title",
    "start_sec",
    "end_sec",
    "duration_sec",
    "clip_path",
    "clip_uri",
    "contact_sheet_path",
    "clip_sha256",
    "clip_bytes",
    "visual_fingerprint",
    "visual_signature_version",
    "visual_summary",
    "segment_reason",
    "segment_confidence",
    "case_id",
    "prompt",
    "category",
    "prompt_revisions",
    "status",
    "created_at",
    "updated_at",
    "cv_duplicate_candidates",
    "duplicate_review",
    "rejection_reason",
    "reviewed_at",
    "accept_reason",
    "duplicate_of_clip_id",
    "delete_reason",
    "multi_example",
    "truncated",
)

CLIP_CATEGORIES = ("rigid", "deformable_bodies", "cloth")


class DatasetStore:
    """Small JSON-backed index for source videos, clips, review events, and prompt style memory."""

    def __init__(self, root: Path = DEFAULT_DATA_ROOT):
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def contact_sheets_dir(self) -> Path:
        return self.root / "contact_sheets"

    @property
    def timelines_dir(self) -> Path:
        return self.root / "timelines"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def ensure_dirs(self) -> None:
        for path in (
            self.root,
            self.videos_dir,
            self.clips_dir,
            self.contact_sheets_dir,
            self.timelines_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        manifest = load_json_object(self.manifest_path)
        if manifest is None:
            return self.empty_manifest()
        return self.normalize_manifest(manifest)

    def save(self, manifest: dict[str, Any]) -> None:
        manifest = self.normalize_manifest(manifest)
        manifest["updated_at"] = now_iso()
        self.ensure_dirs()
        temp_path = self.manifest_path.with_suffix(".json.tmp")
        dump_json(manifest, temp_path)
        temp_path.replace(self.manifest_path)

    def empty_manifest(self) -> dict[str, Any]:
        timestamp = now_iso()
        return {
            "manifest_version": MANIFEST_VERSION,
            "created_at": timestamp,
            "updated_at": timestamp,
            "source_videos": [],
            "clips": [],
            "review_events": [],
            "style_memory": [],
            "review_state": {},
        }

    def normalize_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(manifest)
        normalized.setdefault("manifest_version", MANIFEST_VERSION)
        normalized.setdefault("created_at", now_iso())
        normalized.setdefault("updated_at", now_iso())
        for key in ("source_videos", "clips", "review_events", "style_memory"):
            value = normalized.get(key)
            normalized[key] = value if isinstance(value, list) else []
        if not isinstance(normalized.get("review_state"), dict):
            normalized["review_state"] = {}
        normalized["clips"] = [
            self.normalize_clip_record(clip) if isinstance(clip, dict) else clip for clip in normalized["clips"]
        ]
        return normalized

    def accepted_count(self, manifest: dict[str, Any] | None = None) -> int:
        manifest = self.load() if manifest is None else manifest
        return sum(1 for clip in manifest["clips"] if clip.get("status") == "accepted")

    def status_summary(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        manifest = self.load() if manifest is None else manifest
        statuses: dict[str, int] = {}
        for clip in manifest["clips"]:
            status = str(clip.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "data_root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "source_videos": len(manifest["source_videos"]),
            "clips": len(manifest["clips"]),
            "accepted_clips": statuses.get("accepted", 0),
            "clip_statuses": statuses,
            "review_events": len(manifest["review_events"]),
            "style_memory": len(manifest["style_memory"]),
            "updated_at": manifest.get("updated_at"),
        }

    def unique_source_id(self, manifest: dict[str, Any], seed: str) -> str:
        return self._unique_id({str(item.get("id")) for item in manifest["source_videos"]}, seed, "source")

    def unique_clip_id(self, manifest: dict[str, Any], seed: str) -> str:
        return self._unique_id({str(item.get("id")) for item in manifest["clips"]}, seed, "clip")

    def source_url_seen(self, manifest: dict[str, Any], url: str) -> bool:
        return any(
            str(source.get("url") or source.get("video_url")) == url and source.get("status") != "failed"
            for source in manifest["source_videos"]
        )

    def source_hash_seen(self, manifest: dict[str, Any], sha256: str) -> bool:
        return any(source.get("sha256") == sha256 for source in manifest["source_videos"])

    def clip_hash_seen(self, manifest: dict[str, Any], sha256: str) -> bool:
        return any(clip.get("clip_sha256") == sha256 for clip in manifest["clips"])

    def near_duplicate_clip(
        self,
        manifest: dict[str, Any],
        fingerprint: str | None = None,
        *,
        signature: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        candidates = self.duplicate_candidates(manifest, fingerprint=fingerprint, signature=signature, max_results=1)
        return candidates[0]["clip"] if candidates else None

    def duplicate_candidates(
        self,
        manifest: dict[str, Any],
        fingerprint: str | None = None,
        *,
        signature: dict[str, Any] | None = None,
        max_results: int = 3,
        exclude_clip_id: str | None = None,
    ) -> list[dict[str, Any]]:
        current = _normalize_visual_signature(fingerprint=fingerprint, signature=signature)
        if not _has_visual_signal(current):
            return []
        candidates = []
        for clip in manifest["clips"]:
            if clip.get("id") == exclude_clip_id or clip.get("status") != "accepted":
                continue
            existing = _normalize_visual_signature(
                fingerprint=clip.get("visual_fingerprint") if isinstance(clip.get("visual_fingerprint"), str) else None,
                signature=clip,
            )
            evaluation = _visual_duplicate_evaluation(current, existing)
            if evaluation["is_candidate"]:
                candidates.append(
                    {
                        "clip": clip,
                        "clip_id": clip.get("id"),
                        "score": evaluation["score"],
                        "reason": evaluation["reason"],
                        "metrics": evaluation["metrics"],
                    }
                )
        candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return candidates[:max_results]

    def add_source_video(self, manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        manifest["source_videos"].append(record)
        return record

    def add_clip(self, manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        manifest["clips"].append(record)
        return record

    def accept_clip(self, clip_id: str, *, reason: str | None = None) -> dict[str, Any]:
        manifest = self.load()
        clip = self._find_clip(manifest, clip_id)
        before_status = clip.get("status")
        clip["status"] = "accepted"
        clip["accept_reason"] = reason or "accepted"
        clip["reviewed_at"] = now_iso()
        clip["updated_at"] = now_iso()
        event = self._review_event(
            clip_id,
            "accept",
            reason=reason or "accepted",
            before_status=before_status,
            before_prompt=clip.get("prompt"),
        )
        manifest["review_events"].append(event)
        self.save(manifest)
        return event

    def reject_clip(self, clip_id: str, *, reason: str, avoid_similarity_note: str | None = None) -> dict[str, Any]:
        manifest = self.load()
        clip = self._find_clip(manifest, clip_id)
        before_status = clip.get("status")
        clip["status"] = "rejected"
        clip["rejection_reason"] = reason
        clip["updated_at"] = now_iso()
        event = self._review_event(
            clip_id,
            "reject",
            reason=reason,
            before_status=before_status,
            before_prompt=clip.get("prompt"),
            avoid_similarity_note=avoid_similarity_note or reason,
        )
        manifest["review_events"].append(event)
        self.save(manifest)
        return event

    def edit_clip(self, clip_id: str, *, prompt: str, reason: str | None = None) -> dict[str, Any]:
        case_id, case_prompt = split_case_prompt(prompt)
        manifest = self.load()
        clip = self._find_clip(manifest, clip_id)
        before_prompt = clip.get("prompt")
        before_case_id = clip.get("case_id")
        before_status = clip.get("status")
        revision = {
            "timestamp": now_iso(),
            "before_case_id": before_case_id,
            "before_prompt": before_prompt,
            "after_case_id": case_id,
            "after_prompt": case_prompt,
            "reason": reason,
        }
        clip.setdefault("prompt_revisions", []).append(revision)
        clip["case_id"] = case_id
        clip["prompt"] = case_prompt
        clip["status"] = "accepted"
        clip["updated_at"] = now_iso()
        event = self._review_event(
            clip_id,
            "edit",
            reason=reason,
            before_status=before_status,
            before_prompt=before_prompt,
            after_prompt=case_prompt,
        )
        manifest["review_events"].append(event)
        manifest["style_memory"].append(
            {
                "clip_id": clip_id,
                "case_id": case_id,
                "prompt": case_prompt,
                "reason": reason,
                "timestamp": now_iso(),
            }
        )
        self.save(manifest)
        return event

    def set_clip_category(self, clip_id: str, *, category: str, reason: str | None = None) -> dict[str, Any]:
        normalized_category = normalize_clip_category(category)
        manifest = self.load()
        clip = self._find_clip(manifest, clip_id)
        before_category = clip.get("category")
        before_status = clip.get("status")
        clip["category"] = normalized_category
        clip["updated_at"] = now_iso()
        event = self._review_event(
            clip_id,
            "set_category",
            reason=reason or "human category label",
            before_status=before_status,
            before_prompt=clip.get("prompt"),
        )
        event["before_category"] = before_category
        event["after_category"] = normalized_category
        manifest["review_events"].append(event)
        self.save(manifest)
        return event

    def delete_duplicate_clip(
        self,
        clip_id: str,
        *,
        duplicate_of_clip_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        event = self.delete_clip_without_negative_memory(
            clip_id,
            status="duplicate_deleted",
            event_type="delete_duplicate",
            reason=reason or "duplicate clip removed from active dataset",
            metadata={"duplicate_of_clip_id": duplicate_of_clip_id},
        )
        return event

    def delete_multi_example_clip(self, clip_id: str, *, reason: str | None = None) -> dict[str, Any]:
        return self.delete_clip_without_negative_memory(
            clip_id,
            status="multi_example_deleted",
            event_type="delete_multi_example",
            reason=reason or "clip contains multiple independent examples",
            metadata={"multi_example": True},
        )

    def delete_truncated_clip(self, clip_id: str, *, reason: str | None = None) -> dict[str, Any]:
        return self.delete_clip_without_negative_memory(
            clip_id,
            status="truncated_deleted",
            event_type="delete_truncated",
            reason=reason or "clip is truncated or incomplete",
            metadata={"truncated": True},
        )

    def delete_clip_without_negative_memory(
        self,
        clip_id: str,
        *,
        status: str,
        event_type: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = self.load()
        clip = self._find_clip(manifest, clip_id)
        before_status = clip.get("status")
        metadata = metadata or {}
        clip["status"] = status
        clip["delete_reason"] = reason
        for key, value in metadata.items():
            clip[key] = value
        clip["updated_at"] = now_iso()
        event = self._review_event(
            clip_id,
            event_type,
            reason=reason,
            before_status=before_status,
            before_prompt=clip.get("prompt"),
        )
        event.update(metadata)
        event["negative_memory"] = False
        manifest["review_events"].append(event)
        self.save(manifest)
        return event

    def record_review_position(
        self,
        clip_id: str,
        *,
        manifest_index: int,
        note: str | None = None,
    ) -> dict[str, Any]:
        manifest = self.load()
        state = {
            "last_reviewed_clip_id": clip_id,
            "last_reviewed_manifest_index": manifest_index,
            "updated_at": now_iso(),
        }
        if note:
            state["note"] = note
        manifest["review_state"] = state
        self.save(manifest)
        return state

    def export_cases(self, out_path: Path) -> int:
        manifest = self.load()
        lines = []
        for clip in sorted(manifest["clips"], key=lambda item: str(item.get("id"))):
            if clip.get("status") != "accepted":
                continue
            case_id = str(clip.get("case_id") or clip.get("id"))
            prompt = str(clip.get("prompt") or "").strip()
            if prompt:
                lines.append(f"{case_id}|{prompt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    def relpath(self, path: Path) -> str:
        return safe_relpath(path, self.root)

    def abspath(self, path_text: str) -> Path:
        path = Path(path_text)
        return path if path.is_absolute() else self.root / path

    def file_uri(self, path: Path) -> str:
        return path.resolve().as_uri()

    def add_clip_open_reference(self, clip: dict[str, Any]) -> None:
        clip_path = clip.get("clip_path")
        if isinstance(clip_path, str) and clip_path:
            clip["clip_uri"] = self.file_uri(self.abspath(clip_path))

    def normalize_clip_record(self, clip: dict[str, Any]) -> dict[str, Any]:
        self.add_clip_open_reference(clip)
        ordered: dict[str, Any] = {}
        for key in CLIP_SHORT_FIELD_ORDER:
            if key in clip:
                ordered[key] = clip[key]
        for key, value in clip.items():
            if key not in ordered and key not in CLIP_LONG_FIELD_ORDER:
                ordered[key] = value
        for key in CLIP_LONG_FIELD_ORDER:
            if key in clip:
                ordered[key] = clip[key]
        return ordered

    def _find_clip(self, manifest: dict[str, Any], clip_id: str) -> dict[str, Any]:
        for clip in manifest["clips"]:
            if clip.get("id") == clip_id:
                return clip
        raise KeyError(f"Unknown clip id: {clip_id}")

    def _review_event(
        self,
        clip_id: str,
        event_type: str,
        *,
        reason: str | None,
        before_status: object = None,
        before_prompt: object = None,
        after_prompt: object = None,
        avoid_similarity_note: str | None = None,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        return {
            "event_id": f"{event_type}_{slugify(clip_id)}_{short_hash(timestamp)}",
            "clip_id": clip_id,
            "type": event_type,
            "timestamp": timestamp,
            "reason": reason,
            "before_status": before_status,
            "before_prompt": before_prompt,
            "after_prompt": after_prompt,
            "avoid_similarity_note": avoid_similarity_note,
        }

    def _unique_id(self, existing: set[str], seed: str, fallback: str) -> str:
        base = slugify(seed, fallback=fallback)
        if base not in existing:
            return base
        suffix = short_hash(seed)
        candidate = f"{base}_{suffix}"
        index = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}_{index}"
            index += 1
        return candidate


def split_case_prompt(text: str) -> tuple[str, str]:
    if "|" in text:
        case_id, prompt = text.split("|", 1)
        case_id = slugify(case_id, fallback="case")
        prompt = prompt.strip()
    else:
        prompt = text.strip()
        case_id = slugify(prompt[:64], fallback="case")
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    return case_id, prompt


def normalize_clip_category(category: str) -> str:
    normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rigid_body": "rigid",
        "rigid_bodies": "rigid",
        "deformable": "deformable_bodies",
        "deformable_body": "deformable_bodies",
        "deformable_bodies": "deformable_bodies",
        "soft_body": "deformable_bodies",
        "soft_bodies": "deformable_bodies",
        "cloth": "cloth",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CLIP_CATEGORIES:
        valid = ", ".join(CLIP_CATEGORIES)
        raise ValueError(f"Unknown clip category {category!r}. Expected one of: {valid}.")
    return normalized


def _normalize_visual_signature(
    *,
    fingerprint: str | None,
    signature: dict[str, Any] | None,
) -> dict[str, Any]:
    signature = signature or {}
    visual_fingerprint = signature.get("visual_fingerprint")
    if not isinstance(visual_fingerprint, str):
        visual_fingerprint = fingerprint
    frame_fingerprints = signature.get("frame_fingerprints")
    if not isinstance(frame_fingerprints, list):
        frame_fingerprints = []
    frame_fingerprints = [item for item in frame_fingerprints if isinstance(item, str)]
    histogram = signature.get("color_histogram")
    if not isinstance(histogram, list):
        histogram = []
    histogram = [float(item) for item in histogram if isinstance(item, int | float)]
    foreground_component_fingerprints = signature.get("foreground_component_fingerprints")
    if not isinstance(foreground_component_fingerprints, list):
        foreground_component_fingerprints = []
    foreground_component_fingerprints = [item for item in foreground_component_fingerprints if isinstance(item, str)]
    foreground_component_phashes = signature.get("foreground_component_phashes")
    if not isinstance(foreground_component_phashes, list):
        foreground_component_phashes = []
    foreground_component_phashes = [item for item in foreground_component_phashes if isinstance(item, str)]
    foreground_histogram = signature.get("foreground_color_histogram")
    if not isinstance(foreground_histogram, list):
        foreground_histogram = []
    foreground_histogram = [float(item) for item in foreground_histogram if isinstance(item, int | float)]
    return {
        "visual_fingerprint": visual_fingerprint,
        "frame_fingerprints": frame_fingerprints,
        "color_histogram": histogram,
        "foreground_component_fingerprints": foreground_component_fingerprints,
        "foreground_component_phashes": foreground_component_phashes,
        "foreground_color_histogram": foreground_histogram,
    }


def _visual_signatures_are_near_duplicates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(_visual_duplicate_evaluation(left, right)["is_candidate"])


def _visual_duplicate_evaluation(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    score = 0.0

    distance = hamming_distance_hex(left.get("visual_fingerprint"), right.get("visual_fingerprint"))
    metrics["aggregate_hamming"] = distance
    if distance is not None and distance <= NEAR_DUPLICATE_FINGERPRINT_DISTANCE:
        score = max(score, 1.0)
        reasons.append("aggregate_fingerprint")

    left_frames = left.get("frame_fingerprints") or []
    right_frames = right.get("frame_fingerprints") or []
    if left_frames and right_frames:
        match_fraction = _frame_hash_match_fraction(left_frames, right_frames)
        histogram_distance = _histogram_l1_distance(
            left.get("color_histogram") or [], right.get("color_histogram") or []
        )
        metrics["frame_match_fraction"] = match_fraction
        metrics["color_histogram_l1"] = histogram_distance
        if (
            match_fraction >= NEAR_DUPLICATE_FRAME_MATCH_FRACTION
            and histogram_distance is not None
            and histogram_distance <= NEAR_DUPLICATE_HISTOGRAM_DISTANCE
        ):
            score = max(score, 0.86 + min(0.08, match_fraction * 0.08))
            reasons.append("full_frame_hashes")

    foreground_match = max(
        _bidirectional_hash_match_score(
            left.get("foreground_component_fingerprints") or [],
            right.get("foreground_component_fingerprints") or [],
            max_distance=NEAR_DUPLICATE_COMPONENT_DISTANCE,
        ),
        _bidirectional_hash_match_score(
            left.get("foreground_component_phashes") or [],
            right.get("foreground_component_phashes") or [],
            max_distance=NEAR_DUPLICATE_COMPONENT_PHASH_DISTANCE,
        ),
    )
    foreground_histogram_distance = _histogram_l1_distance(
        left.get("foreground_color_histogram") or [], right.get("foreground_color_histogram") or []
    )
    metrics["foreground_component_match_fraction"] = foreground_match
    metrics["foreground_histogram_l1"] = foreground_histogram_distance
    if (
        foreground_match >= NEAR_DUPLICATE_COMPONENT_MATCH_FRACTION
        and foreground_histogram_distance is not None
        and foreground_histogram_distance <= NEAR_DUPLICATE_FOREGROUND_HISTOGRAM_DISTANCE
    ):
        histogram_bonus = (
            NEAR_DUPLICATE_FOREGROUND_HISTOGRAM_DISTANCE - foreground_histogram_distance
        ) / NEAR_DUPLICATE_FOREGROUND_HISTOGRAM_DISTANCE
        score = max(score, 0.72 + min(0.18, foreground_match * 0.18) + max(0.0, histogram_bonus) * 0.08)
        reasons.append("foreground_components")
    elif foreground_match >= 0.75 and (foreground_histogram_distance is None or foreground_histogram_distance <= 0.45):
        score = max(score, 0.72)
        reasons.append("strong_foreground_components")

    return {
        "is_candidate": score >= CV_DUPLICATE_MIN_SCORE,
        "score": round(score, 4),
        "reason": "+".join(reasons) if reasons else "no_cv_duplicate_signal",
        "metrics": metrics,
    }


def _frame_hash_match_fraction(
    left_frames: list[str],
    right_frames: list[str],
    *,
    max_distance: int = NEAR_DUPLICATE_FRAME_DISTANCE,
) -> float:
    if not left_frames or not right_frames:
        return 0.0
    matches = 0
    for left_hash in left_frames:
        distances = [
            distance
            for right_hash in right_frames
            if (distance := hamming_distance_hex(left_hash, right_hash)) is not None
        ]
        if distances and min(distances) <= max_distance:
            matches += 1
    return matches / len(left_frames)


def _bidirectional_hash_match_score(left_hashes: list[str], right_hashes: list[str], *, max_distance: int) -> float:
    return max(
        _frame_hash_match_fraction(left_hashes, right_hashes, max_distance=max_distance),
        _frame_hash_match_fraction(right_hashes, left_hashes, max_distance=max_distance),
    )


def _has_visual_signal(signature: dict[str, Any]) -> bool:
    return any(
        signature.get(key)
        for key in (
            "visual_fingerprint",
            "frame_fingerprints",
            "foreground_component_fingerprints",
            "foreground_component_phashes",
        )
    )


def _histogram_l1_distance(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return sum(abs(left_value - right_value) for left_value, right_value in zip(left, right, strict=False))


def load_manifest_text(path: Path) -> str:
    if not path.exists():
        return "{}"
    try:
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
    except (OSError, json.JSONDecodeError):
        return "{}"
