from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable
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
    "category_source",
    "split",
    "split_source",
    "split_group_id",
    "split_group_title",
    "split_assigned_at",
    "trained",
    "trained_at",
    "trained_run_id",
    "training_history",
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
DEPRECATED_CLIP_FIELDS = (
    "prompt_from_paper",
    "prompt_from_paper_revisions",
)

CLIP_CATEGORIES = ("rigid", "deformable_bodies", "cloth")
PERMANENT_DATASET_SPLITS = ("train", "test")
TMP_DATASET_SPLITS = ("train-tmp", "test-tmp")
DATASET_SPLITS = (*PERMANENT_DATASET_SPLITS, *TMP_DATASET_SPLITS)


class DatasetStore:
    """Small JSON-backed index for source videos, clips, review events, and prompt style memory."""

    def __init__(self, root: Path = DEFAULT_DATA_ROOT):
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".manifest.lock"

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
        return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, Any]:
        manifest = load_json_object(self.manifest_path)
        if manifest is None:
            return self.empty_manifest()
        return self.normalize_manifest(manifest)

    def save(self, manifest: dict[str, Any]) -> None:
        with self.manifest_lock():
            self._save_unlocked(manifest)

    def save_merged(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Merge a long-lived in-memory manifest into the latest on-disk manifest.

        Dataset builds hold a manifest object while slow Codex/download/ffmpeg work happens. Before writing, this method
        locks the manifest, reloads the current file, and merges records by stable ids so review/train updates made by
        other processes are not overwritten by the build's stale copy.
        """

        with self.manifest_lock():
            current = self._load_unlocked()
            merged = merge_manifests(current, manifest)
            self._save_unlocked(merged)
        manifest.clear()
        manifest.update(merged)
        return manifest

    def update_manifest(self, updater: Callable[[dict[str, Any]], Any]) -> Any:
        with self.manifest_lock():
            manifest = self._load_unlocked()
            result = updater(manifest)
            self._save_unlocked(manifest)
            return result

    @contextlib.contextmanager
    def manifest_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_unlocked(self, manifest: dict[str, Any]) -> None:
        manifest = self.normalize_manifest(manifest)
        manifest["updated_at"] = now_iso()
        self.ensure_dirs()
        temp_path = self.manifest_path.with_suffix(f".json.{os.getpid()}.tmp")
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
        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
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
            return event

        return self.update_manifest(updater)

    def reject_clip(self, clip_id: str, *, reason: str, avoid_similarity_note: str | None = None) -> dict[str, Any]:
        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
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
            return event

        return self.update_manifest(updater)

    def edit_clip(self, clip_id: str, *, prompt: str, reason: str | None = None) -> dict[str, Any]:
        case_id, case_prompt = split_case_prompt(prompt)

        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
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
            return event

        return self.update_manifest(updater)

    def set_clip_category(self, clip_id: str, *, category: str, reason: str | None = None) -> dict[str, Any]:
        normalized_category = normalize_clip_category(category)

        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
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
            clip["category_source"] = "human"
            manifest["review_events"].append(event)
            return event

        return self.update_manifest(updater)

    def set_clip_split(self, clip_id: str, *, split: str, reason: str | None = None) -> dict[str, Any]:
        normalized_split = normalize_dataset_split(split)

        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
            clip = self._find_clip(manifest, clip_id)
            before_split = clip.get("split")
            before_status = clip.get("status")
            clip["split"] = normalized_split
            clip["split_source"] = "human"
            clip["split_assigned_at"] = now_iso()
            clip["updated_at"] = now_iso()
            event = self._review_event(
                clip_id,
                "set_split",
                reason=reason or "human split label",
                before_status=before_status,
                before_prompt=clip.get("prompt"),
            )
            event["before_split"] = before_split
            event["after_split"] = normalized_split
            manifest["review_events"].append(event)
            return event

        return self.update_manifest(updater)

    def backfill_auto_categories(self, manifest: dict[str, Any]) -> int:
        changed = 0
        for clip in manifest.get("clips", []):
            if not isinstance(clip, dict) or clip.get("status") != "accepted":
                continue
            if clip.get("category") in CLIP_CATEGORIES:
                continue
            category = infer_clip_category(clip)
            clip["category"] = category
            clip["category_source"] = "auto_prompt_heuristic"
            clip["updated_at"] = now_iso()
            changed += 1
        return changed

    def assign_train_test_splits(
        self,
        manifest: dict[str, Any],
        *,
        test_fraction: float = 0.30,
        temporary: bool = True,
        include_unset: bool = True,
        overwrite_permanent: bool = False,
    ) -> dict[str, Any]:
        self.backfill_auto_categories(manifest)
        sources_by_id = {
            str(source.get("id")): source
            for source in manifest.get("source_videos", [])
            if isinstance(source, dict) and source.get("id") is not None
        }
        accepted = [
            clip for clip in manifest.get("clips", []) if isinstance(clip, dict) and clip.get("status") == "accepted"
        ]
        if temporary:
            target_train, target_test = "train-tmp", "test-tmp"
            eligible = [
                clip
                for clip in accepted
                if clip.get("split") in TMP_DATASET_SPLITS
                or (include_unset and clip.get("split") not in DATASET_SPLITS)
            ]
            split_source = "auto_tmp_paper_grouped"
        else:
            target_train, target_test = "train", "test"
            eligible = [
                clip
                for clip in accepted
                if overwrite_permanent
                or clip.get("split") in TMP_DATASET_SPLITS
                or (include_unset and clip.get("split") not in DATASET_SPLITS)
            ]
            split_source = "auto_final_paper_grouped"

        groups: dict[str, dict[str, Any]] = {}
        for clip in eligible:
            source = sources_by_id.get(str(clip.get("source_video_id")))
            group_id, group_title = paper_group_for_clip(clip, source)
            group = groups.setdefault(group_id, {"id": group_id, "title": group_title, "clips": []})
            group["clips"].append(clip)

        target_by_category = {
            category: round(
                sum(1 for clip in eligible if clip.get("category") == category) * max(0.0, min(test_fraction, 1.0))
            )
            for category in CLIP_CATEGORIES
        }
        current_test = Counter()

        assignable_groups = []
        for group in groups.values():
            counts = Counter(clip.get("category") for clip in group["clips"])
            group["category_counts"] = counts
            assignable_groups.append(group)

        selected_test = _choose_test_group_ids(
            assignable_groups,
            current_test=current_test,
            target_by_category=target_by_category,
            test_fraction=test_fraction,
            accepted_count=len(eligible),
        )

        timestamp = now_iso()
        changed = 0
        for group in groups.values():
            target_split = target_test if group["id"] in selected_test else target_train
            for clip in group["clips"]:
                if (
                    clip.get("split") != target_split
                    or clip.get("split_source") != split_source
                    or clip.get("split_group_id") != group["id"]
                ):
                    clip["split"] = target_split
                    clip["split_source"] = split_source
                    clip["split_group_id"] = group["id"]
                    clip["split_group_title"] = group["title"]
                    clip["split_assigned_at"] = timestamp
                    clip["updated_at"] = timestamp
                    if target_split == "train":
                        clip.setdefault("trained", False)
                    changed += 1

        return self.split_summary(manifest) | {
            "changed": changed,
            "temporary": temporary,
            "eligible": len(eligible),
            "target_test_by_category": target_by_category,
        }

    def finalize_tmp_splits(self, manifest: dict[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        changed = 0
        for clip in manifest.get("clips", []):
            if not isinstance(clip, dict) or clip.get("status") != "accepted":
                continue
            before_split = clip.get("split")
            if before_split == "train-tmp":
                clip["split"] = "train"
                clip.setdefault("trained", False)
            elif before_split == "test-tmp":
                clip["split"] = "test"
            else:
                continue
            clip["split_source"] = "tmp_promoted"
            clip["split_assigned_at"] = timestamp
            clip["updated_at"] = timestamp
            changed += 1
        return self.split_summary(manifest) | {"changed": changed}

    def drop_paper_prompts(self, manifest: dict[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        changed = 0
        for clip in manifest.get("clips", []):
            if not isinstance(clip, dict):
                continue
            removed = False
            for key in DEPRECATED_CLIP_FIELDS:
                if key in clip:
                    clip.pop(key, None)
                    removed = True
            if removed:
                clip["updated_at"] = timestamp
                changed += 1
        return {"changed": changed, "deprecated_fields": list(DEPRECATED_CLIP_FIELDS)}

    def split_summary(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        manifest = self.load() if manifest is None else manifest
        sources_by_id = {
            str(source.get("id")): source
            for source in manifest.get("source_videos", [])
            if isinstance(source, dict) and source.get("id") is not None
        }
        accepted = [
            clip for clip in manifest.get("clips", []) if isinstance(clip, dict) and clip.get("status") == "accepted"
        ]
        counts: dict[str, dict[str, int]] = {
            split: {category: 0 for category in CLIP_CATEGORIES} | {"unknown": 0, "total": 0}
            for split in DATASET_SPLITS
        }
        counts["unset"] = {category: 0 for category in CLIP_CATEGORIES} | {"unknown": 0, "total": 0}
        training = {"train_total": 0, "trained": 0, "untrained": 0}
        groups: dict[str, dict[str, Any]] = {}
        for clip in accepted:
            split = clip.get("split") if clip.get("split") in DATASET_SPLITS else "unset"
            category = clip.get("category") if clip.get("category") in CLIP_CATEGORIES else "unknown"
            counts[split][category] += 1
            counts[split]["total"] += 1
            if split == "train":
                training["train_total"] += 1
                if clip.get("trained") is True:
                    training["trained"] += 1
                else:
                    training["untrained"] += 1
            source = sources_by_id.get(str(clip.get("source_video_id")))
            group_id, group_title = paper_group_for_clip(clip, source)
            group = groups.setdefault(group_id, {"title": group_title, "splits": Counter(), "clips": []})
            group["splits"].update([split])
            group["clips"].append(clip)

        complete_test_groups = {
            group_id: group
            for group_id, group in groups.items()
            if group["splits"] == Counter({"test": len(group["clips"])})
        }
        mixed_groups = {
            group_id: group
            for group_id, group in groups.items()
            if len([split for split, count in group["splits"].items() if count]) > 1
        }
        return {
            "accepted": len(accepted),
            "counts": counts,
            "paper_groups": len(groups),
            "complete_test_papers": len(complete_test_groups),
            "complete_test_clips": sum(len(group["clips"]) for group in complete_test_groups.values()),
            "mixed_papers": len(mixed_groups),
            "training": training,
        }

    def make_run_batch(
        self,
        *,
        mode: str,
        count: int,
        out_path: Path,
        seed: int | None = None,
        mark_trained: bool = True,
    ) -> dict[str, Any]:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"train", "test"}:
            raise ValueError("mode must be either 'train' or 'test'.")
        manifest = self.load()
        accepted = [
            clip
            for clip in manifest.get("clips", [])
            if isinstance(clip, dict)
            and clip.get("status") == "accepted"
            and clip.get("split") == normalized_mode
            and str(clip.get("prompt") or "").strip()
        ]
        requested = max(0, int(count))
        if normalized_mode == "train":
            eligible = [clip for clip in accepted if clip.get("trained") is not True]
            selected = eligible[:requested]
        else:
            eligible = list(accepted)
            rng = random.Random(seed)
            selected = rng.sample(eligible, k=min(requested, len(eligible)))
            selected.sort(key=lambda clip: str(clip.get("id") or ""))

        lines = [case_line_for_clip(clip) for clip in selected]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        run_id = f"{normalized_mode}_{now_iso().replace(':', '').replace('-', '').replace('+', '_')}"

        return {
            "mode": normalized_mode,
            "requested": requested,
            "selected": len(selected),
            "eligible": len(eligible),
            "out": str(out_path),
            "seed": seed,
            "marked_trained": False,
            "mark_trained_requested": bool(normalized_mode == "train" and mark_trained),
            "training_marker_policy": "pass_only",
            "run_id": run_id if selected else None,
            "clip_ids": [clip.get("id") for clip in selected],
        }

    def mark_train_results_from_suite(
        self,
        summary: dict[str, Any],
        *,
        summary_path: Path | None = None,
    ) -> dict[str, Any]:
        """Mark train clips as trained only after their suite result passes."""

        results = summary.get("results")
        if not isinstance(results, list):
            return {
                "changed": 0,
                "passed_cases": 0,
                "matched_train_clips": 0,
                "already_marked": 0,
                "blocked_by_retrain_request": 0,
                "missing_train_case_ids": [],
                "run_id": _suite_train_run_id(summary, summary_path),
            }

        passed_by_case: dict[str, dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict) or not _suite_result_passed(item):
                continue
            case_id = item.get("case_id")
            if isinstance(case_id, str) and case_id.strip():
                passed_by_case[case_id.strip()] = item

        run_id = _suite_train_run_id(summary, summary_path)
        if not passed_by_case:
            return {
                "changed": 0,
                "passed_cases": 0,
                "matched_train_clips": 0,
                "already_marked": 0,
                "blocked_by_retrain_request": 0,
                "missing_train_case_ids": [],
                "run_id": run_id,
            }

        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
            timestamp = now_iso()
            changed = 0
            matched_train_case_ids: set[str] = set()
            already_marked = 0
            blocked_by_retrain_request = 0

            for clip in manifest.get("clips", []):
                if not isinstance(clip, dict) or clip.get("status") != "accepted" or clip.get("split") != "train":
                    continue
                case_id = clip.get("case_id")
                if not isinstance(case_id, str) or case_id not in passed_by_case:
                    continue
                matched_train_case_ids.add(case_id)
                result = passed_by_case[case_id]
                history = clip.get("training_history")
                if not isinstance(history, list):
                    history = []
                    clip["training_history"] = history
                if _clip_has_retrain_block_for_run(history, run_id):
                    blocked_by_retrain_request += 1
                    continue
                already_has_history = isinstance(history, list) and any(
                    isinstance(item, dict) and item.get("mode") == "train_passed" and item.get("run_id") == run_id
                    for item in history
                )
                fields_already_marked = (
                    clip.get("trained") is True
                    and clip.get("trained_run_id") == run_id
                    and clip.get("trained_at") is not None
                )
                if fields_already_marked and already_has_history:
                    already_marked += 1
                    continue

                before = {
                    "trained": clip.get("trained"),
                    "trained_at": clip.get("trained_at"),
                    "trained_run_id": clip.get("trained_run_id"),
                }
                clip["trained"] = True
                clip["trained_at"] = timestamp
                clip["trained_run_id"] = run_id
                clip["updated_at"] = timestamp
                if isinstance(history, list) and not already_has_history:
                    history.append(
                        {
                            "run_id": run_id,
                            "timestamp": timestamp,
                            "mode": "train_passed",
                            "suite_summary_path": None if summary_path is None else str(summary_path),
                            "case_dir": result.get("case_dir"),
                            "status": result.get("status"),
                            "verdict": result.get("verdict"),
                            "before": before,
                        }
                    )
                changed += 1

            return {
                "changed": changed,
                "passed_cases": len(passed_by_case),
                "matched_train_clips": len(matched_train_case_ids),
                "already_marked": already_marked,
                "blocked_by_retrain_request": blocked_by_retrain_request,
                "missing_train_case_ids": sorted(set(passed_by_case) - matched_train_case_ids),
                "run_id": run_id,
            }

        return self.update_manifest(updater)

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
        metadata = metadata or {}

        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
            clip = self._find_clip(manifest, clip_id)
            before_status = clip.get("status")
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
            return event

        return self.update_manifest(updater)

    def record_review_position(
        self,
        clip_id: str,
        *,
        manifest_index: int,
        note: str | None = None,
    ) -> dict[str, Any]:
        def updater(manifest: dict[str, Any]) -> dict[str, Any]:
            state = {
                "last_reviewed_clip_id": clip_id,
                "last_reviewed_manifest_index": manifest_index,
                "updated_at": now_iso(),
            }
            if note:
                state["note"] = note
            manifest["review_state"] = state
            return state

        return self.update_manifest(updater)

    def export_cases(self, out_path: Path) -> int:
        manifest = self.load()
        lines = []
        for clip in sorted(manifest["clips"], key=lambda item: str(item.get("id"))):
            if clip.get("status") != "accepted":
                continue
            line = case_line_for_clip(clip)
            if line:
                lines.append(line)
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
        if clip.get("split") == "train" and "trained" not in clip:
            clip = {**clip, "trained": False}
        self.add_clip_open_reference(clip)
        ordered: dict[str, Any] = {}
        for key in CLIP_SHORT_FIELD_ORDER:
            if key in clip:
                ordered[key] = clip[key]
        for key, value in clip.items():
            if key not in ordered and key not in CLIP_LONG_FIELD_ORDER and key not in DEPRECATED_CLIP_FIELDS:
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


def case_line_for_clip(clip: dict[str, Any]) -> str:
    case_id = str(clip.get("case_id") or clip.get("id") or "case").strip()
    prompt = str(clip.get("prompt") or "").strip()
    return f"{case_id}|{prompt}" if prompt else ""


def merge_manifests(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged["source_videos"] = _merge_records_by_id(
        current.get("source_videos", []),
        incoming.get("source_videos", []),
        incoming_wins=True,
    )
    merged["clips"] = _merge_clips_by_id(current.get("clips", []), incoming.get("clips", []))
    merged["review_events"] = _merge_unique_records(current.get("review_events", []), incoming.get("review_events", []))
    merged["style_memory"] = _merge_unique_records(current.get("style_memory", []), incoming.get("style_memory", []))
    if not isinstance(merged.get("review_state"), dict):
        merged["review_state"] = incoming.get("review_state") if isinstance(incoming.get("review_state"), dict) else {}
    return merged


def _merge_records_by_id(
    current_records: object,
    incoming_records: object,
    *,
    incoming_wins: bool,
) -> list[Any]:
    current_list = list(current_records) if isinstance(current_records, list) else []
    incoming_list = list(incoming_records) if isinstance(incoming_records, list) else []
    merged = [dict(item) if isinstance(item, dict) else item for item in current_list]
    by_id = {
        str(item.get("id")): index
        for index, item in enumerate(merged)
        if isinstance(item, dict) and item.get("id") is not None
    }
    seen_idless = {_record_identity(item) for item in merged if not (isinstance(item, dict) and item.get("id"))}
    for item in incoming_list:
        if not isinstance(item, dict) or item.get("id") is None:
            identity = _record_identity(item)
            if identity not in seen_idless:
                seen_idless.add(identity)
                merged.append(item)
            continue
        record_id = str(item.get("id"))
        if record_id in by_id:
            if incoming_wins:
                merged[by_id[record_id]] = dict(item)
            continue
        by_id[record_id] = len(merged)
        merged.append(dict(item))
    return merged


def _merge_clips_by_id(current_records: object, incoming_records: object) -> list[Any]:
    current_list = list(current_records) if isinstance(current_records, list) else []
    incoming_list = list(incoming_records) if isinstance(incoming_records, list) else []
    merged = [dict(item) if isinstance(item, dict) else item for item in current_list]
    by_id = {
        str(item.get("id")): index
        for index, item in enumerate(merged)
        if isinstance(item, dict) and item.get("id") is not None
    }
    for item in incoming_list:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        clip_id = str(item.get("id"))
        if clip_id in by_id:
            current = merged[by_id[clip_id]]
            if isinstance(current, dict):
                merged[by_id[clip_id]] = _merge_existing_clip(current, item)
            continue
        by_id[clip_id] = len(merged)
        merged.append(dict(item))
    return merged


def _merge_existing_clip(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in incoming.items():
        if key not in merged or _is_empty_manifest_value(merged.get(key)):
            merged[key] = value

    if _can_accept_incoming_auto_category(current, incoming):
        for key in ("category", "category_source"):
            if key in incoming:
                merged[key] = incoming[key]

    if _can_accept_incoming_tmp_split(current, incoming):
        for key in ("split", "split_source", "split_group_id", "split_group_title", "split_assigned_at"):
            if key in incoming:
                merged[key] = incoming[key]
        if incoming.get("split") == "train" and "trained" not in merged:
            merged["trained"] = False

    if _record_timestamp(incoming) > _record_timestamp(current):
        merged["updated_at"] = incoming.get("updated_at")
    return merged


def _can_accept_incoming_auto_category(current: dict[str, Any], incoming: dict[str, Any]) -> bool:
    if incoming.get("category") not in CLIP_CATEGORIES:
        return False
    current_source = current.get("category_source")
    return current.get("category") not in CLIP_CATEGORIES or current_source in {None, "", "auto_prompt_heuristic"}


def _can_accept_incoming_tmp_split(current: dict[str, Any], incoming: dict[str, Any]) -> bool:
    if incoming.get("split") not in TMP_DATASET_SPLITS:
        return False
    current_split = current.get("split")
    current_source = current.get("split_source")
    return current_split not in PERMANENT_DATASET_SPLITS and current_source != "human"


def _merge_unique_records(current_records: object, incoming_records: object) -> list[Any]:
    merged = list(current_records) if isinstance(current_records, list) else []
    seen = {_record_identity(item) for item in merged}
    for item in incoming_records if isinstance(incoming_records, list) else []:
        identity = _record_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged


def _record_identity(record: object) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)


def _record_timestamp(record: dict[str, Any]) -> str:
    value = record.get("updated_at") or record.get("created_at")
    return str(value) if value is not None else ""


def _is_empty_manifest_value(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _suite_result_passed(result: dict[str, Any]) -> bool:
    return result.get("verdict") == "pass" or result.get("status") == "pass"


def _suite_train_run_id(summary: dict[str, Any], summary_path: Path | None = None) -> str:
    out_dir = summary.get("out_dir")
    if isinstance(out_dir, str) and out_dir.strip():
        return f"train_suite_{Path(out_dir).name}"
    if summary_path is not None:
        return f"train_suite_{summary_path.resolve().parent.name}"
    tasks_file = summary.get("tasks_file")
    if isinstance(tasks_file, str) and tasks_file.strip():
        return f"train_suite_{Path(tasks_file).resolve().parent.name}"
    return "train_suite_unknown"


def _clip_has_retrain_block_for_run(history: list[Any], run_id: str) -> bool:
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if item.get("mode") != "retrain_requested":
            continue
        if item.get("block_train_passed_run_id") == run_id:
            return True
    return False


def _choose_test_group_ids(
    groups: list[dict[str, Any]],
    *,
    current_test: Counter,
    target_by_category: dict[str, int],
    test_fraction: float,
    accepted_count: int,
) -> set[str]:
    if not groups:
        return set()

    states: dict[tuple[int, ...], tuple[int, ...]] = {(0,) * len(CLIP_CATEGORIES): ()}
    for group_index, group in enumerate(groups):
        counts = group.get("category_counts")
        if not isinstance(counts, Counter):
            counts = Counter(counts or {})
        vector = tuple(int(counts.get(category, 0)) for category in CLIP_CATEGORIES)
        if not any(vector):
            continue
        for state, selected_indices in list(states.items()):
            next_state = tuple(state[index] + vector[index] for index in range(len(CLIP_CATEGORIES)))
            if next_state not in states:
                states[next_state] = selected_indices + (group_index,)

    target_total = round(accepted_count * max(0.0, min(test_fraction, 1.0)))

    def score(state: tuple[int, ...]) -> tuple[float, float, int, int, int]:
        final_counts = {
            category: int(current_test.get(category, 0)) + state[index]
            for index, category in enumerate(CLIP_CATEGORIES)
        }
        category_error = sum(
            ((final_counts[category] - target_by_category[category]) / max(target_by_category[category], 1)) ** 2
            for category in CLIP_CATEGORIES
        )
        total = sum(final_counts.values())
        total_error = ((total - target_total) / max(target_total, 1)) ** 2
        overshoot = sum(max(final_counts[category] - target_by_category[category], 0) for category in CLIP_CATEGORIES)
        return (category_error, total_error, abs(total - target_total), overshoot, len(states[state]))

    best_state = min(states, key=score)
    return {str(groups[index]["id"]) for index in states[best_state]}


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


def normalize_dataset_split(split: str) -> str:
    normalized = split.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "training": "train",
        "train-set": "train",
        "training-set": "train",
        "testing": "test",
        "test-set": "test",
        "eval": "test",
        "evaluation": "test",
        "train-temporary": "train-tmp",
        "temporary-train": "train-tmp",
        "test-temporary": "test-tmp",
        "temporary-test": "test-tmp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in DATASET_SPLITS:
        valid = ", ".join(DATASET_SPLITS)
        raise ValueError(f"Unknown dataset split {split!r}. Expected one of: {valid}.")
    return normalized


def infer_clip_category(clip: dict[str, Any]) -> str:
    text = " ".join(
        str(clip.get(key) or "") for key in ("case_id", "prompt", "title", "visual_summary", "segment_reason")
    )
    text = text.lower()
    scores = {
        "cloth": _keyword_score(
            text,
            (
                "cloth",
                "fabric",
                "garment",
                "shirt",
                "skirt",
                "towel",
                "curtain",
                "ribbon",
                "thin-shell",
                "thin shell",
                "sheet",
                "drape",
                "wrinkle",
                "fold",
                "sewing",
                "knit",
                "woven",
                "textile",
            ),
        ),
        "rigid": _keyword_score(
            text,
            (
                "pure rigid",
                "rigid-body",
                "rigid body",
                "rigid bodies",
                "articulated",
                "multibody",
                "mechanism",
                "chain",
                "gear",
                "pulley",
                "ragdoll",
                "jenga",
                "block stack",
                "house of cards",
                "tumbler",
                "granular",
                "grains",
                "dice",
                "trebuchet",
                "vehicle",
            ),
        ),
        "deformable_bodies": _keyword_score(
            text,
            (
                "fem+ipc",
                "deformable",
                "soft body",
                "soft-body",
                "hyperelastic",
                "elastic body",
                "elastic bodies",
                "rubber",
                "jelly",
                "gel",
                "tactile",
                "plastic",
                "putty",
                "volume",
                "volumetric",
            ),
        ),
    }
    if "cloth" in text or "fem.cloth" in text:
        scores["cloth"] += 3
    if text.startswith("create a rigid") or "pure rigid" in text:
        scores["rigid"] += 3
    if text.startswith("create a fem+ipc") or "hyperelastic" in text:
        scores["deformable_bodies"] += 2
    priority = {"cloth": 3, "deformable_bodies": 2, "rigid": 1}
    return max(("cloth", "deformable_bodies", "rigid"), key=lambda category: (scores[category], priority[category]))


def paper_group_for_clip(clip: dict[str, Any], source: dict[str, Any] | None) -> tuple[str, str]:
    source = source or {}
    title = _optional_text(source.get("paper_title")) or _optional_text(source.get("title"))
    key_text = (
        _optional_text(source.get("paper_url"))
        or _optional_text(source.get("paper_title"))
        or _optional_text(source.get("project_url"))
        or _optional_text(source.get("source_url"))
        or _optional_text(source.get("url"))
        or _optional_text(source.get("id"))
        or _optional_text(clip.get("source_video_id"))
        or _optional_text(clip.get("id"))
        or "unknown_paper"
    )
    return slugify(key_text, fallback="paper_group"), title or key_text


def _keyword_score(text: str, keywords: tuple[str, ...]) -> int:
    return sum(text.count(keyword) for keyword in keywords)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
