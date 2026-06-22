from __future__ import annotations

from pathlib import Path
from typing import Any

from code_agent.dataset import agents
from code_agent.dataset.media import (
    build_contact_sheet,
    cut_clip,
    describe_download,
    detect_scene_segments,
    discover_video_urls_from_page,
    download_video,
    visual_signature,
)
from code_agent.dataset.models import BuildConfig, BuildSummary, SegmentCandidate, SimilaritySeed, SourceCandidate
from code_agent.dataset.seeds import collect_similarity_seeds
from code_agent.dataset.store import DatasetStore, infer_clip_category
from code_agent.dataset.utils import (
    first_nonempty,
    is_probably_video_url,
    is_ytdlp_supported_url,
    now_iso,
    sha256_file,
)


def build_dataset(config: BuildConfig) -> BuildSummary:
    store = DatasetStore(config.data_root)
    store.ensure_dirs()
    manifest = store.load()
    if _backfill_clip_visual_signatures(store, manifest):
        store.save(manifest)
    split_summary = store.assign_train_test_splits(manifest)
    if split_summary.get("changed"):
        store.save(manifest)
    sources = _load_sources(config)
    similarity_seeds = collect_similarity_seeds(config, manifest)
    target = max(0, int(config.target_clips))
    summary = BuildSummary(
        status="already_complete",
        data_root=store.root,
        manifest_path=store.manifest_path,
        target_clips=target,
        accepted_clips=store.accepted_count(manifest),
        similarity_seeds=len(similarity_seeds),
    )
    if summary.accepted_clips >= target:
        store.assign_train_test_splits(manifest)
        store.save(manifest)
        return summary

    run_dir = store.logs_dir / f"build_{now_iso().replace(':', '').replace('-', '').replace('+', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    needed = target - summary.accepted_clips
    explicit_candidates = _explicit_source_candidates(sources)
    all_sources_are_explicit = bool(sources) and len(explicit_candidates) == len(sources)
    codex_candidates = []
    if not all_sources_are_explicit:
        codex_candidates = agents.scout_sources(
            store=store,
            manifest=manifest,
            sources=sources,
            needed_clips=needed,
            logs_dir=run_dir,
            similarity_seeds=similarity_seeds,
            run_codex=config.run_codex,
        )
    candidates = _dedupe_candidates([*codex_candidates, *explicit_candidates])
    if config.max_candidates is not None:
        candidates = candidates[: max(0, config.max_candidates)]
    curated = agents.curate_sources(
        candidates=candidates,
        manifest=manifest,
        logs_dir=run_dir,
        similarity_seeds=similarity_seeds,
        run_codex=config.run_codex,
    )
    curated = _expand_project_page_candidates(curated)
    summary.candidates_seen = len(candidates)
    download_limit = config.max_downloads if config.max_downloads is not None else len(curated)

    for candidate in curated:
        if store.accepted_count(manifest) >= target:
            break
        if summary.videos_downloaded >= download_limit:
            summary.skipped.append(f"download_limit_reached:{download_limit}")
            break
        if store.source_url_seen(manifest, candidate.video_url):
            summary.skipped.append(f"source_url_seen:{candidate.video_url}")
            continue
        source_id = store.unique_source_id(manifest, first_nonempty(candidate.title, candidate.video_url))
        try:
            video_path = download_video(candidate, out_dir=store.videos_dir, source_id=source_id)
            download = describe_download(video_path)
        except Exception as exc:  # noqa: BLE001 - record per-source failure and continue building.
            failure_record = _failed_source_record(store, source_id=source_id, candidate=candidate, error=str(exc))
            store.add_source_video(manifest, failure_record)
            store.save(manifest)
            summary.failures.append(f"{candidate.video_url}: {type(exc).__name__}: {exc}")
            continue

        if store.source_hash_seen(manifest, download.sha256):
            summary.skipped.append(f"source_hash_seen:{candidate.video_url}")
            continue

        summary.videos_downloaded += 1
        source_record = _source_record(store, source_id=source_id, candidate=candidate, download=download)
        store.add_source_video(manifest, source_record)
        store.save(manifest)

        try:
            timeline_sheet = store.timelines_dir / f"{source_id}.jpg"
            build_contact_sheet(video_path, timeline_sheet, max_frames=16, thumb_width=180)
            deterministic_segments = detect_scene_segments(video_path, source_id=source_id)
            segments = agents.segment_video(
                source_record=source_record,
                deterministic_segments=deterministic_segments,
                timeline_sheet=timeline_sheet,
                logs_dir=run_dir,
                similarity_seeds=similarity_seeds,
                run_codex=config.run_codex,
            )
            added = _materialize_segments(
                store=store,
                manifest=manifest,
                source_record=source_record,
                source_path=video_path,
                segments=_valid_segments(segments, duration=float(source_record.get("duration_sec") or 0.0)),
                target=target,
                logs_dir=run_dir,
                similarity_seeds=similarity_seeds,
                run_codex=config.run_codex,
            )
            summary.clips_added += added
            store.save(manifest)
        except Exception as exc:  # noqa: BLE001 - keep successful source metadata and continue.
            summary.failures.append(f"{source_id}: {type(exc).__name__}: {exc}")
            source_record["status"] = "failed"
            source_record["error"] = str(exc)
            store.save(manifest)

    summary.accepted_clips = store.accepted_count(manifest)
    summary.status = "complete" if summary.accepted_clips >= target else "partial"
    if not candidates:
        summary.status = "blocked_no_candidates"
    store.assign_train_test_splits(manifest)
    store.save(manifest)
    return summary


def _materialize_segments(
    *,
    store: DatasetStore,
    manifest: dict[str, Any],
    source_record: dict[str, Any],
    source_path: Path,
    segments: list[SegmentCandidate],
    target: int,
    logs_dir: Path,
    similarity_seeds: list[SimilaritySeed],
    run_codex: bool,
) -> int:
    added = 0
    source_id = str(source_record["id"])
    for index, segment in enumerate(segments, start=1):
        if store.accepted_count(manifest) >= target:
            break
        clip_id = store.unique_clip_id(manifest, f"{source_id}_{segment.title_slug or index}")
        clip_path = store.clips_dir / f"{clip_id}.mp4"
        cut_clip(source_path, start_sec=segment.start_sec, end_sec=segment.end_sec, out_path=clip_path)
        clip_sha256 = sha256_file(clip_path)
        if store.clip_hash_seen(manifest, clip_sha256):
            _delete_clip_artifacts(clip_path)
            continue
        contact_sheet_path = store.contact_sheets_dir / f"{clip_id}.jpg"
        build_contact_sheet(clip_path, contact_sheet_path, max_frames=8, thumb_width=180)
        signature = visual_signature(clip_path)
        clip_record = _clip_record(
            store,
            clip_id=clip_id,
            source_record=source_record,
            segment=segment,
            clip_path=clip_path,
            contact_sheet_path=contact_sheet_path,
            clip_sha256=clip_sha256,
            signature=signature,
        )
        duplicate_candidates = store.duplicate_candidates(manifest, signature=signature)
        if duplicate_candidates:
            clip_record["cv_duplicate_candidates"] = [
                _duplicate_candidate_record(candidate) for candidate in duplicate_candidates
            ]
            top_candidate = duplicate_candidates[0]
            existing_sheet = _existing_contact_sheet_path(store, top_candidate.get("clip"))
            duplicate_review = agents.review_duplicate_clip(
                clip_record=clip_record,
                duplicate_candidate=top_candidate,
                current_sheet=contact_sheet_path,
                existing_sheet=existing_sheet,
                logs_dir=logs_dir,
                run_codex=run_codex,
            )
            clip_record["duplicate_review"] = duplicate_review
            if duplicate_review.get("decision") != "distinct":
                _delete_clip_artifacts(clip_path, contact_sheet_path)
                continue
        case_id, prompt = agents.write_prompt(
            clip_record=clip_record,
            source_record=source_record,
            manifest=manifest,
            clip_sheet=contact_sheet_path,
            logs_dir=logs_dir,
            similarity_seeds=similarity_seeds,
            run_codex=run_codex,
        )
        clip_record["case_id"] = case_id
        clip_record["prompt"] = prompt
        clip_record["category"] = infer_clip_category(clip_record)
        clip_record["category_source"] = "auto_prompt_heuristic"
        clip_record["status"] = "accepted"
        store.add_clip(manifest, clip_record)
        store.save(manifest)
        added += 1
    return added


def _load_sources(config: BuildConfig) -> list[str]:
    sources = [source.strip() for source in config.sources if source.strip()]
    if config.source_file is not None and config.source_file.exists():
        for line in config.source_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line)
    return sources


def _explicit_source_candidates(sources: list[str]) -> list[SourceCandidate]:
    candidates = []
    for index, source in enumerate(sources, start=1):
        if not (is_probably_video_url(source) or is_ytdlp_supported_url(source) or Path(source).expanduser().exists()):
            continue
        candidates.append(
            SourceCandidate(
                candidate_id=f"explicit_{index:03d}",
                video_url=source,
                title=Path(source.split("?", 1)[0]).stem or f"explicit_source_{index:03d}",
                source_url=source,
                source_policy_notes="Explicit user-provided source.",
                confidence=1.0,
            )
        )
    return candidates


def _dedupe_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    seen: set[str] = set()
    deduped: list[SourceCandidate] = []
    for candidate in candidates:
        key = candidate.video_url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _expand_project_page_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    expanded: list[SourceCandidate] = []
    for candidate in candidates:
        if _is_downloadable_candidate(candidate):
            expanded.append(candidate)
            continue
        resolved_urls = discover_video_urls_from_page(candidate.video_url)
        if not resolved_urls:
            expanded.append(candidate)
            continue
        for index, video_url in enumerate(resolved_urls, start=1):
            expanded.append(
                SourceCandidate(
                    candidate_id=f"{candidate.candidate_id}_video_{index:02d}",
                    video_url=video_url,
                    title=(
                        f"{candidate.title} video {index:02d}"
                        if candidate.title and len(resolved_urls) > 1
                        else candidate.title
                    ),
                    project_url=candidate.project_url or candidate.video_url,
                    paper_url=candidate.paper_url,
                    paper_title=candidate.paper_title,
                    venue=candidate.venue,
                    source_url=candidate.source_url or candidate.video_url,
                    license_notes=candidate.license_notes,
                    source_policy_notes=candidate.source_policy_notes,
                    notes=first_nonempty(
                        candidate.notes,
                        f"Resolved from project page {candidate.video_url}",
                    ),
                    confidence=candidate.confidence,
                )
            )
    return _dedupe_candidates(expanded)


def _is_downloadable_candidate(candidate: SourceCandidate) -> bool:
    local_path = Path(candidate.video_url).expanduser()
    return local_path.exists() or is_probably_video_url(candidate.video_url) or is_ytdlp_supported_url(candidate.video_url)


def _valid_segments(segments: list[SegmentCandidate], *, duration: float) -> list[SegmentCandidate]:
    valid = []
    for segment in segments:
        start = max(0.0, min(float(segment.start_sec), duration))
        end = max(start, min(float(segment.end_sec), duration))
        if end - start < 0.5:
            continue
        valid.append(
            SegmentCandidate(
                title_slug=segment.title_slug,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                visual_summary=segment.visual_summary,
                reason=segment.reason,
                confidence=segment.confidence,
            )
        )
    return valid


def _has_paper_search_context(source_record: dict[str, Any]) -> bool:
    if source_record.get("paper_title"):
        return True
    for key in ("paper_url", "project_url", "source_url", "url"):
        value = source_record.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return True
    return False


def _backfill_clip_visual_signatures(store: DatasetStore, manifest: dict[str, Any]) -> bool:
    changed = False
    for clip in manifest.get("clips", []):
        if not isinstance(clip, dict) or not _needs_visual_signature_backfill(clip):
            continue
        clip_path = clip.get("clip_path")
        if not isinstance(clip_path, str) or not clip_path:
            continue
        path = store.abspath(clip_path)
        if not path.exists():
            continue
        signature = visual_signature(path)
        _apply_visual_signature(clip, signature)
        clip["updated_at"] = now_iso()
        changed = True
    return changed


def _needs_visual_signature_backfill(clip: dict[str, Any]) -> bool:
    return (
        clip.get("visual_signature_version") != 2
        or "foreground_component_fingerprints" not in clip
        or "foreground_component_phashes" not in clip
        or "foreground_color_histogram" not in clip
    )


def _apply_visual_signature(clip: dict[str, Any], signature: dict[str, object]) -> None:
    clip["visual_fingerprint"] = signature.get("visual_fingerprint")
    clip["frame_fingerprints"] = signature.get("frame_fingerprints", [])
    clip["color_histogram"] = signature.get("color_histogram", [])
    clip["foreground_component_fingerprints"] = signature.get("foreground_component_fingerprints", [])
    clip["foreground_component_phashes"] = signature.get("foreground_component_phashes", [])
    clip["foreground_color_histogram"] = signature.get("foreground_color_histogram", [])
    clip["visual_signature_version"] = signature.get("signature_version")


def _duplicate_candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    clip = candidate.get("clip") if isinstance(candidate.get("clip"), dict) else {}
    return {
        "clip_id": candidate.get("clip_id") or clip.get("id"),
        "case_id": clip.get("case_id"),
        "clip_path": clip.get("clip_path"),
        "contact_sheet_path": clip.get("contact_sheet_path"),
        "score": candidate.get("score"),
        "reason": candidate.get("reason"),
        "metrics": candidate.get("metrics"),
    }


def _existing_contact_sheet_path(store: DatasetStore, clip: object) -> Path | None:
    if not isinstance(clip, dict):
        return None
    path_text = clip.get("contact_sheet_path")
    if not isinstance(path_text, str) or not path_text:
        return None
    path = store.abspath(path_text)
    return path if path.exists() else None


def _delete_clip_artifacts(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _source_record(store: DatasetStore, *, source_id: str, candidate: SourceCandidate, download: Any) -> dict[str, Any]:
    return {
        "id": source_id,
        "candidate_id": candidate.candidate_id,
        "url": candidate.video_url,
        "project_url": candidate.project_url,
        "paper_url": candidate.paper_url,
        "paper_title": candidate.paper_title,
        "venue": candidate.venue,
        "source_url": candidate.source_url,
        "title": candidate.title,
        "license_notes": candidate.license_notes,
        "source_policy_notes": candidate.source_policy_notes,
        "notes": candidate.notes,
        "status": "ready",
        "path": store.relpath(download.path),
        "sha256": download.sha256,
        "bytes": download.bytes,
        "duration_sec": download.info.duration_sec,
        "width": download.info.width,
        "height": download.info.height,
        "created_at": now_iso(),
    }


def _failed_source_record(
    store: DatasetStore,
    *,
    source_id: str,
    candidate: SourceCandidate,
    error: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "candidate_id": candidate.candidate_id,
        "url": candidate.video_url,
        "project_url": candidate.project_url,
        "paper_title": candidate.paper_title,
        "venue": candidate.venue,
        "source_url": candidate.source_url,
        "title": candidate.title,
        "status": "failed",
        "error": error,
        "source_policy_notes": candidate.source_policy_notes,
        "created_at": now_iso(),
        "path": None,
        "sha256": None,
        "bytes": None,
        "duration_sec": None,
    }


def _clip_record(
    store: DatasetStore,
    *,
    clip_id: str,
    source_record: dict[str, Any],
    segment: SegmentCandidate,
    clip_path: Path,
    contact_sheet_path: Path,
    clip_sha256: str,
    signature: dict[str, object],
) -> dict[str, Any]:
    return {
        "id": clip_id,
        "source_video_id": source_record["id"],
        "source_url": source_record.get("url"),
        "title": segment.title_slug,
        "start_sec": segment.start_sec,
        "end_sec": segment.end_sec,
        "duration_sec": round(segment.end_sec - segment.start_sec, 3),
        "clip_path": store.relpath(clip_path),
        "clip_uri": store.file_uri(clip_path),
        "contact_sheet_path": store.relpath(contact_sheet_path),
        "clip_sha256": clip_sha256,
        "clip_bytes": clip_path.stat().st_size,
        "visual_fingerprint": signature.get("visual_fingerprint"),
        "visual_signature_version": signature.get("signature_version"),
        "visual_summary": segment.visual_summary,
        "segment_reason": segment.reason,
        "segment_confidence": segment.confidence,
        "case_id": None,
        "prompt": None,
        "prompt_revisions": [],
        "status": "candidate",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "frame_fingerprints": signature.get("frame_fingerprints", []),
        "color_histogram": signature.get("color_histogram", []),
        "foreground_component_fingerprints": signature.get("foreground_component_fingerprints", []),
        "foreground_component_phashes": signature.get("foreground_component_phashes", []),
        "foreground_color_histogram": signature.get("foreground_color_histogram", []),
    }
