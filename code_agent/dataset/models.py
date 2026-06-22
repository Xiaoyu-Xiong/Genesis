from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ClipStatus = Literal[
    "accepted",
    "rejected",
    "candidate",
    "failed",
    "duplicate",
    "duplicate_deleted",
    "multi_example_deleted",
    "truncated_deleted",
]


@dataclass(slots=True, frozen=True)
class BuildConfig:
    target_clips: int
    data_root: Path
    sources: tuple[str, ...] = ()
    source_file: Path | None = None
    similar_to: tuple[str, ...] = ()
    similar_to_file: Path | None = None
    similarity_seed_limit: int = 12
    max_candidates: int | None = None
    max_downloads: int | None = None
    run_codex: bool = True


@dataclass(slots=True)
class BuildSummary:
    status: str
    data_root: Path
    manifest_path: Path
    target_clips: int
    accepted_clips: int
    candidates_seen: int = 0
    videos_downloaded: int = 0
    clips_added: int = 0
    similarity_seeds: int = 0
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "data_root": str(self.data_root),
            "manifest_path": str(self.manifest_path),
            "target_clips": self.target_clips,
            "accepted_clips": self.accepted_clips,
            "candidates_seen": self.candidates_seen,
            "videos_downloaded": self.videos_downloaded,
            "clips_added": self.clips_added,
            "similarity_seeds": self.similarity_seeds,
            "skipped": self.skipped,
            "failures": self.failures,
        }


@dataclass(slots=True, frozen=True)
class SourceCandidate:
    candidate_id: str
    video_url: str
    title: str = ""
    project_url: str | None = None
    paper_url: str | None = None
    paper_title: str | None = None
    venue: str | None = None
    source_url: str | None = None
    license_notes: str | None = None
    source_policy_notes: str | None = None
    notes: str | None = None
    confidence: float | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, fallback_id: str) -> "SourceCandidate | None":
        video_url = _clean_str(data.get("video_url") or data.get("url") or data.get("source_url"))
        if not video_url:
            return None
        return cls(
            candidate_id=_clean_str(data.get("candidate_id") or data.get("id")) or fallback_id,
            video_url=video_url,
            title=_clean_str(data.get("title")) or fallback_id,
            project_url=_clean_optional_str(data.get("project_url")),
            paper_url=_clean_optional_str(data.get("paper_url")),
            paper_title=_clean_optional_str(data.get("paper_title") or data.get("paper")),
            venue=_clean_optional_str(data.get("venue")),
            source_url=_clean_optional_str(data.get("source_url")),
            license_notes=_clean_optional_str(data.get("license_notes")),
            source_policy_notes=_clean_optional_str(data.get("source_policy_notes")),
            notes=_clean_optional_str(data.get("notes")),
            confidence=float(data["confidence"]) if isinstance(data.get("confidence"), int | float) else None,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "video_url": self.video_url,
            "title": self.title,
            "project_url": self.project_url,
            "paper_url": self.paper_url,
            "paper_title": self.paper_title,
            "venue": self.venue,
            "source_url": self.source_url,
            "license_notes": self.license_notes,
            "source_policy_notes": self.source_policy_notes,
            "notes": self.notes,
            "confidence": self.confidence,
        }


@dataclass(slots=True, frozen=True)
class SimilaritySeed:
    seed_id: str
    prompt: str
    source: str
    case_id: str | None = None
    notes: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "case_id": self.case_id,
            "prompt": self.prompt,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(slots=True, frozen=True)
class SegmentCandidate:
    title_slug: str
    start_sec: float
    end_sec: float
    visual_summary: str = ""
    reason: str = ""
    confidence: float | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, fallback_slug: str) -> "SegmentCandidate | None":
        try:
            start_sec = float(data.get("start_sec"))
            end_sec = float(data.get("end_sec"))
        except (TypeError, ValueError):
            return None
        if end_sec <= start_sec:
            return None
        return cls(
            title_slug=_clean_str(data.get("title_slug") or data.get("id") or data.get("title")) or fallback_slug,
            start_sec=start_sec,
            end_sec=end_sec,
            visual_summary=_clean_str(data.get("visual_summary")),
            reason=_clean_str(data.get("reason")),
            confidence=float(data["confidence"]) if isinstance(data.get("confidence"), int | float) else None,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "title_slug": self.title_slug,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "visual_summary": self.visual_summary,
            "reason": self.reason,
            "confidence": self.confidence,
        }


def _clean_optional_str(value: object) -> str | None:
    text = _clean_str(value)
    return text or None


def _clean_str(value: object) -> str:
    return str(value).strip() if value is not None else ""
