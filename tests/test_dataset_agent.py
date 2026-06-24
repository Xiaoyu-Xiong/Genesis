from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from code_agent.dataset import review_tui
from code_agent.dataset import agents
from code_agent.dataset import builder as dataset_builder
from code_agent.dataset.builder import _materialize_segments, build_dataset
from code_agent.dataset.media import (
    DownloadedVideo,
    average_video_hash,
    build_contact_sheet,
    cut_clip,
    detect_scene_segments,
    discover_video_urls_from_page,
    resolve_ytdlp_js_runtime_args,
    probe_video,
    resolve_ytdlp_command,
    VideoInfo,
    visual_signature,
)
from code_agent.dataset.models import BuildConfig, SegmentCandidate, SourceCandidate
from code_agent.dataset.review_tui import (
    case_line,
    resolve_review_start_index,
    reviewable_clip_positions,
)
from code_agent.dataset.seeds import collect_similarity_seeds
from code_agent.dataset.store import DatasetStore
from code_agent.dataset.utils import is_ytdlp_supported_url
from code_agent.utils.codex import CodexExecResult


def test_dataset_store_round_trip_reject_edit_and_export(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_source_video(
        manifest,
        {
            "id": "source_demo",
            "url": "https://example.test/demo.mp4",
            "status": "ready",
            "sha256": "abc",
            "path": "videos/source_demo.mp4",
        },
    )
    store.add_clip(
        manifest,
        {
            "id": "clip_demo",
            "source_video_id": "source_demo",
            "status": "accepted",
            "case_id": "old_case",
            "prompt": "Create a FEM+IPC scene inspired by an old demo. Render 10s behavior.",
            "clip_path": "clips/clip_demo.mp4",
            "clip_sha256": "cliphash",
            "visual_fingerprint": "ff00ff00ff00ff00",
            "frame_fingerprints": ["ff00ff00ff00ff00"],
            "color_histogram": [0.1] * 24,
            "foreground_component_fingerprints": ["ff00ff00ff00ff00"],
            "foreground_component_phashes": ["00ff00ff00ff00ff"],
            "foreground_color_histogram": [0.2] * 24,
            "prompt_revisions": [],
        },
    )
    store.save(manifest)

    loaded = store.load()
    assert loaded["source_videos"][0]["id"] == "source_demo"
    assert loaded["clips"][0]["clip_uri"].startswith("file://")
    assert loaded["clips"][0]["clip_uri"].endswith("/clips/clip_demo.mp4")
    assert list(loaded["clips"][0])[-5:] == [
        "frame_fingerprints",
        "color_histogram",
        "foreground_component_fingerprints",
        "foreground_component_phashes",
        "foreground_color_histogram",
    ]
    assert store.accepted_count(loaded) == 1
    assert store.clip_hash_seen(loaded, "cliphash")
    assert store.near_duplicate_clip(loaded, "ff00ff00ff00ff01")["id"] == "clip_demo"

    reject_event = store.reject_clip("clip_demo", reason="not suitable")
    assert reject_event["type"] == "reject"
    assert store.load()["clips"][0]["status"] == "rejected"

    edit_event = store.edit_clip(
        "clip_demo",
        prompt=(
            "soft_braid_demo|Create a FEM+IPC scene inspired by a soft braid demo: two rods twist, contact, "
            "and settle through real endpoint actuation. Render 10s behavior."
        ),
        reason="human rewrite",
    )
    edited = store.load()
    assert edit_event["type"] == "edit"
    assert edited["clips"][0]["case_id"] == "soft_braid_demo"
    assert edited["clips"][0]["status"] == "accepted"
    assert edited["style_memory"][0]["case_id"] == "soft_braid_demo"

    out_path = tmp_path / "cases.txt"
    count = store.export_cases(out_path)
    assert count == 1
    assert out_path.read_text(encoding="utf-8").startswith("soft_braid_demo|Create a FEM+IPC scene")


def test_dataset_store_detects_visual_near_duplicates_from_frame_signature(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_a",
            "status": "accepted",
            "case_id": "clip_a",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "visual_fingerprint": "0000000000000000",
            "frame_fingerprints": [
                "ffffffff00000000",
                "fffffff000000000",
                "0fffffff00000000",
                "00ffffff00000000",
            ],
            "color_histogram": [0.05] * 24,
        },
    )

    duplicate = store.near_duplicate_clip(
        manifest,
        signature={
            "visual_fingerprint": "aaaaaaaaaaaaaaaa",
            "frame_fingerprints": [
                "ffffffff00000001",
                "fffffff000000001",
                "0fffffff00000001",
                "00ffffff00000001",
            ],
            "color_histogram": [0.05] * 24,
        },
    )

    assert duplicate["id"] == "clip_a"


def test_dataset_store_detects_visual_near_duplicates_from_foreground_components(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_a",
            "status": "accepted",
            "case_id": "clip_a",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "visual_fingerprint": "0000000000000000",
            "frame_fingerprints": ["1111111111111111"],
            "color_histogram": [0.01] * 24,
            "foreground_component_fingerprints": [
                "f9f9f840086d4f4f",
                "3f0fcfe7f3f9fcfe",
            ],
            "foreground_component_phashes": [
                "49ceb5931fa4d291",
                "383898cfe7c2e9c4",
            ],
            "foreground_color_histogram": [0.05] * 24,
        },
    )

    candidates = store.duplicate_candidates(
        manifest,
        signature={
            "visual_fingerprint": "aaaaaaaaaaaaaaaa",
            "frame_fingerprints": ["2222222222222222"],
            "color_histogram": [0.02] * 24,
            "foreground_component_fingerprints": [
                "f9f9f840086d4f4e",
                "3f0fcfe7f3f9fcff",
            ],
            "foreground_component_phashes": [
                "49ceb5931fa4d290",
                "383898cfe7c2e9c5",
            ],
            "foreground_color_histogram": [0.05] * 24,
        },
    )

    assert candidates[0]["clip_id"] == "clip_a"
    assert candidates[0]["reason"] == "foreground_components"


def test_discover_video_urls_from_project_page(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size):
            return (
                b'<iframe src="//www.youtube.com/embed/abc123"></iframe>'
                b'<a href="videos/demo.mp4">demo</a>'
                b'<img src="thumb.jpg">'
            )

    def fake_urlopen(_request, timeout):
        assert timeout == 30.0
        return FakeResponse()

    monkeypatch.setattr("code_agent.dataset.media.urllib.request.urlopen", fake_urlopen)

    assert discover_video_urls_from_page("https://example.test/project/index.html") == [
        "https://www.youtube.com/embed/abc123",
        "https://example.test/project/videos/demo.mp4",
    ]


def test_delete_duplicate_removes_clip_without_negative_memory(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_dup",
            "status": "accepted",
            "case_id": "clip_dup",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "clip_path": "clips/clip_dup.mp4",
        },
    )
    store.save(manifest)

    event = store.delete_duplicate_clip("clip_dup", duplicate_of_clip_id="clip_original", reason="same demo")
    manifest = store.load()

    assert event["type"] == "delete_duplicate"
    assert event["negative_memory"] is False
    assert manifest["clips"][0]["status"] == "duplicate_deleted"
    assert manifest["clips"][0]["duplicate_of_clip_id"] == "clip_original"
    assert not manifest["style_memory"]
    assert all(item.get("type") != "reject" for item in manifest["review_events"])
    out_path = tmp_path / "cases.txt"
    assert store.export_cases(out_path) == 0


def test_delete_multi_example_removes_clip_without_negative_memory(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_mixed",
            "status": "accepted",
            "case_id": "clip_mixed",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "clip_path": "clips/clip_mixed.mp4",
        },
    )
    store.save(manifest)

    event = store.delete_multi_example_clip("clip_mixed", reason="contains two demos")
    manifest = store.load()

    assert event["type"] == "delete_multi_example"
    assert event["negative_memory"] is False
    assert manifest["clips"][0]["status"] == "multi_example_deleted"
    assert manifest["clips"][0]["multi_example"] is True
    assert all(item.get("type") != "reject" for item in manifest["review_events"])
    assert store.export_cases(tmp_path / "cases.txt") == 0


def test_set_clip_category_records_review_event(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_category",
            "status": "accepted",
            "case_id": "clip_category",
            "prompt": "Create a FEM+IPC scene inspired by a soft body demo. Render 10s behavior.",
            "clip_path": "clips/clip_category.mp4",
        },
    )
    store.save(manifest)

    event = store.set_clip_category("clip_category", category="deformable bodies", reason="review label")
    manifest = store.load()

    assert event["type"] == "set_category"
    assert event["before_category"] is None
    assert event["after_category"] == "deformable_bodies"
    assert manifest["clips"][0]["category"] == "deformable_bodies"
    assert manifest["clips"][0]["status"] == "accepted"
    assert manifest["review_events"][-1]["reason"] == "review label"


def test_deprecated_paper_prompt_fields_are_dropped_and_splits_support_tmp(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_paper",
            "status": "accepted",
            "case_id": "visual_case",
            "prompt": "Create a FEM+IPC scene inspired by a visual soft body demo. Render 10s behavior.",
            "clip_path": "clips/clip_paper.mp4",
            "prompt_from_paper": {"status": "generated", "case_id": "paper_case", "prompt": "old"},
            "prompt_from_paper_revisions": [{"prompt": "older"}],
        },
    )
    store.save(manifest)

    manifest = store.load()
    summary = store.drop_paper_prompts(manifest)
    store.save(manifest)
    split_event = store.set_clip_split("clip_paper", split="test-tmp", reason="holdout candidate")
    manifest = store.load()
    clip = manifest["clips"][0]

    assert summary["changed"] in {0, 1}
    assert split_event["type"] == "set_split"
    assert clip["case_id"] == "visual_case"
    assert "prompt_from_paper" not in clip
    assert "prompt_from_paper_revisions" not in clip
    assert clip["split"] == "test-tmp"
    assert clip["split_source"] == "human"


def test_assign_train_test_splits_groups_whole_papers_and_backfills_category(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    manifest["source_videos"].extend(
        [
            {"id": "paper_a", "paper_title": "Rigid Paper", "url": "https://example.test/a.mp4"},
            {"id": "paper_b", "paper_title": "Cloth Paper", "url": "https://example.test/b.mp4"},
            {"id": "paper_c", "paper_title": "Soft Paper", "url": "https://example.test/c.mp4"},
        ]
    )
    for index in range(3):
        store.add_clip(
            manifest,
            {
                "id": f"rigid_{index}",
                "source_video_id": "paper_a",
                "status": "accepted",
                "case_id": f"rigid_{index}",
                "prompt": "Create a pure rigid multibody block stack scene. Render 10s behavior.",
            },
        )
    for index in range(3):
        store.add_clip(
            manifest,
            {
                "id": f"cloth_{index}",
                "source_video_id": "paper_b",
                "status": "accepted",
                "case_id": f"cloth_{index}",
                "prompt": "Create a FEM.Cloth fabric sheet draping scene. Render 10s behavior.",
            },
        )
    for index in range(4):
        store.add_clip(
            manifest,
            {
                "id": f"soft_{index}",
                "source_video_id": "paper_c",
                "status": "accepted",
                "case_id": f"soft_{index}",
                "prompt": "Create a FEM+IPC hyperelastic soft body compression scene. Render 10s behavior.",
            },
        )

    summary = store.assign_train_test_splits(manifest, test_fraction=0.3, temporary=True)

    assert summary["accepted"] == 10
    assert summary["temporary"] is True
    assert all(clip.get("category") in {"rigid", "cloth", "deformable_bodies"} for clip in manifest["clips"])
    assert all(clip.get("split") in {"train-tmp", "test-tmp"} for clip in manifest["clips"])
    for source_id in ("paper_a", "paper_b", "paper_c"):
        splits = {
            clip["split"]
            for clip in manifest["clips"]
            if clip.get("source_video_id") == source_id and clip.get("status") == "accepted"
        }
        assert len(splits) == 1

    promoted = store.finalize_tmp_splits(manifest)
    assert promoted["changed"] == 10
    assert all(clip.get("split") in {"train", "test"} for clip in manifest["clips"])
    assert all(clip.get("trained") is False for clip in manifest["clips"] if clip.get("split") == "train")


def test_make_run_batch_uses_only_permanent_splits_without_marking_trained(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    for index, split in enumerate(("train", "train", "train-tmp", "test", "test-tmp"), start=1):
        store.add_clip(
            manifest,
            {
                "id": f"clip_{index}",
                "status": "accepted",
                "case_id": f"case_{index}",
                "prompt": f"Create a FEM+IPC scene inspired by demo {index}. Render 10s behavior.",
                "split": split,
                "trained": index == 1,
            },
        )
    store.save(manifest)

    train_summary = store.make_run_batch(mode="train", count=2, out_path=tmp_path / "train.txt")
    manifest = store.load()

    assert train_summary["clip_ids"] == ["clip_2"]
    assert train_summary["marked_trained"] is False
    assert train_summary["training_marker_policy"] == "pass_only"
    assert "case_2|Create a FEM+IPC scene" in (tmp_path / "train.txt").read_text(encoding="utf-8")
    assert next(clip for clip in manifest["clips"] if clip["id"] == "clip_2")["trained"] is False
    assert next(clip for clip in manifest["clips"] if clip["id"] == "clip_3")["trained"] is False

    test_summary = store.make_run_batch(mode="test", count=5, out_path=tmp_path / "test.txt", seed=7)

    assert test_summary["clip_ids"] == ["clip_4"]
    assert "case_4|Create a FEM+IPC scene" in (tmp_path / "test.txt").read_text(encoding="utf-8")


def test_mark_train_results_marks_only_passed_train_cases(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    for clip_id, case_id, split in (
        ("clip_pass", "case_pass", "train"),
        ("clip_fail", "case_fail", "train"),
        ("clip_incomplete", "case_incomplete", "train"),
        ("clip_test_pass", "case_test_pass", "test"),
    ):
        store.add_clip(
            manifest,
            {
                "id": clip_id,
                "status": "accepted",
                "case_id": case_id,
                "prompt": f"Create a FEM+IPC scene inspired by {case_id}. Render 10s behavior.",
                "split": split,
                "trained": False,
            },
        )
    store.save(manifest)

    suite_root = tmp_path / "dataset_train_batch_20260622_123456"
    summary = {
        "out_dir": str(suite_root),
        "tasks_file": str(suite_root / "tasks.txt"),
        "num_cases": 4,
        "num_completed": 3,
        "results": [
            {"case_id": "case_pass", "verdict": "pass", "status": "pass", "case_dir": str(suite_root / "case_pass")},
            {"case_id": "case_fail", "verdict": "fail", "status": "fail", "case_dir": str(suite_root / "case_fail")},
            {
                "case_id": "case_test_pass",
                "verdict": "pass",
                "status": "pass",
                "case_dir": str(suite_root / "case_test_pass"),
            },
        ],
    }

    result = store.mark_train_results_from_suite(summary, summary_path=suite_root / "summary.json")
    manifest = store.load()

    assert result["changed"] == 1
    assert result["passed_cases"] == 2
    assert result["matched_train_clips"] == 1
    assert result["missing_train_case_ids"] == ["case_test_pass"]
    by_id = {clip["id"]: clip for clip in manifest["clips"]}
    assert by_id["clip_pass"]["trained"] is True
    assert by_id["clip_pass"]["trained_run_id"] == "train_suite_dataset_train_batch_20260622_123456"
    assert by_id["clip_pass"]["training_history"][-1]["mode"] == "train_passed"
    assert by_id["clip_fail"]["trained"] is False
    assert by_id["clip_incomplete"]["trained"] is False
    assert by_id["clip_test_pass"]["trained"] is False

    history_count = len([item for item in by_id["clip_pass"]["training_history"] if item["mode"] == "train_passed"])
    second = store.mark_train_results_from_suite(summary, summary_path=suite_root / "summary.json")
    manifest = store.load()
    updated_pass_clip = next(clip for clip in manifest["clips"] if clip["id"] == "clip_pass")
    assert second["changed"] == 0
    assert second["already_marked"] == 1
    assert (
        len([item for item in updated_pass_clip["training_history"] if item["mode"] == "train_passed"]) == history_count
    )

    manifest = store.load()
    blocked_clip = next(clip for clip in manifest["clips"] if clip["id"] == "clip_pass")
    blocked_clip["trained"] = False
    blocked_clip.pop("trained_at", None)
    blocked_clip.pop("trained_run_id", None)
    blocked_clip["prompt"] = "Create a manually edited prompt that must be preserved. Render 10s behavior."
    blocked_clip["training_history"].append(
        {
            "mode": "retrain_requested",
            "timestamp": "2026-06-22T12:45:00Z",
            "reason": "human requested retrain",
            "block_train_passed_run_id": "train_suite_dataset_train_batch_20260622_123456",
        }
    )
    store.save(manifest)

    blocked = store.mark_train_results_from_suite(summary, summary_path=suite_root / "summary.json")
    manifest = store.load()
    blocked_clip = next(clip for clip in manifest["clips"] if clip["id"] == "clip_pass")
    assert blocked["changed"] == 0
    assert blocked["blocked_by_retrain_request"] == 1
    assert blocked_clip["trained"] is False
    assert "trained_at" not in blocked_clip
    assert "trained_run_id" not in blocked_clip
    assert blocked_clip["prompt"] == "Create a manually edited prompt that must be preserved. Render 10s behavior."


def test_save_merged_preserves_concurrent_review_and_train_updates(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    for clip_id, case_id in (("clip_review", "case_review"), ("clip_train", "case_train")):
        store.add_clip(
            manifest,
            {
                "id": clip_id,
                "status": "accepted",
                "case_id": case_id,
                "prompt": f"Create a FEM+IPC scene inspired by {case_id}. Render 10s behavior.",
                "split": "train",
                "trained": False,
            },
        )
    store.save(manifest)

    stale_builder_manifest = store.load()
    store.reject_clip("clip_review", reason="manual reject while builder is running")
    store.mark_train_results_from_suite(
        {
            "out_dir": str(tmp_path / "dataset_train_batch_20260623_111111"),
            "results": [{"case_id": "case_train", "status": "pass", "verdict": "pass"}],
        }
    )

    stale_builder_manifest["source_videos"].append(
        {
            "id": "new_source",
            "url": "https://example.test/new.mp4",
            "status": "ready",
            "sha256": "new_sha",
        }
    )
    stale_builder_manifest["clips"][0]["split"] = "train-tmp"
    stale_builder_manifest["clips"][0]["split_source"] = "auto_tmp_paper_grouped"
    stale_builder_manifest["clips"].append(
        {
            "id": "clip_new",
            "source_video_id": "new_source",
            "status": "accepted",
            "case_id": "case_new",
            "prompt": "Create a rigid scene inspired by a new demo. Render 10s behavior.",
            "split": "train-tmp",
        }
    )

    store.save_merged(stale_builder_manifest)
    merged = store.load()
    by_id = {clip["id"]: clip for clip in merged["clips"]}

    assert by_id["clip_review"]["status"] == "rejected"
    assert by_id["clip_train"]["trained"] is True
    assert by_id["clip_train"]["trained_run_id"] == "train_suite_dataset_train_batch_20260623_111111"
    assert by_id["clip_new"]["status"] == "accepted"
    assert any(source["id"] == "new_source" for source in merged["source_videos"])


def test_review_tui_start_position_helpers_follow_last_reviewed_clip(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    for clip_id in ("clip_a", "clip_b", "clip_c"):
        store.add_clip(
            manifest,
            {
                "id": clip_id,
                "status": "accepted",
                "case_id": clip_id,
                "prompt": f"Create a FEM+IPC scene inspired by {clip_id}. Render 10s behavior.",
            },
        )

    assert [clip["id"] for _index, clip in reviewable_clip_positions(manifest)] == ["clip_a", "clip_b", "clip_c"]
    assert resolve_review_start_index(manifest) == 0
    assert resolve_review_start_index(manifest, "2") == 1
    assert resolve_review_start_index(manifest, "clip_c") == 2
    assert case_line(manifest["clips"][0]).startswith("clip_a|Create a FEM+IPC scene")

    manifest["review_state"] = {
        "last_reviewed_clip_id": "clip_a",
        "last_reviewed_manifest_index": 0,
    }
    assert resolve_review_start_index(manifest) == 1

    manifest["review_state"] = {
        "last_reviewed_clip_id": "clip_c",
        "last_reviewed_manifest_index": 2,
    }
    assert resolve_review_start_index(manifest) == 3


def test_review_actions_accept_record_position_and_delete_truncated(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_review",
            "status": "accepted",
            "case_id": "clip_review",
            "prompt": "Create a FEM+IPC scene inspired by a review demo. Render 10s behavior.",
            "clip_path": "clips/clip_review.mp4",
        },
    )
    store.save(manifest)

    accept_event = store.accept_clip("clip_review", reason="looks good")
    state = store.record_review_position("clip_review", manifest_index=0, note="accept")
    manifest = store.load()

    assert accept_event["type"] == "accept"
    assert manifest["clips"][0]["status"] == "accepted"
    assert manifest["clips"][0]["reviewed_at"]
    assert manifest["clips"][0]["accept_reason"] == "looks good"
    assert state["last_reviewed_clip_id"] == "clip_review"
    assert manifest["review_state"]["last_reviewed_manifest_index"] == 0

    truncated_event = store.delete_truncated_clip("clip_review", reason="cut from the middle")
    manifest = store.load()

    assert truncated_event["type"] == "delete_truncated"
    assert truncated_event["negative_memory"] is False
    assert manifest["clips"][0]["status"] == "truncated_deleted"
    assert manifest["clips"][0]["truncated"] is True
    assert all(item.get("type") != "reject" for item in manifest["review_events"])


def test_review_tui_prefers_windows_file_opener_on_wsl(tmp_path, monkeypatch):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"demo")

    def fake_which(name):
        return {
            "xdg-open": "/usr/bin/xdg-open",
            "explorer.exe": "/mnt/c/Windows/explorer.exe",
        }.get(name)

    monkeypatch.setattr(review_tui.shutil, "which", fake_which)
    monkeypatch.setattr(review_tui, "_wsl_windows_path", lambda path: r"\\wsl.localhost\Ubuntu\tmp\clip.mp4")

    label, command = review_tui._open_command_for_path(clip_path, env={"WSL_DISTRO_NAME": "Ubuntu"})

    assert label == "explorer.exe"
    assert command == ["/mnt/c/Windows/explorer.exe", r"\\wsl.localhost\Ubuntu\tmp\clip.mp4"]


def test_review_tui_defaults_to_modern_waiting_editor(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)

    def fake_which(name):
        return {"code": "/usr/bin/code", "vi": "/usr/bin/vi"}.get(name)

    monkeypatch.setattr(review_tui.shutil, "which", fake_which)

    command = review_tui._resolve_editor_command(None, env={})

    assert command == "/usr/bin/code --wait --reuse-window"


def test_bilibili_urls_are_routed_to_ytdlp():
    assert is_ytdlp_supported_url("https://www.bilibili.com/video/BV1964y1275C/")
    assert is_ytdlp_supported_url("https://b23.tv/example")
    assert is_ytdlp_supported_url("https://www.youtube.com/watch?v=demo")
    assert is_ytdlp_supported_url("https://www.youtube.com/embed/demo")
    assert not is_ytdlp_supported_url("https://www.youtube.com/user/DisneyResearchHub")
    assert not is_ytdlp_supported_url("https://www.youtube.com/channel/UCdemo")


def test_failed_source_urls_are_retryable(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    url = "https://www.youtube.com/watch?v=demo"
    store.add_source_video(manifest, {"id": "failed_demo", "url": url, "status": "failed"})
    assert not store.source_url_seen(manifest, url)

    store.add_source_video(manifest, {"id": "ready_demo", "url": url, "status": "ready"})
    assert store.source_url_seen(manifest, url)


def test_ytdlp_resolver_prefers_current_python_module():
    assert resolve_ytdlp_command()[1:] == ["-m", "yt_dlp"]


def test_ytdlp_js_runtime_args_select_supported_runtime(monkeypatch):
    def fake_which(name):
        return {"node": "/usr/bin/node", "deno": "/usr/bin/deno"}.get(name)

    def fake_version_text(path, executable_name):
        return {"deno": "deno 2.3.1\nv8 13.0", "node": "v18.19.1"}.get(executable_name, "")

    monkeypatch.setattr("code_agent.dataset.media.shutil.which", fake_which)
    monkeypatch.setattr("code_agent.dataset.media._runtime_version_text", fake_version_text)

    assert resolve_ytdlp_js_runtime_args() == ["--js-runtimes", "deno:/usr/bin/deno"]


def test_ytdlp_js_runtime_args_rejects_unsupported_node(monkeypatch):
    monkeypatch.setattr(
        "code_agent.dataset.media.shutil.which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr("code_agent.dataset.media._runtime_version_text", lambda path, executable_name: "v18.19.1")

    assert resolve_ytdlp_js_runtime_args() == []


def test_ytdlp_js_runtime_args_accepts_new_node(monkeypatch):
    monkeypatch.setattr(
        "code_agent.dataset.media.shutil.which",
        lambda name: "/usr/local/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr("code_agent.dataset.media._runtime_version_text", lambda path, executable_name: "v22.2.0")

    assert resolve_ytdlp_js_runtime_args() == ["--js-runtimes", "node:/usr/local/bin/node"]


def test_curator_memory_marks_failed_sources_retryable(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    retry_url = "https://www.youtube.com/watch?v=retry"
    ready_url = "https://www.youtube.com/watch?v=ready"
    manifest["source_videos"].extend(
        [
            {"id": "failed_source", "url": retry_url, "status": "failed", "error": "old yt-dlp failed"},
            {"id": "ready_source", "url": ready_url, "status": "ready", "sha256": "abc"},
        ]
    )

    prompt = agents._curator_prompt(  # noqa: SLF001 - this is the unit boundary for prompt policy.
        candidates=[
            SourceCandidate(candidate_id="retry", video_url=retry_url, title="Retry"),
            SourceCandidate(candidate_id="ready", video_url=ready_url, title="Ready"),
        ],
        manifest=manifest,
        similarity_seeds=[],
    )

    assert "retryable" in prompt
    assert "retryable_failed_sources" in prompt
    assert "old yt-dlp failed" in prompt


def test_codex_agent_wrappers_parse_mocked_schema_outputs(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()

    def fake_run_codex_exec(request):
        payload = _payload_for_role(request.role)
        request.final_message_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.01,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            stderr_path=str(stderr_path),
            started_at_unix=time.time(),
            ended_at_unix=time.time(),
        )

    monkeypatch.setattr(agents, "run_codex_exec", fake_run_codex_exec)
    logs_dir = tmp_path / "logs"

    candidates = agents.scout_sources(
        store=store,
        manifest=manifest,
        sources=[],
        needed_clips=1,
        logs_dir=logs_dir,
    )
    assert candidates[0].candidate_id == "cand_1"

    curated = agents.curate_sources(candidates=candidates, manifest=manifest, logs_dir=logs_dir)
    assert curated == candidates

    segments = agents.segment_video(
        source_record={"id": "source_demo", "duration_sec": 2.0},
        deterministic_segments=[],
        timeline_sheet=tmp_path / "missing.jpg",
        logs_dir=logs_dir,
    )
    assert segments[0].title_slug == "soft_demo"

    case_id, prompt = agents.write_prompt(
        clip_record={"id": "clip_demo", "start_sec": 0.0, "end_sec": 2.0, "visual_summary": "soft demo"},
        manifest=manifest,
        clip_sheet=tmp_path / "missing.jpg",
        logs_dir=logs_dir,
    )
    assert case_id == "soft_demo"
    assert prompt.startswith("Create a FEM+IPC scene inspired by")

    duplicate_review = agents.review_duplicate_clip(
        clip_record={"id": "clip_demo", "title": "soft demo"},
        duplicate_candidate={
            "clip": {"id": "clip_existing", "case_id": "soft_demo"},
            "score": 0.84,
            "reason": "foreground_components",
            "metrics": {},
        },
        current_sheet=tmp_path / "missing_current.jpg",
        existing_sheet=tmp_path / "missing_existing.jpg",
        logs_dir=logs_dir,
    )
    assert duplicate_review["decision"] == "duplicate"
    assert duplicate_review["duplicate_of_clip_id"] == "clip_existing"


def test_similarity_seeds_collect_from_cli_file_and_manifest(tmp_path):
    cases_path = tmp_path / "cases.txt"
    cases_path.write_text(
        "\n".join(
            [
                "soft_twist|Create a FEM+IPC scene inspired by a twisting rods demo. Render 10s behavior.",
                "soft_squeeze|Create a FEM+IPC scene inspired by a squeeze-through-gap demo. Render 10s behavior.",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "clips": [
            {
                "id": "clip_1",
                "status": "accepted",
                "case_id": "accepted_case",
                "prompt": "Create a FEM+IPC scene inspired by a refined accepted clip. Render 10s behavior.",
            }
        ],
        "style_memory": [
            {
                "clip_id": "clip_2",
                "case_id": "edited_case",
                "prompt": "Create a FEM+IPC scene inspired by a human edited prompt. Render 10s behavior.",
                "reason": "better wording",
            }
        ],
    }

    seeds = collect_similarity_seeds(
        BuildConfig(
            target_clips=1,
            data_root=tmp_path / "dataset",
            similar_to=("soft_cli|Create a FEM+IPC scene inspired by a CLI seed. Render 10s behavior.",),
            similar_to_file=cases_path,
            similarity_seed_limit=10,
        ),
        manifest,
    )

    prompts = [seed.prompt for seed in seeds]
    assert any("CLI seed" in prompt for prompt in prompts)
    assert any("twisting rods" in prompt for prompt in prompts)
    assert any("human edited prompt" in prompt for prompt in prompts)
    assert any(seed.case_id == "accepted_case" for seed in seeds)


def test_scout_prompt_receives_similarity_seeds(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    captured_prompt = {}

    def fake_run_codex_exec(request):
        captured_prompt["text"] = request.prompt
        payload = _payload_for_role(request.role)
        request.final_message_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.01,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            stderr_path=str(stderr_path),
            started_at_unix=time.time(),
            ended_at_unix=time.time(),
        )

    monkeypatch.setattr(agents, "run_codex_exec", fake_run_codex_exec)
    seeds = collect_similarity_seeds(
        BuildConfig(
            target_clips=1,
            data_root=store.root,
            similar_to=(
                "soft_braid|Create a FEM+IPC scene inspired by a braided rod contact demo. Render 10s behavior.",
            ),
        ),
        manifest,
    )

    agents.scout_sources(
        store=store,
        manifest=manifest,
        sources=[],
        needed_clips=1,
        logs_dir=tmp_path / "logs",
        similarity_seeds=seeds,
    )

    assert "Similarity targets from previously tuned successful prompts" in captured_prompt["text"]
    assert "braided rod contact demo" in captured_prompt["text"]


def test_segment_and_prompt_writer_prompts_warn_against_truncated_examples(tmp_path, monkeypatch):
    captured_prompts: list[str] = []

    def fake_run_codex_exec(request):
        captured_prompts.append(request.prompt)
        payload = _payload_for_role(request.role)
        request.final_message_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.01,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            stderr_path=str(stderr_path),
            started_at_unix=time.time(),
            ended_at_unix=time.time(),
        )

    monkeypatch.setattr(agents, "run_codex_exec", fake_run_codex_exec)
    logs_dir = tmp_path / "logs"

    agents.segment_video(
        source_record={"id": "source_demo", "duration_sec": 4.0},
        deterministic_segments=[],
        timeline_sheet=tmp_path / "missing_timeline.jpg",
        logs_dir=logs_dir,
    )
    agents.write_prompt(
        clip_record={"id": "clip_demo", "start_sec": 1.0, "end_sec": 3.0, "visual_summary": "partial demo"},
        source_record={
            "paper_title": "Example Contact Paper",
            "paper_url": "https://example.test/paper.pdf",
            "project_url": "https://example.test/project",
        },
        manifest={},
        clip_sheet=tmp_path / "missing_clip.jpg",
        logs_dir=logs_dir,
    )

    segment_prompt, writer_prompt = captured_prompts
    assert "Do not cut a complete demo from the middle" in segment_prompt
    assert "omit that segment rather than" in segment_prompt
    assert "returning a partial mid-demo clip" in segment_prompt
    assert "starts after the main setup" in writer_prompt
    assert "cuts off before" in writer_prompt
    assert "the main outcome/settling" in writer_prompt
    assert "Source/paper metadata" in writer_prompt
    assert "Example Contact Paper" in writer_prompt
    assert "Do not create a separate paper-only prompt" in writer_prompt


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video tests")
def test_media_helpers_segment_and_clip_synthetic_video(tmp_path):
    video_path = _make_two_color_video(tmp_path / "two_color.mp4")

    info = probe_video(video_path)
    assert info.duration_sec >= 1.5

    sheet_path = tmp_path / "sheet.jpg"
    build_contact_sheet(video_path, sheet_path, max_frames=4, thumb_width=80)
    assert sheet_path.exists()

    segments = detect_scene_segments(
        video_path, source_id="two_color", threshold=0.2, min_segment_sec=0.5, sample_fps=4
    )
    assert len(segments) >= 2

    clip_path = tmp_path / "clip.mp4"
    cut_clip(video_path, start_sec=segments[0].start_sec, end_sec=segments[0].end_sec, out_path=clip_path)
    assert clip_path.exists()
    assert average_video_hash(clip_path)
    signature = visual_signature(clip_path)
    assert signature["visual_fingerprint"]
    assert signature["frame_fingerprints"]
    assert signature["signature_version"] == 2
    assert "foreground_component_fingerprints" in signature


def test_real_rigid_ipc_duplicate_pairs_detected_by_cv_when_available(tmp_path):
    pairs = [
        (
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_main_video_anchor_chain_threading_disc.mp4"
            ),
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_supplemental_video_intersection_free_rigid_75343efa11.mp4"
            ),
        ),
        (
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_main_video_sliding_plate_ramp_comparison.mp4"
            ),
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_supplemental_video_intersection_free_rigid_3dfe7b75c2.mp4"
            ),
        ),
    ]
    missing = [path for pair in pairs for path in pair if not path.exists()]
    if missing:
        pytest.skip("real dataset calibration clips are not present")

    store = DatasetStore(tmp_path / "dataset")
    for index, (existing_path, current_path) in enumerate(pairs, start=1):
        manifest = store.empty_manifest()
        existing_signature = visual_signature(existing_path)
        store.add_clip(
            manifest,
            {
                "id": f"existing_{index}",
                "status": "accepted",
                "case_id": f"existing_{index}",
                "prompt": "Create a rigid contact scene. Render 10s behavior.",
                **existing_signature,
            },
        )
        candidates = store.duplicate_candidates(manifest, signature=visual_signature(current_path))
        assert candidates, f"{existing_path.name} vs {current_path.name} should be a CV duplicate candidate"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video tests")
def test_materialize_segments_deletes_codex_rejected_duplicate_artifacts(tmp_path, monkeypatch):
    video_path = _make_two_color_video(tmp_path / "source.mp4")
    store = DatasetStore(tmp_path / "dataset")
    store.ensure_dirs()
    manifest = store.empty_manifest()
    source_record = {
        "id": "source_demo",
        "url": str(video_path),
        "duration_sec": 2.0,
    }

    def fake_duplicate_candidates(
        self, manifest, fingerprint=None, *, signature=None, max_results=3, exclude_clip_id=None
    ):
        return [
            {
                "clip": {"id": "existing_clip", "contact_sheet_path": None},
                "clip_id": "existing_clip",
                "score": 0.9,
                "reason": "test_duplicate",
                "metrics": {},
            }
        ]

    def fake_review_duplicate_clip(**kwargs):
        return {
            "decision": "duplicate",
            "duplicate_of_clip_id": "existing_clip",
            "reason": "same demo",
            "confidence": 0.99,
        }

    def fail_write_prompt(**kwargs):
        raise AssertionError("duplicate clips should be skipped before prompt writing")

    monkeypatch.setattr(DatasetStore, "duplicate_candidates", fake_duplicate_candidates)
    monkeypatch.setattr(agents, "review_duplicate_clip", fake_review_duplicate_clip)
    monkeypatch.setattr(agents, "write_prompt", fail_write_prompt)

    added = _materialize_segments(
        store=store,
        manifest=manifest,
        source_record=source_record,
        source_path=video_path,
        segments=[
            SegmentCandidate(
                title_slug="duplicate_segment",
                start_sec=0.0,
                end_sec=1.0,
                visual_summary="duplicate segment",
                reason="test",
            )
        ],
        target=1,
        logs_dir=tmp_path / "logs",
        similarity_seeds=[],
        run_codex=True,
    )

    assert added == 0
    assert not list(store.clips_dir.glob("*.mp4"))
    assert not list(store.contact_sheets_dir.glob("*.jpg"))
    assert not manifest["clips"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video tests")
def test_build_dataset_no_codex_from_explicit_local_video_is_incremental(tmp_path):
    video_path = _make_two_color_video(tmp_path / "explicit.mp4")
    data_root = tmp_path / "dataset"

    summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=(str(video_path),),
            run_codex=False,
        )
    )
    assert summary.status == "complete"
    assert summary.accepted_clips == 1
    assert summary.clips_added == 1

    second_summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=(str(video_path),),
            run_codex=False,
        )
    )
    assert second_summary.status == "already_complete"
    assert second_summary.clips_added == 0

    manifest = DatasetStore(data_root).load()
    accepted = [clip for clip in manifest["clips"] if clip["status"] == "accepted"]
    assert accepted[0]["prompt"].startswith("Create a FEM+IPC scene inspired by")
    assert accepted[0]["clip_uri"].startswith("file://")
    assert accepted[0]["clip_uri"].endswith(".mp4")


def test_build_dataset_direct_sources_skip_scout_but_keep_codex_steps(tmp_path, monkeypatch):
    video_path = _make_two_color_video(tmp_path / "explicit.mp4")
    data_root = tmp_path / "dataset"
    called: list[str] = []

    def fail_scout(**kwargs):
        raise AssertionError("Direct source builds should not call scout")

    def fake_curate_sources(**kwargs):
        called.append("curator")
        return kwargs["candidates"]

    def fake_segment_video(**kwargs):
        called.append("segmenter")
        return [
            SegmentCandidate(
                title_slug="single_demo",
                start_sec=0.0,
                end_sec=2.0,
                visual_summary="one complete synthetic demo",
                reason="test segment",
                confidence=1.0,
            )
        ]

    def fake_write_prompt(**kwargs):
        called.append("prompt_writer")
        return ("single_demo", "Create a FEM+IPC scene inspired by a synthetic contact demo. Render 2s behavior.")

    monkeypatch.setattr(agents, "scout_sources", fail_scout)
    monkeypatch.setattr(agents, "curate_sources", fake_curate_sources)
    monkeypatch.setattr(agents, "segment_video", fake_segment_video)
    monkeypatch.setattr(agents, "write_prompt", fake_write_prompt)

    summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=(str(video_path),),
            run_codex=True,
        )
    )

    assert summary.status == "complete"
    assert called == ["curator", "segmenter", "prompt_writer"]


def test_build_dataset_repeats_scout_until_target_after_seen_url(tmp_path, monkeypatch):
    data_root = tmp_path / "dataset"
    store = DatasetStore(data_root)
    manifest = store.empty_manifest()
    old_url = "https://example.test/already_seen.mp4"
    new_url = "https://example.test/new_demo.mp4"
    store.add_source_video(
        manifest,
        {
            "id": "old_source",
            "url": old_url,
            "status": "ready",
            "sha256": "old_sha",
            "path": "videos/old_source.mp4",
        },
    )
    store.save(manifest)
    scout_calls: list[int] = []

    def fake_scout_sources(**kwargs):
        scout_calls.append(kwargs["needed_clips"])
        if len(scout_calls) == 1:
            return [SourceCandidate(candidate_id="old", video_url=old_url, title="Already Seen")]
        return [SourceCandidate(candidate_id="new", video_url=new_url, title="New Demo")]

    def fake_curate_sources(**kwargs):
        return kwargs["candidates"]

    def fake_download_video(candidate, *, out_dir, source_id, timeout_sec=120.0):
        path = out_dir / f"{source_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake video")
        return path

    def fake_describe_download(path):
        return DownloadedVideo(
            path=path,
            sha256=f"sha_{path.stem}",
            bytes=path.stat().st_size,
            info=VideoInfo(duration_sec=2.0, width=160, height=120),
        )

    def fake_build_contact_sheet(video_path, out_path, *, max_frames=12, thumb_width=180):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake sheet")

    def fake_segment_video(**kwargs):
        return [
            SegmentCandidate(
                title_slug="new_single_demo",
                start_sec=0.0,
                end_sec=2.0,
                visual_summary="one complete new demo",
                reason="test",
            )
        ]

    def fake_materialize_segments(*, store, manifest, source_record, source_path, segments, target, **kwargs):
        store.add_clip(
            manifest,
            {
                "id": "new_clip",
                "source_video_id": source_record["id"],
                "status": "accepted",
                "case_id": "new_clip",
                "prompt": "Create a rigid scene inspired by a new demo. Render 2s behavior.",
                "category": "rigid",
                "prompt_revisions": [],
            },
        )
        return 1

    monkeypatch.setattr(agents, "scout_sources", fake_scout_sources)
    monkeypatch.setattr(agents, "curate_sources", fake_curate_sources)
    monkeypatch.setattr(agents, "segment_video", fake_segment_video)
    monkeypatch.setattr(dataset_builder, "download_video", fake_download_video)
    monkeypatch.setattr(dataset_builder, "describe_download", fake_describe_download)
    monkeypatch.setattr(dataset_builder, "build_contact_sheet", fake_build_contact_sheet)
    monkeypatch.setattr(dataset_builder, "detect_scene_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(dataset_builder, "_materialize_segments", fake_materialize_segments)

    summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=("https://kesen.realtimerendering.com/",),
            max_scout_rounds=2,
        )
    )

    assert summary.status == "complete"
    assert summary.scout_rounds == 2
    assert summary.empty_scout_rounds == 1
    assert summary.candidates_seen == 2
    assert summary.clips_added == 1
    assert any(item == f"source_url_seen:{old_url}" for item in summary.skipped)
    assert DatasetStore(data_root).accepted_count() == 1


def _payload_for_role(role: str) -> dict[str, object]:
    if role == "dataset_scout":
        return {
            "candidate_sources": [
                {
                    "candidate_id": "cand_1",
                    "video_url": "https://example.test/demo.mp4",
                    "title": "Demo",
                    "project_url": None,
                    "paper_url": "https://example.test/paper.pdf",
                    "paper_title": "Paper",
                    "venue": "SIGGRAPH",
                    "source_url": "https://example.test/project",
                    "license_notes": None,
                    "source_policy_notes": "public project page",
                    "notes": "good deformable demo",
                    "confidence": 0.9,
                }
            ]
        }
    if role == "dataset_curator":
        return {
            "decisions": [
                {"candidate_id": "cand_1", "status": "accept", "reason": "usable", "avoid_similarity_note": None}
            ]
        }
    if role.startswith("dataset_segmenter"):
        return {
            "segments": [
                {
                    "title_slug": "soft_demo",
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                    "visual_summary": "soft body squeezes through a gap",
                    "reason": "one self-contained demo",
                    "confidence": 0.8,
                }
            ]
        }
    if role.startswith("dataset_duplicate_reviewer"):
        return {
            "decision": "duplicate",
            "duplicate_of_clip_id": "clip_existing",
            "reason": "same dominant visual demo",
            "confidence": 0.9,
        }
    return {
        "case_id": "soft_demo",
        "prompt": (
            "Create a FEM+IPC scene inspired by a soft body squeeze demo: a hyperelastic toy is pushed through a "
            "narrow rigid gap, visibly squashes, slides, rebounds, and settles through contact. Render 10s behavior."
        ),
        "coverage": "deformable",
        "notes": None,
    }


def _make_two_color_video(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=1:r=24",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=1:r=24",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            str(path),
        ],
        check=True,
        timeout=30,
    )
    return path
