from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_agent.dataset.builder import build_dataset
from code_agent.dataset.models import BuildConfig
from code_agent.dataset.review_tui import run_review_tui
from code_agent.dataset.store import DEFAULT_DATA_ROOT, DatasetStore


def _cmd_build(args: argparse.Namespace) -> None:
    summary = build_dataset(
        BuildConfig(
            target_clips=args.target_clips,
            data_root=args.data_root,
            sources=tuple(args.source or ()),
            source_file=args.source_file,
            similar_to=tuple(args.similar_to or ()),
            similar_to_file=args.similar_to_file,
            similarity_seed_limit=args.similarity_seed_limit,
            max_candidates=args.max_candidates,
            max_downloads=args.max_downloads,
            run_codex=not args.no_codex,
        )
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


def _cmd_status(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    print(json.dumps(store.status_summary(), indent=2, ensure_ascii=False))


def _cmd_reject(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.reject_clip(args.clip_id, reason=args.reason, avoid_similarity_note=args.avoid_similarity_note)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_edit(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.edit_clip(args.clip_id, prompt=args.prompt, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_set_category(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.set_clip_category(args.clip_id, category=args.category, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_delete_duplicate(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.delete_duplicate_clip(
        args.clip_id,
        duplicate_of_clip_id=args.duplicate_of,
        reason=args.reason,
    )
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_delete_multi_example(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.delete_multi_example_clip(args.clip_id, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_review_tui(args: argparse.Namespace) -> None:
    run_review_tui(
        args.data_root,
        start=args.start,
        editor=args.editor,
        auto_play=not args.no_auto_play,
    )


def _cmd_export_cases(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    count = store.export_cases(args.out)
    print(f"Exported {count} accepted cases to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m code_agent.dataset.cli")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Incrementally build video-clip to code_agent-prompt pairs.")
    build.add_argument("--target-clips", type=int, required=True)
    build.add_argument(
        "--source", action="append", help="Explicit project page, direct video URL, or local video path."
    )
    build.add_argument("--source-file", type=Path, default=None)
    build.add_argument(
        "--similar-to",
        action="append",
        help="Previously tuned case prompt, case_id|prompt line, or behavior description to search around.",
    )
    build.add_argument(
        "--similar-to-file",
        type=Path,
        default=None,
        help="Text cases file or JSON manifest/summary containing successful prompts to use as similarity targets.",
    )
    build.add_argument(
        "--similarity-seed-limit",
        type=int,
        default=12,
        help="Maximum number of similarity target prompts injected into Codex subagents.",
    )
    build.add_argument("--max-candidates", type=int, default=None)
    build.add_argument("--max-downloads", type=int, default=None)
    build.add_argument("--no-codex", action="store_true", help="Use only explicit sources and deterministic fallbacks.")
    build.set_defaults(func=_cmd_build)

    status = sub.add_parser("status", help="Print dataset manifest status.")
    status.set_defaults(func=_cmd_status)

    reject = sub.add_parser("reject", help="Reject a clip and record future-avoidance memory.")
    reject.add_argument("--clip-id", required=True)
    reject.add_argument("--reason", required=True)
    reject.add_argument("--avoid-similarity-note", default=None)
    reject.set_defaults(func=_cmd_reject)

    edit = sub.add_parser("edit", help="Edit a clip prompt and add the edit to style memory.")
    edit.add_argument("--clip-id", required=True)
    edit.add_argument("--prompt", required=True)
    edit.add_argument("--reason", default=None)
    edit.set_defaults(func=_cmd_edit)

    category = sub.add_parser("set-category", help="Set a broad category label on a clip.")
    category.add_argument("--clip-id", required=True)
    category.add_argument(
        "--category",
        required=True,
        help="Broad clip category label: rigid, deformable bodies, or cloth.",
    )
    category.add_argument("--reason", default="human category label")
    category.set_defaults(func=_cmd_set_category)

    delete_duplicate = sub.add_parser(
        "delete-duplicate",
        help="Remove a leaked duplicate from the active dataset without adding negative rejection memory.",
    )
    delete_duplicate.add_argument("--clip-id", required=True)
    delete_duplicate.add_argument("--duplicate-of", default=None)
    delete_duplicate.add_argument("--reason", default=None)
    delete_duplicate.set_defaults(func=_cmd_delete_duplicate)

    delete_multi_example = sub.add_parser(
        "delete-multi-example",
        help="Remove a clip that contains multiple independent examples without adding negative rejection memory.",
    )
    delete_multi_example.add_argument("--clip-id", required=True)
    delete_multi_example.add_argument("--reason", default=None)
    delete_multi_example.set_defaults(func=_cmd_delete_multi_example)

    review_tui = sub.add_parser("review-tui", help="Interactively review accepted dataset clips.")
    review_tui.add_argument(
        "--start",
        default=None,
        help="1-based accepted item index or accepted clip id. Defaults to after the last reviewed clip.",
    )
    review_tui.add_argument(
        "--editor",
        default=None,
        help="Editor command for prompt edits; defaults to $VISUAL/$EDITOR, then code/cursor/nano/vi.",
    )
    review_tui.add_argument(
        "--no-auto-play",
        action="store_true",
        help="Do not automatically open the clip video when moving to a new example.",
    )
    review_tui.set_defaults(func=_cmd_review_tui)

    export = sub.add_parser("export-cases", help="Export accepted clips as case_id|prompt lines.")
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=_cmd_export_cases)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
