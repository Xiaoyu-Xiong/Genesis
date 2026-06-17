from __future__ import annotations

import curses
import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from code_agent.dataset.store import DEFAULT_DATA_ROOT, DatasetStore

CATEGORY_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "rigid", "rigid"),
    ("2", "deformable bodies", "deformable_bodies"),
    ("3", "cloth", "cloth"),
)

REJECT_REASONS: tuple[tuple[str, str, str], ...] = (
    ("r", "Reject: not suitable", "not suitable for Genesis-feasible case generation"),
    ("m", "Reject: precision mesh", "requires precise mesh geometry or exact scanned/model-specific assets"),
    ("s", "Reject: too simple", "too simple, static, or lacks meaningful physical interaction"),
    ("u", "Reject: unsupported physics", "unsupported or too specialized for the current Genesis target set"),
    ("o", "Reject: cloth/render/fluid/RL-only", "cloth-only, rendering-only, fluid/smoke, hair, or RL-heavy demo"),
)

ACTION_HELP: tuple[tuple[str, str], ...] = (
    ("a", "Accept and mark reviewed"),
    ("e", "Edit prompt in $EDITOR"),
    ("g", "Choose category label"),
    *tuple((key, f"Tag: {label}") for key, label, _category in CATEGORY_ACTIONS),
    *tuple((key, label) for key, label, _reason in REJECT_REASONS),
    ("c", "Reject with custom reason"),
    ("d", "Exclude duplicate, no negative memory"),
    ("x", "Exclude multi-example clip, no negative memory"),
    ("t", "Exclude truncated/incomplete clip, no negative memory"),
    ("v", "Open clip video"),
    ("i", "Open contact sheet"),
    ("n", "Next accepted clip"),
    ("q", "Quit"),
)


def reviewable_clip_positions(manifest: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, clip)
        for index, clip in enumerate(manifest.get("clips") or [])
        if isinstance(clip, dict) and clip.get("status") == "accepted"
    ]


def resolve_review_start_index(manifest: dict[str, Any], start: str | None = None) -> int:
    reviewable = reviewable_clip_positions(manifest)
    if not reviewable:
        return 0
    if start is None or not str(start).strip():
        return _next_after_saved_review_state(manifest, reviewable)

    start_text = str(start).strip()
    if start_text.isdecimal():
        position = int(start_text)
        if 1 <= position <= len(reviewable):
            return position - 1
        raise ValueError(f"Start position must be between 1 and {len(reviewable)}.")

    for index, (_manifest_index, clip) in enumerate(reviewable):
        if clip.get("id") == start_text:
            return index
    raise ValueError(f"Start clip id is not an accepted clip in the manifest: {start_text}")


def next_review_index_after_manifest_index(manifest: dict[str, Any], manifest_index: int) -> int:
    reviewable = reviewable_clip_positions(manifest)
    for index, (candidate_manifest_index, _clip) in enumerate(reviewable):
        if candidate_manifest_index > manifest_index:
            return index
    return len(reviewable)


def previous_review_index_before_manifest_index(manifest: dict[str, Any], manifest_index: int) -> int | None:
    reviewable = reviewable_clip_positions(manifest)
    for index in range(len(reviewable) - 1, -1, -1):
        candidate_manifest_index, _clip = reviewable[index]
        if candidate_manifest_index < manifest_index:
            return index
    return None


def _clip_at_manifest_index(manifest: dict[str, Any], manifest_index: int) -> dict[str, Any] | None:
    clips = manifest.get("clips") or []
    if 0 <= manifest_index < len(clips) and isinstance(clips[manifest_index], dict):
        return clips[manifest_index]
    return None


def _review_index_for_manifest_index(
    reviewable: list[tuple[int, dict[str, Any]]],
    manifest_index: int,
) -> int | None:
    for index, (candidate_manifest_index, _clip) in enumerate(reviewable):
        if candidate_manifest_index == manifest_index:
            return index
    return None


def case_line(clip: dict[str, Any]) -> str:
    case_id = str(clip.get("case_id") or clip.get("id") or "case").strip()
    prompt = str(clip.get("prompt") or "").strip()
    return f"{case_id}|{prompt}"


def run_review_tui(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    start: str | None = None,
    editor: str | None = None,
    auto_play: bool = True,
) -> None:
    store = DatasetStore(data_root)
    manifest = store.load()
    initial_index = resolve_review_start_index(manifest, start)
    curses.wrapper(_review_loop, store, initial_index, editor, auto_play)


def _next_after_saved_review_state(
    manifest: dict[str, Any],
    reviewable: list[tuple[int, dict[str, Any]]],
) -> int:
    state = manifest.get("review_state")
    if not isinstance(state, dict):
        return 0

    manifest_index = state.get("last_reviewed_manifest_index")
    if isinstance(manifest_index, int):
        for index, (candidate_manifest_index, _clip) in enumerate(reviewable):
            if candidate_manifest_index > manifest_index:
                return index
        return len(reviewable)

    clip_id = state.get("last_reviewed_clip_id")
    if isinstance(clip_id, str) and clip_id:
        for index, (_manifest_index, clip) in enumerate(reviewable):
            if clip.get("id") == clip_id:
                return min(index + 1, len(reviewable))
    return 0


def _review_loop(
    stdscr: curses.window,
    store: DatasetStore,
    current_index: int,
    editor: str | None,
    auto_play: bool,
) -> None:
    curses.curs_set(0)
    status_message = "Ready."
    last_auto_opened_clip_id: str | None = None
    current_manifest_index: int | None = None
    while True:
        manifest = store.load()
        reviewable = reviewable_clip_positions(manifest)
        if current_manifest_index is None:
            if current_index < len(reviewable):
                current_manifest_index = reviewable[current_index][0]
            else:
                _draw_done(stdscr, len(reviewable), status_message)
                key = _read_key(stdscr)
                if key in {"q", "Q", "\x1b"}:
                    return
                if not reviewable:
                    current_index = 0
                    continue
                current_index = max(0, len(reviewable) - 1)
                current_manifest_index = reviewable[current_index][0]
                continue

        clip = _clip_at_manifest_index(manifest, current_manifest_index)
        if clip is None:
            current_manifest_index = None
            current_index = min(current_index, len(reviewable))
            status_message = "Current clip disappeared from manifest."
            continue

        accepted_index = _review_index_for_manifest_index(reviewable, current_manifest_index)
        current_index = (
            accepted_index
            if accepted_index is not None
            else next_review_index_after_manifest_index(manifest, current_manifest_index)
        )

        if current_index >= len(reviewable) and clip.get("status") == "accepted":
            _draw_done(stdscr, len(reviewable), status_message)
            key = _read_key(stdscr)
            if key in {"q", "Q", "\x1b"}:
                return
            current_index = max(0, len(reviewable) - 1)
            current_manifest_index = reviewable[current_index][0] if reviewable else None
            continue

        manifest_index = current_manifest_index
        clip_id = str(clip.get("id"))
        if auto_play and clip_id != last_auto_opened_clip_id:
            status_message = _open_clip_path(store, clip, "clip_path")
            last_auto_opened_clip_id = clip_id
        _draw_clip(stdscr, store, clip, current_index, len(reviewable), manifest_index, status_message)
        key = _read_key(stdscr)
        status_message = ""

        if key in {"q", "Q", "\x1b"}:
            return
        if key in {"n", "N", "KEY_RIGHT"}:
            store.record_review_position(clip_id, manifest_index=manifest_index, note="manual next")
            fresh_manifest = store.load()
            fresh_reviewable = reviewable_clip_positions(fresh_manifest)
            current_index = next_review_index_after_manifest_index(fresh_manifest, manifest_index)
            current_manifest_index = (
                fresh_reviewable[current_index][0] if current_index < len(fresh_reviewable) else None
            )
            status_message = "Moved to next accepted clip."
            continue
        if key in {"KEY_LEFT"}:
            previous_index = previous_review_index_before_manifest_index(manifest, manifest_index)
            if previous_index is None:
                status_message = "Already at the first accepted clip."
            else:
                current_index = previous_index
                current_manifest_index = reviewable[current_index][0]
                status_message = "Moved back one accepted clip."
            continue
        if key in {"v", "V"}:
            status_message = _open_clip_path(store, clip, "clip_path")
            continue
        if key in {"i", "I"}:
            status_message = _open_clip_path(store, clip, "contact_sheet_path")
            continue
        if key in {"a", "A"}:
            store.accept_clip(clip_id, reason="accepted in review TUI")
            status_message = f"Accepted {clip_id}. Press n/right for next."
            continue
        if key in {"e", "E"}:
            edited = _edit_prompt(stdscr, case_line(clip), editor)
            if edited is None:
                status_message = "Edit cancelled."
                continue
            if edited.strip() == case_line(clip).strip():
                status_message = "Prompt unchanged; no edit saved."
                continue
            reason = _prompt_line(stdscr, "Edit reason", default="human rewrite")
            store.edit_clip(clip_id, prompt=edited, reason=reason or "human rewrite")
            status_message = f"Edited {clip_id}. Press n/right for next."
            continue
        category = _category_for_key(key)
        if category is not None:
            store.set_clip_category(clip_id, category=category, reason="human category label")
            status_message = f"Tagged {clip_id} as {category}. Press n/right for next."
            continue
        if key in {"g", "G"}:
            category = _choose_category(stdscr, current=str(clip.get("category") or ""))
            if category is None:
                status_message = "Category unchanged."
                continue
            reason = _prompt_line(stdscr, "Category reason", default="human category label")
            store.set_clip_category(clip_id, category=category, reason=reason or "human category label")
            status_message = f"Tagged {clip_id} as {category}. Press n/right for next."
            continue

        reject_reason = _reject_reason_for_key(key)
        if reject_reason is not None:
            store.reject_clip(clip_id, reason=reject_reason, avoid_similarity_note=reject_reason)
            store.record_review_position(clip_id, manifest_index=manifest_index, note=f"reject: {reject_reason}")
            fresh_manifest = store.load()
            fresh_reviewable = reviewable_clip_positions(fresh_manifest)
            current_index = next_review_index_after_manifest_index(fresh_manifest, manifest_index)
            current_manifest_index = (
                fresh_reviewable[current_index][0] if current_index < len(fresh_reviewable) else None
            )
            status_message = f"Rejected {clip_id}; moved to next accepted clip."
            continue
        if key in {"c", "C"}:
            reason = _prompt_line(stdscr, "Reject reason", default="not suitable")
            store.reject_clip(clip_id, reason=reason, avoid_similarity_note=reason)
            store.record_review_position(clip_id, manifest_index=manifest_index, note=f"reject: {reason}")
            fresh_manifest = store.load()
            fresh_reviewable = reviewable_clip_positions(fresh_manifest)
            current_index = next_review_index_after_manifest_index(fresh_manifest, manifest_index)
            current_manifest_index = (
                fresh_reviewable[current_index][0] if current_index < len(fresh_reviewable) else None
            )
            status_message = f"Rejected {clip_id}; moved to next accepted clip."
            continue
        if key in {"d", "D"}:
            duplicate_of = _prompt_line(stdscr, "Duplicate of clip id (optional)", default="")
            reason = _prompt_line(stdscr, "Delete reason", default="duplicate clip removed from active dataset")
            store.delete_duplicate_clip(clip_id, duplicate_of_clip_id=duplicate_of or None, reason=reason)
            store.record_review_position(clip_id, manifest_index=manifest_index, note=f"delete duplicate: {reason}")
            fresh_manifest = store.load()
            fresh_reviewable = reviewable_clip_positions(fresh_manifest)
            current_index = next_review_index_after_manifest_index(fresh_manifest, manifest_index)
            current_manifest_index = (
                fresh_reviewable[current_index][0] if current_index < len(fresh_reviewable) else None
            )
            status_message = f"Deleted duplicate {clip_id} without negative memory; moved to next accepted clip."
            continue
        if key in {"x", "X"}:
            reason = _prompt_line(stdscr, "Delete reason", default="clip contains multiple independent examples")
            store.delete_multi_example_clip(clip_id, reason=reason)
            store.record_review_position(clip_id, manifest_index=manifest_index, note=f"delete multi-example: {reason}")
            fresh_manifest = store.load()
            fresh_reviewable = reviewable_clip_positions(fresh_manifest)
            current_index = next_review_index_after_manifest_index(fresh_manifest, manifest_index)
            current_manifest_index = (
                fresh_reviewable[current_index][0] if current_index < len(fresh_reviewable) else None
            )
            status_message = f"Deleted multi-example clip {clip_id} without negative memory; moved to next accepted clip."
            continue
        if key in {"t", "T"}:
            reason = _prompt_line(stdscr, "Delete reason", default="clip is truncated or incomplete")
            store.delete_truncated_clip(clip_id, reason=reason)
            store.record_review_position(clip_id, manifest_index=manifest_index, note=f"delete truncated: {reason}")
            fresh_manifest = store.load()
            fresh_reviewable = reviewable_clip_positions(fresh_manifest)
            current_index = next_review_index_after_manifest_index(fresh_manifest, manifest_index)
            current_manifest_index = (
                fresh_reviewable[current_index][0] if current_index < len(fresh_reviewable) else None
            )
            status_message = f"Deleted truncated clip {clip_id} without negative memory; moved to next accepted clip."
            continue

        status_message = f"Unknown action: {key!r}"


def _reject_reason_for_key(key: str) -> str | None:
    key = key.lower()
    for action_key, _label, reason in REJECT_REASONS:
        if key == action_key:
            return reason
    return None


def _category_for_key(key: str) -> str | None:
    key = key.lower()
    for action_key, _label, category in CATEGORY_ACTIONS:
        if key == action_key:
            return category
    return None


def _choose_category(stdscr: curses.window, *, current: str = "") -> str | None:
    height, width = stdscr.getmaxyx()
    lines = ["Choose category label:"]
    lines.extend(f"{key}  {label}" for key, label, _category in CATEGORY_ACTIONS)
    lines.append("Esc/q  cancel")
    if current:
        lines.insert(1, f"Current: {current}")
    box_width = min(max(len(line) for line in lines) + 4, max(20, width - 2))
    box_height = min(len(lines) + 2, max(4, height - 2))
    start_y = max(0, (height - box_height) // 2)
    start_x = max(0, (width - box_width) // 2)
    for offset in range(box_height):
        _addstr(stdscr, start_y + offset, start_x, " " * box_width, curses.A_REVERSE)
    for offset, line in enumerate(lines[: box_height - 1], start=1):
        _addstr(stdscr, start_y + offset, start_x + 2, line, curses.A_REVERSE)
    stdscr.refresh()
    while True:
        key = _read_key(stdscr)
        if key in {"q", "Q", "\x1b"}:
            return None
        category = _category_for_key(key)
        if category is not None:
            return category


def _draw_clip(
    stdscr: curses.window,
    store: DatasetStore,
    clip: dict[str, Any],
    current_index: int,
    total: int,
    manifest_index: int,
    status_message: str,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    y = 0
    _addstr(stdscr, y, 0, "Dataset Review TUI", curses.A_BOLD)
    _addstr(stdscr, y, max(0, width - 18), "q: quit")
    y += 1
    status = str(clip.get("status") or "unknown")
    if status == "accepted" and current_index < total:
        position_text = f"Accepted item {current_index + 1}/{total}"
    else:
        next_text = "none" if current_index >= total else f"{current_index + 1}/{total}"
        position_text = f"Current status {status}; next accepted {next_text}"
    _addstr(
        stdscr,
        y,
        0,
        f"{position_text} | manifest index {manifest_index} | clip {clip.get('id')}",
    )
    y += 2

    fields = [
        ("case_id", clip.get("case_id")),
        ("status", clip.get("status")),
        ("category", clip.get("category") or "(unset)"),
        ("source", clip.get("source_url") or clip.get("source_video_id")),
        ("time", _format_time_range(clip)),
        ("clip", clip.get("clip_uri") or clip.get("clip_path")),
        ("sheet", _absolute_text(store, clip.get("contact_sheet_path"))),
    ]
    for label, value in fields:
        if y >= height - 2:
            break
        _addstr(stdscr, y, 0, f"{label}: ", curses.A_BOLD)
        _addstr(stdscr, y, len(label) + 2, str(value or ""))
        y += 1

    y += 1
    if y < height - 2:
        _addstr(stdscr, y, 0, "case_id|prompt:", curses.A_BOLD)
        y += 1
    action_columns = 2 if width >= 88 else 1
    action_rows = (len(ACTION_HELP) + action_columns - 1) // action_columns
    reserved = action_rows + 4
    prompt_height = max(3, height - y - reserved)
    for line in _wrap_text(case_line(clip), max(20, width - 2))[:prompt_height]:
        _addstr(stdscr, y, 0, line)
        y += 1

    action_y = max(y + 1, height - action_rows - 2)
    _addstr(stdscr, action_y, 0, "Choose action:", curses.A_BOLD)
    column_width = max(24, width // action_columns)
    for index, (key, label) in enumerate(ACTION_HELP):
        row = index % action_rows
        column = index // action_rows
        x = column * column_width
        _addstr(stdscr, action_y + row + 1, x, f"{key:>2}  {label}")

    if status_message:
        _addstr(stdscr, height - 1, 0, status_message[: max(0, width - 1)], curses.A_REVERSE)
    stdscr.refresh()


def _draw_done(stdscr: curses.window, total: int, status_message: str) -> None:
    stdscr.erase()
    _addstr(stdscr, 0, 0, "Dataset Review TUI", curses.A_BOLD)
    _addstr(stdscr, 2, 0, f"No more accepted clips after the current review cursor. Total accepted clips: {total}.")
    _addstr(stdscr, 4, 0, "Press q to quit, or any other key to return to the last accepted clip.")
    if status_message:
        height, width = stdscr.getmaxyx()
        _addstr(stdscr, height - 1, 0, status_message[: max(0, width - 1)], curses.A_REVERSE)
    stdscr.refresh()


def _edit_prompt(stdscr: curses.window, initial_text: str, editor: str | None) -> str | None:
    editor_command = _resolve_editor_command(editor)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", prefix="dataset_prompt_", delete=False
        ) as file:
            temp_path = Path(file.name)
            file.write(initial_text.rstrip() + "\n")

        curses.def_prog_mode()
        curses.endwin()
        command = _editor_command(editor_command, temp_path)
        result = subprocess.run(command, check=False)
        curses.reset_prog_mode()
        curses.curs_set(0)
        stdscr.refresh()
        if result.returncode != 0:
            return None
        return temp_path.read_text(encoding="utf-8").strip()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _resolve_editor_command(editor: str | None, *, env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    if editor and editor.strip():
        return editor.strip()
    for key in ("VISUAL", "EDITOR"):
        command = env.get(key)
        if command and command.strip():
            return command.strip()

    for binary, args in (
        ("code", ("--wait", "--reuse-window")),
        ("cursor", ("--wait", "--reuse-window")),
        ("micro", ()),
        ("nano", ()),
        ("notepad.exe", ()),
        ("vi", ()),
    ):
        executable = shutil.which(binary)
        if executable:
            return " ".join([shlex.quote(executable), *args])
    return "vi"


def _editor_command(editor_command: str, temp_path: Path) -> list[str]:
    parts = shlex.split(editor_command)
    if not parts:
        return ["vi", str(temp_path)]
    temp_text = str(temp_path)
    if any("{}" in part for part in parts):
        return [part.replace("{}", temp_text) for part in parts]
    return [*parts, temp_text]


def _prompt_line(stdscr: curses.window, prompt: str, *, default: str = "") -> str:
    height, width = stdscr.getmaxyx()
    label = f"{prompt} [{default}]: " if default else f"{prompt}: "
    curses.curs_set(1)
    curses.echo()
    try:
        stdscr.move(height - 1, 0)
        stdscr.clrtoeol()
        _addstr(stdscr, height - 1, 0, label)
        raw = stdscr.getstr(height - 1, min(len(label), max(0, width - 1)), max(1, width - len(label) - 1))
    finally:
        curses.noecho()
        curses.curs_set(0)
    text = raw.decode("utf-8", errors="replace").strip()
    return text or default


def _read_key(stdscr: curses.window) -> str:
    key = stdscr.getkey()
    return key


def _open_clip_path(store: DatasetStore, clip: dict[str, Any], key: str) -> str:
    value = clip.get(key)
    if not isinstance(value, str) or not value:
        return f"No {key} recorded for this clip."
    path = store.abspath(value)
    if not path.exists():
        return f"Missing file: {path}"
    opener = _open_command_for_path(path)
    if opener is None:
        return f"No file opener found. Path: {path}"
    label, command = opener
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{label} failed: {exc}. Path: {path}"
    if result.returncode != 0:
        detail = _first_output_line(result.stderr) or _first_output_line(result.stdout) or "no error output"
        return f"{label} failed with exit {result.returncode}: {detail}. Path: {path}"
    return f"Opened via {label}: {path}"


def _open_command_for_path(path: Path, *, env: Mapping[str, str] | None = None) -> tuple[str, list[str]] | None:
    env = os.environ if env is None else env
    if _is_wsl(env):
        if opener := shutil.which("wslview"):
            return "wslview", [opener, str(path)]
        windows_path = _wsl_windows_path(path)
        if windows_path:
            if opener := shutil.which("explorer.exe"):
                return "explorer.exe", [opener, windows_path]
            if opener := shutil.which("powershell.exe"):
                return (
                    "powershell.exe",
                    [
                        opener,
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        "Start-Process -FilePath $args[0]",
                        windows_path,
                    ],
                )
            if opener := shutil.which("cmd.exe"):
                return "cmd.exe", [opener, "/C", "start", "", windows_path]

    if opener := shutil.which("xdg-open"):
        return "xdg-open", [opener, str(path)]
    if opener := shutil.which("open"):
        return "open", [opener, str(path)]
    return None


def _is_wsl(env: Mapping[str, str]) -> bool:
    return bool(env.get("WSL_DISTRO_NAME") or env.get("WSL_INTEROP"))


def _wsl_windows_path(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _first_output_line(text: str | None) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _absolute_text(store: DatasetStore, value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return str(store.abspath(value))


def _format_time_range(clip: dict[str, Any]) -> str:
    start = clip.get("start_sec")
    end = clip.get("end_sec")
    duration = clip.get("duration_sec")
    if isinstance(start, int | float) and isinstance(end, int | float):
        if isinstance(duration, int | float):
            return f"{start:.2f}s-{end:.2f}s ({duration:.2f}s)"
        return f"{start:.2f}s-{end:.2f}s"
    return ""


def _wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(raw_line, width=width, replace_whitespace=False) or [""])
    return lines


def _addstr(stdscr: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x < 0 or x >= width:
        return
    max_len = max(0, width - x - 1)
    if max_len <= 0:
        return
    try:
        stdscr.addstr(y, x, str(text)[:max_len], attr)
    except curses.error:
        pass
