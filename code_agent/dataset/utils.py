from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slugify(value: str, *, fallback: str = "item", max_len: int = 80) -> str:
    text = value.strip().lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    return text[:max_len].strip("_") or fallback


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: str, *, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def is_probably_video_url(url: str) -> bool:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    return suffix in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def is_youtube_or_vimeo_url(url: str) -> bool:
    return is_ytdlp_supported_url(url)


def is_ytdlp_supported_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(domain in host for domain in ("youtube.com", "youtu.be", "vimeo.com", "bilibili.com", "b23.tv"))


def hamming_distance_hex(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    try:
        left_int = int(left, 16)
        right_int = int(right, 16)
    except ValueError:
        return None
    return (left_int ^ right_int).bit_count()


def first_nonempty(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""
