from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from urllib import error, parse, request

from .models import MeshyApiConfig, MeshyGenerationConfig, MeshyRequestError


class MeshyClient:
    def __init__(self, config: MeshyApiConfig) -> None:
        self.config = config

    def submit_text_to_mesh(self, generation: MeshyGenerationConfig) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": "preview",
            "prompt": generation.prompt,
            "ai_model": generation.ai_model,
            "art_style": generation.art_style,
            "should_remesh": generation.should_remesh,
            "topology": generation.topology,
            "symmetry_mode": generation.symmetry_mode,
            "moderation": generation.moderation,
            "target_formats": [generation.mesh_format],
            "auto_size": generation.auto_size,
        }
        if generation.target_polycount is not None:
            payload["target_polycount"] = generation.target_polycount
        if generation.negative_prompt is not None:
            payload["negative_prompt"] = generation.negative_prompt
        if generation.origin_at is not None:
            payload["origin_at"] = generation.origin_at
        payload.update(generation.extra_payload)
        return self._post_json(self.config.text_to_3d_path, payload)

    def wait_for_preview_completion(
        self,
        *,
        preview_task_id: str,
        poll_interval_sec: float,
        max_wait_sec: float,
    ) -> dict[str, object]:
        deadline = time.monotonic() + max_wait_sec
        while True:
            if time.monotonic() > deadline:
                raise MeshyRequestError(
                    f"Meshy preview task `{preview_task_id}` timed out after {max_wait_sec:.1f}s."
                )
            response = self._get_json(f"{self.config.text_to_3d_path}/{preview_task_id}")
            status = _status_of(response)
            if status in MESHY_READY_SET:
                return response
            if status in MESHY_FAILED_SET:
                message = _task_error_message(response)
                raise MeshyRequestError(
                    f"Meshy preview task `{preview_task_id}` failed with status `{status}`. {message}".strip()
                )
            time.sleep(poll_interval_sec)

    def download_mesh(self, *, task_response: dict[str, object], output_dir: Path, mesh_format: str) -> Path:
        model_urls = task_response.get("model_urls")
        if not isinstance(model_urls, dict):
            raise MeshyRequestError("Meshy task response did not contain `model_urls`.")

        download_url = model_urls.get(mesh_format)
        if not isinstance(download_url, str) or not download_url.strip():
            available = ", ".join(sorted(str(key) for key in model_urls))
            raise MeshyRequestError(
                f"Meshy task response did not contain a `{mesh_format}` model URL. Available keys: {available}"
            )

        downloads_dir = output_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        out_path = downloads_dir / f"model.{mesh_format}"
        self._download_file(download_url, out_path)

        if mesh_format == "obj":
            mtl_url = model_urls.get("mtl")
            if isinstance(mtl_url, str) and mtl_url.strip():
                self._download_file(mtl_url, downloads_dir / "model.mtl")
        return out_path

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        url = _join_url(self.config.base_url, path)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, method="POST", data=body, headers=self.config.auth_headers())
        return self._load_json_response(req, label="submit")

    def _get_json(self, path: str) -> dict[str, object]:
        url = _join_url(self.config.base_url, path)
        req = request.Request(url=url, method="GET", headers=self.config.auth_headers())
        return self._load_json_response(req, label="status")

    def _load_json_response(self, req: request.Request, *, label: str) -> dict[str, object]:
        try:
            with request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise MeshyRequestError(f"Meshy {label} HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise MeshyRequestError(f"Meshy {label} request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise MeshyRequestError(f"Meshy {label} request timed out after {self.config.timeout_sec:.1f}s.") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MeshyRequestError(f"Meshy {label} response is not valid JSON: {raw[:500]}") from exc
        if not isinstance(parsed, dict):
            raise MeshyRequestError(f"Meshy {label} response root is not an object.")
        return parsed

    def _download_file(self, url: str, out_path: Path) -> None:
        req = request.Request(url=url, method="GET", headers={"Authorization": f"Bearer {self.config.api_key}"})
        try:
            with request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                out_path.write_bytes(resp.read())
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise MeshyRequestError(f"Meshy download HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise MeshyRequestError(f"Meshy download failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise MeshyRequestError(
                f"Meshy download timed out after {self.config.timeout_sec:.1f}s."
            ) from exc


def _join_url(base_url: str, path: str) -> str:
    return parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _status_of(payload: dict[str, object]) -> str:
    value = payload.get("status")
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _task_error_message(payload: dict[str, object]) -> str:
    task_error = payload.get("task_error")
    if isinstance(task_error, dict):
        message = task_error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return ""


MESHY_READY_SET = frozenset({"SUCCEEDED"})
MESHY_FAILED_SET = frozenset({"FAILED", "CANCELED", "CANCELLED"})
