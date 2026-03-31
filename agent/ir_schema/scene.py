from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import QuatWXYZ, StrictModel, Vec3


class SimIR(StrictModel):
    dt: float = Field(default=0.01, gt=0.0, le=1.0)
    gravity: Vec3 = (0.0, 0.0, -9.81)


class ViewerIR(StrictModel):
    camera_pos: Vec3 = (3.5, 0.5, 2.5)
    camera_lookat: Vec3 = (0.0, 0.0, 0.5)
    camera_fov: float = Field(default=40.0, gt=1.0, lt=179.0)


class CollisionIR(StrictModel):
    friction: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description=(
            "Contact friction coefficient. Higher values resist sliding more strongly, but do not guarantee perfectly non-slipping contact. Adjust this value to make contacts more or less slippery, which can be useful for tuning contact behaviors between objects."
        ),
    )
    coup_friction: float | None = Field(default=None, ge=0.0)
    coup_restitution: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Impact restitution / bounciness. Higher values make contacts rebound more and can reduce stability. Adjust this value to make contacts more or less bouncy, which can be useful for tuning contact behaviors between objects."
        ),
    )
    contact_resistance: float | None = Field(default=None, gt=0.0)
    sol_params: tuple[float, float, float, float, float, float, float] | None = None


class FollowEntityCameraIR(StrictModel):
    entity: str
    fixed_axis: tuple[float | None, float | None, float | None] = Field(
        default=(None, None, None),
        description=(
            "Optional per-axis lock for the follow target. Use null on an axis to follow the entity on that axis, "
            "or set a number to keep that axis fixed."
        ),
    )
    smoothing: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "Temporal smoothing factor for follow-camera motion. Higher values produce smoother motion but more lag."
        ),
    )
    fix_orientation: bool = False


class RenderIR(StrictModel):
    output_video: str = "rigid.mp4"
    fps: int = Field(default=60, ge=1, le=240)
    res: tuple[int, int] = (640, 480)
    camera_pos: Vec3 = (3.5, 0.5, 2.5)
    camera_lookat: Vec3 = (0.0, 0.0, 0.5)
    camera_up: Vec3 = (0.0, 0.0, 1.0)
    camera_fov: float = Field(default=40.0, gt=1.0, lt=179.0)
    near: float = Field(default=0.1, gt=0.0)
    far: float = Field(default=20.0, gt=0.0)
    gui: bool = False
    render_every_n_steps: int = Field(default=1, ge=1, le=100_000)
    include_initial_frame: bool = True
    force_render: bool = False
    follow_entity: FollowEntityCameraIR | None = None

    @field_validator("res")
    @classmethod
    def _check_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("`res` must contain positive width and height.")
        return value

    @model_validator(mode="after")
    def _check_near_far(self) -> "RenderIR":
        if self.far <= self.near:
            raise ValueError("`far` must be greater than `near`.")
        return self


class SceneIR(StrictModel):
    backend: Literal["cpu", "gpu"] = "cpu"
    show_viewer: bool = False
    add_ground: bool = True
    ground_collision: CollisionIR | None = None
    sim: SimIR = Field(default_factory=SimIR)
    viewer: ViewerIR | None = None
    render: RenderIR | None = None


class PoseIR(StrictModel):
    pos: Vec3 = (0.0, 0.0, 1.0)
    quat: QuatWXYZ = (1.0, 0.0, 0.0, 0.0)
