from __future__ import annotations

from typing import Any

from kinova_middleware.backend.controller import KinovaController
from kinova_middleware.backend.kinova_backend import KinovaBackend


def build_backend(mode: str, **backend_kwargs: Any) -> KinovaBackend:
    """Build a backend instance for the requested runtime mode."""

    mode_key = mode.strip().lower()
    if mode_key == "sim":
        from kinova_middleware.backend.adapters.mujoco_arm_adapter import KinovaMuJoCoBackend

        scene_path = backend_kwargs.pop("scene_path", None)
        if scene_path is not None:
            backend_kwargs.setdefault("model_path", scene_path)
        return KinovaMuJoCoBackend(**backend_kwargs)

    if mode_key == "real":
        from kinova_middleware.backend.kinova_sdk_backend import KinovaSDKBackend

        if backend_kwargs:
            raise TypeError(
                "The real-robot backend factory path is intentionally deferred; "
                "do not pass backend kwargs here yet."
            )
        return KinovaSDKBackend()

    raise ValueError(f"mode must be 'sim' or 'real', got {mode!r}")


def build_controller(
    mode: str,
    *,
    enforce_safety_wrapper: bool = True,
    **backend_kwargs: Any,
) -> KinovaController:
    backend = build_backend(mode, **backend_kwargs)
    return KinovaController(backend, enforce_safety_wrapper=enforce_safety_wrapper)


def build_scene_controller(
    scene_path: str,
    *,
    mode: str = "sim",
    enforce_safety_wrapper: bool = True,
    target_speed_rad_s: float | None = None,
    viewer: bool = True,
    ee_site: str | None = "ee_marker",
    **backend_kwargs: Any,
) -> KinovaController:
    mode_key = mode.strip().lower()
    if mode_key == "sim":
        backend = build_backend(
            mode_key,
            scene_path=scene_path,
            target_speed_rad_s=target_speed_rad_s,
            viewer=viewer,
            ee_site=ee_site,
            **backend_kwargs,
        )
    else:
        backend = build_backend(mode_key, **backend_kwargs)

    controller = KinovaController(backend, enforce_safety_wrapper=enforce_safety_wrapper)
    controller.init()
    return controller
