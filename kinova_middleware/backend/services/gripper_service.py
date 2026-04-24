from __future__ import annotations


def __getattr__(name: str):
    if name == "MuJoCoGripperService":
        from kinova_middleware.backend.mujoco_control import MuJoCoGripperService

        return MuJoCoGripperService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MuJoCoGripperService"]
