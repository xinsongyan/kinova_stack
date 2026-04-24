from __future__ import annotations


def __getattr__(name: str):
    if name == "KinovaMuJoCoBackend":
        from kinova_middleware.backend.kinova_mujoco_backend import KinovaMuJoCoBackend

        return KinovaMuJoCoBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KinovaMuJoCoBackend"]
