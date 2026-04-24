from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class BackendCapability(str, Enum):
    """Named backend features used for runtime capability checks."""

    ARM_MOTION = "arm_motion"
    IK_SOLVER = "ik_solver"
    GRIPPER_CONTROL = "gripper_control"
    SCENE_CONTROL = "scene_control"
    OBJECT_QUERY = "object_query"


CapabilityLike = BackendCapability | str


@runtime_checkable
class CapabilityProvider(Protocol):
    """Protocol for objects that can report their supported capabilities."""

    def supported_capabilities(self) -> frozenset[BackendCapability]:
        ...


def _coerce_capability(capability: CapabilityLike) -> BackendCapability:
    if isinstance(capability, BackendCapability):
        return capability
    return BackendCapability(str(capability))


def get_supported_capabilities(obj: object) -> frozenset[BackendCapability]:
    """Return the declared capability set for a backend/controller-like object."""

    if isinstance(obj, CapabilityProvider):
        return obj.supported_capabilities()
    return frozenset()


def supports_capability(obj: object, capability: CapabilityLike) -> bool:
    """Return True when the object explicitly advertises the requested feature."""

    return _coerce_capability(capability) in get_supported_capabilities(obj)
