from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Sequence

import mujoco

from kinova_sim.sim_env import SimEnv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))

EnvFactory = Callable[[str, bool], SimEnv]


@dataclass(frozen=True, slots=True)
class MuJoCoRuntimeBindings:
    joint_ids: tuple[int, ...]
    qpos_adr: tuple[int, ...]
    qvel_adr: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    joint_limits: tuple[tuple[float, float], ...]
    continuous_indices: tuple[int, ...]
    ee_body_id: int
    ee_site_id: int | None


class MuJoCoRuntimeAdapter:
    """Own MuJoCo env/model/data plus the resolved robot bindings."""

    def __init__(
        self,
        *,
        model_path: str,
        joint_names: Sequence[str],
        ee_body_name: str,
        ee_site_name: str | None,
        viewer: bool,
        site_candidates: Sequence[str] | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self._model_path = str(model_path)
        self._joint_names = tuple(joint_names)
        self._ee_body_name = str(ee_body_name)
        self._ee_site_name = ee_site_name
        self._viewer = bool(viewer)
        self._site_candidates = tuple(site_candidates or ())
        self._env_factory = env_factory or SimEnv
        self._env: SimEnv | None = None
        self._bindings: MuJoCoRuntimeBindings | None = None

    @property
    def env(self) -> SimEnv:
        if self._env is None:
            raise RuntimeError("MuJoCoRuntimeAdapter.open() must be called before use.")
        return self._env

    @property
    def model(self) -> mujoco.MjModel:
        return self.env.model

    @property
    def data(self) -> mujoco.MjData:
        return self.env.data

    @property
    def joint_ids(self) -> tuple[int, ...]:
        return self._require_bindings().joint_ids

    @property
    def qpos_adr(self) -> tuple[int, ...]:
        return self._require_bindings().qpos_adr

    @property
    def qvel_adr(self) -> tuple[int, ...]:
        return self._require_bindings().qvel_adr

    @property
    def actuator_ids(self) -> tuple[int, ...]:
        return self._require_bindings().actuator_ids

    @property
    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        return self._require_bindings().joint_limits

    @property
    def continuous_indices(self) -> tuple[int, ...]:
        return self._require_bindings().continuous_indices

    @property
    def ee_body_id(self) -> int:
        return self._require_bindings().ee_body_id

    @property
    def ee_site_id(self) -> int | None:
        return self._require_bindings().ee_site_id

    def is_open(self) -> bool:
        return self._env is not None

    def open(self, initial_keyframe: str | None = None) -> None:
        if self._env is not None:
            return

        model_path = self.resolve_model_path(self._model_path)
        env = self._env_factory(model_path, viewer=self._viewer)
        if initial_keyframe:
            env.set_model_keyframe(initial_keyframe)

        self._env = env
        self._bindings = self._build_bindings(env.model)

    def close(self) -> None:
        if self._env is None:
            return
        self._env.close()
        self._env = None
        self._bindings = None

    def reset(self, initial_keyframe: str | None = None) -> None:
        env = self.env
        env.reset()
        if initial_keyframe:
            env.set_model_keyframe(initial_keyframe)

    def step_n(self, n_substeps: int) -> None:
        self.env.step_n(n_substeps)

    def get_model_keyframe(self, name: str):
        return self.env.get_model_keyframe(name)

    @staticmethod
    def resolve_model_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(_ROOT_DIR, path))

    def _build_bindings(self, model: mujoco.MjModel) -> MuJoCoRuntimeBindings:
        joint_ids = tuple(self._require_joint_id(model, name) for name in self._joint_names)
        qpos_adr = tuple(int(model.jnt_qposadr[jid]) for jid in joint_ids)
        qvel_adr = tuple(int(model.jnt_dofadr[jid]) for jid in joint_ids)
        actuator_ids = tuple(
            self._require_actuator_id(model, f"motor_{name}") for name in self._joint_names
        )

        if len(set(actuator_ids)) != len(actuator_ids):
            raise ValueError("Actuator mapping contains duplicates; check actuator names and ordering.")
        for actuator_id in actuator_ids:
            if actuator_id < 0 or actuator_id >= model.nu:
                raise ValueError(
                    f"Actuator id {actuator_id} out of range for nu={model.nu}; check model actuators."
                )

        joint_limits: list[tuple[float, float]] = []
        continuous_indices: list[int] = []
        for idx, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                low, high = model.jnt_range[jid]
            else:
                low, high = -float("inf"), float("inf")
                continuous_indices.append(idx)
            joint_limits.append((float(low), float(high)))

        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._ee_body_name)
        if ee_body_id < 0:
            raise ValueError(f"End-effector body '{self._ee_body_name}' not found in MuJoCo model.")

        return MuJoCoRuntimeBindings(
            joint_ids=joint_ids,
            qpos_adr=qpos_adr,
            qvel_adr=qvel_adr,
            actuator_ids=actuator_ids,
            joint_limits=tuple(joint_limits),
            continuous_indices=tuple(continuous_indices),
            ee_body_id=ee_body_id,
            ee_site_id=self._resolve_ee_site_id(model),
        )

    def _resolve_ee_site_id(self, model: mujoco.MjModel) -> int | None:
        candidates: list[str] = []
        if self._ee_site_name is not None:
            candidates.append(self._ee_site_name)
        else:
            candidates.extend(self._site_candidates)
            candidates.append(f"{self._ee_body_name}_site")
        for name in candidates:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                return site_id
        if self._ee_site_name is not None:
            raise ValueError(f"End-effector site '{self._ee_site_name}' not found in MuJoCo model.")
        return None

    def _require_bindings(self) -> MuJoCoRuntimeBindings:
        if self._bindings is None:
            raise RuntimeError("MuJoCoRuntimeAdapter.open() must be called before use.")
        return self._bindings

    @staticmethod
    def _require_joint_id(model: mujoco.MjModel, name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model.")
        return joint_id

    @staticmethod
    def _require_actuator_id(model: mujoco.MjModel, name: str) -> int:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Actuator '{name}' not found in MuJoCo model.")
        return actuator_id
