# Kinova Stack Repository Guide

This document is a report-oriented technical guide to the repository. It is intended to give enough architectural, implementation, and contextual detail to support an individual project report without having to reverse-engineer the codebase from scratch.

## 1. Repository Purpose

At a high level, this repository implements a Kinova robot manipulation stack built around:

- MuJoCo simulation of a Kinova arm and gripper.
- A middleware layer that exposes a stable controller/back-end API.
- An MCP server that turns robot capabilities into callable tools.
- LLM clients that use those tools to perform manipulation tasks such as grasping, sorting, and stacking.
- A hardware path for communicating with a real Kinova arm through the Kinova USB SDK.

The repository is therefore not just "a simulator" and not just "a controller". It is an integrated experimentation stack covering:

- robot model description,
- simulation and control,
- inverse kinematics,
- task scenes,
- tool-based robotic APIs,
- and LLM-driven task execution and benchmarking.

## 2. High-Level Architecture

The main end-to-end software flow is:

```text
User / Demo Script / LLM Client
        ->
FastMCP client calls
        ->
kinova_middleware/backend/mcp_kinova_server.py
        ->
KinovaController
        ->
SafetyWrapperBackend
        ->
Selected backend
    - KinovaMuJoCoBackend for simulation
    - KinovaSDKBackend for real hardware
        ->
Robot state update / actuation / IK / scene interaction
```

The repository is organised so that the controller-facing API is mostly backend-agnostic. In other words, higher-level code talks to a `KinovaController`, not directly to MuJoCo or to the USB SDK.

## 3. Repository At A Glance

Approximate file counts observed during analysis:

| Path | Role | Approx. file count |
|---|---|---:|
| `kinova_middleware/` | Main application layer | 64 |
| `kinova_middleware/backend/` | Core controller and backend implementation | 20 |
| `kinova_middleware/llm_clients/` | LLM-driven task agents and helpers | 14 |
| `kinova_middleware/demo_scripts/` | Demo and experiment scripts | 8 |
| `kinova_middleware/scenes/` | MuJoCo task scenes | 7 |
| `kinova_sim/` | Lower-level simulation/control utilities | 11 |
| `kinova_description/` | Robot description assets, MJCF, URDF, meshes | 97 |
| `kinova-api-python/` | ctypes wrapper around Kinova SDK shared libraries | 13 |
| `kinova_firmware/` | Firmware and vendor assets | 7 |

Important top-level files:

- `README.md`: setup and basic structure.
- `ik_verification.py`: standalone IK benchmark and evaluation harness.
- `debug_startup.py`: import/startup debugging helper for the stack.
- `requirements.txt`: currently empty, so dependency documentation lives mainly in `README.md` and in the imports.

## 4. Major Directories And What They Do

### `kinova_middleware/`

This is the main application layer and the best place to start reading the project.

It contains:

- `backend/`: the controller facade, abstract backend, safety wrapper, MuJoCo backend, SDK backend, and MCP server.
- `llm_clients/`: LLM agents that call the MCP server to perform tasks.
- `demo_scripts/`: scripted experiments and demonstrations.
- `scenes/`: MuJoCo scene XML files for different manipulation tasks.

### `kinova_middleware/backend/`

This is the architectural core of the repository.

Main files:

- `kinova_backend.py`
  - Defines `CartesianPose`.
  - Defines the abstract `KinovaBackend` interface.
  - Implements `SafetyWrapperBackend`.
  - Provides `make_kinova_api(mode=...)`.
- `kinova_controller.py`
  - Thin facade class that exposes a stable, backend-agnostic API.
- `kinova_mujoco_backend.py`
  - Main simulation backend.
  - Contains the most important implementation logic: MuJoCo model setup, joint mapping, trajectory control, finger control, IK, end-effector pose extraction, and scene reset.
- `kinova_sdk_backend.py`
  - Hardware backend for real Kinova devices.
  - Uses the Kinova USB SDK wrapper for actuation and feedback.
  - Reuses a MuJoCo backend instance as an IK solver.
- `mcp_kinova_server.py`
  - Launches the FastMCP server and exposes robot tools.
- `mcp_server/tools.py`
  - Registers the MCP tool functions.
- `mcp_server/prompts.py`
  - Registers task prompts used by LLM agents.

### `kinova_sim/`

This is a lower-level control/simulation utility layer used by the MuJoCo backend and by some standalone experiments.

Main files:

- `controller.py`
  - `PDController`
  - `ComputedTorqueController`
- `trajectory.py`
  - jerk-limited trajectory generator
- `governor.py`
  - saturation-aware reference governor for Joint 1 torque limiting
- `sim_env.py`
  - MuJoCo environment wrapper
- `main.py`
  - older standalone simulation demo for control tuning

### `kinova_description/`

This directory contains the robot model data.

Observed purpose:

- mesh assets,
- URDF and xacro definitions,
- MuJoCo-converted MJCF,
- support scripts for URDF-to-MJCF preparation.

The README states this description package is derived from the Kinova ROS repository:

- `kinova_description` is from `https://github.com/Kinovarobotics/kinova-ros/tree/noetic-devel`

The active MuJoCo model used by the middleware is:

- `kinova_description/mjcf/m1n4s300_standalone.mjcf`

### `kinova-api-python/`

This directory wraps the Kinova USB SDK with Python `ctypes`.

Main files:

- `kinova_api.py`
  - shared library loading,
  - struct definitions,
  - device discovery,
  - joint and Cartesian readback,
  - finger actuation helpers.
- `demo_kinova.py`
  - minimal hardware communication test script.

### `kinova_firmware/`

This looks like a vendor/support directory containing firmware packages and documentation rather than actively executed repository code.

## 5. The Core Software Design

The project uses a layered design:

| Layer | Main code | Purpose |
|---|---|---|
| Description layer | `kinova_description/` | Robot geometry, joints, meshes, MJCF, URDF |
| Low-level control layer | `kinova_sim/` | trajectory generation, inverse dynamics, governor, MuJoCo wrapper |
| Backend layer | `kinova_middleware/backend/` | common API over simulation and hardware |
| Tool/API layer | `mcp_kinova_server.py`, `mcp_server/tools.py` | expose robot functions as MCP tools |
| Agent/task layer | `kinova_middleware/llm_clients/` | LLM-based task execution and benchmarking |
| Evaluation layer | `ik_verification.py`, demos | experiments, verification, task testing |

This is a strong design choice for a report because it separates:

- robot modelling,
- controller design,
- system integration,
- and task-level autonomy.

## 6. Backend Abstraction

### 6.1 `KinovaBackend`

`kinova_middleware/backend/kinova_backend.py` defines the abstract interface all backends must satisfy.

Key responsibilities:

- backend lifecycle: `init()`, `close()`
- homing: `move_home()`
- joint commands: `send_joint_position_rad()`
- Cartesian state: `get_end_effector_pose()`
- inverse kinematics: `solve_ik()` and `solve_ik_position_only()`
- stepping and convergence checks: `step()`, `is_reached()`
- scene reset: `reset_scene()`
- state queries: joint angles, target angles, velocities
- gripper control and gripper state helpers

This abstraction is important because the rest of the stack can call the same API for:

- simulated execution,
- or real hardware execution.

### 6.2 `CartesianPose`

`CartesianPose` is a small but useful design detail:

- stores `(x, y, z, qx, qy, qz, qw)`,
- validates finite values,
- normalises quaternions on construction,
- offers helpers for position/quaternion access.

That means pose validation happens early instead of being left to downstream IK code.

### 6.3 `SafetyWrapperBackend`

The safety wrapper sits around another backend and applies constraints before commands reach it.

Supported safety mechanisms:

- joint limits,
- maximum joint step size,
- maximum joint velocity,
- workspace validation and clipping,
- Cartesian bounds,
- Cartesian no-go boxes,
- Cartesian no-go spheres,
- custom Cartesian validators,
- configurable violation policy: `clip` or `reject`.

This is one of the most report-worthy features in the repository because it shows explicit engineering for safe command filtering rather than blindly forwarding planner outputs.

### 6.4 `KinovaController`

`kinova_middleware/backend/kinova_controller.py` is a facade, not a heavy controller implementation.

Its role is to:

- wrap the selected backend,
- optionally enforce safety wrapping,
- expose a stable API to callers,
- keep higher-level code independent of backend details.

This is effectively the public control surface for:

- demo scripts,
- the MCP server,
- and any future client code.

## 7. MuJoCo Backend Deep Dive

The main simulation implementation is in:

- `kinova_middleware/backend/kinova_mujoco_backend.py`

This is the largest and most important implementation file in the repository.

### 7.1 Active Robot Model

The default simulation model is:

- `kinova_description/mjcf/m1n4s300_standalone.mjcf`

Important observed characteristics:

- 4 arm joints: `joint_1` to `joint_4`
- 6 finger joints:
  - `joint_finger_1`
  - `joint_finger_tip_1`
  - `joint_finger_2`
  - `joint_finger_tip_2`
  - `joint_finger_3`
  - `joint_finger_tip_3`
- 10 actuators total
- end-effector reference site: `ee_marker`

The backend separates:

- arm indices,
- finger indices,
- joint position addresses,
- joint velocity addresses,
- actuator ids.

This makes the controller logic independent of raw MuJoCo indexing.

### 7.2 Initialisation Sequence

During `init()` the backend:

1. resolves the model path,
2. creates a `SimEnv`,
3. optionally applies the `home` keyframe,
4. maps joint ids, qpos/qvel addresses, and actuators,
5. infers joint limits and continuous joints,
6. resolves the end-effector body/site,
7. computes control substeps from physics timestep,
8. initialises desired joint targets,
9. constructs:
   - `TrajectoryGenerator`
   - `ComputedTorqueController`
   - `PDController` for fingers
   - `ReferenceGovernor`

### 7.3 Motion Control Strategy

The simulation backend is not position-control-only. It uses a more structured pipeline:

1. A joint-space trajectory generator produces smooth desired position, velocity, and acceleration.
2. A computed-torque controller uses MuJoCo inverse dynamics terms to convert those desired states into torques.
3. A reference governor monitors raw Joint 1 torque demand and scales future trajectory limits if needed.
4. Finger joints are controlled by a separate PD torque controller.
5. Actuator forces are saturated before being written into MuJoCo controls.

This makes the stack closer to a model-based torque-controlled system than a simple "set joint target and hope" simulator.

### 7.4 `step()` Control Loop

Each backend `step()` effectively does:

1. read current joint positions and velocities,
2. split arm and finger state,
3. apply the governor scale from the previous tick,
4. advance the jerk-limited trajectory generator,
5. compute arm torques using the computed-torque controller,
6. update the Joint 1 reference governor with raw torque demand,
7. compute finger torques with PD control,
8. saturate torques and write controls,
9. perform `n_substeps` MuJoCo physics steps,
10. check whether the desired state has been reached.

This is a clean closed-loop control design and is a central point for a methodology or software-architecture chapter.

## 8. Control Components In `kinova_sim/`

### 8.1 `TrajectoryGenerator`

Implemented in `kinova_sim/trajectory.py`.

Main characteristics:

- jerk-limited,
- acceleration-limited,
- velocity-limited,
- uses a trapezoidal-style accelerate/cruise/brake profile,
- checks braking feasibility before advancing,
- prevents overshoot,
- snaps to the target when near enough and slow enough.

This component produces:

- `q_des`
- `qd_des`
- `qdd_des`

for the arm joints.

### 8.2 `ComputedTorqueController`

Implemented in `kinova_sim/controller.py`.

The controller computes:

```text
qdd_cmd = qdd_des + Kp * (q_des - q) + Kd * (qd_des - qd)
tau = M(q) * qdd_cmd + qfrc_bias + damping/friction feedforward
```

Key technical details:

- uses `mujoco.mj_fullM` to recover the dense mass matrix,
- uses `data.qfrc_bias` for gravity and Coriolis terms,
- adds feedforward compensation for damping and friction,
- clips commanded accelerations before torque computation,
- performs strong sanity checks for non-finite values.

This is stronger than a basic PD joint controller and is worth discussing explicitly in a report.

### 8.3 `ReferenceGovernor`

Implemented in `kinova_sim/governor.py`.

Purpose:

- monitor raw Joint 1 torque utilisation,
- smooth that utilisation using an exponential moving average,
- reduce future trajectory limits when torque demand becomes too high,
- recover gradually when torque demand falls.

This is a useful system-level feature because it tries to prevent aggressive commands from repeatedly saturating the most critical joint.

### 8.4 `SimEnv`

Implemented in `kinova_sim/sim_env.py`.

It is a lightweight wrapper around MuJoCo:

- loads `MjModel` and `MjData`,
- optionally launches a passive viewer,
- offers `step()`, `step_n()`, `reset()`, `set_model_keyframe()`,
- exposes current simulation time and state.

## 9. Inverse Kinematics Implementation

The repository contains two related IK implementations:

- the production IK used by `KinovaMuJoCoBackend`,
- a standalone evaluation-oriented IK implementation in `ik_verification.py`.

### 9.1 Main Production IK

The production solver lives inside:

- `kinova_middleware/backend/kinova_mujoco_backend.py`

Class:

- `LevenbergMarquardtIK`

The solver is a damped least-squares / Levenberg-Marquardt solver based on MuJoCo Jacobians.

Main properties:

- supports position-only IK,
- supports position plus partial orientation alignment,
- uses multiple seeds,
- handles continuous joints with angle wrapping,
- clamps solutions to joint limits,
- selects among candidate solutions using geometric and joint-space criteria.

### 9.2 What Orientation Is Actually Constrained

A very important detail:

- the solver does not fully track an arbitrary 3D end-effector orientation.
- instead, it aligns a direction vector associated with the end-effector.

In the production backend, the target quaternion is converted into a target X-axis direction, and IK tries to align the end-effector X-axis with that vector.

Implication:

- orientation control is reduced from full 6D pose control to position plus a directional constraint,
- which is a sensible compromise for a 4-DOF arm,
- but it should be described accurately in the report to avoid overstating the robot's orientation capability.

### 9.3 Seeding Strategy

The production IK uses a multi-start strategy:

- systematic yaw/pitch combinations,
- random seeds,
- optional caller-provided seed inserted first.

This is used to reduce local-minimum problems and configuration flips.

### 9.4 Solution Selection Strategy

When multiple candidate solutions exist, the backend prefers:

1. low position error,
2. then low orientation error,
3. then low weighted joint-space distance from the current configuration.

There is explicit extra weighting to discourage unnecessary configuration flips, especially around the base joint.

### 9.5 Position-Only And Wrist-Free Modes

The code supports:

- `solve_ik_position_only(...)`
- `move_wrist=False`

When `move_wrist=False`, the solver reduces active DOFs by freezing the wrist joint so that vertical lifting or retreat can be done without trying to satisfy an infeasible orientation target.

This is a practical design choice used extensively by the tool layer and task prompts.

### 9.6 Standalone IK Benchmark In `ik_verification.py`

`ik_verification.py` is effectively an evaluation harness for the IK/control stack.

It:

- loads the dedicated `ik_verification.xml` scene,
- discovers `target_*` sites,
- runs a second LM IK implementation,
- commands the arm to each target using the same low-level control components,
- measures position error, orientation error, and downward alignment,
- prints a table and writes CSV output.

This is useful report material because it demonstrates that the project includes verification and not just implementation.

### 9.7 Hardware IK Reuse

The hardware backend does not implement its own analytic or numerical IK from scratch.

Instead:

- `KinovaSDKBackend` creates a MuJoCo backend internally for IK,
- solves IK in simulation,
- then sends the joint solution to the real arm via the USB SDK.

This is a notable architectural decision:

- it avoids duplicating IK logic,
- but it also means real-hardware IK quality depends on how closely the MuJoCo model matches the physical arm.

## 10. MCP Server And Tool Layer

The MCP server is launched by:

- `kinova_middleware/backend/mcp_kinova_server.py`

### 10.1 Startup Behaviour

At startup the server:

1. chooses a scene using `scene_selector.py`,
2. creates a controller,
3. initialises the selected backend,
4. moves the arm home,
5. then exposes the tool surface over FastMCP using `streamable-http`.

The default client/server URL used across the repository is:

- `http://127.0.0.1:8000/mcp`

### 10.2 Motion Execution Model

The server exposes blocking robot tools.

The important behaviour is:

- tool calls do not just command a target and return immediately,
- they typically command motion and then block until the arm is settled or timed out,
- the main helper for this is `_run_until_reached(...)`.

This makes the tool interface easier for LLM agents because tools behave more like complete actions than low-level setpoints.

### 10.3 Concurrency And Locking

The server uses:

- `_motion_lock`
- `_physics_lock`

The intent is:

- only one motion command can execute at a time,
- and calls into controller/physics state are serialised.

Also, the code comment notes that the earlier continuous stepper thread has been removed, so simulation now advances only during tool calls and explicit waits.

### 10.4 Registered MCP Tools

Current tools implemented in `mcp_server/tools.py` are:

| Tool | Purpose |
|---|---|
| `reset_scene` | reset the current simulation scene |
| `move_home` | send the arm to home and block until stable |
| `get_end_effector_pose` | read current Cartesian pose |
| `set_gripper` | open/close gripper and wait for settling |
| `move_pose` | solve IK, send joint target, and block |
| `rotate_wrist` | rotate the last arm joint by a relative angle |
| `get_object_pose` | read the pose of a body or site, or all free bodies |
| `compute_grasp_height` | estimate top surface height from geometry and orientation |
| `compute_wrist_alignment` | compute wrist yaw adjustment relative to an object's axis |

Useful details:

- `move_pose` treats a near-zero quaternion as a signal for position-only motion when fallback is allowed.
- `move_pose` currently hard-fails only if final position error exceeds `0.04 m`; orientation hard-fail checks are commented out.
- `get_object_pose("all")` returns all free-joint bodies, which is how task scripts discover movable objects in scenes.
- `compute_grasp_height` adds an extra `0.03 m` offset, so it behaves as a practical grasp-planning helper rather than a pure geometry calculation.

## 11. Prompt Layer

The MCP prompt registry lives in:

- `kinova_middleware/backend/mcp_server/prompts.py`

Defined prompts include:

- `grab_shapes`
- `sort_cubes`
- `stack_cubes`
- `pick_up_block`
- `place_block`

These prompts act as:

- reusable task-specific SOPs,
- prompt templates that LLM clients can fetch at runtime.

This is an interesting design choice because it centralises robot task procedure instructions on the server side instead of hardcoding all task logic inside each client.

## 12. LLM Client Layer

The LLM-side orchestration lives in:

- `kinova_middleware/llm_clients/`

### 12.1 `ultimate_llm.py`

This is a unified task router.

It:

- accepts a free-form task,
- chooses between `grab_shapes`, `sort_cubes`, and `stack_cubes`,
- fetches the corresponding MCP prompt,
- binds tools through `langchain_openai`,
- executes one tool call at a time in a controlled loop.

### 12.2 Dedicated Workflow Clients

There are dedicated clients for specific workflows:

- `grab_shapes.py`
- `sort_cubes.py`
- `stack_cubes.py`

These are more specialised than `ultimate_llm.py` and are useful for repeatable experiments.

### 12.3 `helper_functions.py`

This file is effectively the client-side orchestration utility module.

It provides:

- extra local-only tool definitions such as:
  - `check_sorting_status`
  - `check_stacking_status`
  - `verify_object_lift`
  - `finish_task`
- functions to load MCP tools/prompts and translate them into OpenAI/LangChain schemas,
- local evaluation logic for sorting and stacking outcomes,
- official reporting for grab/lift benchmarks,
- raw tool-call fallback parsing for models that emit malformed tool text.

This file is important because it shows that the repository does not rely only on "whatever the LLM said". It also adds explicit verification and structured retry behaviour.

### 12.4 Benchmarking Scripts

The LLM experimentation layer also includes:

- `run_grab_multiple_providers.py`
  - runs the grab benchmark sequentially over several model providers and writes per-run log files.
- `benchmark_nvidia_models.py`
  - benchmarks OpenAI-compatible NVIDIA API models by latency / TTFT.

The log files currently visible in the workspace indicate this benchmark layer is actively used to compare model providers for robot tasks.

## 13. Scene Layer

Task scenes are defined in:

- `kinova_middleware/scenes/`

Key scenes:

| Scene | Purpose |
|---|---|
| `single_cube.xml` | simplest pick-and-place scene |
| `multi_cubes.xml` | multiple movable cubes |
| `shapes.xml` | mixed geometry scene with box, sphere, cylinder |
| `sorting_task.xml` | coloured cubes plus bins for sorting |
| `ik_verification.xml` | target sites for IK evaluation |

General scene design pattern:

- include the standalone Kinova MJCF,
- add free-joint objects,
- define target sites when needed,
- define a `home` keyframe with robot and object state.

This is a strong setup for evaluation because scenes can be swapped independently of the control stack.

### Scene Selection

`scene_selector.py` supports:

- environment variable selection through `KINOVA_SCENE`,
- or an interactive menu.

That allows the same server to be reused across different tasks.

## 14. Robot Description Layer

The active MuJoCo robot description is:

- `kinova_description/mjcf/m1n4s300_standalone.mjcf`

Notable contents:

- arm kinematics and inertial parameters,
- finger subtrees,
- contact/friction properties,
- actuator force limits,
- `ee_marker` and axis visualisation sites,
- ground plane.

The MJCF clearly supports end-effector visualisation via:

- `ee_marker`
- `ee_axis_x`
- `ee_axis_y`
- `ee_axis_z`

This is useful because the IK implementation reasons about an end-effector axis, not just about position.

The `kinova_description/urdf/` directory contains many other Kinova variants, so the repository has broader model assets than the currently active middleware path.

## 15. Hardware Integration Path

Real-hardware support is implemented through:

- `kinova_middleware/backend/kinova_sdk_backend.py`
- `kinova-api-python/kinova_api.py`

`kinova_api.py` provides:

- shared library loading,
- device discovery,
- active-device selection,
- end-effector pose readback,
- joint angle readback,
- finger open/close and percentage mapping,
- `SendBasicTrajectory` usage for angular commands.

This means the repository is not purely simulated. It has a path to physical execution, although the simulated path is clearly the richer and more developed one.

## 16. Key Entry Points

Good scripts to cite in a report:

| File | Why it matters |
|---|---|
| `kinova_middleware/backend/mcp_kinova_server.py` | main tool-serving robot interface |
| `kinova_middleware/backend/kinova_mujoco_backend.py` | core simulation backend |
| `kinova_middleware/backend/kinova_backend.py` | abstract API and safety wrapper |
| `kinova_middleware/backend/kinova_controller.py` | stable control facade |
| `ik_verification.py` | quantitative IK evaluation |
| `kinova_sim/controller.py` | computed-torque controller |
| `kinova_sim/trajectory.py` | motion profile generator |
| `kinova_sim/governor.py` | torque-aware reference governor |
| `kinova_middleware/llm_clients/ultimate_llm.py` | generic LLM task router |
| `kinova_middleware/llm_clients/grab_shapes.py` | benchmark-oriented LLM grasp client |
| `kinova_middleware/demo_scripts/test_hold_cube_60s.py` | targeted grasp stability experiment |
| `kinova-api-python/demo_kinova.py` | hardware communication smoke test |

## 17. Suggested Report Framing

One sensible way to describe this repository in a report is:

### Software Architecture Chapter

Describe the repository as a layered robotic middleware stack with:

- a robot description layer,
- a control/simulation layer,
- a backend abstraction layer,
- a tool-serving API layer,
- and an LLM-driven task layer.

### Control And Simulation Chapter

Focus on:

- MuJoCo model choice,
- computed-torque control,
- jerk-limited trajectories,
- torque-aware reference governor,
- gripper torque control,
- end-effector pose extraction through sites.

### Inverse Kinematics Chapter

Explain:

- Levenberg-Marquardt numerical IK,
- Jacobian-based updates,
- multi-start seeding,
- joint limit clipping,
- reduced orientation control through axis alignment,
- dedicated verification in `ik_verification.py`.

### Autonomy / Human-Robot Interface Chapter

Explain:

- MCP server tools as the robot API,
- prompts as server-side SOPs,
- LLM clients as tool users,
- client-side verification helpers that measure actual task completion.

### Evaluation Chapter

Use:

- `ik_verification.py` for kinematic accuracy experiments,
- demo/task scenes for behavioural validation,
- LLM benchmark scripts for provider/model comparison.

## 18. Important Nuances And Caveats

These are especially worth mentioning so the report stays accurate.

### 18.1 The Active Simulation Model Is 4-DOF

The main MuJoCo backend uses a 4-DOF arm model:

- `m1n4s300_standalone.mjcf`

So although the repository contains broader Kinova assets, the current manipulation pipeline is designed around a reduced-DOF simulated arm.

### 18.2 The Hardware Backend Assumes A Different Arm Configuration

`KinovaSDKBackend` reports:

- `dof = 7`
- `arm_dof = 7`

This differs from the 4-DOF simulation path.

So the repository mixes:

- a 4-DOF simulation workflow,
- and a 7-DOF real-hardware workflow.

That is not necessarily wrong, but it is an important architectural nuance.

### 18.3 Gripper Description Is Inconsistent Across Files

Observed mismatch:

- some prompt/demo text says "2-finger gripper",
- the active MJCF model contains three finger branches and six finger joints.

For a report, it is safer to describe the simulation as using:

- a Kinova hand model with three finger branches in the MJCF,

while noting that some higher-level prompt text appears to use simplified language.

### 18.4 Some Scripts/Prompts Reference Older MCP Tools

Current tool registry does not define:

- `get_joint_state`
- `move_joints`

but these names still appear in:

- `kinova_middleware/demo_scripts/demo_reach_cube.py`
- commented prompt material in `mcp_server/prompts.py`

This suggests the repository has evolved and some demo/prompt text is stale relative to the current MCP interface.

### 18.5 `requirements.txt` Is Empty

The dependency file exists but is empty.

Practical dependency information is spread across:

- `README.md`
- imports in the source code

This means environment reproducibility is currently under-documented.

### 18.6 Dependency Documentation Looks Incomplete

The README lists a minimal setup, but the code also imports packages such as:

- `langchain_core`
- `langchain_openai`
- `scipy`

These do not appear in `requirements.txt`, so the documented setup is probably incomplete.

### 18.7 `benchmark_nvidia_models.py` Appears To Contain Two Concatenated Scripts

The file contains:

- one script for running a provider-wise grab benchmark,
- and then a second benchmarking script appended into the same file.

The transition point includes a broken `if __name__ ==` boundary, so this file currently looks more experimental than production-ready.

### 18.8 There Is Some Duplicate IK Logic

IK is implemented in:

- the production MuJoCo backend,
- and separately in `ik_verification.py`.

This is understandable for experimentation, but it does mean some solver logic exists in two places.

### 18.9 Some Run Instructions And Demos Look Out Of Date

Two concrete examples observed during analysis:

- the README demo command points to `kinova_middleware/demo_controller_sim.py`, but the current file lives under `kinova_middleware/demo_scripts/demo_controller_sim.py`,
- `demo_controller_sim.py` calls `_run_steps(..., target_joint_rad=q_target, ...)` even though `_run_steps` does not accept that argument.

That suggests some repository documentation and demo code has drifted behind the current structure.

## 19. What The Repository Is Strongest At

From an engineering perspective, the strongest aspects of the repository are:

- clear separation between backend abstraction and backend implementation,
- integration of simulation and tool-serving APIs,
- model-based joint control rather than naive kinematic setpointing,
- explicit task scenes for manipulation experiments,
- inclusion of an evaluation-oriented IK benchmark,
- practical LLM integration with verification helpers rather than pure free-form chat control.

## 20. One-Sentence Summary You Can Reuse

If you need a concise report sentence:

> This repository implements a layered Kinova manipulation stack that combines MuJoCo-based robot simulation, model-based joint control, Jacobian-based inverse kinematics, an MCP tool server, and LLM-driven task execution for grasping, sorting, stacking, and benchmarking experiments.
