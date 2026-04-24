# Refactor Plan For The Kinova Stack

This is a MuJoCo-first refactor plan.

It is intentionally scoped to:

- refactoring the simulation/backend structure,
- introducing proper polymorphism and capability-based interfaces,
- and making it easy to plug in the real SDK later.

It is intentionally not scoped to:

- implementing or repairing the current `KinovaSDKBackend`,
- validating hardware behavior,
- or doing real-robot integration work in this phase.

This document describes how I would refactor the current repository so that:

- the codebase is easier to reason about,
- the backend abstraction is real rather than nominal,
- the MCP server stops depending on MuJoCo internals,
- the simulator code becomes reusable across robot models,
- and pieces of the stack can be reused for another robot without rewriting the whole system.

The plan is based on the current repository structure and the current implementation in:

- `kinova_middleware/backend/`
- `kinova_middleware/backend/mcp_server/`
- `kinova_sim/`
- `kinova_middleware/llm_clients/`

## 1. Main Diagnosis

The biggest problem in the current repo is not just "messy code". It is that the architecture claims to be backend-agnostic, but large parts of the system are still tightly coupled to one specific robot, one specific simulator layout, and one specific runtime flow.

In practice:

- `KinovaController` and `KinovaBackend` suggest a clean abstraction.
- The MCP server and tool layer still reach into MuJoCo-specific private fields.
- The MuJoCo backend mixes too many responsibilities into one class.
- The future hardware path does not yet have a clean seam to plug into.
- Several modules are only importable because of `sys.path` mutation rather than proper package structure.

That means the current stack is only partially abstracted. It works more like a tightly coupled Kinova demo system than a reusable robotics middleware layer.

## 2. Refactor Goals

The refactor should aim for five concrete outcomes:

1. A polymorphic runtime contract that MuJoCo can satisfy now and the real SDK can satisfy later.
2. Separation of robot-agnostic simulation/control code from Kinova-specific configuration.
3. Capability-based MCP tooling so the server only exposes tools that the selected backend actually supports.
4. Removal of private-attribute reach-through such as `_backend`, `_inner`, and `_env` from higher layers.
5. A package structure that can be imported and tested normally.

## 2.1 Scope Boundary

For this refactor, I would treat the SDK path as a future integration target, not an implementation target.

That means:

- I would design interfaces with the SDK in mind.
- I would not spend time fixing the current hardware backend now.
- I would prove the design using the MuJoCo runtime plus fake test doubles.
- I would leave a clean adapter seam where the real robot code can be attached later.

## 2.2 Refactor Constraint: The System Must Stay Runnable

The refactor should be staged so that after each major step you can still:

1. start the MCP server,
2. connect one of the LLM clients,
3. exercise the MuJoCo backend,
4. and confirm that the sim/backend behavior still works before continuing.

This matters more than architectural neatness. A good refactor here is not one giant cleanup. It is a sequence of small structural changes where the system remains demonstrably alive at each checkpoint.

That means each stage should:

- preserve a working startup path,
- preserve at least one end-to-end client workflow,
- avoid changing multiple deep seams at once,
- and end with a runnable validation gate.

## 3. Key Architectural Problems To Fix

### 3.1 The current backend abstraction is too Kinova-specific

`KinovaBackend` currently mixes:

- arm motion,
- gripper control,
- IK,
- scene reset,
- wrist rotation,
- and simulation stepping.

This is too much for one interface.

A different robot may:

- not have a wrist joint,
- not expose a percent-based gripper,
- not have scene reset,
- not use stepping,
- or use a different IK strategy entirely.

This violates interface segregation and reduces reuse.

### 3.2 `KinovaMuJoCoBackend` is a god object

The MuJoCo backend currently owns:

- model loading,
- joint discovery,
- actuator mapping,
- controller setup,
- trajectory generation,
- torque saturation,
- IK,
- end-effector extraction,
- reset logic,
- gripper state logic,
- and some configuration policy.

That violates single responsibility. It also makes it hard to test and hard to swap out one part without touching the whole class.

### 3.3 The MCP server depends on simulator internals

The MCP tools currently inspect private controller/backend state to reach MuJoCo `model` and `data`.

That is one of the most important things to remove.

The high-level server should not know:

- whether the backend uses MuJoCo,
- where object data lives,
- or what private fields exist inside the backend.

Instead, object/scene inspection must be exposed through an explicit interface.

### 3.4 The future hardware path does not have a clean seam yet

The current repo structure makes future SDK integration harder than it should be.

Right now, the real path is entangled conceptually with the MuJoCo path instead of being isolated behind a clean interface boundary.

That means:

- the abstraction is not trustworthy,
- the future `real` path will be fragile if the same structure is kept,
- and there is no clean separation between "real robot transport" and "shared motion services".

### 3.5 Packaging/import structure is fragile

Several modules rely on:

- bare imports like `from kinova_backend import ...`
- direct `sys.path.append(...)`

This prevents the repo from behaving like a proper Python package and makes reuse of individual modules much harder.

## 4. OOP Principles Being Violated

These are the main principle violations I would explicitly address.

### SRP: Single Responsibility Principle

Violations:

- `KinovaMuJoCoBackend` does too much.
- `mcp_kinova_server.py` owns startup, scene selection, controller lifecycle, blocking execution policy, and server registration.
- `mcp_server/tools.py` mixes parsing, validation, application policy, motion orchestration, and simulator scene inspection.

Refactor response:

- split lifecycle, simulation adapter, control logic, IK, and scene/query services into separate classes.

### LSP: Liskov Substitution Principle

Violations:

- the current abstraction is too large to support clean substitution,
- the declared interface is not satisfied consistently even in concept,
- tool behavior assumes simulator-only features exist.

Refactor response:

- define smaller contracts and test every implementation against them.

### ISP: Interface Segregation Principle

Violations:

- a single backend interface forces every backend to support unrelated capabilities.

Refactor response:

- replace one large backend ABC with capability-based interfaces.

### DIP: Dependency Inversion Principle

Violations:

- high-level MCP logic depends on concrete backend internals,
- future hardware code would be forced to depend on MuJoCo-specific assumptions if the design is not changed.

Refactor response:

- inject interfaces such as `IKSolver`, `SceneObjectProvider`, and `RobotArmRuntime`.

### Encapsulation / Law of Demeter

Violations:

- code reaches through `controller -> backend -> inner -> env -> model/data`.

Refactor response:

- expose explicit public methods or services for scene/object queries.

## 5. Target Architecture

The architecture I would move toward is composition-based, capability-based, and package-safe.

```text
LLM Client / Demo Script
        ->
Application Service Layer
        ->
Capability Interfaces
    - ArmMotion
    - GripperControl
    - IKSolver
    - SceneControl
    - ObjectQuery
        ->
Concrete Runtime Adapters
    - MuJoCoArmRuntime
    - FutureSDKRuntime
        ->
Robot Configuration + Shared Services
    - RobotModelConfig
    - Trajectory Service
    - Low-level Controller
    - IK Service
```

The important change is this:

- the server depends on capabilities,
- the capabilities are backed by composed services,
- and robot-specific data is configuration, not hard-coded behavior.

## 5.1 Primary Validation Loop

To keep the refactor safe, I would pick one primary end-to-end smoke test and one secondary regression test.

### Primary smoke test

Use `grab_shapes.py` as the default validation client during most stages because it exercises:

- MCP server startup,
- controller initialization,
- MuJoCo runtime stepping,
- object pose queries,
- IK,
- gripper control,
- and prompt/tool integration.

Recommended server command:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

Recommended client command:

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 20
```

Recommended pass criteria:

- the server starts without import/runtime errors,
- the client connects and loads tools/prompts,
- the arm can move home,
- object pose queries still work,
- at least one grasp sequence progresses through approach, descend, gripper actuation, and lift attempt,
- and no new backend/MuJoCo exceptions appear.

### Secondary regression test

Use `sort_cubes.py` after any stage that changes:

- object query,
- scene switching,
- MCP tool registration,
- or prompt/tool orchestration.

Recommended server command:

```bash
KINOVA_SCENE=sorting_task.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

Recommended client command:

```bash
python kinova_middleware/llm_clients/sort_cubes.py --max-steps 25
```

Recommended pass criteria:

- the client connects,
- sorting status or object lookup still works,
- arm motion and gripper tools still execute,
- and the workflow can make forward progress instead of failing immediately on tool/schema/backend errors.

### Tertiary spot check

Use `stack_cubes.py` only after the lower layers are stable, mainly to catch regressions in:

- repeated pick/place motions,
- placement geometry,
- and object query consistency.

## 6. Proposed Module Structure

I would gradually move toward something like this:

```text
kinova_middleware/
  backend/
    interfaces/
      arm.py
      gripper.py
      ik.py
      scene.py
      object_query.py
    config/
      robot_model.py
      kinova_gen3_lite.py
    runtime/
      mujoco_runtime.py
    services/
      motion_service.py
      gripper_service.py
      scene_service.py
      object_query_service.py
    adapters/
      mujoco_arm_adapter.py
    controller.py
    factory.py
    mcp_server/
      app.py
      tool_registry.py
      toolsets/
        motion_tools.py
        scene_tools.py
        task_prompts.py
```

This does not need to happen in one commit. It is a target shape, not a mandatory first move.

## 7. Concrete Refactor Strategy

The safest way to execute this refactor is to make every phase end in a runnable checkpoint.

Each phase below is designed so that:

- the code still has a working MCP server entrypoint,
- the existing LLM client path is still usable,
- and the refactor can be validated before the next phase starts.

## Phase 1: Make The Existing Code Importable And Testable

This is the first phase because nothing else is safe until imports and contracts are stable.

### 7.1 Fix package imports

Replace bare imports like:

```python
from kinova_backend import KinovaBackend
```

with package-safe imports like:

```python
from kinova_middleware.backend.kinova_backend import KinovaBackend
```

or relative imports where appropriate.

Remove all `sys.path.append(...)` workarounds from runtime code.

### Phase 1 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

In another shell:

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 10
```

Stage goal:

- startup/import cleanup only,
- no behavioral changes intended,
- same MCP tools and same client entrypoints should still work.

### 7.2 Add capability contract tests

Create tests that validate the new polymorphic seams using:

- the MuJoCo runtime,
- fake runtime implementations,
- and small test doubles for individual capabilities.

Do not make SDK implementation a dependency of this phase.

At minimum, add tests for:

- import safety,
- capability conformance,
- controller wrapping,
- factory construction,
- and MCP tool behavior against fake capabilities.

### Phase 2 validation gate

After adding the tests, rerun the same live path:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 10
```

Stage goal:

- tests exist without forcing a behavior change,
- the live MuJoCo/MCP path still runs exactly as before.

### 7.3 Stabilize the sim path and isolate the hardware path

Before larger redesign, make the current structure internally honest on the MuJoCo side:

- align the sim-facing interfaces,
- remove package/import fragility,
- and prevent the MCP layer from depending on simulator private state.

For the SDK code, only do the minimum needed to avoid contaminating the new abstractions. Do not invest in fixing or completing it in this phase.

### Phase 3 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 15
```

Stage goal:

- MuJoCo-side interfaces are stabilized,
- hardware code is isolated from the new sim-side design,
- the same client still proves the sim backend is operational.

## Phase 2: Replace One Big Backend Interface With Capability Interfaces

This is the most important architectural change.

### 7.4 Introduce capability interfaces

Define smaller interfaces such as:

```python
class ArmMotion(Protocol):
    def move_home(self) -> None: ...
    def send_joint_position_rad(self, q_des: Sequence[float]) -> None: ...
    def get_joint_angles_rad(self) -> list[float]: ...
    def get_joint_vel_rad(self) -> list[float]: ...

class IKSolver(Protocol):
    def solve_ik(...) -> list[float]: ...
    def solve_ik_position_only(...) -> list[float]: ...

class GripperControl(Protocol):
    def set_gripper_percent(self, percent: float) -> None: ...
    def get_gripper_state(self) -> dict: ...

class SceneControl(Protocol):
    def reset_scene(self) -> None: ...

class ObjectQuery(Protocol):
    def get_object_pose(self, name: str) -> dict: ...
```

The point is not the exact names. The point is that each service should depend only on what it needs.

Implementation note:

- introduce the new interfaces first,
- keep the existing controller/backend facade as an adapter layer temporarily,
- and do not cut over MCP tools directly to the new interfaces in the same step.

That preserves a runnable system while the new abstractions are being introduced.

### 7.5 Refactor the controller into a capability facade

The current `KinovaController` is mostly a thin pass-through. That is fine, but it should depend on explicit capabilities rather than one oversized backend.

Possible direction:

- a `RobotController` composed from `ArmMotion`, `IKSolver`, and optional `GripperControl`
- runtime capability checks
- no private reach-through in higher layers

### Phase 4 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 15
```

Stage goal:

- the new polymorphic interfaces exist,
- the old entrypoints still route through adapters,
- no LLM client changes should be required yet.

## Phase 3: Break Up `KinovaMuJoCoBackend`

This class should be decomposed by responsibility.

### 7.6 Extract robot configuration

Create a `RobotModelConfig` dataclass containing:

- model path,
- joint names,
- arm joint names,
- finger joint names,
- end-effector site/body,
- home keyframe,
- gain arrays,
- velocity/acceleration/jerk limits,
- and optional torque saturation settings.

This eliminates hard-coded robot assumptions from the class body.

Example:

```python
@dataclass
class RobotModelConfig:
    model_path: str
    arm_joint_names: list[str]
    finger_joint_names: list[str]
    ee_site: str | None
    ee_body: str
    home_keyframe: str | None
    kp_arm: np.ndarray
    kd_arm: np.ndarray
    v_max_arm: np.ndarray
    a_max_arm: np.ndarray
    j_max_arm: np.ndarray
```

Then create one concrete config for the current Kinova model instead of baking it into the backend.

### Phase 5 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 15
```

Stage goal:

- configuration has moved out,
- behavior should be functionally unchanged,
- the client should still drive the same grasp flow.

### 7.7 Extract MuJoCo model/runtime adapter

Create a focused class that owns:

- `MjModel`
- `MjData`
- joint and actuator mapping
- stepping and reset
- viewer sync

This adapter should not own IK policy, trajectory generation, or task logic.

### Phase 6 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 15
```

Stage goal:

- MuJoCo model/data ownership is isolated,
- but the server/client path still goes through the same user-facing commands.

### 7.8 Extract the IK solver into its own service

`LevenbergMarquardtIK` is already close to being a reusable service. It should be moved out of the backend and made to depend on a runtime adapter or a small kinematic model interface.

This lets:

- the MuJoCo backend use it,
- a future hardware backend reuse it through dependency injection,
- and future robots provide a different solver without editing backend logic.

### Phase 7 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 20
```

Optional extra check:

```bash
python kinova_middleware/llm_clients/stack_cubes.py --max-steps 20
```

Stage goal:

- IK has been extracted,
- grasping and lift motions still progress,
- no regression in pose solving or wrist-alignment flow.

### 7.9 Extract arm control and gripper control services

Split:

- trajectory tracking / computed torque control
- gripper command mapping

into separate services.

This will reduce the backend to composition, for example:

```text
MuJoCoRobotBackend =
  MuJoCoRuntime
  + ArmMotionService
  + MuJoCoIKService
  + MuJoCoGripperService
  + OptionalSceneQueryService
```

### 7.10 Remove hard-coded 4-DOF assumptions

Audit and remove assumptions such as:

- fixed-length gain arrays,
- fixed-length seed weights,
- hard-coded joint indices,
- hard-coded "wrist is last arm joint" policy,
- hard-coded finger joint naming patterns.

Anything that is truly robot-specific should live in configuration.

### Phase 8 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 20
```

Optional extra check:

```bash
KINOVA_SCENE=multi_cubes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/stack_cubes.py --max-steps 20
```

Stage goal:

- arm control and gripper control are decomposed,
- 4-DOF assumptions are being moved into config instead of hidden in behavior,
- the main grasping client still works.

## Phase 4: Refactor The MCP Server Around Capabilities

The server should expose tools based on what the selected runtime can actually do.

### 7.11 Introduce an application service layer

Add a thin layer between raw tools and backend/runtime code.

For example:

- `MotionApplicationService`
- `SceneApplicationService`
- `TaskSupportService`

The tools should call these services rather than manipulating controller state directly.

Implementation note:

- move one tool group at a time,
- do not rewrite all tools in one pass,
- and keep the old registrations available until the new service-backed registration has been validated.

### 7.12 Split tool registration into capability-based toolsets

Instead of one `setup_tools(...)`, register grouped toolsets such as:

- motion tools
- gripper tools
- scene tools
- object-query tools
- task/prompt tools

Then enable only the applicable toolsets for the active runtime.

For example:

- a future hardware runtime should not expose MuJoCo-only object-pose tools unless a real perception provider exists,
- simulator mode can expose scene and object query tools because the capability exists.

### 7.13 Stop accessing `_backend`, `_inner`, `_env`

Replace private reach-through with explicit public capabilities.

For example:

- `ObjectQuery.get_object_pose(name)`
- `SceneControl.list_objects()`
- `SceneControl.reset_scene()`

This is one of the highest-value cleanup steps in the whole refactor.

### 7.14 Move task heuristics out of prompts where possible

The prompts currently contain backend-specific workaround knowledge such as:

- zero quaternion meaning position-only IK
- `move_wrist=False` as a special control workaround
- hard-coded grasp margins and height offsets

Some of that belongs in reusable task-support helpers instead:

- grasp planning helpers,
- place planning helpers,
- workspace policy,
- and motion-planning constraints.

Prompts should describe task intent, not encode backend quirks.

### Phase 9 validation gate

Run:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 20
```

Then run the secondary regression path:

```bash
KINOVA_SCENE=sorting_task.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/sort_cubes.py --max-steps 25
```

Stage goal:

- MCP tools no longer depend on private MuJoCo internals,
- scene/object tool registration still works,
- clients can still load prompts and make forward progress.

## Phase 5: Define The Future Hardware Boundary Without Implementing It

This phase is design-only for now.

The goal is to make sure the MuJoCo refactor does not block future SDK work. It is not to build the SDK runtime in this pass.

### 7.15 Separate transport from motion services

When you later integrate the real robot, the hardware backend should not directly construct a MuJoCo backend just to get IK.

Instead:

- create an SDK transport/runtime adapter for command + telemetry later,
- inject a solver service into it later,
- and keep the transport layer independent of simulation details now in the design.

### 7.16 Decide what "is_reached" means on hardware

The current abstraction assumes a control loop with explicit stepping and convergence checks.

That is natural in simulation but not always on real hardware.

Choose one of these approaches when you work on the real robot:

1. Keep `is_reached` as a capability only for runtimes that support it.
2. Implement a hardware convergence monitor based on sensed joint error/velocity.
3. Introduce distinct motion execution modes for synchronous simulated control versus asynchronous hardware execution.

I would recommend option 1 or 2, but not pretending both runtimes work identically if they do not.

### Phase 10 validation gate

Run the primary and secondary sim validations again:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 20
```

```bash
KINOVA_SCENE=sorting_task.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

```bash
python kinova_middleware/llm_clients/sort_cubes.py --max-steps 25
```

Stage goal:

- the code now has a clean future SDK seam,
- but the only validated runtime remains MuJoCo,
- and the live sim path is still intact.

## Phase 6: Consolidate Task Logic And LLM Clients

The LLM clients currently duplicate routing/tool orchestration structure.

That is not the worst problem in the repo, but after backend cleanup it becomes worth consolidating.

### 7.17 Create shared workflow runners

Refactor repeated client logic into:

- shared MCP client setup,
- shared tool-loading,
- shared retry logic,
- shared finish/report flow,
- shared prompt selection logic.

Keep task-specific behavior only where it is actually task-specific.

### 7.18 Separate task semantics from robot control semantics

A useful long-term goal is to separate:

- "pick object"
- "place object"
- "stack block"

from:

- `move_pose`
- `rotate_wrist`
- `set_gripper`

That can be done via application-level task services or macro tools later.

I would not start here, but it becomes much easier once the lower layers are cleaned up.

## 8. Suggested Order Of Work

If I were implementing this refactor, I would do it in this order:

1. Fix imports and remove `sys.path` mutation.
2. Add capability contract tests using MuJoCo and fake runtimes.
3. Introduce capability interfaces alongside the old backend abstraction.
4. Extract robot config from `KinovaMuJoCoBackend`.
5. Extract MuJoCo runtime adapter.
6. Extract IK service.
7. Extract arm/gripper services.
8. Move MCP tools to capability-based registration.
9. Remove private-state access from MCP tools.
10. Consolidate prompts and task helpers.
11. Add a future SDK adapter seam, but do not implement it yet.
12. Rename/generalize classes once the structure is stable.

This order minimizes breakage while steadily improving design.

## 8.1 Recommended Stage Boundaries

To keep the repo runnable, I would avoid mixing these changes inside one stage:

### Stage boundary A

- imports/package cleanup only

Do not combine with:

- MCP rewrites
- IK extraction
- controller redesign

### Stage boundary B

- introduce interfaces and adapters only

Do not combine with:

- deleting the old backend facade
- changing client contracts

### Stage boundary C

- extract MuJoCo internals into runtime/services

Do not combine with:

- prompt rewrites
- tool-registration rewrites

### Stage boundary D

- refactor MCP tools to capability-backed services

Do not combine with:

- deep changes to motion control or IK internals

This separation is what makes the "rerun server + rerun client after every stage" approach practical.

## 9. Proposed Intermediate Compatibility Strategy

This repo is already large enough that a flag day rewrite would be risky.

I would use a transitional strategy:

### Step A

Keep the current public API working while introducing new internal abstractions.

### Step B

Add adapters so old code can call the new services.

### Step C

Migrate one layer at a time:

- first backend internals,
- then MCP tools,
- then clients,
- then demos.

### Step D

Only remove the old interface once:

- tests are passing,
- sim mode is stable,
- and the future SDK integration point is clearly defined.

## 10. Testing Strategy For The Refactor

The refactor will only stick if it is backed by tests.

### 10.1 Contract tests

For every implemented runtime/capability set, verify:

- constructibility,
- lifecycle,
- motion command acceptance,
- gripper capability behavior,
- IK capability behavior,
- optional scene/object capability presence.

### 10.2 Import tests

Add simple smoke tests that import all core modules as normal Python packages.

This catches packaging regressions immediately.

### 10.3 Fake backend tests for MCP tools

Create fake capability implementations so MCP tools can be tested without MuJoCo or hardware.

This will massively improve test speed and confidence.

### 10.4 Integration tests

For simulator mode:

- startup,
- reset,
- move home,
- move pose,
- gripper motion,
- and object pose query.

For the future hardware mode:

- define expected contract tests now,
- but defer implementation until the real robot phase.

## 10.5 Manual End-To-End Checklists

For each phase gate, I would manually record:

- whether the server started cleanly,
- whether the client connected,
- whether tools and prompts loaded,
- whether `move_home` still worked,
- whether `get_object_pose` still returned plausible values,
- whether `move_pose` still progressed without backend exceptions,
- whether `set_gripper` still actuated,
- and whether the client made forward task progress.

I would keep this as a short refactor log during the migration. That way if a phase breaks something, you know exactly which seam caused it.

## 11. What I Would Rename

The current naming makes the code feel more robot-specific than it needs to be.

Over time I would consider renaming:

- `KinovaBackend` -> `RobotBackend` or removing it in favor of capabilities
- `KinovaController` -> `RobotController`
- `KinovaMuJoCoBackend` -> `MuJoCoRobotRuntime` or `MuJoCoRobotBackend`
- `KinovaSDKBackend` -> `KinovaSDKRuntime` or `KinovaHardwareAdapter` later, when you actually integrate the real robot

I would only rename after the architectural split begins. Renaming too early makes the diff noisy without fixing the real problem.

## 12. What Should Stay

Not everything needs replacing.

Useful pieces worth preserving:

- the idea of a controller facade,
- the existing `CartesianPose` value object,
- the current LM IK implementation as a standalone service,
- the computed-torque controller,
- the trajectory generator,
- the reference governor,
- and the general MCP-based tooling concept.

The issue is not that the repo has no structure. The issue is that some good pieces are currently fused together too tightly.

## 13. Minimum Acceptable End State

If the full refactor is too large, the minimum acceptable end state should still achieve these outcomes:

- imports work without `sys.path` hacks,
- the MuJoCo runtime and fake runtimes both satisfy the new capability seams,
- MCP tools never use private backend state,
- simulator-only tools are not exposed in unsupported runtimes,
- MuJoCo-specific code is isolated behind a public interface,
- and robot-specific constants are moved into configuration objects.

If those six things are done, the repo becomes materially more reusable even before a full redesign.

## 14. Recommended First Refactor Ticket Set

If I were turning this into actionable tasks, I would start with these tickets:

1. Package import cleanup across `kinova_middleware/backend/` and `kinova_sim/`.
2. Backend interface audit focused on MuJoCo plus fake runtime test doubles.
3. Introduce `RobotModelConfig` and move MuJoCo constants into config.
4. Extract `LevenbergMarquardtIK` into a dedicated module/service.
5. Add `ObjectQuery` capability and stop MCP private-state access.
6. Split MCP tool registration into motion tools and scene/object tools.
7. Add backend contract tests and fake capability-based MCP tests.

That is the shortest high-value sequence I would recommend.

## 14.1 Recommended First Live Validation Pair

If you want exactly one default pair to rerun after each stage, I would use:

Server:

```bash
KINOVA_SCENE=shapes.xml mjpython kinova_middleware/backend/mcp_kinova_server.py
```

Client:

```bash
python kinova_middleware/llm_clients/grab_shapes.py --max-steps 20
```

Reason:

- it is the smallest full-stack workflow,
- it exercises the most important MuJoCo backend features,
- and it will fail quickly if the backend, MCP server, prompts, or tool schema have been broken.

## 15. Final Recommendation

The correct refactor is not a cosmetic cleanup and not a complete rewrite.

It should be a controlled architectural separation:

- first make the MuJoCo-side contracts honest,
- then replace one oversized abstraction with smaller capability interfaces,
- then decompose the MuJoCo backend into composable services,
- and finally make the MCP layer depend only on explicit public capabilities while leaving a clean future SDK seam.

That will give you a codebase that is:

- easier to maintain,
- easier to test,
- much easier to adapt to another robot,
- and far safer to reuse piece by piece.
