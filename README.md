# kinova_stack

Kinova robot arm simulation and control stack using MuJoCo.

`kinova_description` is from https://github.com/Kinovarobotics/kinova-ros/tree/noetic-devel

## Prerequisites

- **Python 3.11** (used by the virtual environment)
- **MuJoCo** with `mjpython` — install via `pip install mujoco` (provides the `mjpython` launcher for macOS OpenGL support)
- A virtual environment (`.venv`) with the required packages

## Setup

1. **Create and activate the virtual environment** (if not already created):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install mujoco numpy==2.0.2 glfw PyOpenGL fastmcp openai
   ```

   > **Note:** NumPy must be `<2.1` (e.g. `2.0.2`) for compatibility with Python 3.11 and MuJoCo 3.4.0.

## Running the Simulation Demo

1. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

2. **Run the demo with `mjpython`:**

   ```bash
   mjpython kinova_middleware/demo_controller_sim.py
   ```

   > `mjpython` is required on macOS to handle MuJoCo's OpenGL rendering context. Using regular `python` will hang on import.

3. **The demo will:**
   - Initialize the Kinova controller in simulation mode (10 DOF)
   - Move the arm to the home position
   - Open and close the gripper fingers
   - Cycle through joint target positions while printing end-effector poses
   - Open a MuJoCo viewer window — **close the viewer window to exit**

## Project Structure

| Directory / File | Description |
|---|---|
| `kinova_middleware/` | Controller logic, backends (MuJoCo + SDK), and demo scripts |
| `kinova_description/` | URDF/MJCF robot model files |
| `kinova_sim/` | Simulation environment configuration |
| `kinova_firmware/` | Firmware-related files |
| `kinova-api-python/` | Kinova Python API bindings |

