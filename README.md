# kinova_stack

kinova_description is from https://github.com/Kinovarobotics/kinova-ros/tree/noetic-devel

## Environment

To create and activate the conda virtual environment for simulation:
`$ conda env create -f environment.yml`
`$ conda activate kinova_stack`

## Structure
List of main folders:
- kinova_description
- kinova_control
- kinova_env
- scripts

## Debug notes
1. Warning under Windows WSL regarding GLFW: `GLFWError: (65548) b'Wayland: The platform does not provide the window position' warnings.warn(message, GLFWError)`

Solution: add the above lines to your shell config file (`~/.bashrc` or `~/.zshrc`):

`$ export LIBGL_ALWAYS_SOFTWARE=1`
`$ export MESA_GL_VERSION_OVERRIDE=3.3`
`$ source ~/.bashrc  # or ~/.zshrc`

2. Filter GLFW Warnings
`import warnings`
# Suppress GLFW Wayland window position warnings
`warnings.filterwarnings("ignore", message=".*Wayland: The platform does not provide the window position.*")`
`from mujoco.viewer import launch_passive`

## Tasks
* [x] add m1n4s300_standalone.urdf
* add actuators
* modify the urdf to have 4 dof mico
* add gripper descriptions
* [x] Joint position controller
* [x] Cartesian space controller
