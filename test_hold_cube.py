import os
import sys
import time
import math
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
# Put kinova_middleware on sys.path
sys.path.append(os.path.join(_HERE, "kinova_middleware"))
sys.path.append(os.path.join(_HERE, "kinova_middleware", "backend"))

from kinova_mujoco_backend import KinovaMuJoCoBackend
from kinova_controller import KinovaController

def main():
    scene_path = os.path.join(_HERE, "kinova_middleware", "scenes", "01_single_cube.xml")
    
    print(f"Loading backend with {scene_path}...")
    backend = KinovaMuJoCoBackend(
        model_path=scene_path,
        viewer=False,
        ee_site="ee_marker",
        target_speed_rad_s=1.0,
    )
    ctrl = KinovaController(backend)
    ctrl.init()
    
    def wait_until_reached(timeout_s=5.0):
        start_real = time.time()
        steps = 0
        while True:
            reached = ctrl.is_reached(pos_tol_rad=0.08, vel_tol_rad_s=0.2)
            if reached:
                break
            
            elapsed_real = time.time() - start_real
            if elapsed_real > timeout_s:
                raise TimeoutError(f"Target not reached within {timeout_s}s. Curr: {ctrl.get_joint_angles_rad()}")
                
            ctrl.step()
            steps += 1
            if steps % 10 == 0:
                effective_dt = backend._n_substeps * backend._env.model.opt.timestep
                elapsed_sim = steps * effective_dt
                if elapsed_real < elapsed_sim:
                    time.sleep(elapsed_sim - elapsed_real)
    
    def step_for(duration):
        steps = int(duration * 1000)
        start_real = time.time()
        for i in range(steps):
            ctrl.step()
            if i % 10 == 0:
                effective_dt = backend._n_substeps * backend._env.model.opt.timestep
                elapsed_sim = (i + 1) * effective_dt
                elapsed_real = time.time() - start_real
                if elapsed_real < elapsed_sim:
                    time.sleep(elapsed_sim - elapsed_real)

    print("Homing...")
    ctrl.move_home()
    wait_until_reached()
    step_for(0.5)
    
    # 1. Open gripper
    print("Opening gripper...")
    ctrl.set_gripper_percent(1.0) # 1.0 is full open
    step_for(0.5)

    # Get the cube body id to read position
    model = backend._env.model
    data = backend._env.data
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_1")
    
    cube_start_pos = data.xpos[cube_id]
    # Target exact cube center
    cube_x = float(cube_start_pos[0])
    cube_y = float(cube_start_pos[1])
    cube_z = float(cube_start_pos[2])
    
    # 2. Face the cube
    print("Facing the cube...")
    current_q = ctrl.get_joint_angles_rad()
    
    # Calculate world theta, negate for base link rotation (180deg flipped Y frame)
    theta_world = math.atan2(cube_y, cube_x)
    j1_target = -theta_world
    j1_target = math.atan2(math.sin(j1_target), math.cos(j1_target))
    
    current_q[0] = j1_target
    ctrl.send_joint_position_rad(current_q)
    wait_until_reached()
    step_for(0.5)
    
    # Capture perfect vertical orientation from home position (after base rotation)
    _, hover_quat = ctrl.get_end_effector_pose()
    
    # 3. Move above
    print("Moving above cube...")
    hover_pos = [cube_x , cube_y, cube_z + 0.10]
    q_target = backend.solve_ik_position_only(hover_pos)
    
    ctrl.send_joint_position_rad(q_target)
    wait_until_reached()
    step_for(0.5)

    # 4. Descend
    print("Descending to grasp...")
    q_descend = backend.solve_ik_position_only([cube_x , cube_y , cube_z + 0.038])
    ctrl.send_joint_position_rad(q_descend)
    wait_until_reached()
    step_for(0.5)
    
    ee_pos, ee_quat = ctrl.get_end_effector_pose()
    print(f"  -> Reached EE pos: {ee_pos}")
    print(f"  -> Reached EE quat: {ee_quat}")
    
    # 4. Grasp (Progressive squeeze to avoid shooting the block out)
    print("Grasping cube gently at 50%...")
    ctrl.set_gripper_percent(0.50)
    step_for(1.0)
    print("Increasing grip to 30%...")
    ctrl.set_gripper_percent(0.30)
    step_for(1.0)
    print("Locking grip at 10% (max crush force)...")
    ctrl.set_gripper_percent(0.10)
    step_for(1.0)
    print(f"  -> Finger forces: {backend.get_finger_forces()}")
    
    # 5. Lift
    target_lift = [cube_x, cube_y, 0.35]
    print(f"Lifting to {target_lift}...")
    q_lift = backend.solve_ik_position_only(target_lift)
    ctrl.send_joint_position_rad(q_lift)
    wait_until_reached()
    step_for(0.5)
    

    
    # 6. Hold for 30 seconds, report location every 2s
    print("\n--- Starting 30 second HOLD test ---")
    hold_duration = 32.0
    check_interval = 2.0
    elapsed = 0.0
    
    start_z = data.xpos[cube_id][2]
    print(f"Cube Z after lift: {start_z:.3f} (target was ~0.35)")
    if start_z < 0.1:
        print("  [!!] Failed to pick up cube entirely!")
        hold_duration = 0.0
    
    while elapsed < hold_duration:
        step_for(check_interval)
        elapsed += check_interval
        
        c_pos = data.xpos[cube_id]
        print(f"  [Hold {elapsed:4.1f}s] Cube pos = ({c_pos[0]:.3f}, {c_pos[1]:.3f}, {c_pos[2]:.3f})")
        
        if c_pos[2] < start_z - 0.05:
            print("  [!!] Cube dropped!")
            break
            
    # 7. Move to place
    place_pos = [cube_x, cube_y, cube_z + 0.2]
    print(f"\nMoving to place target {place_pos}...")
    q_place = backend.solve_ik_position_only(place_pos)
    ctrl.send_joint_position_rad(q_place)
    wait_until_reached()
    step_for(0.5)

    print("Opening gripper to release...")    
    ctrl.set_gripper_percent(1.0)
    step_for(1.0)
    
    # 8. Retreat and home
    print("Retreating...")
    ctrl.send_joint_position_rad(q_lift)
    wait_until_reached()
    
    print("Homing...")
    ctrl.move_home()
    wait_until_reached()
    print("Done!")

if __name__ == "__main__":
    main()
