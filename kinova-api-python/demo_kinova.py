#!/usr/bin/env python3
import sys
from kinova_api import KinovaAPI, device_str


def main():
    api = KinovaAPI()
    try:
        api.init()

        ver = api.get_api_version()
        print(f"API version: {ver[0]}.{ver[1]:02d}.{ver[2]:02d}")

        devices = api.list_devices()
        print(f"Devices found: {len(devices)}")
        if not devices:
            print("No Kinova device detected. Ensure the arm is connected and permissions are set.")
            return 0

        for idx, dev in enumerate(devices):
            print(f"[{idx}] {device_str(dev)}")

        api.set_active_device(devices[0])
        print("Active device set to index 0")

        api.init_fingers()
        print("Fingers initialized.")

        # Go Home test
        print("Sending MoveHome command...")
        api.move_home()
        print("MoveHome command sent.")

        # Open fingers after homing
        print("Opening fingers...")
        api.open_fingers()
        print("Fingers open command sent.")

        # close fingers
        print("Closing fingers...")
        api.close_fingers()
        print("Fingers closed command sent.")


        # get cartesian position
        get_end_effector_pose = api.get_end_effector_pos_quat()
        print(f"End Effector Pose: {get_end_effector_pose}")


        angles = api.get_joint_angles_rad()
        print(f"Joint Angles: {angles}")
        # [J1, J2, J3, J4, J5, J6, J7]


        finger_percent = api.get_finger_ticks()
        print(f"Finger Percent: {finger_percent}")
        # [F1, F2, F3]
    finally:
        api.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(2)


