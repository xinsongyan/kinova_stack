import os
import ctypes
import math


def euler_xyz_to_quaternion(theta_x: float, theta_y: float, theta_z: float) -> tuple[float, float, float, float]:
    """Convert XYZ intrinsic Euler angles (rad) to quaternion (qx, qy, qz, qw)."""
    cx = math.cos(theta_x * 0.5)
    sx = math.sin(theta_x * 0.5)
    cy = math.cos(theta_y * 0.5)
    sy = math.sin(theta_y * 0.5)
    cz = math.cos(theta_z * 0.5)
    sz = math.sin(theta_z * 0.5)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return (qx, qy, qz, qw)


STRING_LENGTH = 20
MAX_KINOVA_DEVICE = 20
API_VERSION_COUNT = 3
CARTESIAN_SIZE = 6
MAX_ACTUATORS = 7

# POSITION_TYPE enum (subset)
NOMOVEMENT_POSITION = 0
CARTESIAN_POSITION = 1
ANGULAR_POSITION = 2

# HAND_MODE enum (subset)
HAND_NOMOVEMENT = 0
POSITION_MODE = 1
VELOCITY_MODE = 2


class KinovaDevice(ctypes.Structure):
    _fields_ = [
        ("SerialNumber", ctypes.c_char * STRING_LENGTH),
        ("Model", ctypes.c_char * STRING_LENGTH),
        ("VersionMajor", ctypes.c_int),
        ("VersionMinor", ctypes.c_int),
        ("VersionRelease", ctypes.c_int),
        ("DeviceType", ctypes.c_int),
        ("DeviceID", ctypes.c_int),
    ]


class CartesianInfo(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_float),
        ("Y", ctypes.c_float),
        ("Z", ctypes.c_float),
        ("ThetaX", ctypes.c_float),
        ("ThetaY", ctypes.c_float),
        ("ThetaZ", ctypes.c_float),
    ]


class FingersPosition(ctypes.Structure):
    _fields_ = [
        ("Finger1", ctypes.c_float),
        ("Finger2", ctypes.c_float),
        ("Finger3", ctypes.c_float),
    ]


class CartesianPosition(ctypes.Structure):
    _fields_ = [
        ("Coordinates", CartesianInfo),
        ("Fingers", FingersPosition),
    ]


class AngularInfo(ctypes.Structure):
    _fields_ = [
        ("Actuator1", ctypes.c_float),
        ("Actuator2", ctypes.c_float),
        ("Actuator3", ctypes.c_float),
        ("Actuator4", ctypes.c_float),
        ("Actuator5", ctypes.c_float),
        ("Actuator6", ctypes.c_float),
        ("Actuator7", ctypes.c_float),
    ]


class AngularPosition(ctypes.Structure):
    _fields_ = [
        ("Actuators", AngularInfo),
        ("Fingers", FingersPosition),
    ]


class Limitation(ctypes.Structure):
    _fields_ = [
        ("speedParameter1", ctypes.c_float),
        ("speedParameter2", ctypes.c_float),
        ("speedParameter3", ctypes.c_float),
        ("forceParameter1", ctypes.c_float),
        ("forceParameter2", ctypes.c_float),
        ("forceParameter3", ctypes.c_float),
        ("accelerationParameter1", ctypes.c_float),
        ("accelerationParameter2", ctypes.c_float),
        ("accelerationParameter3", ctypes.c_float),
    ]


class UserPosition(ctypes.Structure):
    _fields_ = [
        ("Type", ctypes.c_int),
        ("Delay", ctypes.c_float),
        ("CartesianPosition", CartesianInfo),
        ("Actuators", AngularInfo),
        ("HandMode", ctypes.c_int),
        ("Fingers", FingersPosition),
    ]


class TrajectoryPoint(ctypes.Structure):
    _fields_ = [
        ("Position", UserPosition),
        ("LimitationsActive", ctypes.c_int),
        ("SynchroType", ctypes.c_int),
        ("Limitations", Limitation),
    ]


class KinovaAPI:
    def __init__(self, lib_dir: str | None = None):
        self._usb = None
        self._lib_dir = lib_dir or os.path.dirname(os.path.abspath(__file__))
        self._load_libs()
        self._bind()

    def _load_libs(self) -> None:
        # Support both root placement and within a 'KinovaAPI' subfolder
        candidates = [
            (os.path.join(self._lib_dir, "Kinova.API.CommLayerUbuntu.so"),
             os.path.join(self._lib_dir, "Kinova.API.USBCommandLayerUbuntu.so")),
            (os.path.join(self._lib_dir, "KinovaAPI", "Kinova.API.CommLayerUbuntu.so"),
             os.path.join(self._lib_dir, "KinovaAPI", "Kinova.API.USBCommandLayerUbuntu.so")),
        ]

        comm = usb = None
        for comm_path, usb_path in candidates:
            if os.path.exists(comm_path) and os.path.exists(usb_path):
                comm, usb = comm_path, usb_path
                break

        if comm is None or usb is None:
            raise FileNotFoundError(
                "Could not find Kinova shared libraries. Looked in current directory and 'KinovaAPI/' subfolder."
            )
        ctypes.CDLL(comm, mode=ctypes.RTLD_GLOBAL)
        self._usb = ctypes.CDLL(usb, mode=ctypes.RTLD_GLOBAL)

    def _bind(self) -> None:
        u = self._usb
        u.InitAPI.argtypes = []
        u.InitAPI.restype = ctypes.c_int

        u.CloseAPI.argtypes = []
        u.CloseAPI.restype = ctypes.c_int

        u.GetAPIVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        u.GetAPIVersion.restype = ctypes.c_int

        u.GetDevices.argtypes = [ctypes.POINTER(KinovaDevice), ctypes.POINTER(ctypes.c_int)]
        u.GetDevices.restype = ctypes.c_int

        u.SetActiveDevice.argtypes = [KinovaDevice]
        u.SetActiveDevice.restype = ctypes.c_int

        u.GetCartesianPosition.argtypes = [ctypes.POINTER(CartesianPosition)]
        u.GetCartesianPosition.restype = ctypes.c_int

        u.GetAngularPosition.argtypes = [ctypes.POINTER(AngularPosition)]
        u.GetAngularPosition.restype = ctypes.c_int

        u.GetAngularCurrent.argtypes = [ctypes.POINTER(AngularPosition)]
        u.GetAngularCurrent.restype = ctypes.c_int

        u.MoveHome.argtypes = []
        u.MoveHome.restype = ctypes.c_int

        u.EraseAllTrajectories.argtypes = []
        u.EraseAllTrajectories.restype = ctypes.c_int

        u.SendBasicTrajectory.argtypes = [TrajectoryPoint]
        u.SendBasicTrajectory.restype = ctypes.c_int
        
        u.InitFingers.argtypes = []
        u.InitFingers.restype = ctypes.c_int

    @staticmethod
    def _ensure_ok(rc: int, where: str) -> None:
        if rc != 1:
            raise RuntimeError(f"{where} failed with code {rc}")

    # ---- Lifecycle ----
    def init(self) -> None:
        rc = self._usb.InitAPI()
        self._ensure_ok(rc, "InitAPI")

    def close(self) -> None:
        try:
            self._usb.CloseAPI()
        except Exception:
            pass

    # ---- Info ----
    def get_api_version(self) -> tuple[int, int, int]:
        buf = (ctypes.c_int * API_VERSION_COUNT)()
        rc = self._usb.GetAPIVersion(buf)
        self._ensure_ok(rc, "GetAPIVersion")
        return (buf[0], buf[1], buf[2])

    def list_devices(self) -> list[KinovaDevice]:
        devs = (KinovaDevice * MAX_KINOVA_DEVICE)()
        count = ctypes.c_int(0)
        rc = self._usb.GetDevices(devs, ctypes.byref(count))
        self._ensure_ok(rc, "GetDevices")
        return [devs[i] for i in range(count.value)]

    def set_active_device(self, device: KinovaDevice) -> None:
        rc = self._usb.SetActiveDevice(device)
        self._ensure_ok(rc, "SetActiveDevice")

    # ---- Read ----
    def get_cartesian_position(self) -> CartesianPosition:
        pos = CartesianPosition()
        rc = self._usb.GetCartesianPosition(ctypes.byref(pos))
        self._ensure_ok(rc, "GetCartesianPosition")
        return pos

    def get_end_effector_pose(self) -> list[float]:
        """Return end-effector pose [x, y, z, thetaX, thetaY, thetaZ].

        Units: meters for x,y,z and radians for thetas (as provided by the API).
        """
        cp = self.get_cartesian_position().Coordinates
        return [float(cp.X), float(cp.Y), float(cp.Z), float(cp.ThetaX), float(cp.ThetaY), float(cp.ThetaZ)]

    def get_end_effector_pos_quat(self) -> list[float]:
        """Return end-effector pose with quaternion orientation [x, y, z, qx, qy, qz, qw]."""
        cp = self.get_cartesian_position().Coordinates
        qx, qy, qz, qw = euler_xyz_to_quaternion(float(cp.ThetaX), float(cp.ThetaY), float(cp.ThetaZ))
        return [float(cp.X), float(cp.Y), float(cp.Z)], [qx, qy, qz, qw]

    def get_angular_position(self) -> AngularPosition:
        pos = AngularPosition()
        rc = self._usb.GetAngularPosition(ctypes.byref(pos))
        self._ensure_ok(rc, "GetAngularPosition")
        return pos

    def get_angular_current(self) -> AngularPosition:
        cur = AngularPosition()
        rc = self._usb.GetAngularCurrent(ctypes.byref(cur))
        self._ensure_ok(rc, "GetAngularCurrent")
        return cur

    def get_joint_angles_deg(self) -> list[float]:
        """Return the 7 joint angles as a list of floats (degrees)."""
        pos = self.get_angular_position()
        a = pos.Actuators
        return [
            float(a.Actuator1),
            float(a.Actuator2),
            float(a.Actuator3),
            float(a.Actuator4),
            float(a.Actuator5),
            float(a.Actuator6),
            float(a.Actuator7),
        ]

    def get_joint_angles_rad(self) -> list[float]:
        """Return the 7 joint angles in radians."""
        deg = self.get_joint_angles_deg()
        return [math.radians(x) for x in deg]

    def get_finger_ticks(self) -> list[float]:
        """Return the 3 finger positions as raw ticks (floats)."""
        pos = self.get_angular_position()
        f = pos.Fingers
        return [float(f.Finger1), float(f.Finger2), float(f.Finger3)]

    def get_finger_percent(self) -> list[float]:
        """Return the 3 finger positions as normalized percentages in [0.0, 1.0]."""
        t1, t2, t3 = self.get_finger_ticks()
        return [
            self.finger_ticks_to_percent(t1),
            self.finger_ticks_to_percent(t2),
            self.finger_ticks_to_percent(t3),
        ]

    

    # ---- Motion helpers ----
    def move_home(self) -> None:
        rc = self._usb.MoveHome()
        self._ensure_ok(rc, "MoveHome")

    def erase_all_trajectories(self) -> None:
        rc = self._usb.EraseAllTrajectories()
        self._ensure_ok(rc, "EraseAllTrajectories")

    # ---- Gripper helpers ----
    def init_fingers(self) -> None:
        rc = self._usb.InitFingers()
        self._ensure_ok(rc, "InitFingers")



    def set_fingers_tick(self, t1: float, t2: float, t3: float) -> None:
        """Set finger positions using RAW TICKS, preserving current cartesian pose.

        Inputs f1, f2, f3 are in device ticks (e.g., 0.0=open, ~6800.0=closed on many models).
        If you prefer normalized percentages [0.0, 1.0], use open_fingers/close_fingers
        or convert with percent_to_finger_ticks() before calling this.
        """
        current = CartesianPosition()
        rc = self._usb.GetCartesianPosition(ctypes.byref(current))
        self._ensure_ok(rc, "GetCartesianPosition(before set_fingers)")

        point = TrajectoryPoint()
        point.Position.Type = CARTESIAN_POSITION
        point.Position.HandMode = POSITION_MODE
        point.Position.CartesianPosition = current.Coordinates
        point.Position.Fingers.Finger1 = float(t1)
        point.Position.Fingers.Finger2 = float(t2)
        point.Position.Fingers.Finger3 = float(t3)
        point.LimitationsActive = 0
        point.SynchroType = 0
        rc = self._usb.SendBasicTrajectory(point)
        self._ensure_ok(rc, "SendBasicTrajectory(set_fingers_tick)")

    def set_fingers_percent(self, p1: float, p2: float, p3: float) -> None:
        """Set finger positions using normalized percentages in [0.0, 1.0].

        Values outside [0.0, 1.0] are clamped. This preserves the current cartesian pose.
        """
        t1 = self.percent_to_finger_ticks(p1)
        t2 = self.percent_to_finger_ticks(p2)
        t3 = self.percent_to_finger_ticks(p3)
        self.set_fingers_tick(t1, t2, t3)

    def close_fingers(self, percent: float = 1.0) -> None:
        """Close all fingers to the provided percentage (0.0-1.0), preserving pose."""
        ticks = self.percent_to_finger_ticks(percent)
        self.set_fingers_tick(ticks, ticks, ticks)

    def open_fingers(self, percent: float = 0.0) -> None:
        """Open fingers to the provided percentage (0.0-1.0), preserving pose."""
        ticks = self.percent_to_finger_ticks(percent)
        self.set_fingers_tick(ticks, ticks, ticks)

    @staticmethod
    def finger_ticks_to_percent(raw: float, max_ticks: float = 6800.0) -> float:
        """Convert raw finger ticks to normalized percent in [0.0, 1.0]."""
        return max(0.0, min(1.0, (raw / max_ticks)))

    @staticmethod
    def percent_to_finger_ticks(percent: float, max_ticks: float = 6800.0) -> float:
        """Convert normalized percent (0.0-1.0) to raw ticks, clamped to range."""
        p = max(0.0, min(1.0, percent))
        return p * max_ticks



def device_str(dev: KinovaDevice) -> str:
    serial = dev.SerialNumber.decode(errors="ignore").rstrip("\x00")
    model = dev.Model.decode(errors="ignore").rstrip("\x00")
    return f"Serial={serial} Model={model} FW={dev.VersionMajor}.{dev.VersionMinor}.{dev.VersionRelease}"



