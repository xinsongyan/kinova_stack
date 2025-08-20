# kinova_min.py
import ctypes
from ctypes import c_int

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


class FingersPosition(ctypes.Structure):
    _fields_ = [
        ("Finger1", ctypes.c_float),
        ("Finger2", ctypes.c_float),
        ("Finger3", ctypes.c_float),
    ]
    
class AngularPosition(ctypes.Structure):
    _fields_ = [
        ("Actuators", AngularInfo),
        ("Fingers", FingersPosition),
    ]

def finger_to_percent(raw, max_ticks=6800.0):
    return max(0.0, min(100.0, (raw / max_ticks) * 100.0))

class KinovaAPI:
    """
    Minimal ctypes wrapper for Kinova JACO SDK on Linux.
    Default uses the USB Command Layer. Switch to ETH by passing the ETH .so.
    """
    def __init__(self, so_path="/opt/JACO-SDK/API/Kinova.API.USBCommandLayerUbuntu.so"):
        print(f"Loading Kinova API shared library from: {so_path}")
        self.lib = ctypes.cdll.LoadLibrary(so_path)

        # --- declare function signatures (add more as needed) ---
        self.lib.InitAPI.restype = c_int
        self.lib.InitAPI.argtypes = []

        self.lib.CloseAPI.restype = c_int
        self.lib.CloseAPI.argtypes = []

        # Often available in CommandLayer:
        #   int MoveHome(void);
        # If your header shows a different signature, update here.
        self.lib.MoveHome.restype = c_int
        self.lib.MoveHome.argtypes = []
        
        self.lib.GetAngularPosition.restype = ctypes.c_int
        self.lib.GetAngularPosition.argtypes = [ctypes.POINTER(AngularPosition)]
        

    def init(self) -> int:
        return self.lib.InitAPI()

    def close(self) -> int:
        return self.lib.CloseAPI()

    def move_home(self) -> int:
        return self.lib.MoveHome()
    
    def get_angular_position(self) -> AngularPosition:
        position = AngularPosition()
        rc = self.lib.GetAngularPosition(ctypes.byref(position))
        if rc != 1:
            raise RuntimeError(f"GetAngularPosition failed with code {rc}")
        return position


if __name__ == "__main__":
    # Example usage
    api = KinovaAPI("/opt/JACO-SDK/API/Kinova.API.USBCommandLayerUbuntu.so")
    rc = api.init()
    print("InitAPI:", rc)
    if rc == 1:  # many Kinova APIs return 1 on success; check your header/docs
        try:
            print("MoveHome:", api.move_home())
            position = api.get_angular_position()
            print("Angular Position:", 
                  position.Actuators.Actuator1, 
                  position.Actuators.Actuator2,
                  position.Actuators.Actuator3, 
                  position.Actuators.Actuator4,
                  position.Actuators.Actuator5, 
                  position.Actuators.Actuator6)
            print("Finger raw values:", 
                  finger_to_percent(position.Fingers.Finger1), 
                  finger_to_percent(position.Fingers.Finger2), 
                  finger_to_percent(position.Fingers.Finger3))
        finally:
            print("CloseAPI:", api.close())
    else:
        print("InitAPI failed")