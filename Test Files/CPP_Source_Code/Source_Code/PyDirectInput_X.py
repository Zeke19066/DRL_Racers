MapVirtualKey = ctypes.windll.user32.MapVirtualKeyW

# KeyBdInput Flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_UNICODE = 0x0004

_fields_ = [("ki", KeyBdInput),

    'up': MapVirtualKey(0x26, MAPVK_VK_TO_VSC),
    'left': MapVirtualKey(0x25, MAPVK_VK_TO_VSC),
    'down': MapVirtualKey(0x28, MAPVK_VK_TO_VSC),
    'right': MapVirtualKey(0x27, MAPVK_VK_TO_VSC),
    

 class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

                
keybdFlags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY  
extra = ctypes.c_ulong(0)
hexKeyCode = KEYBOARD_MAPPING[key]

ii_.ki = KeyBdInput(0, hexKeyCode, keybdFlags, 0, ctypes.pointer(extra))

SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))