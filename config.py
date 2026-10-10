# ── Analog steering constants ──────────────────────────────────────────────────

# Deadzone: steering angles smaller than this (°) are treated as straight
DEADZONE_DEG = 5

# Full-lock: steering angles beyond this (°) saturate to ±1.0 stick deflection
MAX_STEER_DEG = 40


# ── Hand gesture (accelerate / brake) constants ────────────────────────────────

# Fingertip landmark indices (thumb, index, middle, ring, pinky)
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Ratio thresholds for gesture classification
FIST_THRESHOLD = 0.65
OPEN_THRESHOLD = 0.85

# How many consecutive frames the gesture must hold before it fires
GESTURE_HOLD_FRAMES = 4


# ── Hand skeleton connections (MediaPipe) ──────────────────────────────────────

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ── MediaPipe model path ───────────────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
