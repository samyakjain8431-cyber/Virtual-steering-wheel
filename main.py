import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import vgamepad as vg
from pynput.keyboard import Key, Controller as KeyboardController



# ── Analog steering constants ──────────────────────────────────────────────────

# Deadzone: steering angles smaller than this (°) are treated as straight
DEADZONE_DEG = 5

# Full-lock: steering angles beyond this (°) saturate to ±1.0 stick deflection
MAX_STEER_DEG = 40



# ── Hand gesture (accelerate / brake) constants ────────────────────────────────

# Fingertip landmark indices (thumb, index, middle, ring, pinky)
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Ratio of (avg fingertip-to-wrist distance) / (hand span from wrist to middle-MCP)
# Below FIST_THRESHOLD  → fist  (accelerate)
# Above OPEN_THRESHOLD  → open  (brake)
# Between the two       → neutral (coasting)
FIST_THRESHOLD = 0.65   # tweak if needed
OPEN_THRESHOLD = 0.85   # tweak if needed

# How many consecutive frames the gesture must hold before it fires
GESTURE_HOLD_FRAMES = 4



# Hand Connections (MediaPipe Hand Skeleton)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]



# ── Virtual Xbox Controller ────────────────────────────────────────────────────

gamepad = vg.VX360Gamepad()

# Make sure stick starts centred
gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
gamepad.update()

# ── Keyboard controller for W / S key simulation ──────────────────────────────
kb = KeyboardController()

# Track which keys are currently held so we don't spam press/release
_key_held = {'w': False, 's': False}

def set_key(key_char: str, pressed: bool):
    """Press or release a single key only when its state actually changes."""
    if _key_held[key_char] == pressed:
        return          # already in the right state
    if pressed:
        kb.press(key_char)
    else:
        kb.release(key_char)
    _key_held[key_char] = pressed



# Load MediaPipe Hand Landmarker

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)


neutral_angle = None
smooth_angle = 0
analog_x = 0.0          # current stick X position (-1.0 … +1.0)

# ── Gesture state ──────────────────────────────────────────────────────────────
# Per-hand gesture vote each frame: 'FIST' | 'OPEN' | 'NEUTRAL'
gesture_votes   = []          # collected across both hands each frame
gesture_counter = {'FIST': 0, 'OPEN': 0, 'NEUTRAL': 0}  # rolling hold counts
current_gesture = 'NEUTRAL'   # last committed gesture
smooth_trigger  = 0.0         # smoothed trigger value sent to gamepad



# ── Virtual Steering Wheel drawing helper ──────────────────────────────────────

def draw_steering_wheel(frame, center, radius, rotation_deg, analog_x, direction):
    """
    Draw an animated virtual steering wheel on *frame* in-place.

    Parameters
    ----------
    center       : (cx, cy)  – pixel position of the wheel hub
    radius       : int       – outer radius in pixels
    rotation_deg : float     – current steering rotation (degrees)
    analog_x     : float     – stick value in [-1, +1], used for colour tint
    direction    : str       – 'LEFT' | 'RIGHT' | 'STRAIGHT'
    """
    cx, cy = center
    rot_rad = math.radians(rotation_deg)

    # ── Colour palette based on direction ─────────────────────────────────────
    if direction == "LEFT":
        rim_color   = (255, 140,  40)   # warm orange
        spoke_color = (255, 180,  80)
        hub_color   = (200, 100,  20)
    elif direction == "RIGHT":
        rim_color   = ( 40, 140, 255)   # cool blue
        spoke_color = ( 80, 180, 255)
        hub_color   = ( 20, 100, 200)
    else:
        rim_color   = ( 50, 220,  80)   # neutral green
        spoke_color = ( 80, 255, 120)
        hub_color   = ( 30, 160,  50)

    # ── Semi-transparent backing disc ─────────────────────────────────────────
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius + 4, (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Outer rim (thick circle) ───────────────────────────────────────────────
    rim_thickness = max(6, radius // 8)
    cv2.circle(frame, (cx, cy), radius, rim_color, rim_thickness)

    # ── Slight 3-D bevel on rim ────────────────────────────────────────────────
    highlight = tuple(min(255, c + 80) for c in rim_color)
    shadow    = tuple(max(0,   c - 80) for c in rim_color)
    cv2.circle(frame, (cx, cy), radius,          highlight, 1)
    cv2.circle(frame, (cx, cy), radius - rim_thickness + 1, shadow, 1)

    # ── Three spokes at 0°, 120°, 240° (rotated by rot_rad) ──────────────────
    inner_r = radius // 5          # spoke starts near hub
    outer_r = radius - rim_thickness // 2
    spoke_w = max(3, radius // 12)

    for spoke_offset_deg in (0, 120, 240):
        angle = rot_rad + math.radians(spoke_offset_deg)
        x1 = int(cx + inner_r * math.cos(angle))
        y1 = int(cy + inner_r * math.sin(angle))
        x2 = int(cx + outer_r * math.cos(angle))
        y2 = int(cy + outer_r * math.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), spoke_color, spoke_w, cv2.LINE_AA)

    # ── Hub cap ────────────────────────────────────────────────────────────────
    hub_r = max(6, radius // 6)
    cv2.circle(frame, (cx, cy), hub_r,     hub_color,       -1)
    cv2.circle(frame, (cx, cy), hub_r,     highlight,        1)
    cv2.circle(frame, (cx, cy), hub_r - 3, (30, 30, 30),    -1)

    # ── Steering-angle arc indicator around the rim ───────────────────────────
    #   Draw a coloured arc proportional to how far the wheel is turned.
    arc_span = int(abs(analog_x) * 135)   # max 135° arc
    if arc_span > 2:
        start_angle = int(math.degrees(rot_rad)) - arc_span // 2
        end_angle   = start_angle + arc_span
        cv2.ellipse(
            frame, (cx, cy),
            (radius + rim_thickness // 2 + 3,
             radius + rim_thickness // 2 + 3),
            0, start_angle, end_angle,
            highlight, 2, cv2.LINE_AA
        )

    # ── Direction label below the wheel ───────────────────────────────────────
    label_y = cy + radius + rim_thickness + 18
    font_scale = max(0.45, radius / 90.0)
    cv2.putText(
        frame, direction,
        (cx - 30, label_y),
        cv2.FONT_HERSHEY_DUPLEX, font_scale,
        rim_color, 1, cv2.LINE_AA
    )




# ── Gesture detection helper ───────────────────────────────────────────────────

def classify_hand_gesture(hand_landmarks, frame_w, frame_h):
    """
    Return 'FIST', 'OPEN', or 'NEUTRAL' for a single hand.

    Strategy
    --------
    Compute the ratio:
        avg_tip_dist / hand_scale
    where
        avg_tip_dist = mean Euclidean distance (px) from each fingertip to wrist
        hand_scale   = distance (px) from wrist (0) to middle-finger MCP (9)

    Small ratio → fingers curled → FIST
    Large ratio → fingers extended → OPEN
    """
    wrist = hand_landmarks[0]
    wx = wrist.x * frame_w
    wy = wrist.y * frame_h

    # Hand scale: wrist → middle-finger MCP (landmark 9)
    mid_mcp = hand_landmarks[9]
    scale = math.hypot(
        (mid_mcp.x * frame_w) - wx,
        (mid_mcp.y * frame_h) - wy
    )
    if scale < 1e-6:          # degenerate hand – skip
        return 'NEUTRAL'

    # Average fingertip distance from wrist
    tip_distances = []
    for tip_id in FINGERTIP_IDS:
        tip = hand_landmarks[tip_id]
        d = math.hypot(
            (tip.x * frame_w) - wx,
            (tip.y * frame_h) - wy
        )
        tip_distances.append(d)

    ratio = (sum(tip_distances) / len(tip_distances)) / scale

    if ratio < FIST_THRESHOLD:
        return 'FIST'
    elif ratio > OPEN_THRESHOLD:
        return 'OPEN'
    else:
        return 'NEUTRAL'


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()



while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Slight blur
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    h, w, _ = frame.shape

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = landmarker.detect(mp_image)

    left_wrist = None
    right_wrist = None
    gesture_votes = []   # reset each frame


    
    # Hand Detection
    
    if result.hand_landmarks:

        for hand_index, hand in enumerate(result.hand_landmarks):

            
            # Draw Hand Skeleton
            
            for start, end in HAND_CONNECTIONS:

                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


            
            # Draw Landmarks
            
            for i, landmark in enumerate(hand):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            
            # Thumb Landmarks
            
            thumb_tip = hand[4]
            thumb_ip = hand[3]
            thumb_mcp = hand[2]

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)


            
            # ACTUAL WRIST POINT
            # Landmark 0 = Wrist
            
            wrist = hand[0]

            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            # Highlight wrist
            cv2.circle(
                frame,
                (wrist_x, wrist_y),
                7,
                (0, 255, 0),
                -1
            )


            
            # Left / Right Hand
            
            hand_name = result.handedness[
                hand_index
            ][0].category_name


            cv2.putText(
                frame,
                f"{hand_name} TipY:{thumb_tip.y:.3f} "
                f"IPY:{thumb_ip.y:.3f}",
                (10, 150 if hand_name == "Left" else 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


            
            # Store Wrist Position
            
            if hand_name == "Left":

                left_wrist = (wrist_x, wrist_y)

            else:

                right_wrist = (wrist_x, wrist_y)


            # ── Classify gesture for this hand ────────────────────────────────
            gesture = classify_hand_gesture(hand, w, h)
            gesture_votes.append(gesture)


    
    # STEERING
    
    if left_wrist is not None and right_wrist is not None:

        
        # Draw line between the two wrist points
        
        cv2.line(
            frame,
            left_wrist,
            right_wrist,
            (255, 0, 0),
            2
        )


        
        # Center point between wrists
        
        center_x = (
            left_wrist[0] + right_wrist[0]
        ) // 2

        center_y = (
            left_wrist[1] + right_wrist[1]
        ) // 2

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )


        
        # Calculate angle using WRISTS
        
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        angle = math.degrees(
            math.atan2(dy, dx)
        )


        
        # Smooth angle
        
        alpha = 0.1

        smooth_angle = (
            alpha * angle
            + (1 - alpha) * smooth_angle
        )


        cv2.putText(
            frame,
            f"Angle: {smooth_angle:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        
        # Set Neutral Position
        
        if neutral_angle is None:

            neutral_angle = smooth_angle


        steering_angle = (
            smooth_angle - neutral_angle
        )


        # ── Analog stick mapping ───────────────────────────────────────────────
        #
        #  steering_angle > 0  →  hands tilted LEFT  →  steer LEFT  → stick X < 0
        #  steering_angle < 0  →  hands tilted RIGHT  →  steer RIGHT → stick X > 0
        #
        #  1. Apply deadzone  – tiny wobbles produce zero output
        #  2. Remap remaining range to [-1, +1] relative to MAX_STEER_DEG
        #  3. Clamp to [-1, +1] so full-lock is always possible

        if abs(steering_angle) < DEADZONE_DEG:
            raw_x = 0.0
        else:
            # Remove deadzone offset, then scale
            sign = 1.0 if steering_angle > 0 else -1.0
            effective = abs(steering_angle) - DEADZONE_DEG
            usable_range = MAX_STEER_DEG - DEADZONE_DEG
            raw_x = sign * (effective / usable_range)

        # Clamp to [-1, +1]
        raw_x = max(-1.0, min(1.0, raw_x))

        # Invert: positive steering_angle = left turn = negative stick X
        target_x = -raw_x

        # Smooth the analog output a little (optional, reduces jitter)
        stick_alpha = 0.3
        analog_x = stick_alpha * target_x + (1 - stick_alpha) * analog_x

        # Send to virtual controller
        gamepad.left_joystick_float(
            x_value_float=analog_x,
            y_value_float=0.0
        )
        gamepad.update()

        # ── Derive direction label for display only ────────────────────────────
        if steering_angle > DEADZONE_DEG:
            direction = "LEFT"
        elif steering_angle < -DEADZONE_DEG:
            direction = "RIGHT"
        else:
            direction = "STRAIGHT"

        
        # Display Steering Information
        
        cv2.putText(
            frame,
            f"Steering Angle: {steering_angle:.2f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Direction: {direction}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Analog stick bar visualisation
        bar_center_x = w // 2
        bar_y = h - 30
        bar_half = 100
        fill_x = int(analog_x * bar_half)

        cv2.rectangle(frame, (bar_center_x - bar_half, bar_y - 10),
                      (bar_center_x + bar_half, bar_y + 10), (60, 60, 60), -1)
        cv2.rectangle(frame,
                      (bar_center_x, bar_y - 8),
                      (bar_center_x + fill_x, bar_y + 8),
                      (0, 200, 255), -1)
        cv2.rectangle(frame, (bar_center_x - bar_half, bar_y - 10),
                      (bar_center_x + bar_half, bar_y + 10), (200, 200, 200), 1)
        cv2.putText(frame, f"Stick X: {analog_x:+.2f}",
                    (bar_center_x - bar_half, bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    else:
        # No hands detected – centre the stick
        analog_x = 0.0
        gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gamepad.update()


    # ── Aggregate gesture votes from both hands ────────────────────────────────
    # Majority vote across detected hands this frame
    if gesture_votes:
        fist_count = gesture_votes.count('FIST')
        open_count = gesture_votes.count('OPEN')
        if fist_count > open_count:
            frame_gesture = 'FIST'
        elif open_count > fist_count:
            frame_gesture = 'OPEN'
        else:
            frame_gesture = 'NEUTRAL'
    else:
        frame_gesture = 'NEUTRAL'

    # Hysteresis: require GESTURE_HOLD_FRAMES consecutive frames
    for g in ('FIST', 'OPEN', 'NEUTRAL'):
        if g == frame_gesture:
            gesture_counter[g] = min(
                gesture_counter[g] + 1, GESTURE_HOLD_FRAMES
            )
        else:
            gesture_counter[g] = max(gesture_counter[g] - 1, 0)

    if gesture_counter['FIST'] >= GESTURE_HOLD_FRAMES:
        current_gesture = 'FIST'
    elif gesture_counter['OPEN'] >= GESTURE_HOLD_FRAMES:
        current_gesture = 'OPEN'
    else:
        current_gesture = 'NEUTRAL'

    # ── Map gesture → keyboard W / S  +  gamepad triggers ───────────────────
    #   FIST    : hold W  (accelerate)    + right trigger 255
    #   NEUTRAL : hold W  (accelerate)    + right trigger 255  (always gas)
    #   OPEN    : hold S  (brake/reverse) + left  trigger 255
    TARGET_ACCEL = 255
    TARGET_BRAKE = 255

    if current_gesture == 'OPEN':
        target_trigger = -TARGET_BRAKE
        set_key('w', False)
        set_key('s', True)
    else:
        # FIST or NEUTRAL → always accelerate
        target_trigger = TARGET_ACCEL
        set_key('w', True)
        set_key('s', False)

    # Smooth & send gamepad triggers (works for gamepad-compatible games)
    trigger_alpha = 0.25
    smooth_trigger = trigger_alpha * target_trigger + (1 - trigger_alpha) * smooth_trigger

    rt_val = int(max(0, smooth_trigger))
    lt_val = int(max(0, -smooth_trigger))

    gamepad.right_trigger(value=rt_val)
    gamepad.left_trigger(value=lt_val)
    gamepad.update()

    # ── Gesture HUD ───────────────────────────────────────────────────────────
    gest_colors = {
        'FIST':    (0,   255,  80),   # green  → accelerating
        'OPEN':    (0,   80,  255),   # blue -> braking
        'NEUTRAL': (0,   220,  80),   # green -> accelerating (same as FIST)
    }
    gest_labels = {
        'FIST':    '[FIST]  ACCELERATE',
        'OPEN':    '[OPEN]  BRAKE',
        'NEUTRAL': '[AUTO]  ACCELERATE',
    }
    gest_col = gest_colors[current_gesture]
    gest_lbl = gest_labels[current_gesture]

    cv2.putText(
        frame, gest_lbl,
        (10, 210),
        cv2.FONT_HERSHEY_DUPLEX, 0.65,
        gest_col, 1, cv2.LINE_AA
    )

    # Pedal bar (vertical, right side of frame)
    bar_top    = 60
    bar_bottom = h - 60
    bar_height = bar_bottom - bar_top
    bar_x      = w - 30

    # Background track
    cv2.rectangle(
        frame,
        (bar_x - 10, bar_top),
        (bar_x + 10, bar_bottom),
        (40, 40, 40), -1
    )
    cv2.rectangle(
        frame,
        (bar_x - 10, bar_top),
        (bar_x + 10, bar_bottom),
        (120, 120, 120), 1
    )

    # Throttle fill (green, grows upward from centre)
    if rt_val > 0:
        fill_h = int((rt_val / 255.0) * (bar_height // 2))
        cv2.rectangle(
            frame,
            (bar_x - 8, bar_top + bar_height // 2 - fill_h),
            (bar_x + 8, bar_top + bar_height // 2),
            (0, 220, 80), -1
        )

    # Brake fill (blue, grows downward from centre)
    if lt_val > 0:
        fill_h = int((lt_val / 255.0) * (bar_height // 2))
        cv2.rectangle(
            frame,
            (bar_x - 8, bar_top + bar_height // 2),
            (bar_x + 8, bar_top + bar_height // 2 + fill_h),
            (0, 80, 220), -1
        )

    # Labels
    cv2.putText(frame, 'GAS',   (bar_x - 14, bar_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 80), 1)
    cv2.putText(frame, 'BRK',   (bar_x - 14, bar_bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 220), 1)


    # ── Draw virtual steering wheel overlay ───────────────────────────────────
    # Position: bottom-right corner, sized ~18 % of frame width
    wheel_radius = max(55, int(w * 0.13))
    wheel_cx = w - wheel_radius - 20
    wheel_cy = h - wheel_radius - 20

    # Determine direction for colouring even when both hands aren't detected
    if left_wrist is not None and right_wrist is not None:
        disp_direction = direction          # set during steering block
        disp_analog    = analog_x
        disp_rotation  = -(smooth_angle - (neutral_angle if neutral_angle else smooth_angle))
    else:
        disp_direction = "STRAIGHT"
        disp_analog    = 0.0
        disp_rotation  = 0.0

    draw_steering_wheel(
        frame,
        center=(wheel_cx, wheel_cy),
        radius=wheel_radius,
        rotation_deg=disp_rotation,
        analog_x=disp_analog,
        direction=disp_direction,
    )

    cv2.imshow(
        "Virtual Steering Wheel",
        frame
    )

    cv2.resizeWindow(
        "Virtual Steering Wheel",
        640,
        480
    )


    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# Cleanup – release keys, centre stick, close everything

set_key('w', False)
set_key('s', False)

gamepad.right_trigger(value=0)
gamepad.left_trigger(value=0)
gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
gamepad.update()

cap.release()
cv2.destroyAllWindows()

landmarker.close()