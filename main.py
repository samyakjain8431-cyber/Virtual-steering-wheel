import cv2
import numpy as np

from config import GESTURE_HOLD_FRAMES
from tracker import create_landmarker, detect_hands
from controller import gamepad, set_key, release_all_keys, reset_gamepad
from gesture import classify_hand_gesture, majority_vote, update_gesture_state
from steering import compute_steering
from drawing import (
    draw_hand, draw_wrist, draw_hand_info,
    draw_steering_hud, draw_gesture_hud, draw_pedal_bar,
    draw_steering_wheel
)


# ── State ──────────────────────────────────────────────────────────────────────

neutral_angle   = None
smooth_angle    = 0.0
analog_x        = 0.0

gesture_counter = {'FIST': 0, 'OPEN': 0, 'NEUTRAL': 0}
current_gesture = 'NEUTRAL'
smooth_trigger  = 0.0

TARGET_ACCEL = 255
TARGET_BRAKE = 255
TRIGGER_ALPHA = 0.25


# ── Setup ──────────────────────────────────────────────────────────────────────

landmarker = create_landmarker()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()


# ── Main Loop ──────────────────────────────────────────────────────────────────

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = detect_hands(landmarker, rgb_frame)

    left_wrist    = None
    right_wrist   = None
    gesture_votes = []

    # ── Hand detection & drawing ───────────────────────────────────────────────
    if result.hand_landmarks:
        for hand_index, hand in enumerate(result.hand_landmarks):

            draw_hand(frame, hand, w, h)

            wrist_x = int(hand[0].x * w)
            wrist_y = int(hand[0].y * h)
            draw_wrist(frame, wrist_x, wrist_y)

            hand_name = result.handedness[hand_index][0].category_name
            draw_hand_info(frame, hand_name, hand[4], hand[3])

            if hand_name == "Left":
                left_wrist = (wrist_x, wrist_y)
            else:
                right_wrist = (wrist_x, wrist_y)

            gesture_votes.append(classify_hand_gesture(hand, w, h))

    # ── Steering ───────────────────────────────────────────────────────────────
    if left_wrist and right_wrist:
        cv2.line(frame, left_wrist, right_wrist, (255, 0, 0), 2)
        cv2.circle(frame,
                   ((left_wrist[0] + right_wrist[0]) // 2,
                    (left_wrist[1] + right_wrist[1]) // 2),
                   5, (0, 0, 255), -1)

        steer = compute_steering(
            left_wrist, right_wrist,
            neutral_angle, smooth_angle, analog_x
        )
        smooth_angle  = steer['smooth_angle']
        neutral_angle = steer['neutral_angle']
        analog_x      = steer['analog_x']
        direction     = steer['direction']
        steering_angle = steer['steering_angle']

        gamepad.left_joystick_float(x_value_float=analog_x, y_value_float=0.0)
        gamepad.update()

        draw_steering_hud(frame, smooth_angle, steering_angle, direction, analog_x, w, h)

        disp_direction = direction
        disp_analog    = analog_x
        disp_rotation  = -(smooth_angle - (neutral_angle or smooth_angle))
    else:
        analog_x = 0.0
        gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gamepad.update()
        disp_direction = "STRAIGHT"
        disp_analog    = 0.0
        disp_rotation  = 0.0

    # ── Gesture → controls ────────────────────────────────────────────────────
    frame_gesture = majority_vote(gesture_votes)
    gesture_counter, current_gesture = update_gesture_state(
        frame_gesture, gesture_counter, current_gesture
    )

    if current_gesture == 'OPEN':
        target_trigger = -TARGET_BRAKE
        set_key('w', False)
        set_key('s', True)
    else:
        target_trigger = TARGET_ACCEL
        set_key('w', True)
        set_key('s', False)

    smooth_trigger = TRIGGER_ALPHA * target_trigger + (1 - TRIGGER_ALPHA) * smooth_trigger
    rt_val = int(max(0,  smooth_trigger))
    lt_val = int(max(0, -smooth_trigger))

    gamepad.right_trigger(value=rt_val)
    gamepad.left_trigger(value=lt_val)
    gamepad.update()

    # ── Draw HUD ──────────────────────────────────────────────────────────────
    draw_gesture_hud(frame, current_gesture)
    draw_pedal_bar(frame, rt_val, lt_val, h, w)

    wheel_radius = max(55, int(w * 0.13))
    draw_steering_wheel(
        frame,
        center=(w - wheel_radius - 20, h - wheel_radius - 20),
        radius=wheel_radius,
        rotation_deg=disp_rotation,
        analog_x=disp_analog,
        direction=disp_direction,
    )

    cv2.imshow("Virtual Steering Wheel", frame)
    cv2.resizeWindow("Virtual Steering Wheel", 640, 480)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ── Cleanup ────────────────────────────────────────────────────────────────────

release_all_keys()
reset_gamepad()
cap.release()
cv2.destroyAllWindows()
landmarker.close()