import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import vgamepad as vg



# ── Analog steering constants ──────────────────────────────────────────────────

# Deadzone: steering angles smaller than this (°) are treated as straight
DEADZONE_DEG = 5

# Full-lock: steering angles beyond this (°) saturate to ±1.0 stick deflection
MAX_STEER_DEG = 40



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


    cv2.imshow(
        "Hand Tracking",
        frame
    )

    cv2.resizeWindow(
        "Hand Tracking",
        640,
        480
    )


    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# Cleanup – centre stick before exit

gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
gamepad.update()

cap.release()
cv2.destroyAllWindows()

landmarker.close()