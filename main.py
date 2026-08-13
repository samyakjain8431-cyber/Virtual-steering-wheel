import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
# pyrefly: ignore [missing-import]
import pydirectinput



# Hand Connections (MediaPipe Hand Skeleton)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]



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
current_direction = "STRAIGHT"



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


        
        # Determine Direction
        
        if steering_angle > 20:

            direction = "LEFT"

        elif steering_angle < -20:

            direction = "RIGHT"

        else:

            direction = "STRAIGHT"


        
        # STEERING KEYBOARD CONTROL
        
        if direction != current_direction:

            if direction == "LEFT":

                pydirectinput.keyDown("a")
                pydirectinput.keyUp("d")

            elif direction == "RIGHT":

                pydirectinput.keyDown("d")
                pydirectinput.keyUp("a")

            else:

                pydirectinput.keyUp("a")
                pydirectinput.keyUp("d")


            current_direction = direction


        
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



# Cleanup

pydirectinput.keyUp("a")
pydirectinput.keyUp("d")

cap.release()
cv2.destroyAllWindows()

landmarker.close()