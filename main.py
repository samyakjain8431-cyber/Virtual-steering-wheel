import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import pydirectinput


#not needed in this project as of now, but can be used for future reference
HAND_CONNECTIONS=[
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]
#path of model & add this hand_landmarker.task file in the same directory as this script
MODEL_PATH ="hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)
neutral_angle = None
smooth_angle = 0
current_direction = "STRAIGHT" #official documents of mediapipe hand 

cap = cv2.VideoCapture(0) #0 is for primary webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1) #to get mirror image
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    #frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=5)

    h, w, _ = frame.shape #to define h , w , $ 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert BGR to RGB for mediapipe
    #CONVERT THE FRAME TO MEDIAPIPE IMAGE
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    
    left_wrist = None
    right_wrist = None

    if result.hand_landmarks:
        for hand_index, hand in enumerate(result.hand_landmarks):
            # Thumb landmarks
            thumb_tip = hand[4]
            thumb_ip = hand[3]
            thumb_mcp = hand[2]

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)
            #cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 0, 255), -1)    
                            
            p0 = hand[0]     # Wrist
            p5 = hand[5]     # Index MCP
            p17 = hand[17]   # Pinky MCP

            x = int((p0.x + p5.x + p17.x) / 3 * w)
            y = int((p0.y + p5.y + p17.y) / 3 * h)

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            hand_name = result.handedness[hand_index][0].category_name
            cv2.putText(frame, f"{hand_name} TipY:{thumb_tip.y:.3f} IPY:{thumb_ip.y:.3f}", (10, 150 if hand_name == "Left" else 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if hand_name == "Left":
                left_wrist = (x, y)
            else:
                right_wrist = (x, y)

    if left_wrist is not None and right_wrist is not None:
        cv2.line(frame, left_wrist, right_wrist, (255, 0, 0), 2)

        center_x = (left_wrist[0] + right_wrist[0]) // 2
        center_y = (left_wrist[1] + right_wrist[1]) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)     

        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        angle = math.degrees(math.atan2(dy, dx))
        alpha = 0.1
        smooth_angle = alpha * angle + (1 - alpha) * smooth_angle
        cv2.putText(frame, f"Angle: {smooth_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if neutral_angle is None:
            neutral_angle = smooth_angle

        steering_angle = smooth_angle - neutral_angle
        if steering_angle > 20:
            direction = "LEFT"
        elif steering_angle < -20:
            direction = "RIGHT"
        else:
            direction = "STRAIGHT"

        # STEERING CONTROL
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

        cv2.putText(frame, f"Steering Angle: {steering_angle:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Direction: {direction}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    cv2.resizeWindow("Hand Tracking", 640, 480)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


pydirectinput.keyUp("a")
pydirectinput.keyUp("d")
cap.release()
cv2.destroyAllWindows()
landmarker.close()