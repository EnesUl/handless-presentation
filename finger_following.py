import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pyautogui
import time
from pathlib import Path
from urllib.request import urlretrieve

pyautogui.PAUSE = 0.0

MODEL_PATH = Path.home() / ".mediapipe" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(MODEL_URL, MODEL_PATH)

hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)

TIP_IDS = [4, 8, 12, 16, 20] #parmak ucu 
PIP_IDS = [2, 6, 10, 14, 18] # alt boğum

cap = cv2.VideoCapture(0)

# Eylem kilit süresi (debounce)
last_action_time = 0
COOLDOWN_SECONDS = 0.8
last_playback_gesture = None

def get_finger_status(landmarks, handedness):
    """ 1 fingers open , 0 fingers closed"""
    fingers = []
    
    is_right_hand = handedness[0].category_name == "Right"
    if is_right_hand:
        fingers.append(1 if landmarks[TIP_IDS[0]].x < landmarks[PIP_IDS[0]].x  else 0) # right hand is open if toe left than pıp
    else:
        fingers.append(1 if landmarks[PIP_IDS[0]].x > landmarks[TIP_IDS[0]].x  else 0)
        
    for i in range(1, 5):
        if landmarks[TIP_IDS[i]].y < landmarks[PIP_IDS[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers


while cap.isOpened():
    success , frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    
    results = hands.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    )
    
    action_text = "Waiting"
    
    #İf hand detected
    if results.hand_landmarks and results.handedness:
        hand_landmarks = results.hand_landmarks[0]
        handedness = results.handedness[0]

        for landmark in hand_landmarks:
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, (0, 255, 0), -1)
        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            cv2.line(
                frame,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (255, 0, 0),
                2,
            )
    
        fingers = get_finger_status(hand_landmarks, handedness)
        current_time = time.time()
        
        if current_time - last_action_time > COOLDOWN_SECONDS:
            
            if fingers == [0 , 1 , 0 , 0 ,0]:
                pyautogui.press("right")
                action_text = "next page "
                last_action_time = current_time
                
            elif fingers == [1 , 1 , 0 , 0 ,0 ]:
                pyautogui.press("left")
                action_text = "previous page"
                last_action_time = current_time
                
            elif fingers == [1 , 1 ,1 , 1 , 1 ]:
                if last_playback_gesture != "open":
                    pyautogui.press("playpause")
                    action_text = "video play/pause"
                    last_playback_gesture = "open"
                    last_action_time = current_time
                
            elif fingers == [0 , 0 ,0 , 0 , 0 ]:
                if last_playback_gesture != "closed":
                    pyautogui.press("stop")
                    action_text = "video stopped"
                    last_playback_gesture = "closed"
                    last_action_time = current_time
                  
    cv2.putText(frame, f"Aksiyon: {action_text}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand control panel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()               
cv2.destroyAllWindows()