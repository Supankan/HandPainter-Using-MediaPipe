import mediapipe as mp
import cv2
import math
import time
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open the webcam.")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            # time.sleep(1)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = frame.shape

            # Get the coordinates of the index fingertip.
            xi = int(index_finger_tip.x * w)
            yi = int(index_finger_tip.y * h)

            # Get the coordinates of the thumb tip.
            xt = int(thumb_tip.x * w)
            yt = int(thumb_tip.y * h)

            print(f"Finger Tip Coordinates: {xi}, {yi}")
            print(f"Thumb Tip Coordinates: {xt}, {yt}")
            distance_index_thumb = (xt - xi) * (xt - xi) + (yt - yi) * (yt - yi)
            print(f"Distance = {distance_index_thumb}")
            print("----------")

            is_mouse_down = False
            # screen_width = pyautogui.size().width
            pyautogui.moveTo(xi, yi)

            if distance_index_thumb < 700:
                if not is_mouse_down:
                    pyautogui.mouseDown()
                    is_mouse_down = True
            else:
                if is_mouse_down:
                    pyautogui.mouseUp()
                    is_mouse_down = False

            '''
            for ids, landmark in enumerate(hand_landmarks.landmark):
                print(ids, landmark)
                cx, cy = landmark.x * frame_width, landmark.y * frame_height
                print(cx, cy)
            '''

    cv2.imshow('Webcam Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

