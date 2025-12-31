import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

pyautogui.FAILSAFE = True

cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# ====== TUNING PARAMETERS ======
GAIN_X = 25
GAIN_Y = 18
DEAD_ZONE = 4            # pixels (VERY IMPORTANT)
MAX_STEP = 35            # max pixels per frame
BLINK_THRESH = 0.19
SMOOTHING = 6            # frames
CLICK_COOLDOWN = 0.6

dx_buf = deque(maxlen=SMOOTHING)
dy_buf = deque(maxlen=SMOOTHING)

last_click = 0

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A+B)/(2*C)

print("✅ Stable Eye Mouse Running | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        # Eye region
        left = int(lm[33].x * w)
        right = int(lm[133].x * w)
        top = int(lm[159].y * h)
        bottom = int(lm[145].y * h)

        ix = int(lm[468].x * w)
        iy = int(lm[468].y * h)

        cx = (left + right) // 2
        cy = (top + bottom) // 2

        dx = ix - cx
        dy = iy - cy

        # ===== DEAD ZONE =====
        if abs(dx) < DEAD_ZONE: dx = 0
        if abs(dy) < DEAD_ZONE: dy = 0

        dx_buf.append(dx)
        dy_buf.append(dy)

        sx = int(np.mean(dx_buf))
        sy = int(np.mean(dy_buf))

        mx = np.clip(sx * GAIN_X, -MAX_STEP, MAX_STEP)
        my = np.clip(sy * GAIN_Y, -MAX_STEP, MAX_STEP)

        if mx != 0 or my != 0:
            pyautogui.moveRel(mx, my, duration=0.01)

        # ===== BLINK CLICK =====
        eye = np.array([[lm[i].x*w, lm[i].y*h] for i in
                        [33,160,158,133,153,144]])

        if eye_aspect_ratio(eye) < BLINK_THRESH:
            if time.time() - last_click > CLICK_COOLDOWN:
                pyautogui.click()
                last_click = time.time()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
