import cv2
import pyautogui
import time
import os

os.makedirs("data/raw", exist_ok=True)
labels = open("data/labels_norm.csv", "w")

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

print("LOOK at cursor. Press SPACE to capture. ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if key == 32:  # SPACE
        x, y = pyautogui.position()
        x_n = x / screen_w
        y_n = y / screen_h

        ts = time.time()
        path = f"data/raw/{ts}.png"
        cv2.imwrite(path, frame)

        labels.write(f"{path},{x_n},{y_n}\n")
        labels.flush()

        print("Saved:", path)

cap.release()
labels.close()
cv2.destroyAllWindows()
