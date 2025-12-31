import cv2, os, time

cap = cv2.VideoCapture(0)
os.makedirs("data/blink/open", exist_ok=True)
os.makedirs("data/blink/closed", exist_ok=True)

print("Press O = eyes open | C = blink | ESC = exit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    eye = frame[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.6)]
    eye = cv2.resize(eye,(64,64))

    cv2.imshow("Eye", eye)
    key = cv2.waitKey(1)

    if key == ord('o'):
        cv2.imwrite(f"data/blink/open/{time.time()}.png", eye)
    elif key == ord('c'):
        cv2.imwrite(f"data/blink/closed/{time.time()}.png", eye)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
