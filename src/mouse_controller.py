import pyautogui
pyautogui.FAILSAFE = False

def move(x, y):
    pyautogui.moveTo(int(x), int(y), duration=0)
