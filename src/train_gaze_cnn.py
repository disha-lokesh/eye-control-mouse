import pandas as pd
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load labels
df = pd.read_csv("data/labels_norm.csv", header=None, names=["img","x","y"])

X, y = [], []

for _, row in df.iterrows():
    img_path = row.img

    if not os.path.exists(img_path):
        print(f"❌ Missing file: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64))
    img = img / 255.0

    X.append(img)
    y.append([row.x, row.y])

X = np.array(X).reshape(-1, 64, 64, 1)
y = np.array(y)

print("Training samples:", len(X))
assert len(X) > 0, "❌ No training data loaded!"

# CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="sigmoid")   # 🔴 IMPORTANT
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X, y, epochs=25, batch_size=32)

os.makedirs("models", exist_ok=True)
model.save("models/gaze_model.keras")

print("✅ Model saved: models/gaze_model.keras")
