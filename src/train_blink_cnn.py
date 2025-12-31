import cv2, os, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

X, y = [], []

for label, folder in enumerate(["open","closed"]):
    path = f"data/blink/{folder}"
    for file in os.listdir(path):
        img = cv2.imread(f"{path}/{file}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(64,64)) / 255.0
        X.append(img)
        y.append(label)

X = np.array(X).reshape(-1,64,64,1)
y = np.array(y)

model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(64,64,1)),
    MaxPooling2D(),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y,epochs=10,batch_size=16)

model.save("models/blink_model.keras")
