import pandas as pd

SCREEN_W = 1440   # change if needed
SCREEN_H = 900

df = pd.read_csv("data/labels.csv", header=None,
                 names=["img", "x", "y"])

df["x"] = df["x"] / SCREEN_W
df["y"] = df["y"] / SCREEN_H

df.to_csv("data/labels_norm.csv", index=False)

print("Normalized labels saved to data/labels_norm.csv")
