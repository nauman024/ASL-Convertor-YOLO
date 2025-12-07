import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

DATASET_DIR = "dataset"

X = []
y = []

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(label_path, file))
            X.append(df.values.flatten())
            y.append(label)

X = np.array(X)
y = np.array(y)

# Label encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)

np.save("X.npy", X)
np.save("y.npy", y_enc)

joblib.dump(le, "exported_models/label_encoder.joblib")

print("Dataset prepared successfully!")
