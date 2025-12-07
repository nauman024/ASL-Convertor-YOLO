import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

X = np.load("X.npy")
y = np.load("y.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_scaled, y)

os.makedirs("exported_models", exist_ok=True)
joblib.dump(clf, "exported_models/classifier.joblib")
joblib.dump(scaler, "exported_models/scaler.joblib")

print("Model trained successfully!")
