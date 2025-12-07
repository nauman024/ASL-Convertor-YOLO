import cv2
import numpy as np
import joblib
from ultralytics import YOLO

clf = joblib.load("exported_models/classifier.joblib")
scaler = joblib.load("exported_models/scaler.joblib")
label_encoder = joblib.load("exported_models/label_encoder.joblib")

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame, verbose=False)[0]

    if results.keypoints is not None and len(results.keypoints):
        kpts = results.keypoints[0].xy.cpu().numpy().flatten()

        X = scaler.transform([kpts])
        pred = clf.predict(X)[0]
        label = label_encoder.inverse_transform([pred])[0]

        cv2.putText(frame, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ASL Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
