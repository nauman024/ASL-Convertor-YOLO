import os
import numpy as np
from ultralytics import YOLO
import joblib


# ------------------------------------------------------
# 1. Make sure a directory exists
# ------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------
# 2. Load YOLOv8 pose model
# ------------------------------------------------------
def load_yolo_pose(model_path="yolov8n-pose.pt"):
    """
    Loads YOLOv8 pose/hand model.
    Default uses yolov8n-pose.pt (small & fast).
    """
    return YOLO(model_path)


# ------------------------------------------------------
# 3. Extract 21 hand landmarks from YOLO results
# ------------------------------------------------------
def extract_hand_keypoints(results):
    """
    Extracts 21 (x, y) keypoints from YOLO pose output.
    Returns None if no hand detected.
    """

    if len(results) == 0:
        return None

    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    # YOLO gives keypoints in shape: (1, 21, 3) -> x,y,confidence
    kps = result.keypoints.xy[0].cpu().numpy()

    if kps.shape[0] != 21:
        return None

    # Only take (x, y)
    return kps[:, :2]


# ------------------------------------------------------
# 4. Flatten landmark array (21x2 -> 42)
# ------------------------------------------------------
def flatten_landmarks(landmarks):
    """
    Converts (21,2) landmark array into a 42-length flat vector.
    """
    return landmarks.flatten()


# ------------------------------------------------------
# 5. Normalize landmark coordinates
# ------------------------------------------------------
def normalize_landmarks(landmarks):
    """
    Normalizes keypoints so the scale and position do not matter.
    This helps classification generalize better.
    """

    # center to wrist (landmark 0)
    wrist = landmarks[0]
    centered = landmarks - wrist

    # scale based on max distance
    max_val = np.max(np.abs(centered))
    if max_val != 0:
        centered = centered / max_val

    return centered


# ------------------------------------------------------
# 6. Save & Load joblib files (classifier, scaler, label encoder)
# ------------------------------------------------------
def save_joblib(obj, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_joblib(path):
    return joblib.load(path)


# ------------------------------------------------------
# 7. Letter mapping helpers
# ------------------------------------------------------
def letter_from_index(idx, label_encoder):
    """Given classifier output index → return the actual letter."""
    return label_encoder.inverse_transform([idx])[0]


def index_from_letter(letter, label_encoder):
    """Convert ASL letter to numeric label."""
    return label_encoder.transform([letter])[0]
