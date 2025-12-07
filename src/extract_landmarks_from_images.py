import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# Load YOLO pose model
model = YOLO("yolov8n-pose.pt")  # or yolov8s-pose for higher accuracy

INPUT_DIR = "online_dataset"
OUTPUT_DIR = "dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_label(label_folder):
    input_path = os.path.join(INPUT_DIR, label_folder)
    output_path = os.path.join(OUTPUT_DIR, label_folder)
    os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(input_path)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(images, desc=f"Processing {label_folder}"):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model(img, verbose=False)[0]

        # Skip if no hands detected
        if results.keypoints is None or len(results.keypoints) == 0:
            continue

        # Get first detected hand landmarks
        kpts = results.keypoints[0].xy.cpu().numpy().flatten()

        # Save CSV
        df = pd.DataFrame([kpts])
        csv_name = img_name.replace(".jpg", ".csv") \
                           .replace(".png", ".csv") \
                           .replace(".jpeg", ".csv")
        df.to_csv(os.path.join(output_path, csv_name), index=False)


if __name__ == "__main__":
    label_folders = os.listdir(INPUT_DIR)

    for label in label_folders:
        path = os.path.join(INPUT_DIR, label)
        if os.path.isdir(path):
            process_label(label)
