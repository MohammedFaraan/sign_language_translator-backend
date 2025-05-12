# process_imgs.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
DATA_IMG_DIR = os.path.join("data", "images")
OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each label folder
for label in os.listdir(DATA_IMG_DIR):
    label_folder = os.path.join(DATA_IMG_DIR, label)
    if not os.path.isdir(label_folder):
        continue

    data = []
    print(f"Processing images for label: {label}")
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for point in hand_landmarks.landmark:
                        landmarks.extend([point.x, point.y, point.z])
            # Pad with zeros if less than 126 values (42 points x 3)
            while len(landmarks) < 126:
                landmarks.append(0.0)
            if len(landmarks) == 126:
                data.append(landmarks)
    if data:
        df = pd.DataFrame(data)
        df["label"] = label
        output_file = os.path.join(OUTPUT_DIR, f"{label}_images.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved CSV for label {label} at {output_file}")
