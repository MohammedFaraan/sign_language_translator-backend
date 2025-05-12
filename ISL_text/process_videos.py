import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Directories
DATA_VIDEO_DIR = os.path.join("data", "videos")
OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allowed video extensions
allowed_extensions = [".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv"]

# Config
SEQUENCE_LENGTH = 30   # frames per sequence
FEATURE_DIM = 126      # 42 landmarks × 3 coords

# Iterate over each label folder
for label in os.listdir(DATA_VIDEO_DIR):
    label_dir = os.path.join(DATA_VIDEO_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    print(f"Processing videos for sign '{label}'...")

    sequence = []  # to collect a sequence of frames
    all_sequences = []  # to collect all sequences

    video_files = [
        f for f in os.listdir(label_dir)
        if any(f.lower().endswith(ext) for ext in allowed_extensions)
    ]
    if not video_files:
        print(f"No video files found for sign '{label}'. Skipping...")
        continue

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for video_file in video_files:
            video_path = os.path.join(label_dir, video_file)
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)

            frame_skip = 2  # Skip every 2 frames to avoid near-duplicate frames
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                frame = cv2.resize(frame, (640, 480))
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for point in hand_landmarks.landmark:
                            landmarks.extend([point.x, point.y, point.z])

                if len(landmarks) > FEATURE_DIM:
                    landmarks = landmarks[:FEATURE_DIM]
                while len(landmarks) < FEATURE_DIM:
                    landmarks.append(0.0)

                if len(landmarks) == FEATURE_DIM:
                    sequence.append(landmarks)

                    # Once SEQUENCE_LENGTH frames are collected → save sequence
                    if len(sequence) == SEQUENCE_LENGTH:
                        seq_array = np.array(sequence)
                        seq_flat = seq_array.flatten()  # Flatten (30 frames × 126) → (3780,)
                        all_sequences.append(seq_flat)
                        sequence = []  # Reset for next sequence

            cap.release()

    if all_sequences:
        df = pd.DataFrame(all_sequences)
        df["label"] = label
        output_file = os.path.join(OUTPUT_DIR, f"{label}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved CSV for '{label}' at {output_file}")
    else:
        print(f"No valid sequences for sign '{label}'.")