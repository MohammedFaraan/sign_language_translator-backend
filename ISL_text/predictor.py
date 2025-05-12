import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import os

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH      = "ISL_text/isl_model.h5"
LABEL_MAP_PATH  = "ISL_text/label_map.pkl"
SCALER_PATH     = "ISL_text/scaler.pkl"
THRESHOLD       = 0.7       # minimum confidence to accept prediction
SEQUENCE_LENGTH = 30        # number of frames per sequence
FEATURE_DIM     = 126       # 42 landmarks × 3 coords

# Check if model files exist
for file_path in [MODEL_PATH, LABEL_MAP_PATH, SCALER_PATH]:
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found!")

# ─── Load model, label_map, and scaler ────────────────────────────────────────
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Invert label_map
    idx_to_label = {int(k): v for k, v in (label_map.items() if isinstance(label_map, dict) else enumerate(label_map))}
    print(f"Model loaded successfully. Available labels: {len(idx_to_label)}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# ─── Mediapipe setup ─────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dictionary to store all detected signs
detected_signs = {}

def predict_isl_signs(video_source=0, show_video=True):
    """
    Predict ISL signs from a video source
    
    Args:
        video_source: Camera index (0) or path to video file
        show_video: Whether to display the video during processing
        
    Returns:
        Dictionary of detected signs and their counts
    """
    global detected_signs
    detected_signs = {}  # Reset detected signs
    
    # ─── Video capture setup ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_source)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {video_source}")
    
    print(f"Processing video from {video_source}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    sequence = []            # buffer for landmarks
    collecting = False       # flag for when hand(s) detected
    last_prediction = "Waiting..."
    last_confidence = 0.0
    last_time = time.time()
    frame_count = 0
    
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached")
                break

            # Process every 2nd frame for faster processing if file
            if isinstance(video_source, str) and frame_count % 2 != 0:
                frame_count += 1
                continue
                
            frame_count += 1
            
            # Print progress for files
            if isinstance(video_source, str) and total_frames > 0 and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames})")

            # FPS calculation
            current_time = time.time()
            fps_calculated = 1 / (current_time - last_time) if last_time else 0
            last_time = current_time

            try:
                # Process frame
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if show_video:
                    img_out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                else:
                    img_out = frame.copy() if frame is not None else None

                # Extract landmarks
                landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if show_video and img_out is not None:
                            mp_drawing.draw_landmarks(img_out, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                # Start or continue collecting when hand detected
                if landmarks:
                    # Pad/truncate
                    if len(landmarks) > FEATURE_DIM:
                        landmarks = landmarks[:FEATURE_DIM]
                    else:
                        landmarks += [0.0] * (FEATURE_DIM - len(landmarks))

                    if not collecting:
                        collecting = True
                        sequence = []

                    sequence.append(landmarks)
                    if show_video and img_out is not None:
                        cv2.putText(img_out, "Collecting...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # Stop collecting and predict when hand disappears
                elif collecting:
                    # Only predict if we have any collected frames
                    if sequence:
                        # Prepare fixed-length sequence
                        seq = np.array(sequence)
                        if len(seq) < SEQUENCE_LENGTH:
                            # Only predict if we have enough frames (at least 10)
                            if len(seq) >= 10:
                                pad = np.zeros((SEQUENCE_LENGTH - len(seq), FEATURE_DIM))
                                seq = np.vstack([seq, pad])
                            else:
                                collecting = False
                                sequence = []
                                continue
                        else:
                            seq = seq[:SEQUENCE_LENGTH]

                        try:
                            # Scale and predict
                            seq_scaled = scaler.transform(seq)
                            input_seq = np.expand_dims(seq_scaled, axis=0)
                            probs = model.predict(input_seq, verbose=0)[0]
                            best_idx = np.argmax(probs)
                            best_conf = probs[best_idx]
                            
                            if best_conf >= THRESHOLD:
                                last_prediction = idx_to_label[best_idx]
                                last_confidence = best_conf
                                
                                # Add detected sign to dictionary
                                if last_prediction != "Unknown":
                                    if last_prediction in detected_signs:
                                        detected_signs[last_prediction] += 1
                                    else:
                                        detected_signs[last_prediction] = 1
                                    print(f"Detected sign: {last_prediction} (confidence: {best_conf:.2f})")
                            else:
                                last_prediction = "Unknown"
                                last_confidence = best_conf
                        except Exception as e:
                            print(f"Error during prediction: {str(e)}")

                    collecting = False
                    sequence = []

                # Overlay prediction and FPS
                if show_video and img_out is not None:
                    cv2.putText(img_out, f"Sign: {last_prediction} ({last_confidence:.2f})", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(img_out, f"FPS: {fps_calculated:.1f}", (10, img_out.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

                    cv2.imshow("ISL Sign Prediction", img_out)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested stop (q key)")
                        break
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    
    print(f"Processing complete. Detected {len(detected_signs)} unique signs.")
    return detected_signs

# Run the program and get the detected signs
if __name__ == "__main__":
    signs_dict = predict_isl_signs()
    print("\nDetected Signs:", signs_dict)
    print("\nDetected Signs Summary:")
    for sign, count in signs_dict.items():
        print(f"{sign}: {count} times")
