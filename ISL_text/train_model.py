import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─── Configuration ───────────────────────────────────────────────────────
DATASET_DIR = "dataset"
SEQUENCE_LENGTH = 30
FRAME_FEATURE_DIM = 126  # 42 landmarks × 3 coords
FLATTENED_SEQ_DIM = SEQUENCE_LENGTH * FRAME_FEATURE_DIM

# ─── Prepare dataset ──────────────────────────────────────────────────────
data = []
labels = []
label_map = {}
label_counter = 0

# Shuffle CSV files to avoid ordering bias
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
random.shuffle(files)

for file in files:
    df = pd.read_csv(os.path.join(DATASET_DIR, file))
    unique_labels = df["label"].unique()
    if len(unique_labels) == 0:
        continue
    label_str = unique_labels[0]

    # Assign or reuse label index
    if label_str not in label_map.values():
        label_map[label_counter] = label_str
        current_label = label_counter
        label_counter += 1
    else:
        current_label = [k for k, v in label_map.items() if v == label_str][0]

    # Extract flattened sequences
    features_flat = df.drop(columns=["label"]).values  # shape: (n_sequences, FLATTENED_SEQ_DIM)
    if features_flat.shape[1] != FLATTENED_SEQ_DIM:
        raise ValueError(f"Unexpected feature dimension in {file}: {features_flat.shape[1]} != {FLATTENED_SEQ_DIM}")

    # Reshape each row back to (SEQUENCE_LENGTH, FRAME_FEATURE_DIM)
    for seq_flat in features_flat:
        seq = seq_flat.reshape(SEQUENCE_LENGTH, FRAME_FEATURE_DIM)
        data.append(seq)
        labels.append(current_label)

# Convert to numpy arrays
X = np.array(data)  # shape: (n_samples, 30, 126)
y = np.array(labels)  # shape: (n_samples,)

# Debug print
print(f"Loaded sequences: X={X.shape}, y={y.shape}")

# Ensure X and y lengths match
if X.shape[0] != y.shape[0]:
    raise ValueError(f"Mismatch between X samples ({X.shape[0]}) and y labels ({y.shape[0]})")

# ─── Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
# Flatten frames for scaling
X_flat = X.reshape(-1, FRAME_FEATURE_DIM)  # shape: (n_samples * SEQUENCE_LENGTH, FRAME_FEATURE_DIM)
X_scaled_flat = scaler.fit_transform(X_flat)
# Reshape back
X = X_scaled_flat.reshape(-1, SEQUENCE_LENGTH, FRAME_FEATURE_DIM)

# ─── Train-Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test shape:  X_test={X_test.shape},  y_test={y_test.shape}")

# ─── Build LSTM Model ───────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQUENCE_LENGTH, FRAME_FEATURE_DIM)),
    LSTM(128, return_sequences=False, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(len(label_map), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ─── Train Model ───────────────────────────────────────────────────────────
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# ─── Save Model and Artifacts ───────────────────────────────────────────────
model.save("isl_model.h5")
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Model, label_map, and scaler saved.")
