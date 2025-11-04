import tensorflow as tf

src = r"D:\UVT\UVT\Cambridge\Scripts\Resnet\Resnet Model\2025-09-04_RESNET_100px_72cls__100ep_0.7098901271820068acc-bw.h5"
dst = r"D:\UVT\UVT\Cambridge\Scripts\Resnet\Resnet Model\2025-09-04_RESNET_100px_72cls__100ep_0.7098901271820068acc-bw.keras"

# Key fix: compile=False avoids deserializing legacy losses/metrics (e.g., reduction='auto')
model = tf.keras.models.load_model(src, compile=False)
model.save(dst)
print("Converted to:", dst)



# ============================================================
# ResNet50V2 inference for 1-channel (NCHW) images in folders
# - Expected input: (batch, 1, 100, 100), values ~ 0..255 (float32)
# - Classes: 72
# - Loads a modern .keras full model (no .h5, no h5py)
# - Writes CSV to ./test_results/
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras

# ------------- CONFIG (edit these) ----------------
NUMBER_OF_PIXELS  = 100
NUMBER_OF_CLASSES = 72

# One or more folders with PNGs
FOLDERS = [r"C:\Users\andre\Desktop\stage31_binary2\stage31_binary2"]

# Full saved model in Keras v3 format
KERAS_MODEL = r"D:\UVT\UVT\Cambridge\Scripts\Resnet\Resnet Model\2025-09-04_RESNET_100px_72cls__100ep_0.7098901271820068acc-bw.keras"

BATCH_SIZE = 128
OUTPUT_DIR = Path("test_results")
# --------------------------------------------------

# Make Keras use channels_first to match (1, H, W)
keras.backend.set_image_data_format("channels_first")

def list_pngs(folder_list):
    """Return [(folder_path, filename), ...] for all PNG files in the folders."""
    if isinstance(folder_list, (str, Path)):
        folder_list = [folder_list]
    files = []
    for d in folder_list:
        dpath = Path(d)
        if not dpath.is_dir():
            print(f"[warn] Skipping non-folder: {dpath}")
            continue
        for f in os.listdir(dpath):
            p = dpath / f
            if p.is_file() and f.lower().endswith(".png"):
                files.append((dpath, f))
    return files

def load_image_gray_NCHW(folder: Path, fname: str, N: int) -> np.ndarray:
    """Read PNG as grayscale, resize to NxN, return array of shape (1, N, N)."""
    p = folder / fname
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # HxW
    if img is None:
        raise FileNotFoundError(str(p))
    img = cv2.resize(img, (N, N), interpolation=cv2.INTER_AREA)  # HxW
    return img[np.newaxis, :, :]  # (1, N, N)

def get_test_data(folder_list, N: int):
    pairs = list_pngs(folder_list)
    X = np.empty((len(pairs), 1, N, N), dtype=np.uint8)  # NCHW
    names = []
    for i, (d, f) in enumerate(pairs):
        X[i] = load_image_gray_NCHW(d, f, N)
        names.append(f)
    return X, names

def load_model_keras(keras_path: str):
    """Load a full .keras model (architecture + weights)."""
    kp = Path(keras_path)
    if not kp.is_file():
        raise FileNotFoundError(f".keras model not found at: {kp}")
    print("Loading full model (.keras):", kp)
    # compile=False avoids legacy compile configs from older training setups
    return keras.models.load_model(str(kp), compile=False)

def test_model(folder_test_list, keras_model_path, N: int, n_classes: int, batch_size: int = 128):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model_keras(keras_model_path)

    # Collect images
    print("Collecting images...")
    X_u8, file_list = get_test_data(folder_test_list, N)
    if len(X_u8) == 0:
        raise RuntimeError("No PNG files found in the provided folders.")

    # Match the original pipeline: feed 0..255 (float32), channels_first
    X = X_u8.astype(np.float32)  # no /255.0

    # Predict
    print(f"Running prediction on {len(X)} images...")
    preds = model.predict(X, batch_size=batch_size, verbose=1)
    labels = np.argmax(preds, axis=1)

    # Save CSV
    model_tag = Path(keras_model_path).stem
    out_csv = OUTPUT_DIR / f"classes_{n_classes}_prediction_{model_tag}.csv"
    pd.DataFrame({"filename": file_list, "prediction": labels}).to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return out_csv

# ------------------ RUN ------------------
print("Folders:")
for d in FOLDERS:
    print("  -", d, "| exists:", Path(d).exists())

out_csv = test_model(FOLDERS, KERAS_MODEL, NUMBER_OF_PIXELS, NUMBER_OF_CLASSES, batch_size=BATCH_SIZE)
out_csv
