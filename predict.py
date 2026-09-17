import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import traceback
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "rice_vs_invalid_model.h5")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "rice_disease_model.h5")

CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot"
]

def make_tta_batch(base_img):
    versions = []
    versions.append(base_img)
    versions.append(cv2.flip(base_img, 1))

    bright = np.clip(base_img * 1.2, 0, 255)
    versions.append(bright)

    dark = np.clip(base_img * 0.8, 0, 255)
    versions.append(dark)

    h, w = base_img.shape[:2]
    ch, cw = int(h * 0.9), int(w * 0.9)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cropped = base_img[y0:y0 + ch, x0:x0 + cw]
    zoomed = cv2.resize(cropped, (w, h))
    versions.append(zoomed)

    batch = np.stack(versions, axis=0)
    return preprocess_input(batch)

def predict_disease_api(img_path):
    if not os.path.exists(img_path):
        return {"disease": "Invalid Image", "confidence": "0.0 %", "confidence_val": 0.0}

    try:
        img = cv2.imread(img_path)
        if img is None:
            return {"disease": "Invalid Image", "confidence": "0.0 %", "confidence_val": 0.0}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
        tta_batch = make_tta_batch(img_resized)

        # Stage 1: Binary Filter
        if os.path.exists(BINARY_MODEL_PATH):
            try:
                binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH, compile=False)
                binary_preds = binary_model.predict(tta_batch, verbose=0)[:, 0]
                binary_pred = float(np.mean(binary_preds))
                if binary_pred < 0.40:
                    invalid_confidence = float((1.0 - binary_pred) * 100.0)
                    return {
                        "disease": "Invalid Image",
                        "confidence": f"{round(invalid_confidence, 2)} %",
                        "confidence_val": float(round(invalid_confidence, 2))
                    }
            except Exception as e:
                print(f"Binary model error: {e}", file=sys.stderr)

        # Stage 2: Disease Classification
        if os.path.exists(DISEASE_MODEL_PATH):
            disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)
            tta_predictions = disease_model.predict(tta_batch, verbose=0)
            predictions = np.mean(tta_predictions, axis=0)

            per_version_top = np.argmax(tta_predictions, axis=1)
            agreement = float(np.mean(per_version_top == np.argmax(predictions)))

            sorted_indices = np.argsort(predictions)[::-1]
            top1_idx = int(sorted_indices[0])
            top2_idx = int(sorted_indices[1])

            top1_score = float(predictions[top1_idx] * 100.0)
            top2_score = float(predictions[top2_idx] * 100.0)
            margin = float(top1_score - top2_score)

            if top1_score < 35.0 or margin < 10.0 or agreement < 0.5:
                invalid_conf = float(100.0 - top1_score)
                return {
                    "disease": "Invalid Image",
                    "confidence": f"{round(invalid_conf, 2)} %",
                    "confidence_val": float(round(invalid_conf, 2))
                }

            predicted_disease = str(CLASS_NAMES[top1_idx])
            return {
                "disease": predicted_disease,
                "confidence": f"{round(top1_score, 2)} %",
                "confidence_val": float(round(top1_score, 2))
            }
        else:
            return {"disease": "Error", "confidence": "0.0 %", "confidence_val": 0.0, "details": "Model missing"}

    except Exception as e:
        print("---- ERROR IN PREDICT API ----", file=sys.stderr)
        traceback.print_exc()
        return {"disease": "Error", "confidence": "0.0 %", "confidence_val": 0.0, "details": str(e)}
