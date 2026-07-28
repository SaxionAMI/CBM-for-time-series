import numpy as np
import pandas as pd

def get_window_groups(sensor_groups, idx):
    return [group[idx : idx + 1].copy() for group in sensor_groups]

def inverse_transform_concepts(concept_scaler, concept_values):
    return concept_scaler.inverse_transform(concept_values)

def _inverse_normalize(sensor_windows, sensor_scaler, cols):
    cols = list(cols)
    means = sensor_scaler.mean_[cols].reshape(1, 1, -1)
    stds = sensor_scaler.scale_[cols].reshape(1, 1, -1)
    raw = sensor_windows * stds + means
    return raw, means, stds

def _renormalize(raw_windows, means, stds):
    return (raw_windows - means) / stds

def predict_window(window_groups, concept_scaler, sensors_to_concepts, lr_model, LABEL_NAMES):
    concept_pred = sensors_to_concepts.predict(window_groups, verbose=0)
    concept_pred_ord = inverse_transform_concepts(
        concept_scaler, concept_pred
    )
    probs = lr_model.predict(concept_pred, verbose=0)
    return {
        "concept_pred_norm": concept_pred,
        "concept_pred_ord": concept_pred_ord,
        "probs": probs,
        "label": LABEL_NAMES[int(np.argmax(probs, axis=1)[0])],
        "confidence": float(np.max(probs, axis=1)[0]),
    }

def rotate_chest_acc(chest_acc_windows, sensor_scaler, theta_deg, chest_cols=(0, 1, 2)):
    raw, means, stds = _inverse_normalize(chest_acc_windows, sensor_scaler, chest_cols)
    theta = np.radians(theta_deg)
    rotated = raw.copy()
    rotated[:, :, 0] = raw[:, :, 0] * np.cos(theta) - raw[:, :, 2] * np.sin(theta)
    rotated[:, :, 2] = raw[:, :, 0] * np.sin(theta) + raw[:, :, 2] * np.cos(theta)
    return _renormalize(rotated, means, stds)


def scale_ankle_acc(
    ankle_acc_windows,
    sensor_scaler,
    scale_factor,
    ankle_cols=(5, 6, 7),
):
    raw, means, stds = _inverse_normalize(ankle_acc_windows, sensor_scaler, ankle_cols)
    signal_mean = raw.mean(axis=1, keepdims=True)
    raw_centered = raw - signal_mean
    scaled = signal_mean + raw_centered * scale_factor
    return _renormalize(scaled, means, stds)


def concept_delta_frame(baseline_ord, updated_pred, parameter_name, parameter_value, CONCEPT_NAMES):
    updated_ord = updated_pred["concept_pred_ord"][0]
    frame = pd.DataFrame(
        {
            "concept": CONCEPT_NAMES,
            "baseline": baseline_ord[0],
            "updated": updated_ord,
            "delta": updated_ord - baseline_ord[0],
            parameter_name: parameter_value,
        }
    )
    frame["label"] = updated_pred["label"]
    frame["confidence"] = updated_pred["confidence"]
    return frame


def find_windows_by_label(labels, label_name, LABEL_NAMES, limit=None):
    label_idx = LABEL_NAMES.index(label_name)
    matches = np.where(labels == label_idx)[0]
    if limit is None:
        return matches
    return matches[:limit]