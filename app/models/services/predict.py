import cv2
import numpy as np
from .labels import LABELS


def classify_shape_with_keras(crop, model):
    if crop is None or crop.size == 0:
        return "background", 0.0

    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
    else:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    crop = cv2.resize(crop, (64, 64))
    crop = crop.astype("float32") / 255.0
    inp = np.expand_dims(crop, axis=0)

    preds = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    if isinstance(LABELS, dict):
        label = LABELS.get(idx, "unknown")
    else:
        label = LABELS[idx] if idx < len(LABELS) else "unknown"

    return label, conf


def process_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    elements = []
    if hierarchy is None:
        return img, []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 500:
            continue

        if w > img.shape[1] * 0.9:
            continue

        crop = img[y:y + h, x:x + w]
        label, conf = classify_shape_with_keras(crop, model)

        if label == "background":
            continue

        elements.append({
            "id": i,
            "label": label,
            "bbox": (x, y, x + w, y + h),
            "center": (x + w // 2, y + h // 2),
            "confidence": conf
        })

    elements = sorted(elements, key=lambda x: x["bbox"][1])
    return img, elements