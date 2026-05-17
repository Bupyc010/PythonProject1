import cv2
import os
import numpy as np

IMG_SIZE = 64
CLASSES = ["rectangle", "diamond", "ellipse", "arrow"]

def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return gray

def draw_debug_boxes(image, results, output_path="debug/found_shapes.png"):
    debug_img = image.copy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, r in enumerate(results):
        x1, y1, x2, y2 = r["bbox"]
        label = r.get("label", "unknown")
        conf = r.get("confidence", 0.0)

        color = (0, 255, 0)
        if label == "arrow":
            color = (255, 0, 0)
        elif label == "diamond":
            color = (0, 255, 255)
        elif label == "ellipse":
            color = (255, 255, 0)
        elif label == "rectangle":
            color = (0, 255, 0)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

        text = f"{i}: {label} {conf:.2f}"
        y_text = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            debug_img,
            text,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, debug_img)
    return debug_img

def extract_candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    h, w = image.shape[:2]

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 20 or bh < 20:
            continue
        if bw > 0.95 * w and bh > 0.95 * h:
            continue

        pad = 8
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + bw + pad, w)
        y2 = min(y + bh + pad, h)

        crop = image[y1:y2, x1:x2]
        candidates.append({
            "bbox": (x1, y1, x2, y2),
            "crop": crop
        })

    candidates = sorted(candidates, key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return candidates

def save_debug_contours(image, candidates, output_path="debug/candidates.png"):
    debug_img = image.copy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = c["bbox"]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            debug_img,
            f"cand {i}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, debug_img)
    return debug_img