import cv2
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