import cv2
import numpy as np
import os
import tensorflow as tf

IMG_SIZE = (64, 64)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "flowchart_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_crop(crop):
    crop = cv2.resize(crop, IMG_SIZE)
    crop = crop.astype("float32") / 255.0
    return np.expand_dims(crop, axis=0)

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Здесь предполагается, что у вас уже есть простой поиск контуров
    # Ниже — примерный каркас
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    class_names = ["arrow", "background", "diamond", "ellipse", "hexagon", "rectangle"]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue

        crop = image[y:y+h, x:x+w]
        inp = preprocess_crop(crop)
        pred = model.predict(inp, verbose=0)[0]
        class_id = int(np.argmax(pred))
        label = class_names[class_id]
        conf = float(pred[class_id])

        results.append({
            "label": label,
            "confidence": conf,
            "bbox": [x, y, x+w, y+h]
        })

    return image, results