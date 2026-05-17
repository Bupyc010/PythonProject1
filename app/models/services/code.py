import os
import re
import cv2
from tensorflow.keras.models import load_model

from .predict import process_image
from .ocr_utils import read_text
from .geometry_utils import find_connections
from .codegen import generate_smart_code


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "flowchart_model.keras")


def clean_ocr_text(text):
    if not text:
        return ""

    text = text.replace("х", "x").replace("Х", "X")
    text = text.replace("у", "y").replace("У", "Y")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")

    text = re.sub(r"[^\w\s\(\)\[\]\{\}\+\-\*/%=<>,.:!'\"#&|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def draw_debug(image, results, connections, out_path="debug_result.png"):
    if out_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(base_dir, "debug_result.png")

    vis = image.copy()

    for el in results:
        x1, y1, x2, y2 = el["bbox"]
        label = el["label"]
        conf = el.get("confidence", 0.0)
        text = el.get("text", "")

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2
        )

        if text:
            cv2.putText(
                vis,
                text[:35],
                (x1, min(vis.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 0, 0),
                1
            )

    for src, dst in connections:
        a = results[src]["bbox"]
        b = results[dst]["bbox"]

        ax = (a[0] + a[2]) // 2
        ay = (a[1] + a[3]) // 2
        bx = (b[0] + b[2]) // 2
        by = (b[1] + b[3]) // 2

        cv2.line(vis, (ax, ay), (bx, by), (0, 0, 255), 2)

    cv2.imwrite(out_path, vis)

def cod(IMAGE_PATH):
    if not os.path.exists(IMAGE_PATH):
        print(f"Не найден файл изображения: {IMAGE_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Не найдена модель: {MODEL_PATH}")
        return

    print("1. Загрузка модели...")
    model = load_model(MODEL_PATH)

    print("2. Поиск и классификация блоков...")
    image, results = process_image(IMAGE_PATH, model)
    if image is None:
        print("Не удалось прочитать изображение.")
        return

    print(f"Найдено блоков: {len(results)}")

    print("3. OCR текста...")
    for r in results:
        x1, y1, x2, y2 = r["bbox"]

        pad = 8
        xx1 = max(0, x1 + pad)
        yy1 = max(0, y1 + pad)
        xx2 = min(image.shape[1], x2 - pad)
        yy2 = min(image.shape[0], y2 - pad)

        if xx2 > xx1 and yy2 > yy1:
            crop = image[yy1:yy2, xx1:xx2]
            txt = read_text(crop)
            r["text"] = clean_ocr_text(txt)
        else:
            r["text"] = ""

    print("4. Поиск связей...")
    connections = find_connections(results)

    print("5. Генерация кода...")
    final_code = generate_smart_code(results, connections)

    draw_debug(image, results, connections)
    print(final_code)

    return str(final_code)