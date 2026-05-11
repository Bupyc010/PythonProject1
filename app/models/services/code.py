import cv2
import os

from .predict import process_image
from .ocr_utils import read_text
from .geometry_utils import find_connections
from .codegen import generate_smart_code


def cod(image_path):
    # Проверка, существует ли файл
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не найден.")
        return

    image, results = process_image(image_path)

    for r in results:
        if r["label"] != "arrow":
            x1, y1, x2, y2 = r["bbox"]
            # Обрезка с проверкой границ
            crop = image[max(0, y1-5):min(image.shape[0], y2+5),
                         max(0, x1-5):min(image.shape[1], x2+5)]
            r["text"] = read_text(crop)
        else:
            r["text"] = ""

    connections = find_connections(results)
    final_code = generate_smart_code(results, connections)
    print(final_code)

    return str(final_code)