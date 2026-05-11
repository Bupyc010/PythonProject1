import cv2
import sys
import os
from predict import process_image
from ocr_utils import read_text
from geometry_utils import find_connections
from codegen import generate_smart_code

def main(image_path):
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

    with open("output_flowchart.txt", "w", encoding="utf-8") as f:
        f.write(final_code)

    print("Код успешно сгенерирован:")
    print(final_code)

if __name__ == "__main__":
    # Если аргумент передан — используем его,
    # иначе используем стандартное имя файла (замените 'image.png' на ваше)
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Укажите здесь путь к вашей картинке по умолчанию
        default_path = "26.png"
        print(f"Аргумент не передан. Использую файл по умолчанию: {default_path}")
        main(default_path)