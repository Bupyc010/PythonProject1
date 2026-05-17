import os
from PIL import Image

input_folder = "images"
output_folder = "resized"
target_size = (224, 224)

os.makedirs(output_folder, exist_ok=True)

supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)]

print(f"Найдено файлов: {len(files)}")

counter = 218

for filename in files:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"{counter}.png")

    try:
        with Image.open(input_path) as img:
            img = img.convert("RGBA")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(output_path, "PNG", optimize=True)

        print(f"Готово: {filename} -> {counter}.png")
        counter += 1

    except Exception as e:
        print(f"Ошибка при обработке {filename}: {e}")

print("Обработка завершена")