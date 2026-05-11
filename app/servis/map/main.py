import os
from PIL import Image, ImageOps

input_folder = "images"
output_folder = "resized"

# Создаем папку, если её нет
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_64x64.png")

        try:
            with Image.open(input_path) as img:
                # 1. Изменяем размер точно до 64x64 (обрезая лишнее, если пропорции другие)
                img_64 = ImageOps.fit(img, (64, 64), method=Image.Resampling.LANCZOS)

                # 2. Переводим в адаптивную палитру (256 цветов) для сильного сжатия.
                # Это сохраняет прозрачность и сильно уменьшает размер файла.
                img_optimized = img_64.convert("P", palette=Image.ADAPTIVE, colors=32)

                # 3. Сохраняем с флагом оптимизации
                img_optimized.save(output_path, "PNG", optimize=True)

            print(f"Обработан: {filename}")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

print("\nВсе изображения успешно обработаны и сжаты!")