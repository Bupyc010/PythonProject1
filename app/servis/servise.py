import os
import json
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import math


# ---------------------------
# 0) Токенизатор
# ---------------------------
class CodeTokenizer:
    def __init__(self, max_len=64):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx = len(self.vocab)
        self.pad_id = 0
        self.max_len = max_len

    def fit_on_texts(self, texts):
        for text in texts:
            tokens = self._basic_tokenize(text)
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = self.idx
                    self.idx += 1

    def _basic_tokenize(self, text: str):
        if not isinstance(text, str): return []
        text = text.replace("\n", " <NEWLINE> ").replace("\t", " <TAB> ")
        return text.split()

    def encode(self, text):
        tokens = ["<START>"] + self._basic_tokenize(text) + ["<END>"]
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [self.pad_id] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int32)


# ----------------------------------------
# 1) Обработка изображений
# ----------------------------------------
def get_crop(img, box_norm, size=(64, 64)):
    ih, iw = img.shape[:2]
    x, y, w, h = box_norm
    x1, y1 = int(x * iw), int(y * ih)
    x2, y2 = int((x + w) * iw), int((y + h) * ih)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*size, 3), dtype=np.float32)

    h_c, w_c = crop.shape[:2]
    scale = min(size[0] / h_c, size[1] / w_c)
    nw, nh = int(w_c * scale), int(h_c * scale)
    crop_res = cv2.resize(crop, (nw, nh))

    pad_img = np.zeros((*size, 3), dtype=np.float32)
    dy, dx = (size[0] - nh) // 2, (size[1] - nw) // 2
    pad_img[dy:dy + nh, dx:dx + nw] = crop_res
    return pad_img / 255.0


def pad_to_square(img, target_size=224):
    h, w = img.shape[:2]
    side = max(h, w)
    pad_h, pad_w = (side - h) // 2, (side - w) // 2
    padded = cv2.copyMakeBorder(img, pad_h, side - h - pad_h, pad_w, side - w - pad_w,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (target_size, target_size)).astype(np.float32) / 255.0


def load_image_by_path(img_path, target_size=64):
    if not os.path.exists(img_path):
        return np.zeros((target_size, target_size, 3), dtype=np.float32)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return np.zeros((target_size, target_size, 3), dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return pad_to_square(img_rgb, target_size)


# ----------------------------------------
# 2) Загрузка данных из CSV
# ----------------------------------------
def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден!")

    df = pd.read_csv(csv_path)
    print(f"Загружен CSV: {len(df)} строк")

    # Исправляем пути - убираем лишние кавычки
    paths = []
    for p in df['image_path'].tolist():
        if pd.isna(p):
            paths.append("")
        else:
            # Убираем лишние кавычки и пробелы
            path_str = str(p).strip()
            if path_str.startswith('"') and path_str.endswith('"'):
                path_str = path_str[1:-1]
            if path_str.startswith('\\"') and path_str.endswith('\\"'):
                path_str = path_str[2:-2]
            paths.append(path_str)

    codes = []
    for c in df['code'].tolist():
        if pd.isna(c):
            codes.append("")
        else:
            code_str = str(c).strip()
            if code_str.startswith('"') and code_str.endswith('"'):
                code_str = code_str[1:-1]
            if code_str.startswith('\\"') and code_str.endswith('\\"'):
                code_str = code_str[2:-2]
            codes.append(code_str)

    print(f"Валидных путей к изображениям: {len([p for p in paths if p])}")
    print(f"Валидных кодов: {len([c for c in codes if c])}")
    print(f"Примеры исправленных путей: {paths[:3]}")
    print(f"Примеры исправленных кодов: {codes[:3]}")

    arrows_list = []
    for x in df['arrows']:
        if pd.isna(x) or x == "" or x is None:
            arrows_list.append([])
        else:
            try:
                arrows_list.append(json.loads(str(x)))
            except (json.JSONDecodeError, ValueError):
                arrows_list.append([])

    shapes_list = []
    for x in df['shapes']:
        if pd.isna(x) or x == "" or x is None:
            shapes_list.append([])
        else:
            try:
                shapes_list.append(json.loads(str(x)))
            except (json.JSONDecodeError, ValueError):
                shapes_list.append([])

    return paths, codes, arrows_list, shapes_list


def load_aux_csv(csv_path, base_dir):
    if not os.path.exists(csv_path):
        print(f"Предупреждение: {csv_path} не найден, используются заглушки")
        return {}, {}

    df = pd.read_csv(csv_path)
    img_to_path = {}
    img_to_hint = {}

    for _, row in df.iterrows():
        img_name = row['image_name']
        img_path = os.path.join(base_dir, row['image_path'])

        img_to_path[img_name] = img_path
        img_to_hint[img_name] = load_image_by_path(img_path, target_size=64)

    return img_to_path, img_to_hint


def create_dataset(paths, codes, arrows_list_json, shapes_list_json, tokenizer,
                   arrow_img_cache, shape_img_cache, batch_size=8):
    def generator_fn():
        for i, (p, c, arr_names, shp_names) in enumerate(zip(paths, codes, arrows_list_json, shapes_list_json)):
            if not p or not os.path.exists(p): continue
            try:
                img_bgr = cv2.imread(p)
                if img_bgr is None: continue

                main_img = pad_to_square(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 224)

                # Обработка стрелок
                arr_list = []
                if isinstance(arr_names, str):
                    arr_names_clean = arr_names.replace('"', '').replace('[', '').replace(']', '').strip()
                    arr_list = [name.strip() for name in arr_names_clean.split(',') if name.strip()]
                elif isinstance(arr_names, list):
                    arr_list = arr_names

                arr_hint = arrow_img_cache.get(arr_list[0], np.zeros((64, 64, 3), dtype=np.float32)) if arr_list else np.zeros((64, 64, 3), dtype=np.float32)

                # Обработка фигур
                shp_list = []
                if isinstance(shp_names, str):
                    shp_names_clean = shp_names.replace('"', '').replace('[', '').replace(']', '').strip()
                    shp_list = [name.strip() for name in shp_names_clean.split(',') if name.strip()]
                elif isinstance(shp_names, list):
                    shp_list = shp_names

                shp_hint = shape_img_cache.get(shp_list[0], np.zeros((64, 64, 3), dtype=np.float32)) if shp_list else np.zeros((64, 64, 3), dtype=np.float32)

                code_ids = tokenizer.encode(str(c))
                # code_ids имеет размер max_len (128)
                # code_ids_in = первые max_len-1 токенов
                # target = токены со 2-го до конца

                yield {
                    "image": main_img,
                    "arrow_hint": arr_hint,
                    "shape_hint": shp_hint,
                    "code_ids_in": code_ids[:-1]  # 127 токенов
                }, code_ids[1:]  # 127 токенов (цель)

            except Exception:
                continue

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            {
                "image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                "arrow_hint": tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                "shape_hint": tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
                "code_ids_in": tf.TensorSpec(shape=(127,), dtype=tf.int32),  # 128-1
            },
            tf.TensorSpec(shape=(127,), dtype=tf.int32)
        )
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------
# 3) Модель
# ----------------------------------------
class VisualEncoder(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv = keras.Sequential([
            layers.Conv2D(filters, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(filters * 2, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D()
        ])

    def call(self, x):
        return self.conv(x)


class FlowchartCoder(tf.keras.Model):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.main_enc = VisualEncoder(64)
        self.hint_enc = VisualEncoder(32)

        self.fusion = layers.Dense(512, activation="relu")
        self.embedding = layers.Embedding(vocab_size, 256, mask_zero=True)
        self.lstm = layers.LSTM(512, return_sequences=True)
        self.out = layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # Извлекаем данные из inputs
        img = inputs["image"]
        a_hint = inputs["arrow_hint"]
        s_hint = inputs["shape_hint"]
        c_in = inputs["code_ids_in"]

        # Обработка изображений
        f_main = self.main_enc(img)
        f_arr = self.hint_enc(a_hint)
        f_shp = self.hint_enc(s_hint)

        # Конкатенация признаков
        context = layers.Concatenate()([f_main, f_arr, f_shp])
        context = tf.expand_dims(self.fusion(context), 1)

        # Обработка последовательности кода
        emb = self.embedding(c_in)
        ctx_rep = tf.tile(context, [1, emb.shape[1], 1])
        x = layers.Concatenate()([emb, ctx_rep])

        # LSTM и выходной слой
        lstm_out = self.lstm(x)
        return self.out(lstm_out)


# ----------------------------------------
# 4) Запуск (с проверками)
# ----------------------------------------
def main():
    main_csv = "dataset.csv"
    arrow_images_csv = "arrows.csv"
    shape_images_csv = "shapes.csv"
    base_dir_for_crops = ""

    if not os.path.exists(main_csv):
        print(f"Ошибка: {main_csv} не найден!")
        return

    print("Загрузка данных...")
    paths, codes, arrows_list_json, shapes_list_json = load_data_from_csv(main_csv)

    _, arrow_img_cache = load_aux_csv(arrow_images_csv, base_dir_for_crops)
    _, shape_img_cache = load_aux_csv(shape_images_csv, base_dir_for_crops)

    tokenizer = CodeTokenizer(max_len=128)
    tokenizer.fit_on_texts([str(c) for c in codes if pd.notna(c) and str(c).strip()])

    if len(tokenizer.vocab) < 4:
        print("Ошибка: словарь слишком маленький!")
        return

    print(f"Словарь собран: {len(tokenizer.vocab)} токенов")

    print("Создание датасета...")
    dataset = create_dataset(
        paths, codes, arrows_list_json, shapes_list_json,
        tokenizer, arrow_img_cache, shape_img_cache, batch_size=2
    )

    # ПРОВЕРКА ДАТАСЕТА
    print("Проверка датасета...")
    try:
        for i, (x, y) in enumerate(dataset.take(1)):
            print(f"Батч {i + 1} OK: image_shape={x['image'].shape}, code_shape={y.shape}")
            print(f"Ключи в x: {list(x.keys())}")
        print("Датасет работает корректно!")
    except Exception as e:
        print(f"Ошибка в датасете: {e}")
        return

    # Подсчет примеров в датасете
    try:
        dataset_size = sum(1 for _ in dataset)
        print(f"Размер датасета: {dataset_size} батчей")
        steps_per_epoch = max(1, dataset_size)
    except:
        steps_per_epoch = 10
        print("Не удалось посчитать размер датасета, использую steps_per_epoch=10")

    model = FlowchartCoder(len(tokenizer.vocab), tokenizer.max_len)

    # Тестируем модель на одном примере
    print("Тестирование модели...")
    try:
        for x, y in dataset.take(1):
            output = model(x)
            print(f"Модель работает: output_shape={output.shape}")
            break
    except Exception as e:
        print(f"Ошибка в модели: {e}")
        return

    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("Начало обучения...")
    model.fit(dataset, epochs=20, steps_per_epoch=steps_per_epoch)
    model.save_weights("flowchart_model.weights.h5")
    print("Обучение завершено. Веса сохранены.")


if __name__ == "__main__":
    main()