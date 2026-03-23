import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Импортируем ваши классы (или вставьте их определения выше)
# Копируем CodeTokenizer, VisualEncoder и FlowchartCoder из вашего кода обучения
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

class Predictor:
    def __init__(self, weights_path, codes_for_vocab, max_len=128):
        # 1. Восстанавливаем токенизатор и его словарь
        self.tokenizer = CodeTokenizer(max_len=max_len)
        self.tokenizer.fit_on_texts([str(c) for c in codes_for_vocab])
        vocab_size = len(self.tokenizer.vocab)

        # 2. Создаем модель с той же архитектурой
        self.model = FlowchartCoder(vocab_size, max_len)

        # 3. Инициализируем веса (нужен "холостой" прогон для создания переменных)
        dummy_inputs = {
            "image": np.zeros((1, 224, 224, 3), dtype=np.float32),
            "arrow_hint": np.zeros((1, 64, 64, 3), dtype=np.float32),
            "shape_hint": np.zeros((1, 64, 64, 3), dtype=np.float32),
            "code_ids_in": np.zeros((1, 127), dtype=np.int32)
        }
        _ = self.model(dummy_inputs)

        # 4. Загружаем веса
        self.model.load_weights(weights_path)
        print("✅ Модель успешно загружена")

    def predict(self, img_path, arrow_img=None, shape_img=None):
        # Подготовка основного изображения
        if not os.path.exists(img_path):
            raise FileNotFoundError("Изображение не найдено")

        raw_img = cv2.imread(img_path)
        main_img = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (224, 224))
        main_img = np.expand_dims(main_img, 0)  # Добавляем размер батча

        # Подготовка хинтов (если нет, создаем пустые)
        a_hint = np.expand_dims(arrow_img if arrow_img is not None else np.zeros((64, 64, 3), dtype=np.float32), 0)
        s_hint = np.expand_dims(shape_img if shape_img is not None else np.zeros((64, 64, 3), dtype=np.float32), 0)

        # Генерация текста (Auto-regressive decoding)
        decoded_ids = [self.tokenizer.vocab["<START>"]]

        for _ in range(self.tokenizer.max_len - 1):
            # Подготавливаем текущую последовательность
            curr_in = decoded_ids + [self.tokenizer.pad_id] * (127 - len(decoded_ids))
            curr_in = np.array([curr_in[:127]], dtype=np.int32)

            # Прогон через модель
            preds = self.model({
                "image": main_img,
                "arrow_hint": a_hint,
                "shape_hint": s_hint,
                "code_ids_in": curr_in
            }, training=False)

            # Берем индекс самого вероятного токена для последнего шага
            last_token_idx = len(decoded_ids) - 1
            next_id = np.argmax(preds[0, last_token_idx, :])

            if next_id == self.tokenizer.vocab["<END>"]:
                break

            decoded_ids.append(next_id)

        # Декодируем ID обратно в текст
        id_to_word = {v: k for k, v in self.tokenizer.vocab.items()}
        result_tokens = [id_to_word.get(i, "<UNK>") for i in decoded_ids[1:]]

        return " ".join(result_tokens).replace("<NEWLINE>", "\n").replace("<TAB>", "\t")


# === ЗАПУСКА ===
if __name__ == "__main__":
    main_csv = "dataset.csv"
    df = pd.read_csv(main_csv)
    all_codes = df['code'].fillna("").tolist()

    predictor = Predictor("flowchart_model.weights.h5", all_codes)

    generated_code = predictor.predict("test.png")
    print("\n--- СГЕНЕРИРОВАННЫЙ КОД ---")
    print(generated_code)