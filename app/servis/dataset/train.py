import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Устанавливаем новый размер 224x224
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Настройка генератора (аугментация добавлена для лучшего распознавания стрелок)
train_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,  # небольшие повороты помогут различать криво нарисованные стрелки
    zoom_range=0.1,
    validation_split=0.2  # резервируем 20% данных под валидацию
)

# 2. Загрузка данных
train_data = train_gen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Количество классов теперь рассчитается автоматически исходя из количества папок
num_classes = train_data.num_classes

# 3. Обновленная структура модели под 224x224
model = models.Sequential([
    # Входной слой теперь принимает 224x224
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Добавим еще один слой, так как картинка 224x224 больше и требует более глубокого анализа
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),  # Увеличили количество нейронов
    layers.Dropout(0.5),  # Увеличили Dropout для защиты от переобучения
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Обучение
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15  # Увеличили количество эпох, так как модель стала сложнее
)

# 5. Сохранение и проверка классов
model.save("flowchart_model_v2.keras")

print("\nОбучение завершено!")
print("Финальный список классов и их индексов:")
# Это выведет: {'background':0, 'diamond':1, 'down_arrow':2, 'ellipse':3, ...}
print(train_data.class_indices)

# Создаем список имен для удобства использования в будущем
class_names = list(train_data.class_indices.keys())
print("Список имен классов:", class_names)