import tensorflow as tf

# Укажите полный путь к вашей модели
MODEL_PATH = r'C:\Users\1\PycharmProjects\keras\ai_emotion_model_fer2013_5emotions_1_00_50echoes.keras'

# Загружаем модель
model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Готово! Размер: {len(tflite_model) / 1024:.1f} КБ")