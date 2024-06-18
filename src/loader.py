from tensorflow.keras.models import load_model
import tensorflow as tf
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Parámetros
img_height = 150
img_width = 150
batch_size = 32
val_dir = 'output/val'

# Generador de datos de validación
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Cargar el modelo
model = load_model('model_3.keras')

# Verificar la arquitectura del modelo
model.summary()

loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')