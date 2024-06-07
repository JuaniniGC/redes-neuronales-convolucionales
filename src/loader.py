from tensorflow.keras.models import load_model
import tensorflow as tf
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Directorio de datos
data_dir = 'data' 

# Parámetros
img_height = 150
img_width = 150
batch_size = 32

# Generador de datos de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # Usa el 20% de los datos para validación

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # Usa los datos de entrenamiento

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # Usa los datos de validación

# Cargar el modelo
model = load_model('model_1.keras')

# Verificar la arquitectura del modelo
model.summary()

loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')