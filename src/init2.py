import tensorflow as tf
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Directorio de datos originales
data_dir = 'data'  # Ajusta esta ruta según tu estructura en Google Drive

# Directorios de salida
output_dir = 'output'  # Ajusta esta ruta según tu estructura en Google Drive

# División de datos
splitfolders.ratio(data_dir, output=output_dir, seed=1337, ratio=(.8, .15, .05)) 

# Parámetros
img_height = 150
img_width = 150
batch_size = 32

# Generador de datos de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza los valores de los píxeles
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_val_datagen = ImageDataGenerator(rescale=1./255)  # Solo normalización

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_val_datagen.flow_from_directory(
    os.path.join(output_dir, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_val_datagen.flow_from_directory(
    os.path.join(output_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento

epochs = 20

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

loss, accuracy = model.evaluate(test_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Guardar el modelo
model.save('modelo_entrenado.')
