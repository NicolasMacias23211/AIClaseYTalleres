# Importar las librerías necesarias
import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Directorios de las imágenes (ajusta las rutas si es necesario)
train_dir = '/ruta/a/fruits-360/Training'
test_dir = '/ruta/a/fruits-360/Test'

# Parámetros del modelo
IMG_SIZE = (100, 100)  # Tamaño de las imágenes
BATCH_SIZE = 32  # Tamaño del lote

# Configurar generadores de datos con aumentos de datos
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalización
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Solo normalización para el set de prueba

# Cargar imágenes desde los directorios y aplicar aumentos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Visualizar algunas imágenes aumentadas
filas, columnas = 4, 8
fig, axes = plt.subplots(filas, columnas, figsize=(1.5*columnas, 2*filas))
for X_batch, Y_batch in train_generator:
    for i in range(filas * columnas):
        ax = axes[i // columnas, i % columnas]
        ax.imshow(X_batch[i])
        ax.set_title(f"Label: {np.argmax(Y_batch[i])}")
        ax.axis('off')
    break
plt.tight_layout()
plt.show()

# Definir el modelo de red neuronal convolucional
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Salida ajustada al número de clases
])

# Compilación del modelo
modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Entrenamiento del modelo
print("Entrenando modelo...")
epocas = 15  # Puedes ajustar el número de épocas
history = modelo.fit(
    train_generator,
    epochs=epocas,
    validation_data=test_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=test_generator.samples // BATCH_SIZE
)

print("Modelo entrenado!")
