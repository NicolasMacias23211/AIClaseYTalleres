import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

datos, metadatos = tfds.load('emnist/byclass', as_supervised=True, with_info=True)

datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255  # Escalar los valores de píxeles a rango [0, 1]
    return imagenes, etiquetas

datos_entrenamiento = datos_entrenamiento.map(normalizar).cache().shuffle(10000).batch(32)
datos_pruebas = datos_pruebas.map(normalizar).batch(32)

for imagen, etiqueta in datos_entrenamiento.take(1):
    plt.imshow(imagen[0].numpy().reshape((28, 28)), cmap=plt.cm.binary)
    plt.title(f"Etiqueta: {etiqueta[0].numpy()}")
    plt.show()

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # Pasar de 28x28 a 1D
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(62, activation='softmax')  # 62 clases en total
])

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entrenar el modelo
historial = modelo.fit(datos_entrenamiento,epochs=3,validation_data=datos_pruebas)

test_loss, test_accuracy = modelo.evaluate(datos_pruebas)
print(f"\nPrecisión en el conjunto de prueba: {test_accuracy:.2f}")

# Función para mostrar predicciones de caracteres
def mostrar_predicciones(imagen, etiqueta_real):
    imagen = np.array([imagen])  # Agregar batch dimension
    prediccion = modelo.predict(imagen)
    etiqueta_predicha = np.argmax(prediccion[0])

    plt.imshow(imagen[0].reshape((28, 28)), cmap=plt.cm.binary)
    plt.title(f"Predicción: {etiqueta_predicha} / Real: {etiqueta_real}")
    plt.show()

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagen = imagenes_prueba[0]
    etiqueta_real = etiquetas_prueba[0]
    mostrar_predicciones(imagen, etiqueta_real)
