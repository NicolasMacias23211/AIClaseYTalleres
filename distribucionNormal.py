import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf

# Devuelve 100 valores sobre el intervalo [0.001, 0.999]
x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)

# Calcula los valores de la función de densidad de probabilidad para x
y = norm.pdf(x)

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Modelo entrenado!")

predicciones = modelo.predict(celsius)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, '*', label='Distribución Normal')
plt.title('Distribución Normal (0,1)')
plt.ylabel('f(x)')
plt.xlabel('X')

plt.subplot(1, 2, 2)
plt.plot(celsius, fahrenheit, 'ro', label='Datos Reales')
plt.plot(celsius, predicciones, 'b*', label='Predicciones')
plt.title('Celsius a Fahrenheit')
plt.ylabel('Fahrenheit')
plt.xlabel('Celsius')
plt.legend()

plt.tight_layout()
plt.show()

nuevo_valor_celsius = 100.0
resultado = modelo.predict(np.array([nuevo_valor_celsius]))
print(f"El resultado para {nuevo_valor_celsius}°C es {resultado[0][0]} Fahrenheit!")