import pandas as pd
import matplotlib.pyplot as plt

data = {
    'x1': [0, 2, 2.5, 1, 4, 7],
    'x2': [0, 1, 2, 3, 6, 2],
    'y': [5, 10, 9, 0, 3, 27]
}
df = pd.DataFrame(data)

df['x1^2'] = df['x1'] ** 2
df['x2^2'] = df['x2'] ** 2
df['x1*x2'] = df['x1'] * df['x2']
df['x1*y'] = df['x1'] * df['y']
df['x2*y'] = df['x2'] * df['y']

totals = df.sum()

print(df)
print("\nSuma de cada columna:")
print(totals)

a0, a1, a2 = 4.483, 2.439, 2.561

x1_new = 2.5
x2_new = 2

y_pred = a0 + a1 * x1_new + a2 * x2_new
print(f"Predicción para x1 = {x1_new} y x2 = {x2_new} es: y ≈ {y_pred:.2f}")


#graficas
plt.figure(figsize=(10, 6))
plt.scatter(df['x1'], df['y'], color='blue', label='x1 vs y', s=100, alpha=0.6)
plt.scatter(df['x2'], df['y'], color='red', label='x2 vs y', s=100, alpha=0.6)
plt.title('Gráfica de los Datos de la Tabla 1')
plt.xlabel('Valores de x1 y x2')
plt.ylabel('Valores de y')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()

plt.show()