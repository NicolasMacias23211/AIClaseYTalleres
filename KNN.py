import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

archivo = pd.read_csv("C:/Users/s4ds/Downloads/examen/Musica/Spotify Most Streamed Songs.csv")

print(archivo.head())

print(archivo.dtypes)

archivo['streams'] = pd.to_numeric(archivo['streams'], errors='coerce')

archivo['clase'] = np.where(archivo['streams'] > 100000000, 1, 0)

buenos = archivo[archivo['clase'] == 1]
malos = archivo[archivo['clase'] == 0]

plt.scatter(buenos['bpm'], buenos['danceability_%'], marker='*', s=150, color="skyblue", label="Popular (Clase: 1)")
plt.scatter(malos['bpm'], malos['danceability_%'], marker='*', s=150, color="red", label="No Popular (Clase: 0)")
plt.ylabel("Danceability")
plt.xlabel("BPM")
plt.legend(bbox_to_anchor=(1, 0.2))

datos = archivo[['bpm', 'danceability_%']]
clase = archivo['clase']

escalador = preprocessing.MinMaxScaler()
datos = escalador.fit_transform(datos)

clasificador = KNeighborsClassifier(n_neighbors=3)
clasificador.fit(datos, clase)

nuevo_solicitante = np.array([[70, 50]])
nuevo_solicitante_escalado = escalador.transform(nuevo_solicitante)

print("Clase del nuevo solicitante:", clasificador.predict(nuevo_solicitante_escalado))
print("Probabilidades por clase:", clasificador.predict_proba(nuevo_solicitante_escalado))

plt.scatter(buenos['bpm'], buenos['danceability_%'],
            marker='*', s=150, color="skyblue", label="Popular (Clase: 1)")
plt.scatter(malos['bpm'], malos['danceability_%'],
            marker='*', s=150, color="red", label="No Popular (Clase: 0)")
plt.scatter(nuevo_solicitante[0][0], nuevo_solicitante[0][1], marker="P", s=250, color="green", label="Nuevo Solicitante")
plt.ylabel("Danceability")
plt.xlabel("BPM")
plt.legend(bbox_to_anchor=(1, 0.3))

bpm = np.arange(80, 201, 1)
danceability = np.arange(0, 101, 1)
todos = pd.DataFrame(np.array(np.meshgrid(bpm, danceability)).T.reshape(-1, 2), columns=["bpm", "danceability_%"])

solicitantes = escalador.transform(todos)

clases_resultantes = clasificador.predict(solicitantes)

archivo_filtrado = archivo[(archivo['bpm'] >= 100) & (archivo['bpm'] <= 140) & (archivo['danceability_%'] > 50)]

buenos = archivo_filtrado[archivo_filtrado['clase'] == 1]
malos = archivo_filtrado[archivo_filtrado['clase'] == 0]

plt.scatter(buenos['bpm'], buenos['danceability_%'],
            marker='*', s=150, color="skyblue", label="Popular (Clase: 1)")
plt.scatter(malos['bpm'], malos['danceability_%'],
            marker='*', s=150, color="red", label="No Popular (Clase: 0)")
plt.ylabel("Danceability")
plt.xlabel("BPM")
plt.legend(bbox_to_anchor=(1, 0.2))
plt.show()
