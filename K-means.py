import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = "C:/Users/s4ds/Downloads/archive/Games.csv"
df = pd.read_csv(file_path)


print(df.head())

label_encoder = LabelEncoder()
df['Review'] = label_encoder.fit_transform(df['Review'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Review', 'Score']])

scaled_df = pd.DataFrame(scaled_features, columns=['Review', 'Score'])
print(scaled_df.head())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia (WCSS)')
plt.title('Método del Codo')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_df)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

print("Etiquetas de los clústers para cada punto:", labels)
print("Centroides de los clústers:", centroids)

pca = PCA(n_components=2)
reduced_X = pd.DataFrame(pca.fit_transform(scaled_df), columns=['PCA1', 'PCA2'])

centers = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
plt.scatter(reduced_X['PCA1'], reduced_X['PCA2'], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
plt.title('Clustering de Juegos por Review y Score')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.grid(True)
plt.show()
