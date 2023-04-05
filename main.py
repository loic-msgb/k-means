from drawing import draw
import numpy as np
import random

#lire des données de fichier csv
def readData(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            #sauter la première ligne
            if line[0] == 'x':
                continue
            data.append([float(x) for x in line.split(',')])
    return data
data = np.genfromtxt('data/mock_2d_data.csv', delimiter=',')
data = np.nan_to_num(data, nan=0, posinf=1e9, neginf=-1e9)
print(data.shape)


#initialiser des centroides aléatoires
k = 4  # nombre de clusters à identifier

centroids = data[np.random.choice(range(len(data)), size=k, replace=False), :]
# Initialisation
old_centroids = np.zeros_like(centroids)

# Tant que les centroïdes continuent à changer
while not np.array_equal(old_centroids, centroids):
    # Affecter chaque point au centroïde le plus proche
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)

    # Mettre à jour les centroïdes en prenant la moyenne des points dans chaque cluster
    old_centroids = np.copy(centroids)
    for i in range(k):
        centroids[i] = np.mean(data[clusters == i], axis=0)

# Concaténer les données avec les coordonnées des centroids
result = np.hstack((data, centroids[clusters]))

draw(result, windowSize=1000, offset=(0, 0, 0))