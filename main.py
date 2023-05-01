from drawing import draw
import csv
from fileinput import filename
import random
import numpy as np

# Define the k-means function
def k_means(data, k):
    # Initialize centroids randomly
    centroids = np.array(random.sample(data, k))

    while True:
        # Assign each point to the closest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = np.linalg.norm(point - centroids, axis=1)
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        # Calculate new centroids
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

        # Stop if the centroids haven't moved
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Calculate the average distance from points to their assigned centroids
    distances = []
    for i, cluster in enumerate(clusters):
        for point in cluster:
            distance = np.linalg.norm(point - centroids[i])
            distances.append(distance)

    average_distance = sum(distances) / len(distances)

    return clusters, centroids, average_distance

# Load the data from a CSV file
def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the first line (header)
        for row in reader:
            if len(row) == 2:
                data.append([float(row[0]), float(row[1])])
            elif len(row) == 3:
                data.append([float(row[0]), float(row[1]), float(row[2])])
            else:
                data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    return data

# Save the data to a CSV file with assigned centroids
def save_data(filename, data, centroids):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'centroid_x', 'centroid_y'])
        for i, point in enumerate(data):
            centroid = centroids[i % len(centroids)]
            row = point[:2] + centroid[:2]
            writer.writerow(row)
            affichage.append(row)
# Define the file path
filename = 'data/2d_data.csv'

# Load the data
data = load_data(filename)

# Choose the number of clusters
k = 4 if 'mock' in filename else 10

# Run k-means clustering
clusters, centroids, average_distance = k_means(data, k)

# Print the average distance
print('Average distance:', average_distance)

# Save the data with assigned centroids
affichage = []
if len(data[0]) == 2:
    save_data('output.csv', data, centroids)
else:
    save_data('output.csv', data[:,:3], centroids)

draw(affichage)