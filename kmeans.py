from drawing import draw
import numpy as np
import random
import csv

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
            try:
                point = [float(val) for val in row]
                data.append(point)
            except ValueError:
                print(f"Error: {row}")
                continue
    return data

# format the data for drawing
def format_data(clusters, centroids):
    formatted_data = []
    for i, cluster in enumerate(clusters):
        centroid = centroids[i]
        for point in cluster:
            formatted_row = point.copy()
            formatted_row.extend(centroid)
            formatted_data.append(formatted_row)
    return formatted_data

# Save the data to a CSV file with assigned centroids
def save_data(filename, formatted_data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z', 'centroid_x', 'centroid_y', 'centroid_z'])
        for row in formatted_data:
            writer.writerow(row)

# Define the file path
filename = 'data/3d_data.csv'

# Load the data
data = load_data(filename)

# Choose the number of clusters
k = 10

# Run k-means
clusters, centroids, average_distance = k_means(data, k)

# Print the results
print(f'Number of points: {len(data)}')
print(f'Number of clusters: {k}')
print(f'Average distance: {average_distance}')

# Format the data for drawing
formatted_data = format_data(clusters, centroids)
print(formatted_data[0])
print(len(formatted_data[0]))
# Save the data
save_data('output.csv', formatted_data)

# Draw the data
#draw(formatted_data, windowSize= 1000, offset=(0,0,0))