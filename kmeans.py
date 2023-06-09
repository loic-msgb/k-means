from drawing import draw
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

#Draw the data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def draw_3D(points_list):
    # Extracting the clusters' centroids
    centroids = [(p[3], p[4], p[5]) for p in points_list]
    unique_centroids = list(set(centroids))
    num_clusters = len(unique_centroids)

    # Assigning a color to each cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

    # Creating a figure and an axes object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting each point with its corresponding color
    for i in range(num_clusters):
        points_in_cluster = [p for p in points_list if (p[3], p[4], p[5]) == unique_centroids[i]]
        ax.scatter([p[0] for p in points_in_cluster], [p[1] for p in points_in_cluster], [p[2] for p in points_in_cluster], color=colors[i])

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Showing the plot
    plt.show()


# Define the file path
filename = 'data/3d_data.csv'

# Load the data
data = load_data(filename)

# Choose the number of clusters

k_values = range(7, 11)

average_distances = []
best_clusters, best_centroids, best_average_distance = None, None, float('inf')

for k in k_values:
    # Run k-means 5 times and keep the centroids with the smallest average distance
    for i in range(3):
        clusters, centroids, average_distance = k_means(data, k)
        if average_distance < best_average_distance:
            best_clusters, best_centroids, best_average_distance = clusters, centroids, average_distance
    
    average_distances.append(best_average_distance)

# Print the results
print(f'Data shape: {np.array(data).shape}')
print(f'Number of points: {len(data)}')
print(f'Average distance: {best_average_distance}')


# Format the data for drawing
formatted_data = format_data(best_clusters, best_centroids)

# Save the data
save_data('output.csv', formatted_data)

draw_3D(formatted_data)

# Plot the average distance for each value of k
plt.plot(k_values, average_distances)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Average distance vs. number of clusters')
plt.show()

# Draw the data
#draw(formatted_data, windowSize= 1000, offset=(0,0,0))