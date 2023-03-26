import numpy as np
import random

#Generate Numbers
data = [random.randint(1, 100000) for _ in range(300)]
data = np.array(data).reshape(-1, 1)

# Convert the data list into a numpy array
X = np.array(data)

# Set the number of clusters
k = 5

# Initialize the centroids
centroids = np.array(random.sample(list(X), k)).reshape(k, -1)
print("Selected centroids:")
for i in centroids:
    print(i)
# Keep track of the previous centroids
prev_centroids = np.zeros(centroids.shape)

# Create a list to store the cluster assignments
clusters = np.zeros(len(X))

# Loop until one of the stopping conditions is met
for iteration in range(1000000):
    # Assign each data point to the closest centroid
    for i in range(len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        cluster = np.argmin(distances)
        clusters[i] = cluster
        
    # Update the centroids
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    
    # Check if the centroids have converged
    if np.array_equal(centroids, prev_centroids):
        print("Converged after {} iterations".format(iteration+1))
        break
    # Update the previous centroids
    prev_centroids = centroids.copy()
# Print the final centroids
# Print the final centroids with labels for each cluster
print("Final centroids:")
for i, centroid in enumerate(centroids):
    print("Cluster {}: {}".format(i+1, centroid))

choice = input("\nWould you like to see the clusters? (y/n): ")
if choice == "y":
    print("\nClusters:")
    for i in range(k):
        print("\nCluster {}: {}".format(i+1, [int(x) for x in X[clusters == i]]))
    input("Press enter to exit...")
    print("Goodbye!")
else:
    print("Goodbye!")