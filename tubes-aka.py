import numpy as np
import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

# Define the iterative K-NN algorithm
def knn_iterative(driver, passengers, k):
    distances = []
    for i, p in enumerate(passengers):
        distances.append((i, np.linalg.norm(np.array(driver) - np.array(p))))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

# Define the iterative K-NN algorithm optimized with KD-Tree
def knn_iterative_kdtree(driver, passengers, k):
    tree = KDTree(passengers)
    distances, indices = tree.query([driver], k=k)
    return list(zip(indices[0], distances[0]))

# Define the recursive K-NN algorithm without KD-Tree optimization
def knn_recursive(driver, passengers, k, result=None, indices=None):
    if result is None:
        result = []
    if indices is None:
        indices = set()

    if len(result) == k:
        return result

    min_dist, min_index = float('inf'), None
    for i, p in enumerate(passengers):
        if i not in indices:
            dist = np.linalg.norm(np.array(driver) - np.array(p))
            if dist < min_dist:
                min_dist, min_index = dist, i

    result.append((min_index, min_dist))
    indices.add(min_index)
    return knn_recursive_simple(driver, passengers, k, result, indices)

# Define the recursive K-NN algorithm with KD-Tree optimization
def knn_recursive_kdtree(driver, passengers, k, tree=None, result=None, indices=None):
    if result is None:
        result = []
    if indices is None:
        indices = set()
    if tree is None:
        tree = KDTree(passengers)

    if len(result) == k:
        return result

    distances, indices_tree = tree.query([driver], k=k)
    for idx, dist in zip(indices_tree[0], distances[0]):
        if idx not in indices:
            result.append((idx, dist))
            indices.add(idx)
            break

    return knn_recursive_kdtree(driver, passengers, k, tree, result, indices)

# Generate realistic passenger data (simulated geographic coordinates)
def generate_passengers(size):
    return np.random.uniform(low=[-180, -90], high=[180, 90], size=(size, 2))  # Longitude and latitude

# Measure execution time for different input sizes
input_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]
iterative_times = []
iterative_optimized_times = []
recursive_simple_times = []
recursive_kdtree_times = []
driver_location = (50, 50)  # Example location (latitude, longitude)

for size in input_sizes:
    passengers = generate_passengers(size)
    k = min(5, size)  # Number of nearest neighbors

    # Measure time for iterative K-NN
    start_time = time.time()
    knn_iterative(driver_location, passengers, k)
    iterative_times.append(time.time() - start_time)

    # Measure time for optimized iterative K-NN
    start_time = time.time()
    knn_iterative_kdtree(driver_location, passengers, k)
    iterative_optimized_times.append(time.time() - start_time)

    # Measure time for recursive K-NN without KD-Tree optimization
    start_time = time.time()
    knn_recursive(driver_location, passengers, k)
    recursive_simple_times.append(time.time() - start_time)

    # Measure time for recursive K-NN with KD-Tree optimization
    start_time = time.time()
    knn_recursive_kdtree(driver_location, passengers, k)
    recursive_kdtree_times.append(time.time() - start_time)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(input_sizes, iterative_times, label="Iterative K-NN", marker='o')
plt.plot(input_sizes, iterative_optimized_times, label="Iterative K-NN (KD-Tree)", marker='x')
plt.plot(input_sizes, recursive_simple_times, label="Recursive K-NN", marker='s')
plt.plot(input_sizes, recursive_kdtree_times, label="Recursive K-NN (KD-Tree)", marker='^')
plt.xlabel("Input Size (Number of Passengers)")
plt.ylabel("Execution Time (Seconds)")
plt.title("Execution Time vs Input Size for Iterative and Recursive K-NN")
plt.legend()
plt.grid()
plt.show()

# Print running times for each approach
print("\nRunning Times:")
for size, iter_time, iter_opt_time, recur_simple_time, recur_kdtree_time in zip(input_sizes, iterative_times, iterative_optimized_times, recursive_simple_times, recursive_kdtree_times):
    print(f"Input Size: {size:5d} | Iterative: {iter_time:.6f}s | Iterative (KD-Tree): {iter_opt_time:.6f}s | Recursive: {recur_simple_time:.6f}s | Recursive (KD-Tree): {recur_kdtree_time:.6f}s")
    