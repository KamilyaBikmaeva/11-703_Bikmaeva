import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy


def distance(point1: [float, float], point2: [float, float]):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_random_data(low=1, high=100, size=(100, 2)):
    data = np.random.randint(low, high, size)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    return data


def generate_random_data():
    # Set three centers, the model should predict similar results
    center_1 = np.array([1, 1])
    center_2 = np.array([5, 5])
    center_3 = np.array([8, 1])

    # Generate random data and center it to the three centers
    data_1 = np.random.randn(200, 2) + center_1
    data_2 = np.random.randn(200, 2) + center_2
    data_3 = np.random.randn(200, 2) + center_3

    data = np.concatenate((data_1, data_2, data_3), axis=0)

    plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.show()
    return data


# Finding optimal number of clusters
def find_optimal_k(data, iterations=11):
    wcss = []
    for i in range(1, iterations):
        # kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        # kmeans.fit(data)
        # wcss.append(kmeans.inertia_)
        clusters, centers = k_means_algorithm(data, i)
        wcss.append(calculate_inertia(data, centers, clusters))
    plt.plot(range(1, iterations), wcss)
    plt.show()
    return wcss


def get_optimal_k_automatically(wcss):
    def _d(k):
        return np.abs(wcss[k] - wcss[k + 1]) / np.abs(wcss[k - 1] - wcss[k])
    d = _d(1)
    result = 1
    for i in range(2, len(wcss) - 1):
        if _d(i) < d:
            d = _d(i)
            result = i
    return result


def calculate_inertia(data, centers, clusters):
    result = 0
    for i in range(data.shape[0]):
        result += distance(data[i], centers[clusters[i]])
    return result


def generate_random_centers(data, k, c):
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, c) * std + mean
    return centers


def generate_centroids(data, k):
    r_max = 0
    mean_point = np.array([np.mean(data[:, 0]), np.mean(data[:, 1])])
    x_c = np.mean(data[:, 0])
    y_c = np.mean(data[:, 1])
    delete_me = 0
    for point in data:
        delete_me += 1
        r = distance(point, mean_point)
        if r > r_max:
            r_max = r
    x_cc = [r_max * np.cos(2 * np.pi * i / k) + x_c for i in range(k)]
    y_cc = [r_max * np.sin(2 * np.pi * i / k) + y_c for i in range(k)]
    return np.dstack((x_cc, y_cc))[0]


def k_means_algorithm(data, num_of_clusters=3):
    # Number of clusters
    k = num_of_clusters
    # Number of training data
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]

    # centers = generate_random_centers(k, c)
    centers = generate_centroids(data, k)

    # Plot the data and the centers generated as random
    # plt.scatter(data[:, 0], data[:, 1], s=7)
    # plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)
    # plt.show()

    centers_old = np.zeros(centers.shape)  # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)

    # Plot the data and the centers generated as random

    return clusters, centers_new


def plot_k_means(clusters, centers_new, data):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for point in range(data.shape[0]):
        plt.scatter(data[point][0], data[point][1], s=30, c=colors[clusters[point]])
    # plt.scatter(data[:, 0], data[:, 1], s=7)
    plt.scatter(centers_new[:, 0], centers_new[:, 1], marker='*', c='tab:pink', s=150)
    plt.show()


if __name__ == '__main__':
    # points = get_random_data()
    points = generate_random_data()
    results = find_optimal_k(points)
    print(get_optimal_k_automatically(results))
    # clusters_result, centers_result = \
    #     k_means_algorithm(data=points, num_of_clusters=int(input('Input number of clusters: ')))
    clusters_result, centers_result = \
        k_means_algorithm(data=points, num_of_clusters=get_optimal_k_automatically(results))
    plot_k_means(clusters_result, centers_result, points)
