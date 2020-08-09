from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (16, 9)
plt.style.use("ggplot")

data = pd.read_csv("xclara.csv")

# Getting the values and plotting it
f1 = data["V1"].values
f2 = data["V2"].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c="black", s=7)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def dist_1(a, b, ax=1):
    return np.sqrt(np.sum(np.power(a - b, 2), axis=ax))


# Number of clusters
k = 3
picks = np.random.choice(np.arange(len(X)), k)
C = X[picks]
print(C)


# Plotting along with the Centroids
plt.scatter(f1, f2, c="#050505", s=7)
plt.scatter(C[:, 0], C[:, 1], marker="*", s=200, c="g")

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist_1(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        C[i] = np.mean(X[clusters == i], axis=0)
    error = dist(C, C_old, None)

colors = ["r", "g", "b", "y", "c", "m"]
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker="*", s=200, c="#050505")
plt.show()
