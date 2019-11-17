import operator

import matplotlib.pyplot as plt
import numpy as np

X_train = np.genfromtxt("hw2-data/X_train.csv", delimiter=",")
y_train = np.genfromtxt("hw2-data/y_train.csv")

X_test = np.genfromtxt("hw2-data/X_test.csv", delimiter=",")
y_test = np.genfromtxt("hw2-data/y_test.csv")


def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2), axis=1))


def absolute_distance(vector1, vector2):
    return np.sum(np.absolute(vector1 - vector2), axis=1)


def KNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    for i in range(X_test.shape[0]):
        dists = absolute_distance(X_train, X_test[i])
        order = np.argsort(dists)
        output_classes.append(np.argmax(np.bincount(Y_train[order[:k]].astype(int))))
    return output_classes


def prediction_accuracy(predicted_labels, original_labels):
    return np.sum(predicted_labels == original_labels) / len(predicted_labels)


predicted_classes = {}
final_accuracies = {}
for k in range(1, 21):
    predicted_classes[k] = KNN_test(X_train, X_test, y_train, y_test, k)
    final_accuracies[k] = prediction_accuracy(predicted_classes[k], y_test)

plt.figure(figsize=(15, 6))
plt.plot(list(final_accuracies.keys()), list(final_accuracies.values()))
plt.xticks(list(final_accuracies.keys()))
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title(
    "Plot of the prediction accuracy of KNN Classifier as a function of k (Number of Neighbours)"
)
plt.show()
