import sys
sys.path.append('src/')

from SVM import SVM 
import numpy as np
import matplotlib.pyplot as plt

# Define a linear kernel function
def linear_kernel(x, y):
    return x.dot(y)

# Generate a linearly separable dataset
np.random.seed(42)
data = [(np.random.randn(2) + np.array([1, 1]), 1) for _ in range(50)] + [(np.random.randn(2) + np.array([-1, -1]), -1) for _ in range(50)]

# Instantiate and train the SVM
svm = SVM(kernel=linear_kernel)
svm.fit(data)

# Visualize the data and decision boundary
plt.scatter(*zip(*[x for x, y in data if y == 1]), color='#0e89c5')
plt.scatter(*zip(*[x for x, y in data if y == -1]), color='#ff11bc')

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.array([svm.predict([x, y]) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)

# Predict and plot some points
predicted_points = [
    [0, 0],
    [2, 2],
    [-1, -1],
    [1, -1],
    [-1, 1]
]

predicted_labels = [svm.predict(p) for p in predicted_points]

plt.scatter(*zip(*[p for p, l in zip(predicted_points, predicted_labels) if l == 1]), s=200, color='blue', marker='X')
plt.scatter(*zip(*[p for p, l in zip(predicted_points, predicted_labels) if l == -1]), s=200, color='red', marker='X')

plt.show()