import sys
sys.path.append('src/')

from SVM import SVM 
import numpy as np
import matplotlib.pyplot as plt

# Define an RBF kernel function
def rbf_kernel(x, y, gamma=1.0):
    return np.exp(-gamma * (x.distance(y) ** 2))

# Generate a non-linearly separable dataset (circular pattern)
np.random.seed(42)
data = [(np.random.randn(2) * 0.3, 1) for _ in range(50)] + [(np.random.randn(2) * 1.0, -1) for _ in range(50)]

# Instantiate and train the SVM with an RBF kernel
svm = SVM(kernel=lambda x, y: rbf_kernel(x, y, gamma=1.0))
svm.fit(data)

# Visualize the data and decision boundary
plt.scatter(*zip(*[x for x, y in data if y == 1]), color='#0e89c5')
plt.scatter(*zip(*[x for x, y in data if y == -1]), color='#ff11bc')

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.array([svm.predict([x, y]) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Predict and plot new points
new_points = [
    [0.25, 0.5],
    [-0.5, -0.5],
    [-1.0, -1.0],
    [-0.3, 0],
    [1.0, 1.0],
]

for point in new_points:
    prediction = svm.predict(point)
    if prediction == 1:
        plt.scatter(*point, color='red', marker='X', s=200)
    else:
        plt.scatter(*point, color='blue', marker='X', s=200)

plt.show()