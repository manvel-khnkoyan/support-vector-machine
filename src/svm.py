
from Vector import Vector

# Define the SVM class
class SVM:
    def __init__(self, C=1.0, kernel=None):
        self.C = C
        self.data = None
        self.y = None
        self.alpha = None
        self.bias = 0
        self.kernel = kernel

        self.s_v = [] # support vectors
        self.s_y = [] # support vectors labels
        self.s_a = [] # support vectors alphas

    # get langragian 
    def lagrangian(self, data):
        n = len(self.alpha)
        return sum(self.alpha) - 0.5 * sum(self.alpha[i] *self. alpha[j] * self.y[i] * self.y[j] * self.kernel(data[i], data[j]) for i in range(n) for j in range(n))

    # Fits the SVM model
    def fit(self, data, iterations=1000, step=0.01):
        self.data = [Vector(x) for x, y in data]
        self.y = [y for x, y in data]
        self.alpha = [0] * len(data)

        # Main loop for training the SVM using coordinate descent
        chunk = step/(len(data) - 1)
        max_lagrangian = 0
        for _ in range(iterations):
            for i in range(len(data)):
                alfa_prev = self.alpha[i]

                self.alpha[i] = alfa_prev + step
                for j in range(len(data)):
                    if i != j:
                        self.alpha[j] -= self.y[j] * chunk

                lang = self.lagrangian(self.data)
                if lang > max_lagrangian:
                    max_lagrangian = lang
                    break

                self.alpha[i] = alfa_prev - step
                for j in range(len(data)):
                    if i != j:
                        self.alpha[j] += self.y[j] * chunk    

                lang = self.lagrangian(self.data)
                if lang > max_lagrangian:
                    max_lagrangian = lang
                    break

                self.alpha[i] = alfa_prev

        # Find Suupport vectors        
        for i in range(len(self.alpha)):
            if 0 < self.alpha[i] < self.C:
                self.s_v.append(self.data[i])
                self.s_y.append(self.y[i])
                self.s_a.append(self.alpha[i])

        # Calculate bias
        self.bias = sum(self.y[i] - self.decision_function(self.data[i]) for i in range(len(data))) / len(data)

    # The decision function
    def decision_function(self, x):
        return sum(alpha_i * y_i * self.kernel(x_i, x) for alpha_i, y_i, x_i in zip(self.s_a, self.s_y, self.s_v)) + self.bias

    # Predicts the class for a new sample
    def predict(self, x):
        return 1 if self.decision_function(Vector(x)) > 0 else -1

