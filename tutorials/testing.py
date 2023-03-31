
import sys
import math
sys.path.append('src/')

from SVM import SVM 

if __name__ == "__main__":

    # rbf
    kernel = lambda x, y: math.exp(- 1.0 * (x.distance(y) ** 2))
    ## linear
    # kernel = lambda x, y: x.dot(y)
    ## poly 
    # kernel = lambda x, y: (1.0 * x.dot(y)) ** 2

    # Instantiate the SVM with a regularization parameter C and a kernel type
    svm = SVM(C=1.5, kernel=kernel)

    # Train the SVM on the sample data and labels
    svm.fit([
        [[1, 1], 1],
        [[1, 3], 1],
        [[3, 1], 1], 
        [[3, 3], 1],
        [[0, 0], -1],
        [[0, 4], -1],
        [[4, 0], -1],
        [[4, 4], -1],
        [[4, 2], -1]
    ])


    # Additional test samples
    test_samples = [
        [1, 2],
    ]

    # Predict the class for each test sample
    for sample in test_samples:
        prediction = svm.predict(sample)
        print(f"Prediction for {sample}: {prediction}")
